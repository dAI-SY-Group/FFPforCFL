import psutil
p = psutil.Process()
p.cpu_affinity([i for i in range(psutil.cpu_count())]) # Set CPU affinity to all cores (llvm-openmp package on version 16.0.3 weirdly sets cpu affinity only to two cores when starting subprocesses. Just to make sure all CPUs are utilized...)
#print(p.cpu_affinity())

import argparse
import numpy as np

from src.toolkit.config import build_config
from src.toolkit.history import History
from src.toolkit.system import set_seeds, system_startup

from src.data.datazoo import get_dataloaders
from src.data.datazoo import get_single_class_tst_dataloaders
from src.data.utils import get_unique_targets
from src.models.ModelCapsule import ModelCapsule
from src.training.federated.environments import get_federated_environment



class FederatedTrainingExperiment(object):
    """A class representing a federated training experiment.

    Args:
        config_file (str): Path to the configuration file.
        debug (bool, optional): Indicates whether debug mode is enabled. Defaults to False.

    Attributes:
        eval_history_path (str): Path to the evaluation history file.
        eval_history (History): The evaluation history.
        fed_data (tuple): A tuple containing training, testing, and validation data loaders.
        config (Config): The configuration object.
        model: The machine learning model.
        trainer: The trainer for the machine learning model.
        fed_environment: The federated environment for training.

    Example:
        >>> federated_exp = FederatedTrainingExperiment('config.yaml', debug=True)
        >>> federated_exp.start()
    """
    def __init__(self, config_file, debug=False):
        self.config = build_config(config_file, debug)
        
        self.config.device = system_startup()
        print(f'Experiment name: {self.config.experiment_name}')
        set_seeds(self.config.seed)
    
        self.eval_history_path = self.config.history_path + self.config.experiment_name + '_EVAL'
        if self.config.debug:
            self.eval_history_path += '_debug'
        self.eval_history_path += '.history'
        self.eval_history = History(['Metric', 'ModelSource', 'DataPartition'], savefile=self.eval_history_path)

        self.fed_data = get_dataloaders(self.config.data, True, self.config)
        self.config.data.use_val = list(self.fed_data.val_loaders.values())[0] is not None
        
        self.fed_environment = get_federated_environment(self.config.training.environment, self.fed_data, self.config)

    def start(self):
        print(f'Starting {self.__class__.__name__} {self.config.experiment_name}...')
        self.fed_environment.train()
        self.fed_environment.tune()

    def test(self):
        """
        Run the test phase of the federated training experiment.

        This method evaluates the trained models on the test datasets and saves the evaluation results.

        Specifically, it performs the following steps:

        1. Loads the best global model and evaluation histories.
        2. Sets up the final evaluation mode. (Communication round -1)
        3. Evaluates the models on the training, validation, and test datasets.
        4. Saves the evaluation histories.

        Additionally, if the experiment involves clusters:
        - For each cluster:
            - Loads the best cluster model and relevant metrics.
            - Evaluates the cluster model on cluster-specific test data.
            - Aggregates and records the results.

        Finally, if considering global metrics:
        - Evaluates the best global model on the entire test dataset.
        - Evaluates each client on all available test data for the classes the client had access to during training.

        Note:
            This method assumes that the best global model and evaluation histories can be properly loaded, i.e. that the experiment has been executed and the models are trained.
        """
        print(f'Testing {self.__class__.__name__} {self.config.experiment_name}...')
        print('Preparing test data for experiment evaluation. (START)')

        #prepare global test_data
        if self.config.eval.global_test_data: 
            print('Preparing global test data.')
            global_test_data = []
            for client_id, client in self.fed_environment.clients.items():
                global_test_data.append(client.model_trainer.tst_data)

        #prepare cluster test_data
        if self.config.eval.cluster_test_data:
            print('Preparing cluster test data.')
            all_cluster_test_data = {}
            for cluster_id, cluster in self.fed_environment.clusters.items():
                tmp = []
                for client_id, client in cluster.clients.items():
                    tmp.append(client.model_trainer.tst_data)
                all_cluster_test_data[cluster_id] = tmp
        
        #prepare available_classes_test_data
        if self.config.eval.available_classes_test_data:
            print('Preparing available classes test data.')
            available_class_dataloaders = {}
            #case 1: One dataset was split for multiple Clients
            if self.config.data.client_list is None:
                self.classes_are_unambiguous = True
                dataloaders = get_single_class_tst_dataloaders(self.config) #dict with key: class, value: dataloader for that class
                available_class_dataloaders = dataloaders
            
            #case 2: Multiple datasets were split for multiple Clients
            else:
                self.classes_are_unambiguous = False
                for client in self.config.data.source_client_list:
                    tmp_config = self.config.copy()
                    tmp_config.data.dataset = client
                    dataloaders = get_single_class_tst_dataloaders(tmp_config) #dict with key: class, value: dataloader for that class
                    available_class_dataloaders[client] = dataloaders #dict with key: dataset, value: dict with key: class, value: dataloader for that class

        print('Preparing test data for experiment evaluation. (END)')      

        eval_metrics = list(self.fed_environment.clients.values())[0].model_trainer.metrics
        
        print(f'Loading the best achieved global model and evaluate all datasets for communication round -1 (best).')
        self.best_metrics = {}   
        self.fed_environment.load_best()
        self.fed_environment.load_histories()
        self.fed_environment.set_final_evaluation()

        if self.config.eval.local_splits: #test most specific model on local splits; i.e. local when tuned, cluster when clustered, global when not clusters or tuning
            self.fed_environment.evaluate('TRN')
            if self.config.data.use_val:
                self.fed_environment.evaluate('VAL')
            self.fed_environment.evaluate('TST')
            self.fed_environment.save_histories()

        if self.config.eval.global_model and self.config.training.trainer != 'PFLTrainer':
            print('Evaluating global model. (START)')
            model = list(self.fed_environment.clients.values())[0].model_trainer.model
            model.load_state_dict(self.fed_environment.global_model)
            
            if self.config.eval.global_test_data:
                print('Evaluating global model on global test data.')
                test_results = model.test(global_test_data, eval_metrics, self.config.device, verbose=False)
                self.update_eval_history(test_results, 'TST', 'GLOBAL', 'ALL')

            if self.config.eval.cluster_test_data:
                print('Evaluating global model on clusterwise test data.')
                for cluster_id, cluster_test_data in all_cluster_test_data.items():
                    test_results = model.test(cluster_test_data, eval_metrics, self.config.device, verbose=False)
                    self.update_eval_history(test_results, 'TST', 'GLOBAL', f'CLUSTER_{cluster_id}')
            
            if self.config.eval.available_classes_test_data:
                print('SKIP evaluating global model on available classes test data since all classes are available for the global model.')
                pass

            if self.config.eval.local_splits:
                print('Evaluating global model on local splits.')
                for client_id, client in self.fed_environment.clients.items():
                    test_results = model.test(client.model_trainer.tst_data, eval_metrics, self.config.device, verbose=False)
                    self.update_eval_history(test_results, 'TST', 'GLOBAL', f'CLIENT_{client_id}')

            print('Evaluating global model. (END)')
        elif self.config.training.trainer == 'PFLTrainer':
            print('Global model evaluation is not supported for PFLTrainer. SKIPPING')
        else:
            print('Global model evaluation is disabled.')

        if self.config.eval.cluster_models and self.config.trainer != 'PFLTrainer':
            print('Evaluating cluster models. (START)')

            for cluster_id, cluster in self.fed_environment.clusters.items():
                print(f'Evaluating cluster model for cluster {cluster_id}. (START)')
                cluster.load(best=True)
                model = list(self.fed_environment.clients.values())[0].model_trainer.model
                model.load_state_dict(cluster.cluster_model)

                if self.config.eval.global_test_data:
                    print(f'Evaluating cluster model for cluster {cluster_id} on global test data.')
                    test_results = model.test(global_test_data, eval_metrics, self.config.device, verbose=False)
                    self.update_eval_history(test_results, 'TST', f'CLUSTER_{cluster_id}', 'ALL')
                
                if self.config.eval.cluster_test_data:
                    print(f'Evaluating cluster model for cluster {cluster_id} on cluster test data.')
                    test_results = model.test(all_cluster_test_data[cluster_id], eval_metrics, self.config.device, verbose=False)
                    self.update_eval_history(test_results, 'TST', f'CLUSTER_{cluster_id}', f'CLUSTER')
                
                if self.config.eval.available_classes_test_data and self.classes_are_unambiguous:
                    print(f'Evaluating cluster model for cluster {cluster_id} on available classes in this cluster.')
                    client_unique_targets = []
                    for client_id, client in cluster.clients.items():
                        client_unique_targets.extend(get_unique_targets(client.model_trainer.trn_data))
                    cluster_unique_targets = np.unique(client_unique_targets)
                    active_classes_test_data = [available_class_dataloaders[target] for target in cluster_unique_targets]
                    
                    test_results = model.test(active_classes_test_data, eval_metrics, self.config.device, verbose=False)
                    self.update_eval_history(test_results, 'TST', f'CLUSTER_{cluster_id}', f'AVAILABLE_CLASSES')
                
                if self.config.eval.local_splits:
                    print(f'Evaluating cluster model for cluster {cluster_id} on local splits.')
                    for client_id, client in cluster.clients.items():
                        test_results = model.test(client.model_trainer.tst_data, eval_metrics, self.config.device, verbose=False)
                        self.update_eval_history(test_results, 'TST', f'CLUSTER_{cluster_id}', f'CLIENT_{client_id}')

                print(f'Evaluating cluster model for cluster {cluster_id}. (END)')
            print('Evaluating cluster models. (END)')
        elif self.config.training.trainer == 'PFLTrainer':
            print('Cluster model evaluation is not supported for PFLTrainer. SKIPPING')
        else:
            print('Cluster model evaluation is disabled.')

        if self.config.eval.local_models:
            print('Evaluating local models. (START)')

            for client_id, client in self.fed_environment.clients.items():
                print(f'Evaluating local model for client {client_id}. (START)')
                client.load(best=True)
                #if we do PFL the model consists of the global feature extractor and the local model head
                if self.config.training.trainer == 'PFLTrainer' and client.model_trainer.personal_model_head: 
                    model = ModelCapsule([client.model_trainer.model, client.model_trainer.personal_model_head])
                else:
                    model = client.model_trainer.model

                if self.config.eval.global_test_data:
                    print(f'Evaluating local model for client {client_id} on global test data.')
                    test_results = model.test(global_test_data, eval_metrics, self.config.device, verbose=False)
                    self.update_eval_history(test_results, 'TST', f'CLIENT_{client_id}', 'ALL')

                if self.config.eval.cluster_test_data:
                    corresponding_cluster_id = self.fed_environment.client_cluster_ids[client_id]
                    print(f'Evaluating local model for client {client_id} on cluster test data of corresponding cluster {corresponding_cluster_id}.')
                    test_results = model.test(all_cluster_test_data[corresponding_cluster_id], eval_metrics, self.config.device, verbose=False)
                    self.update_eval_history(test_results, 'TST', f'CLIENT_{client_id}', f'CLUSTER')

                if self.config.eval.available_classes_test_data:
                    print(f'Evaluating local model for client {client_id} on available classes for this client.')
                    client_unique_targets = get_unique_targets(client.model_trainer.trn_data)
                    if self.classes_are_unambiguous:
                        active_classes_test_data = [available_class_dataloaders[target] for target in client_unique_targets]
                    else:
                        ds_name = client_id.split('_')[0]
                        active_classes_test_data = [available_class_dataloaders[ds_name][target] for target in client_unique_targets]
                    
                    test_results = model.test(active_classes_test_data, eval_metrics, self.config.device, verbose=False)
                    self.update_eval_history(test_results, 'TST', f'CLIENT_{client_id}', f'AVAILABLE_CLASSES')

                if self.config.eval.local_splits:
                    print(f'Evaluating local model for client {client_id} on local splits.')
                    trn_results = model.test(client.model_trainer.trn_data, eval_metrics, self.config.device, verbose=False)
                    self.update_eval_history(trn_results, 'TRN', f'CLIENT_{client_id}', 'CLIENT')
                    if self.config.data.use_val:
                        val_results = model.test(client.model_trainer.val_data, eval_metrics, self.config.device, verbose=False)
                        self.update_eval_history(val_results, 'VAL', f'CLIENT_{client_id}', 'CLIENT')
                    test_results = model.test(client.model_trainer.tst_data, eval_metrics, self.config.device, verbose=False)
                    self.update_eval_history(test_results, 'TST', f'CLIENT_{client_id}', 'CLIENT')

                print(f'Evaluating local model for client {client_id}. (END)')
            self.aggregate_metrics('CLIENT')
            print('Evaluating local models. (END)')
        else:
            print('Local model evaluation is disabled.')
    
        #History merging and accumulation to relevant metrics
        print('Evaluating further global metrics. (START)')
        mgh = self.fed_environment.global_history
        global_best_cr = mgh.df.query(f'Metric == "VAL_{self.config.training.loss}_L" and CommunicationRound > 0').sort_values(by='Value').head(1).CommunicationRound.values[0]
        global_tem = mgh.df.query(f'Metric == "TotalExchangedModels" and CommunicationRound == {global_best_cr}').Value.values[0]

        if self.config.training.environment_name in ['HierarchicalEnvironment', 'IsolatedClustersEnvironment']:
            if self.config.training.environment_name == 'IsolatedClustersEnvironment':
                global_tem = 0
            for cluster_id, cluster in self.fed_environment.clusters.items():
                cluster_best_cr = mgh.df.query(f'Metric == "VAL_{self.config.training.loss}_L_CLUSTER_{cluster_id}" and CommunicationRound > 0').sort_values(by='Value').head(1).CommunicationRound.values[0]
                cluster_tem = mgh.df.query(f'Metric == "TotalExchangedModels_CLUSTER_{cluster_id}" and CommunicationRound == {cluster_best_cr}').Value.values[0]
                self.eval_history[('TotalExchangedModels', f'CLUSTER_{cluster_id}', 'CLUSTER')] = cluster_tem
                self.eval_history[('CommunicationRound', f'CLUSTER_{cluster_id}', 'CLUSTER')] = cluster_best_cr
                if self.config.training.environment_name == 'IsolatedClustersEnvironment':
                    global_tem += cluster_tem
    

        self.best_metrics['TotalExchangedModels'] = global_tem
        self.eval_history[('TotalExchangedModels', 'GLOBAL', 'ALL')] = global_tem
        self.best_metrics['CommunicationRound'] = global_best_cr
        self.eval_history[('CommunicationRound', 'GLOBAL', 'ALL')] = global_best_cr

        print('Evaluating further global metrics. (END)')

        self.eval_history.save()
        print(f'Finished experiment evaluation. History saved to {self.eval_history_path}')

    def aggregate_metrics(self, model_source):
        mean_metrics = None
        std_metrics = None
        if model_source == 'CLIENT':
            mean_metrics = self.eval_history.df.query(f'ModelSource.str.contains("CLIENT")').drop(columns=['ModelSource']).groupby(['Metric', 'DataPartition']).mean().reset_index()
            mean_metrics['Metric'] = mean_metrics['Metric'] + '_Mean'
            std_metrics = self.eval_history.df.query(f'ModelSource.str.contains("CLIENT")').drop(columns=['ModelSource']).groupby(['Metric', 'DataPartition']).std().reset_index()
            std_metrics['Metric'] = std_metrics['Metric'] + '_Std'
        else: 
            print('This model source metric aggregation is not yet supported.')
            return
        if mean_metrics is not None:
            for index, row in mean_metrics.iterrows():
                self.best_metrics[f'{row.Metric}'] = row.Value
                self.eval_history[(row.Metric, model_source, row.DataPartition)] = row.Value
        if std_metrics is not None:
            for index, row in std_metrics.iterrows():
                self.best_metrics[f'{row.Metric}'] = row.Value
                self.eval_history[(row.Metric, model_source, row.DataPartition)] = row.Value


    def update_eval_history(self, test_results, split, model_source, data_partition):
        """
        Update the evaluation history with the given results.

        Args:
            test_results (dict): A dictionary containing the evaluation results.
            split (str): The data split the results were obtained on.
            model_source (str): The source of the model.
            data_partition (str): The data partition the results were obtained on.
        """
        for metric, value in test_results.items():
            self.eval_history[(f'{split}_{metric}', model_source, data_partition)] = value

    def summary(self):
        self.fed_environment.summary()
        if self.config.eval.local_models:
            print(f'Best achieved metrics after {self.best_metrics["TotalExchangedModels"]} exchanged models:')
            for metric in self.config.training.metrics:
                if metric == 'SKAccuracy': 
                    metric = 'Accuracy'
                print(f'Test {metric}: {self.best_metrics["TST_"+metric+"_Mean"]:.4f}+-{self.best_metrics["TST_"+metric+"_Std"]:.4f}')

    def shutdown(self):
        print(f'Shutting down {self.__class__.__name__} {self.config.experiment_name}!')
        print('DONE!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run a federated training experiment.')
    parser.add_argument('config', type=str, help='Path to the configuration file.')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode.')

    args = parser.parse_args()

    ftx = FederatedTrainingExperiment(args.config, args.debug)
    ftx.start()
    ftx.test()
    ftx.summary()
    ftx.shutdown()