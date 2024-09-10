import numpy as np

from src.training.federated.environments.bases import CentralizedEnvironment
from src.training.clusters import calculate_clusters
from src.training.federated.clusters import get_cluster_class
from src.data.fingerprint import create_distribution_fingerprints, calculate_fingerprint_similarity        

class FederatedHierarchicalEnvironment(CentralizedEnvironment):
    def __init__(self, fed_dataset, config, *args, **kwargs):
        print('### Initializing FederatedHierarchicalEnvironment (START) ###')
        super().__init__(fed_dataset, config, *args, **kwargs)
        self.cluster_class = get_cluster_class('FedAvgCluster') if self.config.training.clusters.cluster_class is None else get_cluster_class(self.config.training.clusters.cluster_class)
        print(f'Using {self.cluster_class.__name__} as cluster class.')
        self.clusters, self.client_cluster_ids = self.setup_client_clusters()
        print('### Initializing FederatedHierarchicalEnvironment (END) ###')


    def setup_client_clusters(self, *args, **kwargs):
        print('### Setting up client clusters (START) ###')
        cluster_indices = self.config.training.clusters.cluster_indices
        if cluster_indices is None or len(cluster_indices) < 1:
            print('No cluster indices provided. Creating cluster indices based on different cases...')
            #handle edge cases:
            if self.config.training.clusters.cluster_n == 1:
                print('Only one cluster requested. Creating one cluster with all clients.')
                cluster_indices = [[i for i in range(self.num_clients)]]
            elif self.config.training.clusters.cluster_n == self.num_clients:
                print('Number of clusters requested equals number of clients. Creating one cluster per client (isolated training).')
                cluster_indices = [[i] for i in range(self.num_clients)]
            else:
                #create fingerprint based clusters
                print(f'Creating clusters based on fingerprints...')
                ## 1. create (load if exists) fingerprints for all clients
                client_FPs_dict = create_distribution_fingerprints({client_id: client.model_trainer.trn_data for client_id, client in self.clients.items()}, self.data_fingerprint_path, self.config.data_distribution_config.fingerprint.generator_mode, self.config.data_distribution_config.fingerprint.data_mode, self.config.data_distribution_config.fingerprint.feature_extractor, self.config.data_distribution_config.fingerprint.fe_batch_size, self.config.data_distribution_config.fingerprint.reload)
                ## 2. calculate similarity matrix based on fingerprint
                similarity_matrix, _ = calculate_fingerprint_similarity(client_FPs_dict, self.config.training.clusters.clusterer.similarity_calculator)
                ## 3. cluster clients based on similarity matrix
                self.config.training.clusters.clusterer.cluster_n = self.config.training.clusters.cluster_n
                cluster_indices, _ = calculate_clusters(similarity_matrix, self.config.training.clusters.clusterer)
                
        assert len(cluster_indices) > 0, 'Empty cluster!'
        nc = sum([len(l) for l in cluster_indices]) # count number of clients deployed between all clusters
        assert nc == self.num_clients, f'Number of clients in clusters ({nc}) does not match number of clients in dataset ({self.num_clients})!'
        print(f'Deploying {len(cluster_indices)} clusters with client_id grouping: {cluster_indices}')
        if isinstance(self.client_id_map[0], str):
            print(f'This corresponds to the following client_id grouping: {[[self.client_id_map[client_id] for client_id in cluster] for cluster in cluster_indices]}')

        clusters = {}
        client_cluster_ids = {}
        for cluster_id, client_ids in enumerate(cluster_indices):
            clusters[cluster_id] = self.cluster_class(cluster_id, [self.clients[self.client_id_map[client_id]] for client_id in client_ids], self.config)
            for client_id in client_ids:
                client_cluster_ids[self.client_id_map[client_id]] = cluster_id
        print('### Setting up client clusters (END) ###')
        return clusters, client_cluster_ids

    def save(self, path=None, best=False):
        super().save(path, best)
        if best: #only save cluster models if best model is saved
            for cluster in self.clusters.values():
                cluster.save(best=best)

    def load(self, path=None, best=False):
        super().load(path, best)
        for cluster in self.clusters.values():
            cluster.load(best=best)
    
    def train(self, communication_rounds=None, *args, **kwargs):
        return super().train(communication_rounds, *args, **kwargs)
    
    def global_train_step(self, *args, **kwargs):
        cluster_weights = []
        for cluster_id, cluster in self.clusters.items():
            w_cluster = cluster.train(self.global_model, self.current_communication_round)
            self.total_num_exchanged_models += 2 + cluster.one_round_num_exchanged_models
        
            if self.config.training.weigh_sample_quantity:
                cluster_weights.append((cluster.get_sample_number('TRN'), w_cluster))
            else:
                cluster_weights.append((1, w_cluster))
            self.global_history[(f'TotalExchangedModels_CLUSTER{cluster_id}', self.current_communication_round)] = cluster.num_exchanged_models_clients
            
        #update_global_weights
        self.global_model = self.aggregate(cluster_weights)

        self.global_history[('TotalExchangedModels', self.current_communication_round)] = self.total_num_exchanged_models

    #do basic evaluation (in this general case every client evaluates on the global model)
    #also capture the cluster-wise evaluation metrics
    #and handle cluster-wise early stopping
    def evaluate(self, split='VAL', *args, **kwargs):
        client_metrics, client_samples, client_ids = super().evaluate(split, *args, **kwargs)
        for cluster in self.clusters.values():
            if cluster.early_stopper is None or (cluster.early_stopper and not cluster.early_stopper.stop):
                weights_mask = np.array([num_samples if _id in cluster.clients.keys() else 0 for _id, num_samples in zip(client_ids, client_samples)])
                for metric, values in client_metrics.items():
                    if metric in ['CommunicationRound', 'LocalEpoch', 'OverallTrainedEpochs']:
                        continue
                    self.global_history[(metric+f'_CLUSTER{cluster.cluster_id}', cluster.total_cluster_communication_rounds)] = np.average(values, weights=weights_mask)
        return client_metrics, client_samples, client_ids

    def set_final_evaluation(self):
        super().set_final_evaluation()
        for cluster in self.clusters.values():
            cluster.total_cluster_communication_rounds = -1
            if cluster.early_stopper: #to make sure the above eval will be executed
                cluster.early_stopper.reset()