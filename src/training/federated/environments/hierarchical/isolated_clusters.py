from src.training.federated.environments.hierarchical.bases import FederatedHierarchicalEnvironment

class IsolatedClustersEnvironment(FederatedHierarchicalEnvironment):
    def __init__(self, fed_dataset, config, *args, **kwargs):
        print('### Initializing IsolatedClustersEnvironment (START) ###')
        super().__init__(fed_dataset, config, *args, **kwargs)
        print('### Initializing IsolatedClustersEnvironment (END) ###')

    def global_train_step(self, *args, **kwargs):
        for cluster_id, cluster in self.clusters.items():
            if not cluster.early_stopper.stop:
                cluster.train(cluster.cluster_model, self.current_communication_round) #don't do any global aggregation of cluster models. Keep it "cluster local"
                self.total_num_exchanged_models += cluster.one_round_num_exchanged_models
            self.global_history[(f'TotalExchangedModels_CLUSTER{cluster_id}', self.current_communication_round)] = cluster.num_exchanged_models_clients

        self.global_history[('TotalExchangedModels', self.current_communication_round)] = self.total_num_exchanged_models

        if all([cluster.early_stopper.stop for cluster in self.clusters.values()]):
            print('### All clusters are done training. Let global early stopping trigger ###')
            self.early_stopper.stop = True
    
    #make sure that for evaluation every client gets the "cluster local" model state
    def get_client_eval_model_state(self, client, *args, **kwargs):
        cluster_id = self.client_cluster_ids[client.client_id]
        cluster = self.clusters[cluster_id]
        return cluster.cluster_model

    def evaluate(self, split='VAL', *args, **kwargs):
        client_metrics, client_samples, client_ids = super().evaluate(split, *args, **kwargs)
        if self.current_communication_round > 0:
            #EarlyStopping
            for cluster_id, cluster in self.clusters.items():
                if not cluster.early_stopper.stop:
                    cluster.early_stopper(self.global_history[(cluster.early_stopper.metric+f'_CLUSTER{cluster_id}', cluster.total_cluster_communication_rounds)])
                    if cluster.early_stopper.improved:
                        cluster.save(best=True)
                    if cluster.early_stopper.stop:
                        print(f'Early stopping the global federated training since we had no improvement of {cluster.early_stopper.metric} for {cluster.early_stopper.patience} rounds. Training of cluster {cluster.cluster_id} was stopped after {cluster.total_cluster_communication_rounds} CommunicationRounds.')
        return client_metrics, client_samples, client_ids

    #in this case there is no global model to be saved. and cluster model saves are only saved when triggered via early stopping during eval
    #hence only save histories
    def save(self, path=None, best=False):
        self.save_histories()

    #when loading only load best cluster models and client models
    def load(self, path=None, best=False):
        for cluster in self.clusters.values():
            cluster.load(best=best)
        if self.was_tuned:
            for client in self.clients.values():
                client.load(best=best)
    