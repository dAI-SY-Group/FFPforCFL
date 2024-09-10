from abc import ABC, abstractmethod
from collections import defaultdict
import os

import numpy as np
import torch

from src.toolkit.history import History
from src.training.utils.early_stopping import EarlyStopping
from src.training.federated.clients import get_client_class
from src.models.modelzoo import get_model

class FederatedEnvironment(ABC):
    """
    Abstract base class for federated learning environments.

    This class defines the core functionality and structure for distributed learning environments.
    Subclasses must implement the `train` and `aggregate` methods.

    Args:
        fed_dataset (tuple): A tuple containing local training, testing, and validation dataloaders (that are dicts with a structure of client_id: local_split_dataloader).
        config (object): Configuration object containing environment and training settings.

    Attributes:
        config (object): Configuration object containing environment and training settings.
        debug (bool): Flag indicating whether debugging mode is enabled.
        client_class (type): Class for creating client instances. Defaults to 'Client'.
        global_model (object): Global model state (weight parameters).
        total_num_exchanged_models (int): Total number of exchanged models.
        scheduler (None or object): LR scheduler for training.
        early_stopper (None or object): Early stopping criteria.
        file_identifier (str): Identifier for saving files, i.e. histories, checkpoints, dumps ect.
        global_history (object): History object for tracking global metrics.
        checkpoint_path (str): Path for saving checkpoints.
        current_communication_round (int): Current communication round.
        was_tuned (bool): Flag indicating whether the clients have executed local tuning.

    Methods:
        train(*args, **kwargs): Train the federated environment. Must be implemented by subclasses.
        get_client_eval_model_state(client, *args, **kwargs): 
            Get the evaluation model state that is to be evaluated on local client data. 
            Per default: global model state. 
            If model should only be evaluated on local data, latest local client state.

        evaluate(split='VAL', *args, **kwargs): 
            Evaluate the federated environment.
            Iterates over all clients and lets them evaluate a model state given by 'get_client_eval_model_state' on their local data 'split'.

        aggregate(local_weights, *args, **kwargs): Aggregate local weights. Must be implemented by subclasses.

        log_status(*args, **kwargs): Log the current metrics to show the status of the federated environment.
    """
    def __init__(self, fed_dataset, config, *args, **kwargs):
        super().__init__()
        assert config.training.glob is not None, f'You need to provide config.training.glob for a FederatedEnvironment! Got {config.training.glob}'
        print('### Initializing FederatedEnvironment (START) ###')
        self.config = config
        self.debug = self.config.debug
        self.client_class = get_client_class('Client') if self.config.training.client_class is None else get_client_class(self.config.training.client_class)
        print(f'Using {self.client_class.__name__} as client class.')
        self.data_fingerprint_path = fed_dataset.fingerprint_path
        self._setup_clients(fed_dataset)
        
        print('Initializing global model.')
        if self.config.model.DPfy:
            self.global_model = get_model(self.config.model.name, self.config, return_base_model=False).state_dict()
        else:
            self.global_model = get_model(self.config.model.name, self.config, return_base_model=True).state_dict()
        self.total_num_exchanged_models = 0
        self.scheduler = None

        
        if self.config.training.glob.early_stopping:
            print('Initializing global EarlyStopper.')
            self.early_stopper = EarlyStopping(config.training.glob.early_stopping.patience, config.training.glob.early_stopping.delta, config.training.glob.early_stopping.metric, config.training.glob.early_stopping.use_loss, config.training.glob.early_stopping.subject_to, self.config.data.use_val, config.training.glob.early_stopping.verbose) if config.training.glob.early_stopping is not None else None
        else:
            self.early_stopper = None

        self.file_identifier = f'{self.config.experiment_name}_GLOBAL'+self.config.debug_file_suffix
        self.global_history = History(keys=['Metric', 'CommunicationRound'], savefile=os.path.join(self.config.history_path, self.file_identifier))
        self.checkpoint_path = os.path.join(self.config.checkpoint_path, self.file_identifier)

        self.current_communication_round = 0
        self.was_tuned = False
        print('### Initializing FederatedEnvironment (END) ###')

    def _setup_clients(self, fed_dataset, *args, **kwargs):
        print('# Setting up Clients (START) #')
        self.clients = {}
        self.client_id_map = {}
        local_trn_data_dict, local_tst_data_dict, local_val_data_dict = fed_dataset.trn_loaders, fed_dataset.tst_loaders, fed_dataset.val_loaders
        for numerical_id, client_id in enumerate(local_trn_data_dict.keys()):
            self.clients[client_id] = self.client_class(client_id, (local_trn_data_dict[client_id], local_tst_data_dict[client_id], local_val_data_dict[client_id]), self.config)
            self.client_id_map[numerical_id] = client_id
        self.num_clients = len(self.clients)
        print(f'# A total of {self.num_clients} clients were initialized. #')
        print('# Setting up Clients (END) #')

    @abstractmethod
    def train(self, *args, **kwargs):
        raise NotImplementedError(f'train is not implemented for {self.__class__.__name__}! Choose a more specific, fully implemented Environment subclass!')

    def tune(self, *args, **kwargs):
        if self.config.training.tuning:
            print(f'### {self.__class__.__name__} Tuning (START) ###')
            self.load_best()
            for client_id, client in self.clients.items():
                _ = client.tune(self.global_model)
            self.was_tuned = True
            print(f'### {self.__class__.__name__} Tuning (END) ###')
        else:
            print('No tuning since no tuning parameters were set in config.training.tuning.')

    def get_client_eval_model_state(self, client, *args, **kwargs):
        return self.global_model

    def evaluate(self, split='VAL', *args, **kwargs):
        print(f'### {self.__class__.__name__} Testing Model (START) ###')
        client_metrics = defaultdict(list)
        client_samples = []
        client_ids = [] #also track this so we can be sure to have the right order if required
        for client_id, client in self.clients.items():
            local_metrics = client.evaluate(self.current_communication_round, split, self.get_client_eval_model_state(client, *args, **kwargs))
            for metric, value in local_metrics.wandb_dict(step_key='CommunicationRound').items():
                client_metrics[metric].append(value)
            client_samples.append(client.get_sample_number(split))
            client_ids.append(client_id)
        for metric, values in client_metrics.items():
            self.global_history[(metric, self.current_communication_round)] = np.mean(values)
            if metric in ['CommunicationRound', 'LocalEpoch', 'OverallTrainedEpochs']:
                continue
            self.global_history[(metric+'_WEIGHTED', self.current_communication_round)] = np.average(values, weights=client_samples)
        print(f'### {self.__class__.__name__} Testing Model (END) ###')
        return client_metrics, client_samples, client_ids

    def aggregate(self, local_weights, *args, **kwargs):
        raise NotImplementedError(f'aggregate is not implemented for {self.__class__.__name__}! Choose a more specific, fully implemented Environment subclass!')

    def log_status(self, *args, **kwargs):
        metric_dict = self.global_history.wandb_dict(step=self.current_communication_round, step_key='CommunicationRound')
        metric_strings = [f'{metric}: {value}' for metric, value in metric_dict.items() if metric not in ['CommunicationRound']]

        ms = ' | '.join(metric_strings)
        print(f'# Model performance in communication round {self.current_communication_round}: {ms} #')

    def summary(self):
        print('### Global History Summary ###')
        #self.global_history.summary(step_key='CommunicationRound', mode='latest')
        key_value = 'VAL_' + self.config.training.loss  if self.config.data.use_val else 'TST_' + self.config.training.loss
        key_value += '_L'
        self.global_history.summary(key_value=key_value, max_key=False, step_key='CommunicationRound', mode='best')
        if len(self.clients) <= 10:
            for client in self.clients.values():
                client.summary()

    def save_histories(self):
        print('### Saving Histories ###')
        self.global_history.save()
        for client in self.clients.values():
            client.local_history.save()

    def save(self, path=None, best=False):
        self.save_histories()
        assert self.checkpoint_path or path, 'If the Environment object has no default checkpoint_path you have to provide a path!'
        path = self.checkpoint_path if path is None else path
        if best:
            path += '_best'
        statepath = path if path.endswith('.state') else path + '.state'
        torch.save(self.global_model, statepath)
        print(f'Saved global model state to {path}')

    def load(self, path=None, best=False):
        assert self.checkpoint_path or path, 'If the Environment object has no default checkpoint_path you have to provide a path!'
        path = self.checkpoint_path if path is None else path
        if best:
            path += '_best'
        statepath = path if path.endswith('.state') else path + '.state'
        self.global_model = torch.load(statepath)
        print(f'Loaded global model state from {path}')        

        if self.was_tuned:
            for client in self.clients.values():
                client.load(best=best)

    def load_best(self):
        self.load(best=True)

    def load_histories(self):
        print('### Loading Histories ###')
        self.global_history.load()
        for client in self.clients.values():
            client.local_history.load()

    def get_history(self):
        self.global_history.load()
        return self.global_history
    
    def get_client_histories(self):
        for client in self.clients.values():
            client.local_history.load()
        return {client_id: client.local_history for client_id, client in self.clients.items()}

    def set_final_evaluation(self):
        self.current_communication_round = -1
    

class CentralizedEnvironment(FederatedEnvironment):
    """
    Centralized environment for federated learning.

    This class extends the FederatedEnvironment and provides centralized training functionality; 
    i.e. global server distributes global model state to clients (that perform local training).

    Args:
        fed_dataset (tuple): A tuple containing local training, testing, and validation dataloaders (that are dicts with a structure of client_id: local_split_dataloader).
        config (object): Configuration object containing environment and training settings.

    Attributes:
        Inherits attributes from FederatedEnvironment.

    Methods:
        train(communication_rounds=None, *args, **kwargs):
            Training procedure of a centralized federated environment (silo).
            0. Initialize global model. (handled by FederatedEnvironment init)
            In each communication round:
                1. Distribute global model to clients.
                2. Clients train on local data and return updated local weights.
                3. Aggregate local weights to update global model.
                4. Evaluate global model on different data splits.

        global_train_step(*args, **kwargs):
            Perform a global training step. See 1.-3. above.

    """
    def __init__(self, fed_dataset, config, *args, **kwargs):
        print('### Initializing CentralizedEnvironment (START) ###')
        super().__init__(fed_dataset, config, *args, **kwargs)
        print('### Initializing CentralizedEnvironment (END) ###')

    def train(self, communication_rounds=None, *args, **kwargs):
        print(f'### {self.__class__.__name__} Training (START) ###')
        if communication_rounds is None:
            communication_rounds = self.config.training.glob.communication_rounds
        for global_epoch in range(communication_rounds):
            self.current_communication_round += 1
            print(f'# Communication round {self.current_communication_round} #')

            self.global_train_step()

            if self.scheduler and not self.scheduler.after_val:
                self.scheduler.step()
                self.global_history[('LR', self.current_communication_round)] = self.scheduler.lr()
                if self.config.training.glob.globally_derived_lr:
                    for client in self.clients.values():
                        client.set_lr(self.scheduler.lr() * self.config.training.glob.globally_derived_lr.factor)


            if global_epoch % self.config.training.glob.validation_frequency == 0:
                if self.config.data.use_val:
                    self.evaluate('VAL') # evaluate global model on validation data
                if self.config.eval.skip_test_data_during_training:
                    pass
                else:
                    self.evaluate('TST') # evaluate global model on test data

                if self.current_communication_round in self.config.training.glob.save_rounds or self.current_communication_round == communication_rounds:
                    self.save(self.checkpoint_path+f'_CR{self.current_communication_round}')
                
                #EarlyStopping
                if self.early_stopper:
                    self.early_stopper(self.global_history[(self.early_stopper.metric, self.current_communication_round)])

                    if self.early_stopper.improved:
                        self.save(self.checkpoint_path, best=True)
                    if self.early_stopper.stop:
                        print(f'Early stopping the global federated training since we had no improvement of {self.early_stopper.metric} for {self.early_stopper.patience} rounds. Training was stopped after {self.current_communication_round} CommunicationRounds.')
                        break
                else:
                    self.save(self.checkpoint_path, best=True)

                if self.scheduler and self.scheduler.after_val:
                    if hasattr(self.scheduler, 'metric'):
                        self.scheduler.step(self.global_history.get(self.scheduler.metric))
                    else:
                        self.scheduler.step()
                    if self.config.training.glob.globally_derived_lr:
                        for client in self.clients.values():
                            client.set_lr(self.scheduler.lr() * self.config.training.glob.globally_derived_lr.factor)
                    self.global_history[('LR', self.current_communication_round)] = self.scheduler.lr()

            if self.debug and self.current_communication_round >= 2:
                break
        
        print(f'### {self.__class__.__name__} Training (END) ###')
    
    def global_train_step(self, *args, **kwargs):
        local_weights = []
        for client_id, client in self.clients.items():
            client.sent_models += 1
            client.received_models += 1
            w_local = client.train(self.global_model, self.current_communication_round)
            if self.config.training.weigh_sample_quantity:
                local_weights.append((client.get_sample_number('TRN'), w_local))
            else:
                local_weights.append((1, w_local))

        # update global weights
        self.global_model = self.aggregate(local_weights)

        self.total_num_exchanged_models += 2*self.num_clients
        self.global_history[('TotalExchangedModels', self.current_communication_round)] = self.total_num_exchanged_models