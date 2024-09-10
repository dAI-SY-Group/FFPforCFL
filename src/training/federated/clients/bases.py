import os

from src.toolkit.history import History
from src.models.modelzoo import get_model
from src.training.federated.trainers import get_model_trainer

class Client:
    """
    Represents a client in a federated learning setting.

    Args:
    - client_id (str or int): Unique identifier for the client.
    - data (tuple): Tuple of data loaders for training, testing and validation.
    - config: Experiment and especially config.training configurations.

    Attributes:
    - client_id (str or int): Unique identifier for the client.
    - trn_data: Training data loader.
    - tst_data: Test data loader.
    - val_data: Validation data loader.
    - model: SimpleAI Model object that is trained by the client.
    - model_trainer: SimpleAI ModelTrainer object that handles the epoch based model train_step logic for training the model.
    - config: Experiment and especially config.training configurations.
    - trn_samples (int): Number of training samples.
    - tst_samples (int): Number of test samples.
    - val_samples (int): Number of validation samples.
    - file_identifier (str): Identifier for files related to this client.
    - local_history (History): History object for local training history.
    - checkpoint_path (str): Path for storing local client checkpoints.
    - sent_models (int): Number of models sent.
    - received_models (int): Number of models received.

    Methods:
    - get_sample_number(split, get_truth=False): Get the number of samples in a given split.
    - train(w_global, communication_round): Get global model by server/cluster and train the model locally with local data during the given global communication round.
    - _train_output(): Get the output of the training (per default: the local model weights after training).
    - tune(w_global, communication_round=-1): Tune the local model with local data after normal training!!.
    - save(path=None, best=False): Save the local client state.
    - load(path=None, best=False): Load the local client state.
    - set_training_model_state(w_global, communication_round): Set the local model state to the global model state.
    - evaluate(communication_round, split='TST', model_state=None, *args, **kwargs): Evaluate the client's local model if None. Else evaluate the given model state on local data split.
    - summary(): Print a summary of the local history.

    """
    def __init__(self, client_id, data, config):
        self.config = config
        self.client_id = client_id
        trn_data, tst_data, val_data = data
        self.trn_samples = len(trn_data.dataset)
        self.tst_samples = 0 if tst_data == None else len(tst_data.dataset)
        self.val_samples = 0 if val_data == None else len(val_data.dataset)
        
        self._setup_model_trainer(data)
        self.model_trainer.reset_optimizer_state()
        
        self.file_identifier = f'{self.config.experiment_name}_CLIENT{self.client_id}'+self.config.debug_file_suffix
        self.local_history = History(keys=['Metric', 'LocalEpoch', 'CommunicationRound'], savefile=os.path.join(self.config.history_path, self.file_identifier))
        self.checkpoint_path = os.path.join(self.config.checkpoint_path, self.file_identifier)
        self.model_trainer.checkpoint_path = self.checkpoint_path
        
        self.sent_models = 0
        self.received_models = 0
        print(f'# Client {client_id} initialized. local train samples: {self.trn_samples} | local test samples: {self.tst_samples} | local val samples: {self.val_samples} #')

    def _setup_model_trainer(self, data):
        self.model = get_model(self.config.model.name, self.config)
        self.model_trainer = get_model_trainer(self.config.training.trainer, self.model, data, self.config, self.config.device)

    def get_sample_number(self, split, get_truth=False):
        """returns the number of samples in a given split of a dataset

        Args:
            split ([str]): one of trn, tst, val
            get_truth (bool, optional): returns the actual number of samples if True (else val_samples may be modified if no val_data was provided to the client). Defaults to False.

        Returns:
            [int]: number of samples
        """
        if split == 'TRN':
            return self.trn_samples
        elif split == 'TST':
            return self.tst_samples
        elif split == 'VAL':
            if get_truth:
                return self.val_samples
            if self.val_samples == 0: print('The Client was not provided with explicit validation data. Using testdata instead.')
            return self.val_samples if self.val_samples > 0 else self.tst_samples
        else:
            raise ValueError(('This data split is not defined for Clients. Choose one of trn, tst, val'))


    def train(self, w_global, communication_round):
        self.set_training_model_state(w_global, communication_round)
        self.model_trainer.reset_optimizer_state()

        self.model_trainer.train()
        self.model_trainer.history.add_col('CommunicationRound', communication_round)
        self.local_history.update(self.model_trainer.history)#, f'CLIENT_{self.client_id}')
        self.local_history(('SentModels', self.model_trainer.current_epoch, communication_round), self.sent_models)
        self.local_history(('ReceivedModels', self.model_trainer.current_epoch, communication_round), self.received_models)
        return self._train_output()

    def _train_output(self):
        return self.model_trainer.get_model_state()
    
    def tune(self, w_global, communication_round=-1):
        print(f'# Client {self.client_id} tuning (START) #')
        self.set_training_model_state(w_global, communication_round)

        #prepare optimizer for fine tuning
        self.config.training.save_client_models = True #make sure that the best local client model is saved during tuning
        tuning_config = self.model_trainer.config.training.copy()
        tuning_config.optimizer.lr = self.config.training.tuning.lr
        tuning_config.lr_scheduler = False
        tuning_config.early_stopping = False
        self.model_trainer.reset_optimizer_state(training_config=tuning_config)
    
        self.model_trainer.train(self.config.training.tuning.epochs)
        self.model_trainer.history.add_col('CommunicationRound', communication_round) # -1 indicates that this is a tuning run on the "best" global model
        self.local_history.update(self.model_trainer.history)

        
        self.save(best=True)
        print(f'# Client {self.client_id} tuning (END) #')

    def save(self, path=None, best=False):
        self.model_trainer.save(path, best)

    def load(self, path=None, best=False):
        self.model_trainer.load(path, best)

    def set_training_model_state(self, w_global, communication_round):
        self.model_trainer.set_model_state(w_global) # whole model is exchanged!

    def evaluate(self, communication_round, split='TST', model_state=None, *args, **kwargs):
        if model_state is None: #Local training evaluation
            self.model_trainer.evaluate(split, reset_history=True)
            self.model_trainer.history.add_col('CommunicationRound', communication_round)
            self.local_history.update(self.model_trainer.history)
            return None
        else: #an explicit model was given i.e. global aggregated model is to be evaluated
            self.model_trainer.set_model_state(model_state)
            self.model_trainer.evaluate(split, reset_history=True)
            self.model_trainer.history.add_col('CommunicationRound', communication_round)
            self.local_history.update(self.model_trainer.history)
            return self.model_trainer.history

    def set_lr(self, lr):
        self.config.training.optimizer.lr = lr
        self.model_trainer.optimizer.set_lr(lr)
    
    def summary(self):
        print(f'### Local History Summary for Client {self.client_id} ###')
        try:
            key_value = 'VAL_' + self.model_trainer.loss_fn.name  if self.config.data.use_val else 'TST_' + self.model_trainer.loss_fn.name
            key_value += '_L'
            self.local_history.summary(key_value=key_value, max_key=self.model_trainer.loss_fn.subject_to=='max', step_key='CommunicationRound', mode='best')
        except Exception as e:
            print(f'Could not print local history summary for client {self.client_id}! Error: {e}')

