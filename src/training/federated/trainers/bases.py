from collections import defaultdict
import copy
import gc

import torch

from src.toolkit.history import History
from src.training.utils.optimizer import set_optimizer
from src.training.utils.scheduler import set_scheduler
from src.training.utils.early_stopping import EarlyStopping
from src.training.central.trainer import CentralTrainer


class FederatedTrainer(CentralTrainer):
    """
    Basic Federated model trainer for distributed learning scenarios.
    Defines the local behavior of a Client regarding model training and testing
    
    Basic local training is similar to usual central training behavior
    This class extends the CentralTrainer and implements additional functionality for federated training.

    Args:
        model (object): The model to be trained.
        data (tuple): Tuple of data loaders for training, testing and validation.
        config (object): Configuration object containing training settings.
        device (str): Device for training ('cpu' or 'cuda').
        *args, **kwargs: Additional arguments and keyword arguments.

    Attributes:
        Inherits attributes from CentralTrainer.
        overall_trained_epochs (int): Total number of trained epochs across all clients.
        checkpoint_path (str): Path for saving checkpoints.
        muted (bool): Flag indicating if the trainer is muted, e.g. if there are a lot of clients.

    Methods:
        train(epochs=None, *args, **kwargs):
            Train the model for a specified number of epochs.
            Does basic epoch based deep learning training. Main logic see CentralTrainer. 
            Adds some additional logging.
        train_step(*args, **kwargs):
            Perform a single training step.
            Does basic training step. Main logic see CentralTrainer.
            Adds some additional logging.
        save(path=None, best=False):
            Save the model checkpoint.
        evaluate(split='VAL', *args, **kwargs):
            Evaluate the model on a specified dataset split.
        reset_optimizer_state(full_reinitialization=False, training_config=None, *args, **kwargs):
            Reset the optimizer state.
        get_model_state():
            Get the state of the model.
        set_model_state(state_dict):
            Set the state of the model.
    """
    def __init__(self, model, data, config, device, *args, **kwargs):
        super().__init__(model, data, config, device, *args, **kwargs)
        self.overall_trained_epochs = 0
        self.best_model = None
        if config.training.num_clients > 20:
            self.muted = True

    def train(self, epochs=None, *args, **kwargs):
        self.history = History(keys=['Metric', 'LocalEpoch'])

        super().train(epochs, *args, **kwargs)

        self.save('save_best')


    def train_step(self, *args, **kwargs):
        super().train_step(*args, **kwargs)    
        self.overall_trained_epochs += 1
        self.update_history({'OverallTrainedEpochs': self.overall_trained_epochs}, 'TRN')
        torch.cuda.empty_cache()

    #adjust save for local federated training, as we do not want to save every single client model
    def save(self, path=None, best=False):
        if path is None:
            return
        if best:
            self.best_model = copy.deepcopy(self.model.cpu().state_dict())        
        if path == 'save_best' and self.config.training.save_client_models:
            self.model.load_state_dict(self.best_model)
            self.model.save(self.checkpoint_path+f'_best')

    def load(self, path=None, best=False):
        if best and self.config.training.save_client_models:
            self.model.load(self.checkpoint_path+f'_best')
        
    def evaluate(self, split='VAL', *args, **kwargs):
        if 'reset_history' in kwargs.keys() and kwargs['reset_history']:
            self.history = History(keys=['Metric', 'LocalEpoch'])
        super().evaluate(split, *args, **kwargs)
        

    def reset_optimizer_state(self, full_reinitialization=False, training_config=None, *args, **kwargs):
        config = self.config.training if training_config is None else training_config
        self.current_epoch = 0
        if config.optimizer.name in ['SGD'] or full_reinitialization:
            self.optimizer = set_optimizer(self.model, config.optimizer) #MASSIVELY LEAKS GPU MEMORY FOR STATEFUL OPTIMIZERS LIKE ADAM
        else:
            self.optimizer.state = defaultdict(dict) # RESET OPTIMIZER STATE without recreating whole optimizer which might lead to memory leaks
        if config.lr_scheduler:
            self.scheduler = set_scheduler(self.optimizer, config.validation_frequency, config.lr_scheduler, self.use_val)
        else:
            self.scheduler = None
        if full_reinitialization:
            if config.early_stopping:
                self.early_stopper = EarlyStopping(config.early_stopping.patience, config.early_stopping.delta, config.early_stopping.metric, config.early_stopping.use_loss, config.early_stopping.subject_to, self.use_val, config.early_stopping.verbose) if config.early_stopping is not None else None
            else:
                self.early_stopper = None
        gc.collect()
    
    def get_model_state(self, best=False):
        if best:
            return copy.deepcopy(self.best_model.cpu().state_dict())
        else:
            return copy.deepcopy(self.model.cpu().state_dict())

    def set_model_state(self, state_dict):
        self.model.load_state_dict(state_dict)
