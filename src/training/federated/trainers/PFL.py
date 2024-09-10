import copy

from src.training.utils.optimizer import set_optimizer
from src.training.federated.trainers.bases import FederatedTrainer
from src.models.utils import save_model, load_model


class PFLTrainer(FederatedTrainer):
    """Trainer class for Personalized Federated Learning.

    This class extends the FederatedTrainer and includes functionality for handling personal model heads.
    Make sure to use PFLClient for correct initialization of the personal model head (if configured).
    self.model in this case corresponds to the commonly trained feature extractor that is shared across all clients.

    Args:
        model (object): The model to be trained.
        config (object): Configuration object containing training settings.
        device (str): Device for training ('cpu' or 'cuda').
        *args, **kwargs: Additional arguments and keyword arguments.

    Attributes:
        Inherits attributes from FederatedTrainer.
        personal_model_head (ClassifierHead): Personal model head for individualized predictions.
        personal_model_head_optimizer (Optimizer): Optimizer for the personal model head.

    Methods:
        reset_optimizer_state(_all=False, full_reinitialization=False, training_config=None, *args, **kwargs):
            Reset the optimizer state, including that of the personal model head if present.
        _zero_grad():
            Perform gradient zeroing for the main model, global optimizer, and personal model head optimizer if present.
        _model_forward(inputs):
            Perform the forward pass through the main model, optionally passing the outputs through the personal model head.
        _optimizer_step():
            Take an optimizer step for the main model and the personal model head optimizer if present.
        evaluate(split='VAL', *args, **kwargs):
            Evaluate the model on a specified data split, ensuring the personal model head is correctly configured.

    """
    def __init__(self, model, config, device, *args, **kwargs):
        super().__init__(model, config, device, *args, **kwargs)
        self.personal_model_head = None
        self.best_personal_model_head = None

    def train(self, epochs=None, *args, **kwargs):
        if self.config.training.local_adjustment and self.config.training.personal_model_head:
            current_lr = self.personal_model_head_optimizer.lr()
            self.personal_model_head_optimizer.set_lr(self.config.training.local_adjustment.lr)
            self.model.freeze()
            super().train(self.config.training.local_adjustment.epochs, *args, **kwargs)
            self.model.unfreeze()
            self.personal_model_head_optimizer.set_lr(current_lr)

        super().train(epochs, *args, **kwargs)
            
    def reset_optimizer_state(self, full_reinitialization=False, training_config=None, *args, **kwargs):
        config = self.config.training if training_config is None else training_config
        super().reset_optimizer_state(full_reinitialization, config, *args, **kwargs)
        if self.personal_model_head:
            self.personal_model_head_optimizer = set_optimizer(self.personal_model_head, config.optimizer)

    def save(self, path=None, best=False):
        if path is None:
            return
        if self.personal_model_head:
            if best:
                self.best_personal_model_head = copy.deepcopy(self.personal_model_head.cpu().state_dict())
            if path == 'save_best' and self.config.training.save_client_models:
                self.personal_model_head.load_state_dict(self.best_personal_model_head)
                save_model(self.checkpoint_path+f'_pmh_best', self.personal_model_head)

        return super().save(path, best)
    
    def load(self, path=None, best=False):
        super().load(path, best)
        if self.personal_model_head and best and self.config.training.save_client_models:
            self.personal_model_head = load_model(self.checkpoint_path+'_pmh_best', self.personal_model_head)
            self.personal_model_head.to(self.device)

    def train_step(self, *args, **kwargs):
        if self.personal_model_head:
            self.personal_model_head.to(self.device)
            self.personal_model_head.train()
        return super().train_step(*args, **kwargs)

    def _zero_grad(self):
        self.optimizer.zero_grad()
        if self.personal_model_head:
            self.personal_model_head_optimizer.zero_grad()

    def _model_forward(self, inputs):
        outputs = self.model(inputs)
        if self.personal_model_head:
            outputs = self.personal_model_head(outputs)
        return outputs

    def _optimizer_step(self):
        self.optimizer.step()
        if self.personal_model_head:
            self.personal_model_head_optimizer.step()

    def evaluate(self, split='VAL', *args, **kwargs):
        if self.personal_model_head:
            self.personal_model_head.to(self.device)
            self.personal_model_head.eval()
        return super().evaluate(split, *args, **kwargs)
