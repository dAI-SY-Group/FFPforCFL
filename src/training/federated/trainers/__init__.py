from .bases import FederatedTrainer
from .PFL import PFLTrainer

def get_model_trainer(trainer_name, model, data, config, device):
    if trainer_name == 'FederatedTrainer':
        trainer = FederatedTrainer(model, data, config, device)
    elif trainer_name == 'PFLTrainer':
        trainer = PFLTrainer(model, data, config, device)
    else:
        raise NotImplementedError(f'{trainer_name} is not implemented yet..')
    return trainer
