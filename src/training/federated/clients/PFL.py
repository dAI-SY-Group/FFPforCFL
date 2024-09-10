from src.training.federated.clients.bases import Client
from src.models.ModelHead import ClassifierHead

class PFLClient(Client):
    """
    Client class for Personalized Federated Learning.

    This class extends the base Client class and incorporates functionality for handling personalized model heads.
    Initializes personal model head (if configured) based on the number of classes in the local client dataset.
    Make sure to use PFLTrainer since it adjusts state getters, setters as well as training functionalities to make use of the personal model head.

    Args:
        client_id (int): Identifier for the client.
        data (tuple): Tuple of data loaders for training, testing and validation.
        model_trainer (object): Model trainer for the client.

    Attributes:
        Inherits attributes from the Client class.
        num_classes (int): Number of classes in the dataset.
        personal_model_head (ClassifierHead or None): Personalized model head if configured, otherwise None.

    """
    def __init__(self, client_id, data, config):
        self.num_classes = data[0].dataset.num_classes
        super().__init__(client_id, data, config)
        
    
    def _setup_model_trainer(self, data):
        super()._setup_model_trainer(data)
        if self.config.training.personal_model_head:
            print(f'Initializing personal model head for client {self.client_id} with {self.num_classes} classes.')
            self.model_trainer.personal_model_head = ClassifierHead(self.config.training.personal_model_head.input_dim, self.num_classes)
        else:
            self.model_trainer.personal_model_head = None
