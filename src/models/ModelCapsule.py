import torch
from src.models.utils import attach_model_functionalities

class ModelCapsule(torch.nn.Module):
    def __init__(self, model_list):
        """
        Initialize the ModelCapsule object.

        Args:
        - model_list (list): A list of PyTorch models to encapsulate within ModelCapsule.

        The function initializes a ModelCapsule object by setting up a ModuleList with the given list of models.
        Additionally, it attaches model functionalities using the 'attach_model_functionalities' util.
        """
        super().__init__()
        self.model_list = torch.nn.ModuleList(model_list)
        attach_model_functionalities(self, model_list[0]._config)
    
    def forward(self, x):
        """
        Perform a forward pass through the encapsulated models.

        Args:
        - x (torch.Tensor): Input tensor to be passed through the models.

        Returns:
        - torch.Tensor: Output tensor after passing through all the encapsulated models.

        The function iterates through each model in the ModuleList and sequentially applies them to the input tensor 'x'.
        The final output tensor after passing through all models is returned.
        """
        for model in self.model_list:
            x = model(x)
        return x

    def __getitem__(self, index):
        return self.model_list[index]
    
    def __len__(self):
        return len(self.model_list)
    
    def __iter__(self):
        return iter(self.model_list)
    
    def __repr__(self):
        return f'ModelCapsule({self.model_list})'
    
    def __str__(self):
        return f'ModelCapsule({self.model_list})'