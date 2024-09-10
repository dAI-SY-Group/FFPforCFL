from .bases import Client
from .PFL import PFLClient

def get_client_class(client_type):
    return eval(client_type)