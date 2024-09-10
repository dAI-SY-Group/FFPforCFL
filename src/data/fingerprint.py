import copy
import os

import numpy as np
import torch
import torchvision.models as models

def get_feature_extractor(backbone):
    """
    Returns a feature extractor based on the specified backbone architecture.

    Parameters:
        backbone (str): The name of the backbone architecture ('VGG16', 'ResNet18', 'ResNet50', 'MobileNetS', 'ViTB16', 'SWINTiny', 'SWINSmall', 'SWINBase'). 
                        trained on Imagenet.

    Returns:
        torch.nn.Module: The feature extractor module.

    Raises:
        NotImplementedError: If the specified backbone is not implemented.
    """
    if backbone == 'VGG16':
        full_model = models.vgg16(pretrained=True)
    elif backbone == 'ResNet18':
        full_model = models.resnet18(pretrained=True)
    elif backbone == 'ResNet50':
        full_model = models.resnet50(pretrained=True)
    elif backbone == 'MobileNetS':
        full_model = models.mobilenet_v3_small(pretrained=True)
    elif backbone == 'ViTB16':
        full_model = models.vit_b_16(pretrained=True)
    elif backbone == 'SWINTiny':
        full_model = models.swin_t(pretrained=True)
    elif backbone == 'SWINSmall':
        full_model = models.swin_s(pretrained=True)
    elif backbone == 'SWINBase':
        full_model = models.swin_b(pretrained=True)
    else:
        raise NotImplementedError('Feature extractor not implemented!')

    if 'ResNet' in backbone:
        feature_extractor = torch.nn.Sequential(*list(full_model.children())[:-1])
    else:
        feature_extractor = full_model.features
    feature_extractor = feature_extractor.to('cuda')
    feature_extractor.eval()
    return feature_extractor

#small helper dataset class
class FEDS:
    def __init__(self, data, targets) -> None:
        self.data = data
        self.targets = targets

def extract_features(dataloader, fe_model):
    """
    Extracts features from a dataloader using a given feature extraction model.

    Parameters:
        dataloader (torch.utils.data.DataLoader): The dataloader containing input data.
        fe_model (torch.nn.Module): The feature extraction model.

    Returns:
        FEDS: A custom dataset containing extracted features and corresponding labels.

    Note:
        - The feature extraction model (`fe_model`) should be compatible with the input data.

    Raises:
        AssertionError: If the provided `dataloader` is incompatible or if batch size adjustment fails.
    """
    # pass input data through FE to get features and then generate fingerprints based on those features
    data = []
    targets = []
    with torch.no_grad():
        for i, (x, y) in enumerate(dataloader):
            # get features from the pretrained model and flatten to batch dimension OR just flatten "raw features" if no pretrained model is provided
            if fe_model is None:
                data.append(x.reshape(x.shape[0], -1))
            else:
                x = x.to('cuda')
                data.append(fe_model(x).detach().cpu().reshape(x.shape[0], -1))
            targets.append(y)
    feature_dataset = FEDS(torch.cat(data), torch.cat(targets))
    print('Done extracting features.')
    return feature_dataset


class FingerprintGenerator:
    """
    Class for generating and managing fingerprints.

    Args:
        dataset : Dataset
            The dataset for which the fingerprint is generated.
        save_file : str, optional
            File path to save the fingerprint, by default None.

    Attributes:
        dataset : Dataset
            The dataset for which the fingerprint is generated.
        save_file : str or None
            File path to save the fingerprint.
        generation_fn : function
            The function used for generating the fingerprint.
        fingerprint : ndarray or None
            The generated fingerprint.

    Methods:
        __call__(recalculate=False):
            Generates or retrieves the fingerprint.

        generate_fingerprint():
            Generates the fingerprint using the specified generation function.

        save(savefile=None):
            Saves the fingerprint to a file.

        load(savefile=None):
            Loads the fingerprint from a file.

    """
    def __init__(self, dataset, save_file=None):
        self.dataset = dataset
        self.save_file = save_file

        self.generation_fn = None
        self.fingerprint = None

    def __call__(self, recalculate=False):
        if self.fingerprint is None or recalculate:
            self.generate_fingerprint()
        return self.fingerprint
    
    def generate_fingerprint(self):
        if self.generation_fn is None:
            raise NotImplementedError('The fingerprint generator has not been implemented yet. Use a fitting subclass.')
        else:
            self.fingerprint = self.generation_fn()
        return self.fingerprint
    
    def save(self, savefile=None):
        if savefile is None:
            savefile = self.save_file
        if savefile is None:
            raise ValueError('No save file specified.')
        else:
            savefile = savefile + '.npy' if not savefile.endswith('.npy') else savefile
            np.save(savefile, self.fingerprint)

    def load(self, savefile=None):
        if savefile is None:
            savefile = self.save_file
        if savefile is None:
            raise ValueError('No save file specified.')
        else:
            savefile = savefile + '.npy' if not savefile.endswith('.npy') else savefile
            self.fingerprint = np.load(savefile)
        return self.fingerprint
    

class SVDFingerprintGenerator(FingerprintGenerator):
    """
    Subclass of FingerprintGenerator for generating fingerprints using SVD.

    Args:
        dataset : Dataset
            The dataset for which the fingerprint is generated.
        K : int, optional
            Number of components to keep in the SVD, by default 5.
        savefile : str, optional
            File path to save the fingerprint, by default None.

    Attributes:
        dataset : Dataset
            The dataset for which the fingerprint is generated.
        save_file : str or None
            File path to save the fingerprint.
        generation_fn : function
            The function used for generating the fingerprint (calculate_dataset_svd_fingerprint).
        fingerprint : ndarray or None
            The generated fingerprint.
        K : int
            Number of components to keep in the SVD.

    Methods:
        calculate_dataset_svd_fingerprint():
            Calculates the dataset fingerprint using SVD.

    """
    def __init__(self, dataset, K=5, savefile=None):
        super().__init__(dataset, savefile)
        self.K = K
        self.generation_fn = self.calculate_dataset_svd_fingerprint

    # from https://github.com/MMorafah/PACFL (main_PACFL)
    def calculate_dataset_svd_fingerprint(self):
        """
        Calculate the dataset fingerprint using Singular Value Decomposition (SVD). from https://github.com/MMorafah/PACFL (main_PACFL)

        Returns:
            ndarray:
                The generated fingerprint.

        """
        idxs_local = np.arange(len(self.dataset.data))
        labels_local = np.array(self.dataset.targets)
        
        # Sort Labels Train 
        idxs_labels_local = np.vstack((idxs_local, labels_local))
        idxs_labels_local = idxs_labels_local[:, idxs_labels_local[1, :].argsort()]
        idxs_local = idxs_labels_local[0, :]
        labels_local = idxs_labels_local[1, :]
        
        uni_labels, cnt_labels = np.unique(labels_local, return_counts=True)
        
        #print(f'Labels: {uni_labels}, Counts: {cnt_labels}')
        
        nlabels = len(uni_labels)
        cnt = 0
        U_temp = []

        #for each class in the dataset calculate the SVD 
        for j in range(nlabels):
            print('Calculating SVD for class:', j)
            local_ds1 = self.dataset.data[idxs_local[cnt:cnt+cnt_labels[j]]]
            local_ds1 = local_ds1.reshape(cnt_labels[j], -1)
            local_ds1 = local_ds1.T
    
            if self.K > 0: 
                u1_temp, _, _ = np.linalg.svd(local_ds1, full_matrices=False)
                u1_temp=u1_temp/np.linalg.norm(u1_temp, ord=2, axis=0) # trace operator (in this case equals to L2 Norm)
                U_temp.append(u1_temp[:, 0:self.K])
            cnt+=cnt_labels[j]

        return copy.deepcopy(np.hstack(U_temp))
        


def create_distribution_fingerprints(federated_dataloader, fingerprint_path, generator_mode = 'SVD', data_mode = 'raw', feature_extractor = None, fe_batch_size=256, reload = False, *args, **kwargs):
    """
    Creates distribution fingerprints for clients in a federated dataset.

    Parameters:
        federated_dataloader (dict): A dictionary of client IDs and their corresponding dataloaders.
        fingerprint_path (str): File path to save or load the fingerprints.
        generator_mode (str, optional): The mode for generating fingerprints. Defaults to 'SVD'.
        data_mode (str, optional): The mode for handling data ('raw' or 'features'). Defaults to 'raw'.
        feature_extractor (nn.Module or None, optional): The feature extractor module for 'features' mode.
            Required if `data_mode='features'`. Defaults to None.
        fe_batch_size (int, optional): Batch size for feature extraction. Defaults to 256.
        reload (bool, optional): Flag to reload existing fingerprints. Defaults to False.

    Returns:
        dict: A dictionary containing client IDs as keys and their corresponding fingerprints as values.

    Raises:
        NotImplementedError: If generator_mode or data_mode is not implemented yet.

    Note:
        - If reload is set to True, existing fingerprints will be loaded (if available).
        - If reload is set to False, new fingerprints will be generated.
    """
    if os.path.exists(fingerprint_path) and not reload:
        print(f'Fingerprint file {fingerprint_path} already exists! Loading existing fingerprints!')
        client_FPs_dict = torch.load(fingerprint_path)
    else:
        print(f'Fingerprint file {fingerprint_path} does not exist! Generating fingerprint file!')
        client_FPs_dict = {}
        
        if generator_mode == 'SVD':
            fpg_class = SVDFingerprintGenerator
        else: 
            raise NotImplementedError(f'Fingerprint generator {generator_mode} is not implemented yet!')

        if data_mode == 'features':
            assert feature_extractor is not None, 'Feature extractor must be specified for feature mode!'
            fe_model = get_feature_extractor(feature_extractor)

        for client_id, client_dl in federated_dataloader.items():
            print(f'Generating fingerprint for client {client_id} based on {generator_mode} and {data_mode}!')
            if data_mode == 'raw':
                client_dataset = extract_features(client_dl, None)
            elif data_mode == 'features':               
                client_dataset = extract_features(client_dl, fe_model)
            else: 
                raise NotImplementedError(f'Data mode {data_mode} is not implemented yet!')
            
            fingerprint_generator = fpg_class(client_dataset)
            client_FPs_dict[client_id] = fingerprint_generator()
        
        torch.save(client_FPs_dict, fingerprint_path)
    return client_FPs_dict

class FingerprintSimilarityCalculator:
    """
    A class for calculating and visualizing similarity matrices for a set of fingerprints.

    Parameters:
        dataset_fingerprint_dict (dict): A dictionary containing fingerprint data for different clients, i.e. keys: client IDs, values: fingerprints.

    Attributes:
        dataset_fingerprint_dict (dict): The input dictionary containing fingerprint data.
        client_names (list): List of client names derived from the keys of `dataset_fingerprint_dict`.
        num_clients (int): The total number of clients.
        similarity_matrix (numpy.ndarray or None): The similarity matrix calculated for the fingerprints.
            Initialized to None until `__call__()` is invoked.

    Methods:
        __init__(self, dataset_fingerprint_dict)
            Initializes the FingerprintSimilarityCalculator instance.

        __call__(self, recalculate=False, *args, **kwargs)
            Returns the similarity matrix. Calculates if not already available or if `recalculate=True`.

        calculate_similarity_matrix(self)
            [Abstract Method]
            Calculates the similarity matrix based on the fingerprints.
            Must be implemented by a subclass.

    Usage:
        fingerprint_calculator = FingerprintSimilarityCalculator(dataset_fingerprint_dict)
        similarity_matrix = fingerprint_calculator()  # Calls the calculator to get the similarity matrix.
        fingerprint_calculator.plot()  # Plots the similarity matrix as a heatmap.

    Note:
        This is the base class for fingerprint similarity calculators. Use a fitting subclass for actual calculations.
    """
    def __init__(self, dataset_fingerprint_dict):
        self.dataset_fingerprint_dict = dataset_fingerprint_dict
        self.client_names = list(dataset_fingerprint_dict.keys())
        self.num_clients = len(self.client_names)
        self.similarity_matrix = None
    
    def __call__(self, recalculate=False, *args, **kwargs):
        """
        Returns the similarity matrix. Calculates if not already available or if `recalculate=True`.

        Parameters:
            recalculate (bool, optional): Flag to force recalculation of the similarity matrix.
                Defaults to False.

        Returns:
            numpy.ndarray: The similarity matrix.
        """
        if self.similarity_matrix is None or recalculate:
            print('Calculating similarity matrix!')
            self.similarity_matrix = self.calculate_similarity_matrix()
        return self.similarity_matrix

    def calculate_similarity_matrix(self):
        """
        [Abstract Method]
        Calculates the similarity matrix based on the fingerprints.
        Must be implemented by a subclass.

        Raises:
            NotImplementedError: This is the base class for fingerprint similarity calculators.
                Use a fitting subclass!

        Returns:
            numpy.ndarray: The similarity matrix.
        """
        raise NotImplementedError('This is the base class for fingerprint similarity calculators. Use a fitting subclass!')
    

class PrincipalAngleFingerprintSimilarityCalculator(FingerprintSimilarityCalculator):
    def __init__(self, dataset_fingerprint_dict):
        super().__init__(dataset_fingerprint_dict)
        self.high_is_good = False
        
    def calculate_similarity_matrix(self):
        """
        Calculate the similarity matrix between client fingerprints. from https://github.com/MMorafah/PACFL (src/clustering/hierarchical_clustering)
        Low values indicate high similarity, high values indicate low similarity. (Proximity matrix)

        Args:
            dataset_fingerprints (dict): 
                A dictionary containing client IDs as keys and their corresponding fingerprints as values.

        Returns:
            numpy.ndarray:
                A similarity matrix where each element represents the similarity between two clients' fingerprints.

        """
        similarity_matrix = np.zeros([self.num_clients, self.num_clients])
        for idx1 in range(self.num_clients):
            for idx2 in range(self.num_clients):
                if idx1 == idx2: # skip diagonal elements because we can sometimes have numerical instailities here and really do not need to calculate this.
                    continue
                U1 = copy.deepcopy(self.dataset_fingerprint_dict[self.client_names[idx1]])
                U2 = copy.deepcopy(self.dataset_fingerprint_dict[self.client_names[idx2]])
                
                mul = np.clip(U1.T@U2 ,a_min =-1.0, a_max=1.0)
                similarity_matrix[idx1,idx2] = np.min(np.arccos(mul))*180/np.pi
            
        return similarity_matrix



def calculate_fingerprint_similarity(dataset_fingerprint_dict, similarity_calculator, *args, **kwargs): 
    """
    Calculates fingerprint similarity using the specified similarity calculator.

    Parameters:
        dataset_fingerprint_dict (dict): A dictionary of client IDs and their corresponding fingerprints.
        similarity_calculator (str): The name of the similarity calculator class to use.

    Returns:
        tuple: A tuple containing:
            - similarity_matrix (numpy.ndarray): The similarity matrix calculated based on fingerprints.
            - similarity_calculator (FingerprintSimilarityCalculator): The instantiated similarity calculator.

    Note:
        - The `dataset_fingerprint_dict` should be a dictionary with client IDs as keys and fingerprints as values.
        - The `similarity_calculator` parameter should be a string specifying a valid similarity calculator class.
        - The similarity calculator class must implement a __call__ method for similarity calculation.

    Raises:
        NotImplementedError: If the provided similarity_calculator string does not correspond to a valid class.
    """
    if similarity_calculator == 'PrincipalAngles':
        similarity_calculator = PrincipalAngleFingerprintSimilarityCalculator(dataset_fingerprint_dict)
    else: 
        raise NotImplementedError(f'Similarity calculator {similarity_calculator} is not implemented yet!')
    similarity_matrix = similarity_calculator()
    return similarity_matrix, similarity_calculator