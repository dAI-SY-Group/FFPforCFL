import numpy as np
import random



def class_cluster_target_assignment(targets, num_clients, clusters):
    """
    Assigns data samples to clients based on specified label clusters.

    Parameters:
    ----------
    targets : list or array-like
        A list or array of target labels corresponding to the data samples.
    
    num_clients : int
        The total number of clients to which data samples should be assigned.
    
    clusters : array-like of shape (n_clusters, n_classes_per_cluster)
        A 2D array where each row represents a cluster of class labels. Each client will be assigned samples
        from the classes specified in one of these clusters.

    Returns:
    -------
    dict
        A dictionary where keys are client indices (from 0 to num_clients-1) and values are arrays of sample indices 
        assigned to each client.

    Raises:
    ------
    AssertionError
        If the number of unique labels in `targets` does not match the number of unique labels in `clusters`.
        If the number of clients per class cluster is less than 1.

    Notes:
    -----
    - This function ensures that data samples are distributed among clients without overlap, based on the specified clusters.
    - The total number of clients should be divisible by the number of class clusters to ensure even distribution.
    - The indices assigned to each client are shuffled to ensure randomness.
    """
    unique_labels = np.unique(targets)
    assert len(unique_labels) == len(np.unique(clusters)), f'Unique labels and cluster labels do not match! Labels: {unique_labels}, Cluster labels: {np.unique(clusters)}'

    # number of different class label clusters
    nr_class_clusters = clusters.shape[0]
    # number of clients per class cluster
    nr_clients_per_class_cluster = num_clients // nr_class_clusters
    assert nr_clients_per_class_cluster > 0, f'Number of clients per class cluster must be greater than 0!'

    client_classes = np.tile(clusters, (nr_clients_per_class_cluster, 1))[:num_clients]

    #count how often each class needs to be split
    class_counts = np.unique(client_classes, return_counts=True)[1]

    # dictionary to store the indices of the samples for each client
    dict_users = {i: np.array([], dtype='int64') for i in range(num_clients)}
    targets = np.array(targets)

    # for each class...
    for _cls in unique_labels:
        # get the indices of the samples with the current class label
        class_idxs = np.where(targets == _cls)[0]
        # shuffle the indices
        np.random.shuffle(class_idxs)
        # split the indices into nr_clients_per_class_cluster parts
        class_idxs = np.array_split(class_idxs, class_counts[_cls])
        assigned_counter = 0
        # for each client...
        for i in range(num_clients):
            # if this client should get samples of the current class...
            if _cls in client_classes[i]:
                #append those indices to his list
                dict_users[i] = np.concatenate((dict_users[i], class_idxs[assigned_counter]), axis=0)
                assigned_counter += 1
    return dict_users

def generate_disjoint_shard_clusters(num_varying_clusters, num_classes_per_client, total_classes):
    """
    Generates disjoint shard clusters where each class is assigned to exactly one label cluster without overlap.

    Parameters:
    ----------
    num_varying_clusters : int
        The number of label clusters to generate.
    
    num_classes_per_client : int
        The number of classes each client should receive.
    
    total_classes : int
        The total number of classes available.

    Returns:
    -------
    np.ndarray
        A 2D array of shape (num_varying_clusters, num_classes_per_client) where each row represents a cluster of class labels.

    Raises:
    ------
    AssertionError
        If the product of `num_classes_per_client` and `num_varying_clusters` is not equal to `total_classes`.

    Example:
    -------
    >>> generate_disjoint_shard_clusters(5, 2, 10)
    array([[2, 4],
           [1, 5],
           [3, 9],
           [6, 8],
           [0, 7]])
    
    Notes:
    -----
    - This function ensures that all classes are used exactly once across the clusters.
    - The class labels are randomly shuffled before being assigned to clusters.
    """
    assert num_classes_per_client * num_varying_clusters == total_classes, "num_classes_per_client * num_varying_clusters should be equal to total_classes so that all classes are used!"
    cluster = np.zeros((num_varying_clusters, num_classes_per_client), dtype='int64')
    cluster_array = np.random.choice(total_classes, total_classes, replace=False)
    for i in range(num_varying_clusters):
        cluster[i] = cluster_array[i*num_classes_per_client: i*num_classes_per_client + num_classes_per_client]
    return cluster

def generate_overlapping_shard_clusters(num_varying_clusters, num_classes_per_client, total_classes):
    """
    Generates overlapping shard clusters where classes can be assigned to multiple clusters.

    Parameters:
    ----------
    num_varying_clusters : int
        The number of clusters to generate.
    
    num_classes_per_client : int
        The number of classes each client should receive.
    
    total_classes : int
        The total number of classes available.

    Returns:
    -------
    np.ndarray
        A 2D array of shape (num_varying_clusters, num_classes_per_client) where each row represents a cluster of class labels.

    Raises:
    ------
    AssertionError
        If not all classes are assigned to any clients, ensuring each class is included in the clusters.

    Example:
    -------
    >>> generate_overlapping_shard_clusters(5, 3, 10)
    array([ [7, 8, 5],
            [5, 6, 9],
            [9, 1, 2],
            [2, 3, 0],
            [0, 4, 7]])
    
    Notes:
    -----
    - This function allows classes to overlap between clusters, meaning a class can appear in multiple clusters.
    - The class labels are randomly shuffled before being assigned to clusters.
    """
    # Initialize the cluster array with zeros
    cluster = np.zeros((num_varying_clusters, num_classes_per_client), dtype='int64')
    
    # Randomly shuffle the class labels
    lst = np.random.choice(total_classes, total_classes, replace=False)
    
    # Assign classes to clusters
    for i in range(num_varying_clusters):
        start_index = (i * (num_classes_per_client - 1)) % total_classes
        if start_index + num_classes_per_client <= total_classes:
            cluster[i] = lst[start_index:start_index + num_classes_per_client]
        else:
            cluster[i] = np.concatenate((lst[start_index:], lst[:(start_index + num_classes_per_client) % total_classes]))
    
    assert len(np.unique(cluster)) == total_classes, "All classes should be assigned to any clients! Adjust num_varying_clusters or num_classes_per_client accordingly!"
    return cluster



def disjoint_shards_split(targets, num_clients, num_varying_clusters, num_classes_per_client, seed=None):
    """
    Splits the targets into disjoint shards and assigns them to clients.

    Parameters:
    ----------
    targets : list or array-like
        A list or array of target labels corresponding to the data samples.
    
    num_clients : int
        The total number of clients to which data samples should be assigned.
    
    num_varying_clusters : int
        The number of clusters to generate.
    
    num_classes_per_client : int
        The number of classes each client should receive.
    
    seed : int, optional
        A seed for the random number generator to ensure reproducibility. Default is None.

    Returns:
    -------
    dict
        A dictionary where keys are client indices (from 0 to num_clients-1) and values are arrays of sample indices assigned to each client.

    Raises:
    ------
    AssertionError
        If the number of unique labels in `targets` does not match the number of unique labels in the generated clusters.
        If the number of clients per class cluster is less than 1.
    
    Notes:
    -----
    - This function ensures that data samples are distributed among clients without overlap, based on the specified clusters.
    - The total number of clients should be divisible by the number of class clusters to ensure even distribution.
    - The indices assigned to each client are shuffled to ensure randomness.
    """
    if seed:
        np.random.seed(seed)
    num_classes = len(np.unique(targets))   
    clusters = generate_disjoint_shard_clusters(num_varying_clusters, num_classes_per_client, num_classes)
    return class_cluster_target_assignment(targets, num_clients, clusters)


def overlapping_shards_split(targets, num_clients, num_varying_clusters, num_classes_per_client, seed=None):
    """
    Splits the targets into overlapping shards and assigns them to clients.

    Parameters:
    ----------
    targets : list or array-like
        A list or array of target labels corresponding to the data samples.
    
    num_clients : int
        The total number of clients to which data samples should be assigned.
    
    num_varying_clusters : int
        The number of clusters to generate.
    
    num_classes_per_client : int
        The number of classes each client should receive.
    
    seed : int, optional
        A seed for the random number generator to ensure reproducibility. Default is None.

    Returns:
    -------
    dict
        A dictionary where keys are client indices (from 0 to num_clients-1) and values are arrays of sample indices assigned to each client.

    Raises:
    ------
    AssertionError
        If not all classes are assigned to any clients, ensuring each class is included in the clusters.
    
    Notes:
    -----
    - This function allows classes to overlap between clusters, meaning a class can appear in multiple clusters.
    - The total number of clients should be divisible by the number of class clusters to ensure even distribution.
    - The indices assigned to each client are shuffled to ensure randomness.
    """
    if seed:
        np.random.seed(seed)
    num_classes = len(np.unique(targets))
    clusters = generate_overlapping_shard_clusters(num_varying_clusters, num_classes_per_client, num_classes)
    return class_cluster_target_assignment(targets, num_clients, clusters)