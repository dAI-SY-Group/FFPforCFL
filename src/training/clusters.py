from typing import Iterable 

import numpy as np


def calculate_clusters(similarity_matrix, clusterer_config):
    """
    Calculates clusters based on the provided similarity matrix and clusterer configuration.

    Parameters:
        similarity_matrix (numpy.ndarray): The similarity matrix used for clustering.
        clusterer_config (munch.Munch): A configuration object specifying clusterer parameters.

    Returns:
        tuple: A tuple containing:
            - clusters (numpy.ndarray): An array containing cluster labels for each data point.
            - clusterer: The instantiated clusterer.

    Note:
        - The `similarity_matrix` should be a square matrix representing pairwise similarities between data points.
        - The `clusterer_config` should be an object with attributes specifying the type and parameters of the clusterer.

    Raises:
        NotImplementedError: If the specified clusterer type is not implemented.
    """
    if clusterer_config.name == 'PACFL':
        clusterer = PACFLCluster(similarity_matrix, clusterer_config.stopping_threshold, clusterer_config.linkage)
    else:
        raise NotImplementedError(f'Clusterer {clusterer_config.name} not implemented!')
    return clusterer(), clusterer
    

class Cluster:
    def __init__(self, proximity_matrix, *args, **kwargs):
        self.proximity_matrix = proximity_matrix
        self.clusters = None
        self.cluster_labels = None

    def __call__(self, recalculate=False):
        if self.clusters is None or recalculate:
            self.cluster()
        return self.clusters

    def create_cluster_matrix(self, cluster_assignments):
            cluster_matrix = np.zeros_like(self.proximity_matrix, dtype=int)

            for i, cluster in enumerate(cluster_assignments):
                #make sure that all index combinations in the current cluster become the label "i+1"
                for index1 in cluster:
                    for index2 in cluster:
                        cluster_matrix[index1, index2] = i+1
            return cluster_matrix

    

class PACFLCluster(Cluster):
    def __init__(self, proximity_matrix, stopping_threshold=3.5, linkage='average'):
        super().__init__(proximity_matrix)
        self.stopping_threshold = stopping_threshold
        self.linkage = linkage
    
    def cluster_range(self, thresh_step=0.1):
        # test different stopping thresholds and return a dict that maps from a number of clusters to a a list of clusters for that number of clusters
        max_num_clusters = self.proximity_matrix.shape[0]
        clusters = {}
        threshs = {}
        start_thresh = 0
        while (len(clusters) < max_num_clusters) and not 1 in clusters:
            self.stopping_threshold = start_thresh
            self.cluster()
            clusters[len(self.clusters)] = self.clusters
            threshs[len(self.clusters)] = start_thresh
            start_thresh += thresh_step
        self.cluster_range = clusters
        self.cluster_range_threshs = threshs
        print(f'Found {len(clusters)} cluster setups for these stopping thresholds: {threshs}')
        return clusters
                    

    def cluster(self):
        '''
        FROM PACFL repo(https://github.com/MMorafah/PACFL/blob/main/src/clustering/hierarchical_clustering.py)
        Hierarchical Clustering Algorithm. It is based on single linkage, finds the minimum element and merges
        rows and columns replacing the minimum elements. It is working on adjacency matrix. 

        :return: clusters
        '''
        A = self.proximity_matrix.copy()

        label_assg = {i: i for i in range(A.shape[0])}
        
        step = 0
        while A.shape[0] > 1:
            np.fill_diagonal(A,-np.NINF)
            #print(f'step {step} \n {A}')
            step+=1
            ind=np.unravel_index(np.argmin(A, axis=None), A.shape)

            if A[ind[0],ind[1]] > self.stopping_threshold:
                #print('Breaking HC')
                break
            else:
                np.fill_diagonal(A, 0)
                if self.linkage == 'maximum':
                    Z = np.maximum(A[:,ind[0]], A[:,ind[1]])
                elif self.linkage == 'minimum':
                    Z = np.minimum(A[:,ind[0]], A[:,ind[1]])
                elif self.linkage == 'average':
                    Z = (A[:,ind[0]] + A[:,ind[1]])/2
                
                A[:,ind[0]]=Z
                A[:,ind[1]]=Z
                A[ind[0],:]=Z
                A[ind[1],:]=Z
                A = np.delete(A, (ind[1]), axis=0)
                A = np.delete(A, (ind[1]), axis=1)

                if type(label_assg[ind[0]]) == list: 
                    label_assg[ind[0]].append(label_assg[ind[1]])
                else: 
                    label_assg[ind[0]] = [label_assg[ind[0]], label_assg[ind[1]]]

                label_assg.pop(ind[1], None)

                temp = []
                for k,v in label_assg.items():
                    if k > ind[1]: 
                        kk = k-1
                        vv = v
                    else: 
                        kk = k 
                        vv = v
                    temp.append((kk,vv))

                label_assg = dict(temp)

        clusters = []

        def flatten(items):
            """Yield items from any nested iterable; see Reference."""
            for x in items:
                if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
                    for sub_x in flatten(x):
                        yield sub_x
                else:
                    yield x

        for k in label_assg.keys():
            if type(label_assg[k]) == list:
                clusters.append(list(flatten(label_assg[k])))
            elif type(label_assg[k]) == int: 
                clusters.append([label_assg[k]])
                
        self.clusters = clusters