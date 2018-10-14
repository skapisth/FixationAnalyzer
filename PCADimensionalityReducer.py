import numpy as np
from sklearn.decomposition import PCA

class PCADimensionalityReducer(object):
    def __init__(self,
                    n_components,
                    whiten=False):

        self.n_components = n_components
        self.whiten = whiten
