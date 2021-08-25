# State Representation Learning with Task-Irrelevant Factors of Variation in Robotics
# Anonymous Authors 2021

import numpy as np
import tensorflow.keras as keras
from sklearn.neighbors import NearestNeighbors
from collections import Counter

class ClassifyCallback(keras.callbacks.Callback):

    def __init__(self, X_tr, X_te, S_tr, S_te):
        self.X_tr=X_tr
        self.X_te=X_te
        self.S_tr=np.array([str(s) for s in S_tr])
        self.S_te=np.array([str(s) for s in S_te])

    def knn_accuracy(self, Z_tr, Z_te, n):
        nbrs = NearestNeighbors(n_neighbors=n, algorithm='kd_tree').fit(Z_tr)
        distances, indices = nbrs.kneighbors(Z_te)

        #Finds the most common element
        S_pred = np.array([Counter(s).most_common(1)[0][0] for s in self.S_tr[indices]])
        acc= np.array(self.S_te == S_pred).mean()
        return acc


    def compute_accuracy(self):
        model_layer = self.model.get_layer("dense") 

        #Split in batches of 10 to fit in memory
        batches = [self.X_tr[i*10:(i+1)*10] for i in range(self.X_tr.shape[0]//10)]
        Z_tr = []
        for b in batches:
            for z in  model_layer(b):
              Z_tr.append(z)

        Z_tr = np.array(Z_tr)
        Z_te = np.array(model_layer(self.X_te)) 



        acc1 = self.knn_accuracy(Z_tr, Z_te, 1)
        acc3 = self.knn_accuracy(Z_tr, Z_te, 3)

        return (acc1, acc3)



    def on_epoch_end(self, epoch, logs=None):
        pass

