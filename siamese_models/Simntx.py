# State Representation Learning with Task-Irrelevant Factors of Variation in Robotics
# Anonymous Authors 2021

import tensorflow.python.keras.backend as K
import tensorflow as tf

from siamese_models.losses import _dot_simililarity_dim1 as sim_func_dim1, _dot_simililarity_dim2 as sim_func_dim2 
from siamese_models.losses import get_negative_mask

from tensorflow.keras import models , optimizers , losses ,activations , callbacks, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Input , Flatten, Dense, Reshape, Lambda
from tensorflow.keras.optimizers import RMSprop
import tensorflow.keras.layers as layers 
import time
import os
import numpy as np

def manhattan_distance(vects):
        x, y = vects

        return K.sum( K.abs( x-y),axis=1,keepdims=True)

def euclidean_distance(vects):
        x, y = vects

        sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
        return K.sqrt(K.maximum(sum_square, K.epsilon()))

class Recognizer_SC(Model) :


    def __init__(self, img_shape,dense_dim=12):

        input_A = Input(shape=img_shape) 
        input_B = Input(shape=img_shape) 
        base_network = self.create_base_network(img_shape,dense_dim)
        base_network.summary()
        output_A = base_network(input_A)   
        #create the model 
        super(Recognizer_SC, self).__init__(input_A, output_A)

        self.loss_tracker = tf.keras.metrics.Mean(name="loss")

    def compile(self, batch_size):

        self.BATCH_SIZE = batch_size

        rms = RMSprop()
        super(Recognizer_SC,self).compile(optimizer=rms)
        self.summary()

    def setClassifyCallBack(self, fun):
        self.ccb_ = fun

    def getClassifyCallBack(self):
        return self.ccb_

    @tf.function
    def train_step(self, data):

        #hyperpparameters
        criterion = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, 
                                                          reduction=tf.keras.losses.Reduction.SUM)
        temperature = 0.1
        x, y = data 
        xa, xb = x 

        BATCH_SIZE = self.BATCH_SIZE
        negative_mask = get_negative_mask(BATCH_SIZE)

        with tf.GradientTape() as tape:
            zis = self(xa) #forward pass 
            zjs = self(xb) #forward pass 
        
            # normalize projection feature vectors
            zis = tf.math.l2_normalize(zis, axis=1)
            zjs = tf.math.l2_normalize(zjs, axis=1)
    
            l_pos = sim_func_dim1(zis, zjs)
            l_pos = tf.reshape(l_pos, (BATCH_SIZE, 1))
            l_pos /= temperature
    
            negatives = tf.concat([zjs, zis], axis=0)
    
            loss = 0
    
            for positives in [zis, zjs]:
                l_neg = sim_func_dim2(positives, negatives)
    
                labels = tf.zeros(BATCH_SIZE, dtype=tf.int32)
    
                l_neg = tf.boolean_mask(l_neg, negative_mask)
                l_neg = tf.reshape(l_neg, (BATCH_SIZE, -1))
                l_neg /= temperature
    
                logits = tf.concat([l_pos, l_neg], axis=1) 
                loss += criterion(y_pred=logits, y_true=labels)
    
            loss = loss / (2 * BATCH_SIZE)
    
        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Compute our own metrics (ADD the nearest neighbor here as well)
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    @property
    def metrics(self):
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch
        # or at the start of `evaluate()`.
        # If you don't implement this property, you have to call
        # `reset_states()` yourself at the time of your choosing.
        return [self.loss_tracker]

    def create_base_network(self, img_shape,dense_dim): 

        input_x = Input(img_shape) 
        pool_size_1 = (2 , 2)
        x =  MaxPooling2D(pool_size=pool_size_1)(input_x)

        kernel_size_1 = (4, 4)
        kernel_size_2 = (8 ,8)
        kernel_size_3 = (16 ,16)
        pool_size_1 = (2 , 2)
        pool_size_2 = (16 , 16)
        pool_size_3 = (4 , 4)
        
        x =  Conv2D(4, kernel_size=kernel_size_1, activation = "relu" )(x) 
        x =  Conv2D(4, kernel_size=kernel_size_1, activation = "relu" )(x) 
        x =  MaxPooling2D(pool_size=pool_size_2)(x)
        x = Flatten()(x)
            
        x= Dense(dense_dim)(x)
        #Normalize the last layer (this makes the euclidean distance equivalent to the cosine distance) 
        x= Lambda( lambda t:K.l2_normalize(t, axis=1))(x)
            
        return models.Model(input_x, x,name="dense") 
        
    def eucl_dist_shape(self, shapes):
        shape1, shape2 = shapes
        return (shape1[0], 1)

    #def create_base_network(input_shape):
    def fit(self, X, Y ,  hyperparameters  ):
        super(Recognizer_SC,self).fit( X  , Y ,
            batch_size=hyperparameters[ 'batch_size' ] ,
            epochs=hyperparameters[ 'epochs' ] ,
            callbacks=hyperparameters[ 'callbacks'],
            )
     
    def save_model(self , file_path ):
            super(Recognizer_SC,self).save(file_path )
