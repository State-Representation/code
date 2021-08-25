# State Representation Learning with Task-Irrelevant Factors of Variation in Robotics
# Anonymous Authors 2021

import tensorflow.python.keras.backend as K
from tensorflow.keras import models , optimizers , losses ,activations , callbacks
from tensorflow.keras.layers import Conv2D, Dropout, MaxPooling2D, Input , Flatten, Dense, Reshape, Lambda
from tensorflow.keras.optimizers import RMSprop
import tensorflow.keras.layers as layers
import tensorflow.keras.callbacks as callbacks
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

def custom_loss(margin=1, square_loss = 1):
    # Create a loss function that calculates what you want
    def contrastive_loss_in(y_true,y_pred):
        print(margin)
        pred_loss = y_pred
        margin_loss = K.maximum(margin - y_pred, 0)
        if square_loss:
            pred_loss = K.square(pred_loss)
            margin_loss = K.square(margin_loss)

        return K.mean(y_true * pred_loss + (1 - y_true) * margin_loss)

    # Return a function
    return contrastive_loss_in
#

def contrastive_loss(y_true, y_pred, margin=1, square_loss = 1):
        '''Contrastive loss from Hadsell-et-al.'06
        http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
        '''


        pred_loss = y_pred
        margin_loss = K.maximum(margin - y_pred, 0)
        if square_loss:
            pred_loss = K.square(pred_loss)
            margin_loss = K.square(margin_loss)



        return K.mean(y_true * pred_loss + (1 - y_true) * margin_loss)



class Recognizer (object) :

    def __init__( self):
        print("Created Empty Instance of Recognizer")


    
    def compile(self, img_shape,dense_dim=12,distance_type='2'):
        base_network = self.create_base_network(img_shape,dense_dim)
        base_network.summary()

        input_A = Input(shape=img_shape)
        input_B = Input(shape=img_shape)

        output_A = base_network(input_A)
        output_B = base_network(input_B)

        if distance_type == '2':
            distance = Lambda(euclidean_distance)([output_A, output_B])
        if distance_type == '1':
            distance = Lambda(manhattan_distance)([output_A, output_B])



        self._model = models.Model([input_A, input_B], distance)
        self._model.summary()
        self._model.min_dist = 0.5

        rms = RMSprop(learning_rate=0.0005)#learning_rate=0.001
        self._model.compile(loss=contrastive_loss, optimizer=rms, metrics=[self.accuracy])
        #self._model.compile(loss=custom_loss(self._model.min_dist), optimizer=rms, metrics=[self.accuracy])

    def setClassifyCallBack(self, fun):
        self.ccb_ = fun

    def getClassifyCallBack(self):
        return self.ccb_


    def create_base_network(self, img_shape,dense_dim):

        input_x = Input(img_shape)
        kernel_size_1 = (4, 4)
        pool_size_1 = (2 , 2)
        pool_size_2 = (16 , 16)
        x =  MaxPooling2D(pool_size=pool_size_1)(input_x)  
        x =  Conv2D(4, kernel_size=kernel_size_1, activation = "relu" )(x)
        x =  Conv2D(4, kernel_size=kernel_size_1, activation = "relu" )(x)
        x =  MaxPooling2D(pool_size=pool_size_2)(x)

        x =  Flatten()(x)

        x= Dense(dense_dim)(x)
        #Normalize the last layer (this makes the euclidean distance equivalent to the cosine distance) 
        
        x= Lambda( lambda t:K.l2_normalize(t, axis=1))(x)
 

        return models.Model(input_x, x,name="dense")

    def eucl_dist_shape(self, shapes):
        shape1, shape2 = shapes
        return (shape1[0], 1)

    
    def accuracy(self, y_true, y_pred,c_t=0.5):
       '''Compute classification accuracy with a fixed threshold on distances.
       '''
       return K.mean(K.equal(y_true, K.cast(y_pred < c_t, y_true.dtype)))

    def compute_accuracy(self, y_true, y_pred,c_t=0.5):
        '''Compute classification accuracy with a fixed threshold on distances.
        '''
        pred = y_pred.ravel() < c_t
        return np.mean(pred == y_true)



    #def create_base_network(input_shape):
    def fit(self, X, Y ,  hyperparameters  ):

        self._model.my_validation_data  = hyperparameters[ 'val_data' ]
        print(self._model.my_validation_data)
        self._model.fit( X  , Y ,
            batch_size=hyperparameters[ 'batch_size' ] ,
            epochs=hyperparameters[ 'epochs' ] ,
            callbacks=hyperparameters[ 'callbacks'],
            validation_data=hyperparameters[ 'val_data' ]
            )

    def evaluate(self , test_X , test_Y  ) :
        return self._model.evaluate(test_X, test_Y)

    def predict(self, X  ):
        predictions = self._model.predict(X)
        return predictions

    def summary(self):
        self._model.summary()

    def save_model(self , file_path ):
        self._model.save(file_path )

    def load_model(self , file_path ):
        self._model = models.load_model(file_path, custom_objects={'contrastive_loss': contrastive_loss})


    def layer_output(self, X,name ):
        inp = self._model.input                                           # input placeholder
        model = self._model
        model_layer = model.get_layer(name)
        out_latent = model_layer(X)
        return np.array(out_latent)
