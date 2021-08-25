# State Representation Learning with Task-Irrelevant Factors of Variation in Robotics
# Anonymous Authors 2021

import numpy as np
import random
import pickle
import os, os.path
import pickle
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from importlib.machinery import SourceFileLoader
import algorithms as alg

import torch
from torch.autograd import Variable
from matplotlib import pyplot as plt
from matplotlib.axes._axes import _log as matplotlib_axes_logger
matplotlib_axes_logger.setLevel('ERROR')

from siamese_models.Siamese import Recognizer
from siamese_models.ClassCallback import ClassifyCallback 
from siamese_models.Simntx import Recognizer_SC
from sklearn.neighbors import NearestNeighbors
from collections import Counter
from tensorflow.keras import models 

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    print(e)



def load_data(filename):
  data_full = pickle.load(open(filename+".pkl", 'rb'))
  return data_full

def split_train_test_similar_only(data):
    XA = [] 
    XB = [] 
    SA = [] 
    SB = [] 

    
    for (img1, img2, sim, state1, state2) in data: 

        if isinstance(state1, (str)):
            state1 =np.array([int(i) for i in state1.split(" ")]) 
            state2 =np.array([int(i) for i in state2.split(" ")]) 

        #if (state1 == state2).all():
        if (sim == 0):
            if not (state1 == state2).all():
                a=1/0
            SA.append(state1)
            SB.append(state2)
            XA.append(img1/255.)
            XB.append(img2/255.)
   
    N = len(XA)
    print ("Created {} similar states from the dataset".format(N))
    data=[]


    #convert to numpy arrays
    XA = np.array(XA)
    XB = np.array(XB)
    SA = np.array(SA)
    SB = np.array(SB)
    print ("State dims:", SB.shape)
    print ("Image dims:",XB.shape)
    Y = np.zeros([N,1]) #Not need keeping it for compatability
    print (Y.shape)

    #Randomize the input
    p = np.random.permutation(N)
    XA = XA[p]
    XB = XB[p]
    SA = SA[p]
    SB = SB[p]
    Y = Y[p] 

    #Split in Train/Test set
    #Train set is 90% of the datapoints 

    N_tr = int(np.ceil(0.9 * N))
    XA_tr = XA[:N_tr]
    XB_tr = XB[:N_tr]
    SA_tr = SA[:N_tr]
    SB_tr = SB[:N_tr]
    Y_tr = Y[:N_tr] 
   
    XA_te = XA[N_tr:]
    XB_te = XB[N_tr:]
    SA_te = SA[N_tr:]
    SB_te = SB[N_tr:]
    Y_te = Y[N_tr:] 

    return XA_tr, XB_tr, Y_tr, XA_te, XB_te, Y_te, SA_tr, SB_tr, SA_te, SB_te



def split_train_test(data):
    XA = []
    XB = []
    SA = [] 
    SB = [] 
    Y = []

    
    
    count_sim=0
    count_not_sim=0
    for (img1, img2, action, state1, state2) in data:
        if isinstance(state1, (str)):
            state1 =np.array([int(i) for i in state1.split(" ")]) 
            state2 =np.array([int(i) for i in state2.split(" ")]) 

        SA.append(state1)
        SB.append(state2)
        XA.append(img1/255.)
        XB.append(img2/255.)

        sim=-1
        if action==1 :
            sim=0
        if action==0:
            sim=1
        Y.append(float(sim))

        if sim==0:
          count_not_sim+=1
          #assert (state1 != state2).all(), "Sim error in dataset"
        else:
          count_sim+=1
          #assert (state1 == state2).all(), "Sim error in dataset"

   
    print(" got " + str(count_sim) + " sim pairs")
    print(" got " + str(count_not_sim) + " not sim pairs")

    N = len(XA)
    print ("Created {} states from the dataset".format(N))
    data=[]
    #convert to numpy arrays
    XA = np.array(XA)
    XB = np.array(XB)
    SA = np.array(SA)
    SB = np.array(SB)
    Y = np.array(Y)

    #Randomize the input
    p = np.random.permutation(N)
    XA = XA[p]
    XB = XB[p]
    SA = SA[p] 
    SB = SB[p] 
    Y = Y[p]

    #Split in Train/Test set
    #Train set is 90% of the datapoints
    N_tr = int(np.ceil(0.9 * N))
    XA_tr = XA[:N_tr]
    XB_tr = XB[:N_tr]
    SA_tr = SA[:N_tr]
    SB_tr = SB[:N_tr]
    Y_tr = Y[:N_tr]

    XA_te = XA[N_tr:]
    XB_te = XB[N_tr:]
    SA_te = SA[N_tr:]
    SB_te = SB[N_tr:]
    Y_te = Y[N_tr:]

    return XA_tr, XB_tr, Y_tr, XA_te, XB_te, Y_te, SA_tr, SB_tr, SA_te, SB_te

"""
Trains a recognizer to distinquish between similar/disimillar images
"""

def train_siamese(model_name,train_dataset):
    data=load_data("datasets/"+train_dataset)
    print(len(data))
    #preproces for training
    XA_tr, XB_tr, Y_tr, XA_te, XB_te, Y_te, SA_tr, SB_tr, SA_te, SB_te = split_train_test(data)

    #Concatenate in a single form
    X_tr = np.concatenate((XA_tr, XB_tr))
    X_te = np.concatenate((XA_te, XB_te))
    S_tr = np.concatenate((SA_tr, SB_tr))
    S_te = np.concatenate((SA_te, SB_te))

    recognizer = Recognizer()
    recognizer.compile(XA_tr[0].shape)
    recognizer.setClassifyCallBack(ClassifyCallback(X_tr, X_te, S_tr, S_te))

    #we can have this poarameter shere as they should be the same for all the models anyhow
    parameters = {
        'batch_size' : 32,
        'epochs' : 100,
        'callbacks' : [recognizer.getClassifyCallBack()], #DistanceCallback()] , # [ TensorBoard( log_dir='logs/{}'.format( time.time() ) ) ] ,
        'val_data' :( [XA_te, XB_te], Y_te)
    }

    recognizer.fit( [XA_tr, XB_tr], Y_tr, hyperparameters=parameters)
    print(model_name)

    Y_pred = recognizer.predict([XA_tr, XB_tr])
    tr_acc = recognizer.compute_accuracy(Y_tr, Y_pred)
    Y_pred = recognizer.predict([XA_te, XB_te])
    te_acc = recognizer.compute_accuracy(Y_te, Y_pred)

    print('* Accuracy on training set(' + str(len(XA_tr)) + '): %0.2f%%' % (100 * tr_acc))
    print('* Accuracy on test set(' + str(len(XA_te)) + '): %0.2f%%' % (100 * te_acc))

    acc1, acc3 = recognizer.getClassifyCallBack().compute_accuracy() 
    print (" True Accuracy Neighbors:1: {:2.2f} ".format(acc1))
    print (" True Accuracy Neighbors:3: {:2.2f} ".format(acc3))

    #save model
    if (not os.path.exists("./models/"+ model_name)):
      os.mkdir("./models/"+ model_name)

    f = open('models/'+model_name +'/' + model_name+"_logs.txt","w")
    f.write(model_name+'\n')
    f.write('* Accuracy on training set(' + str(len(XA_tr)) + '): %0.2f%%' % (100 * tr_acc) +'\n')
    f.write('* Accuracy on test set(' + str(len(XA_te)) + '): %0.2f%%' % (100 * te_acc) +'\n')
    f.write("*  True Accuracy Neighbors:1: {:2.2f} \n".format(acc1))
    f.write("*  True Accuracy Neighbors:3: {:2.2f} \n".format(acc3))
    f.close()

    f.close()
    print('models/'+model_name +'/'+ model_name+"_logs.txt")

    recognizer.save_model('models/'+model_name +'/'+model_name+'_siamese_model.h5')


def train_simntx(model_name,train_dataset):
    data=load_data("datasets/"+train_dataset)
    print(len(data))
    #preproces for training 
    XA_tr, XB_tr, Y_tr, XA_te, XB_te, Y_te, SA_tr, SB_tr, SA_te, SB_te = split_train_test_similar_only(data)


    #add reduntant samples
    batch_s=32
    needed_data_idx=(int(len(XA_tr)/batch_s))*batch_s+batch_s
    XA_tr=list(XA_tr)
    XB_tr=list(XB_tr)
    Y_tr=list(Y_tr)
    SA_tr=list(SA_tr)
    SB_tr=list(SB_tr)
    defficit=needed_data_idx-len(XA_tr)
    print("-------")
    print(len(XA_tr))
    print(needed_data_idx)
    print(defficit)
    for i in range(defficit):

        XA_tr.append(random.choice(XA_tr))
        XB_tr.append(random.choice(XB_tr))
        Y_tr.append(random.choice(Y_tr))
        SA_tr.append(random.choice(SA_tr))
        SB_tr.append(random.choice(SB_tr))

    XA_tr=np.array(XA_tr)
    XB_tr=np.array(XB_tr)
    Y_tr=np.array(Y_tr)
    SA_tr=np.array(SA_tr)
    SB_tr=np.array(SB_tr)


    X_tr = np.concatenate((XA_tr, XB_tr))
    X_te = np.concatenate((XA_te, XB_te))
    S_tr = np.concatenate((SA_tr, SB_tr))
    S_te = np.concatenate((SA_te, SB_te))
    recognizer = Recognizer_SC(XA_tr[0].shape)
    recognizer.compile(batch_s)
    recognizer.setClassifyCallBack(ClassifyCallback(X_tr, X_te, S_tr, S_te))

    #we can have this poarameter shere as they should be the same for all the models anyhow
    parameters = {
        'batch_size' : batch_s,
        'epochs' : 100,
        'callbacks' : [recognizer.getClassifyCallBack()] ,
    }

    recognizer.fit( [XA_tr, XB_tr], Y_tr, hyperparameters=parameters)

    # compute final accuracy, based on k-nn classification accuracy
    acc1, acc3 = recognizer.getClassifyCallBack().compute_accuracy() 

    print(model_name)
    print (" True Accuracy Neighbors:1: {:2.2f} ".format(acc1))
    print (" True Accuracy Neighbors:3: {:2.2f} ".format(acc3))

    #save model
    if (not os.path.exists("./models/"+ model_name)):
      os.mkdir("./models/"+ model_name)

    f = open('models/'+model_name +'/' + model_name+"_logs.txt","w") 
    f.write(model_name+'\n')
    f.write("*  True Accuracy Neighbors:1: {:2.2f} \n".format(acc1))
    f.write("*  True Accuracy Neighbors:3: {:2.2f} \n".format(acc3))
    f.close()
    print('models/'+model_name +'/'+ model_name+"_logs.txt")

    recognizer.save_model('models/'+model_name +'/'+model_name+'_siamclear_model')


def knn_accuracy(Z_tr, Z_te, S_tr, S_te, n):
        nbrs = NearestNeighbors(n_neighbors=n, algorithm='kd_tree').fit(Z_tr)
        distances, indices = nbrs.kneighbors(Z_te)

        #Finds the most common element
        S_pred = np.array([Counter(s).most_common(1)[0][0] for s in S_tr[indices]])
        acc = np.array(S_te == S_pred).mean()
        return acc

def train_pca(model_name,train_dataset):
    data=load_data("datasets/"+train_dataset)
    print(len(data))
    #preproces for training
    XA_tr, XB_tr, Y_tr, XA_te, XB_te, Y_te, SA_tr, SB_tr, SA_te, SB_te = split_train_test(data)

    #Concatenate in a single form
    X_tr = np.concatenate((XA_tr, XB_tr))
    X_te = np.concatenate((XA_te, XB_te))
    S_tr = np.concatenate((SA_tr, SB_tr))
    S_te = np.concatenate((SA_te, SB_te))


    #Latent space dimention
    
    pca = PCA(n_components=12)

    #Flatten the matrices
    X_tr = np.array([el.flatten() for el in X_tr])
    X_te = np.array([el.flatten() for el in X_te])

    Z_tr = pca.fit_transform(X_tr)
    Z_te = pca.transform(X_te)
    print(model_name)

    #Make the states strings
    S_tr=np.array([str(s) for s in S_tr])
    S_te=np.array([str(s) for s in S_te])


    acc1 = knn_accuracy(Z_tr, Z_te, S_tr, S_te, 1)
    acc3 = knn_accuracy(Z_tr, Z_te, S_tr, S_te, 3)
    print ("True Accuracy Neighbors:1: {:2.2f} ".format(acc1))
    print ("True Accuracy Neighbors:3: {:2.2f} ".format(acc3))

    #save model
    if (not os.path.exists("./models/"+ model_name)):
      os.mkdir("./models/"+ model_name)

    f = open('models/'+model_name +'/' + model_name+"_logs.txt","w")
    f.write(model_name+'\n')
    f.write("* True Accuracy Neighbors:1: {:2.2f} \n".format(acc1))
    f.write("* True Accuracy Neighbors:3: {:2.2f} \n".format(acc3))
    f.close()

    print('models/'+model_name +'/'+ model_name+"_logs.txt")

    
    pickle.dump(pca, open('models/'+model_name +'/'+model_name+'_pca_model.h5',"wb"))




def make_nice_labels(labels):
  print(len(labels))
  nice_labels=[]
  print(labels[0])
  labels=[str(i) for i in labels]
  u_lable=list(np.unique(labels,axis=0))
  u_lable=sorted(u_lable)
  for lable in labels:
    t_lable=u_lable.index(lable)
    nice_labels.append(t_lable)
    if min(nice_labels)<0:
      print("Error")
      a=1/0
  return nice_labels


def nice_colors(num_color):
  colors=[]
  for i in range(num_color):
    colors.append(np.array([random.uniform(0, 1),random.uniform(0, 1),random.uniform(0, 1)]))
  return colors



def t_snes_plot(latent_map,model_name,infer_dataset):
    X_zs=[]
    Y_zs=[]
    target_ids=[]
    print("loaded: " +str(len(latent_map)))
    for pair in latent_map:
        #print(pair)
        X_zs.append(pair[0])
        X_zs.append(pair[1])
        Y_zs.append(pair[3])
        Y_zs.append(pair[4])

    tsne = TSNE(n_components=2, random_state=0)
    Y_zs=make_nice_labels(Y_zs)
    Y_zs=np.array(Y_zs)
    target_ids = np.unique(Y_zs,axis=0)
    X_2d = tsne.fit_transform(np.array(X_zs)) 


    colors=nice_colors(len(target_ids))
    for i, c, label in zip(target_ids, colors, Y_zs):
      #print(i)
      i=np.array(i)
      idx_plot=[]
      for j in range(Y_zs.shape[0]):
          if np.array_equal(i,Y_zs[j]):
              idx_plot.append(j)
      plt.scatter(X_2d[idx_plot, 0], X_2d[idx_plot, 1], c=c, label=i)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left' , ncol=3)

    plt.title(model_name+"_"+infer_dataset)   

    plt.savefig('models/'+ model_name + '/'+model_name+'_t-sne_plot_' +infer_dataset+ '.png',bbox_inches = 'tight',
    pad_inches = 0.1)
    plt.clf()
    plt.close()
    


def infer_pca(model_name,infer_dataset):
    data=load_data("datasets/"+infer_dataset)
    
    pca= pickle.load(open('models/'+ model_name + '/'+model_name+'_pca_model.h5','rb'))

    latent_map=[]
    
    for (img1, img2, sim, state1, state2) in data:

      x1=img1/255.
      x1=np.expand_dims(x1, axis=0)
      x1 = x1.flatten()
      x1 = x1.reshape(1,-1)
      z1 = pca.transform(x1)

      x2=img2/255.
      x2=np.expand_dims(x2, axis=0)
      x2 = x2.flatten()
      x2 = x2.reshape(1,-1)
      z2 = pca.transform(x2)
      latent_map.append((np.squeeze(z1),np.squeeze(z2),sim,state1,state2))
  

    pkl_filename='models/'+ model_name + '/'+model_name+'_z_encodings_' + infer_dataset
    with open(pkl_filename + ".pkl", 'wb') as f:
        pickle.dump(latent_map, f)
    print("inference done")

    #make t-snes plot
    t_snes_plot(latent_map,model_name,infer_dataset)
    print("t-sne done")


def infer_siamese(model_name,infer_dataset):
    data=load_data("datasets/"+infer_dataset)
    recognizer = Recognizer()
    recognizer.load_model('models/'+ model_name + '/'+model_name+'_siamese_model.h5')
    latent_map=[]
    
    for (img1, img2, sim, state1, state2) in data:

      x1=img1/255.
      x1=np.expand_dims(x1, axis=0)
      z1 = recognizer.layer_output(np.array(x1),"dense")
      x2=img2/255.
      x2=np.expand_dims(x2, axis=0)
      z2 = recognizer.layer_output(np.array(x2),"dense")
      latent_map.append((np.squeeze(z1),np.squeeze(z2),sim,state1,state2))


    pkl_filename='models/'+ model_name + '/'+model_name+'_z_encodings_' + infer_dataset
    with open(pkl_filename + ".pkl", 'wb') as f:
        pickle.dump(latent_map, f)
    print("inference done")

    #make t-snes plot
    t_snes_plot(latent_map,model_name,infer_dataset)
    print("t-sne done")



def infer_simntx(model_name,infer_dataset):
    data=load_data("datasets/"+infer_dataset)
    recognizer = models.load_model('models/'+ model_name + '/'+model_name+'_siamclear_model')
                     
    latent_map=[]
    
    for (img1, img2, sim, state1, state2) in data: 

      x1=img1/255.
      x1=np.expand_dims(x1, axis=0)
      z1 = recognizer.get_layer("dense")(np.array(x1))
      x2=img2/255.
      x2=np.expand_dims(x2, axis=0)
      z2 = recognizer.get_layer("dense")(np.array(x2))
      latent_map.append((np.squeeze(z1),np.squeeze(z2),sim,state1,state2))
          
  
    pkl_filename='models/'+ model_name + '/'+model_name+'_z_encodings_' + infer_dataset
    with open(pkl_filename + ".pkl", 'wb') as f:
        pickle.dump(latent_map, f)
    print("inference done")

    #make t-snes plot
    t_snes_plot(latent_map,model_name,infer_dataset)
    print("t-sne done")
