"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
RMDL: Random Multimodel Deep Learning for Classification

* Copyright (C) 2018  Kamran Kowsari <kk7nc@virginia.edu>
* Last Update: Oct 26, 2018
* This file is part of  HDLTex project, University of Virginia.
* Free to use, change, share and distribute source code of RMDL
* Refrenced paper : HDLTex: Hierarchical Deep Learning for Text Classification
* Link: https://doi.org/10.1109/ICMLA.2017.0-134
* Comments and Error: email: kk7nc@virginia.edu
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

from keras.models import Sequential
from keras.models import Model
import numpy as np
from keras.layers import Dense, Input, Flatten
#from keras.layers import Conv1D, MaxPooling1D, Embedding, Merge, Dropout, LSTM, GRU, Bidirectional,SimpleRNN
'''
buildModel_DNN(nFeatures, nClasses, nLayers=3,Numberof_NOde=100, dropout=0.5)
Build Deep neural networks Model for text classification
Shape is input feature space
nClasses is number of classes
nLayers is number of hidden Layer
Number_Node is number of unit in each hidden layer
dropout is dropout value for solving overfitting problem
'''
def buildModel_DNN(Shape, nClasses, nLayers=3,Number_Node=100, dropout=0.5):
    model = Sequential()
    model.add(Dense(Number_Node, input_dim=Shape))
    model.add(Dropout(dropout))
    for i in range(0,nLayers):
        model.add(Dense(Number_Node, activation='relu'))
        model.add(Dropout(dropout))
    model.add(Dense(nClasses, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='RMSprop',
                  metrics=['accuracy'])

    return model

'''
def buildModel_RNN(word_index, embeddings_index, nClasses, MAX_SEQUENCE_LENGTH, EMBEDDING_DIM):
word_index in word index , 
embeddings_index is embeddings index, look at data_helper.py 
nClasses is number of classes, 
MAX_SEQUENCE_LENGTH is maximum lenght of text sequences, 
EMBEDDING_DIM is an int value for dimention of word embedding look at data_helper.py 
'''

'''
def buildModel_CNN(word_index,embeddings_index,nClasses,MAX_SEQUENCE_LENGTH,EMBEDDING_DIM,Complexity=0):
word_index in word index , 
embeddings_index is embeddings index, look at data_helper.py 
nClasses is number of classes, 
MAX_SEQUENCE_LENGTH is maximum lenght of text sequences, 
EMBEDDING_DIM is an int value for dimention of word embedding look at data_helper.py 
Complexity we have two different CNN model as follows 
Complexity=0 is simple CNN with 3 hidden layer
Complexity=2 is more complex model of CNN with filter_length of [3, 4, 5, 6, 7]
'''
