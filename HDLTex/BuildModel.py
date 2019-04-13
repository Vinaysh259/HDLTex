

from keras.models import Sequential
from keras.models import Model
import numpy as np
from keras.layers import Dense, Input, Flatten
from keras.layers import Dropout,Dense
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