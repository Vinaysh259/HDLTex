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

import re
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import WOS_input as WOS
import csv
#import Download_Glove as GloVe
import numpy as np
import os


''' Location of the dataset'''
#path_WOS = WOS.download_and_extract()
#GLOVE_DIR = GloVe.download_and_extract()
#print(GLOVE_DIR)

def clean_str(string):
    """
    Tokenization/string cleaning for dataset
    Every dataset is lower cased except
    """
    string = re.sub(r"\\", "", string)
    string = re.sub(r"\'", "", string)
    string = re.sub(r"\"", "", string)
    return string.strip().lower()

def text_cleaner(text):
    text = text.replace(".", "")
    text = text.replace("[", " ")
    text = text.replace(",", " ")
    text = text.replace("]", " ")
    text = text.replace("(", " ")
    text = text.replace(")", " ")
    text = text.replace("\"", "")
    text = text.replace("-", "")
    text = text.replace("=", "")
    text = text.replace("!", "")
    rules = [
        {r'>\s+': u'>'},  # remove spaces after a tag opens or closes
        {r'\s+': u' '},  # replace consecutive spaces
        {r'\s*<br\s*/?>\s*': u'\n'},  # newline after a <br>
        {r'</(div)\s*>\s*': u'\n'},  # newline after </p> and </div> and <h1/>...
        {r'</(p|h\d)\s*>\s*': u'\n\n'},  # newline after </p> and </div> and <h1/>...
        {r'<head>.*<\s*(/head|body)[^>]*>': u''},  # remove <head> to </head>
        {r'<a\s+href="([^"]+)"[^>]*>.*</a>': r'\1'},  # show links instead of texts
        {r'[ \t]*<[^<]*?/?>': u''},  # remove remaining tags
        {r'^\s+': u''}  # remove spaces at the beginning
    ]
    for rule in rules:
        for (k, v) in rule.items():
            regex = re.compile(k)
            text = regex.sub(v, text)
        text = text.rstrip()
        text = text.strip()
    return text.lower()


def loadData():
    #WOS.download_and_extract()
    #fname = os.path.join(path_WOS,"WebOfScience/WOS5736/X.txt")  ABstract
    #fnamek = os.path.join(path_WOS,"WebOfScience/WOS5736/YL1.txt")  Domain
    #fnameL2 = os.path.join(path_WOS,"WebOfScience/WOS5736/YL2.txt")  keywords

    
	with open("train_data_binary.csv") as csvFile:
  	mpg = list(csv.DictReader(csvFile))

  	labelslist=['VALUE','BUSINESS','CHECKIN','LOCATION','FOOD','CLEANLINESS','ROOMS','SERVICE','NOTRELATED','OTHER']

  	list2 = []
  	for d in mpg:
  		for m in labelslist:
  			if(d[m] == '1'):
  				if(m == 'OTHER'):
  					list2.append(0)
            	elif(m == 'ROOMS'):
                	list2.append(1)
            	elif(m == 'LOCATION'):
                	list2.append(2)
            	else:
                	list2.append(3)
            	break

    sum = 0
	content1 = []
	for d in mpg:
    	sum = 0;
    	for m in labelslist:
        	sum = sum + int(d[m])
    
    	if(sum > 0):
        	content.append(d['SEGMENTS']) 

    content = [text_cleaner(x) for x in content]

    Label = np.matrix(list2, dtype=int)
    Label = np.transpose(Label)

    with open("train.unique.csv") as csvFile:
  		mpg1 = list(csv.DictReader(csvFile))

  	labelslist1=['segmentLabels__VALUE','segmentLabels__BUSINESS','segmentLabels__CHECKIN','segmentLabels__LOCATION','segmentLabels__FOOD','segmentLabels__CLEANLINESS','segmentLabels__ROOMS','segmentLabels__SERVICE','segmentLabels__NOTRELATED','segmentLabels__OTHER']
	
	with open("test.unique.csv") as csvFile:
  		mpg2 = list(csv.DictReader(csvFile))
	
	list3 = []
	for d in mpg1:
    	for m in labelslist1:
        	if(d[m] == 'p' or d[m] == 'ip'):
            	list3.append(0)
            	break
        	elif(d[m] == 'n' or d[m] == 'in'):
            	list3.append(1)
            	break
        	elif(d[m] == 'x' or d[m] == 'ix'):
            	list3.append(2)
            	break
     

    for d in mpg2:
    	for m in labelslist1:
        	if(d[m] == 'p' or d[m] == 'ip'):
            	list3.append(0)
            	break
        	elif(d[m] == 'n' or d[m] == 'in'):
            	list3.append(1)
            	break
        	elif(d[m] == 'x' or d[m] == 'ix'):
            	list3.append(2)
            	break

	list3.append(0)
	list3.append(1)
	list3.append(2)

	Label_L2 = np.matrix(list3, dtype=int)
	Label_L2 = np.transpose(Label_L2)

	np.random.seed(7)
	print(Label.shape)
	print(Label_L2.shape)
	Label = np.column_stack((Label, Label_L2))

	number_of_classes_L1 = np.max(Label)+1

	number_of_classes_L2 = np.zeros(number_of_classes_L1,dtype=int)


                    
    X_train, X_test, y_train, y_test  = train_test_split(content, Label, test_size=0.2,random_state= 0)

    vectorizer_x = CountVectorizer()
    X_train = vectorizer_x.fit_transform(X_train).toarray()
    X_test = vectorizer_x.transform(X_test).toarray()

    L2_Train = []
    L2_Test = []
    content_L2_Train = []
    content_L2_Test = []

    for i in range(0, number_of_classes_L1):
        L2_Train.append([])
        L2_Test.append([])
        content_L2_Train.append([])
        content_L2_Test.append([])


    for i in range(0, X_train.shape[0]):
        L2_Train[y_train[i, 0]].append(y_train[i, 1])
        number_of_classes_L2[y_train[i, 0]] = max(number_of_classes_L2[y_train[i, 0]],(y_train[i, 1]+1))
        content_L2_Train[y_train[i, 0]].append(X_train[i])

    for i in range(0, X_test.shape[0]):
        L2_Test[y_test[i, 0]].append(y_test[i, 1])
        content_L2_Test[y_test[i, 0]].append(X_test[i])

    for i in range(0, number_of_classes_L1):
        L2_Train[i] = np.array(L2_Train[i])
        L2_Test[i] = np.array(L2_Test[i])
        content_L2_Train[i] = np.array(content_L2_Train[i])
        content_L2_Test[i] = np.array(content_L2_Test[i])
    return (X_train,y_train,X_test,y_test,content_L2_Train,L2_Train,content_L2_Test,L2_Test,number_of_classes_L2)

