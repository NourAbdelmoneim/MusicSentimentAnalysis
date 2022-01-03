
# DNN for Music Emotion Classification

from sklearn.metrics import accuracy_score 
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import numpy as np 
import pandas as pd
import math
import pickle

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import Model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.layers import Dropout

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

#Loading and preprocessing data
def preprocessData():
    trainingVectors = np.genfromtxt('condensedData/matrix-training.csv',delimiter = ',')
    print(trainingVectors.shape)
    print("done")

    testingVectors = np.genfromtxt('condensedData/matrix-testing.csv', delimiter = ',')
    print(testingVectors.shape)
    print("done")

    trainingFile = 'condensedData/labels-training.csv'
    trainingLabels = pd.read_csv(trainingFile, header=None)
    trainingLabels[1] = trainingLabels[1].astype('category')
    trainingLabels[1] = trainingLabels[1].cat.codes

    testingFile = 'condensedData/labels-testing.csv'
    testingLabels = pd.read_csv(testingFile, header=None)
    testingLabels[1] = testingLabels[1].astype('category')
    testingLabels[1] = testingLabels[1].cat.codes

    trainingLabels = np.array(trainingLabels[1])
    testingLabels = np.array(testingLabels[1])

    trainingVectors = trainingVectors.astype('float32')
    testingVectors = testingVectors.astype('float32')

    return trainingVectors, trainingLabels, testingVectors, testingLabels


# dumping preprocessed data into pickle files for faster access
def dumpData(trainingVectors, trainingLabels, testingVectors, testingLabels):
    pickle.dump(trainingVectors, open( "trainingVectors.p", "wb" ))
    pickle.dump(testingVectors, open( "testingVectors.p", "wb" ))
    pickle.dump(trainingLabels, open( "trainingLabels.p", "wb" ))
    pickle.dump(testingLabels, open( "testingLabels.p", "wb" ))


# loading data from pickle files
def loadData():
    trainingVectors = pickle.load(open("trainingVectors.p", "rb" ))
    testingVectors = pickle.load(open("testingVectors.p", "rb" ))
    trainingLabels = pickle.load(open("trainingLabels.p", "rb" ))
    testingLabels = pickle.load(open("testingLabels.p", "rb" ))

    trainingLabels = np.reshape(trainingLabels,(len(trainingLabels),1))
    testingLabels = np.reshape(testingLabels,(len(testingLabels),1))

    print(trainingVectors.shape, testingVectors.shape, trainingLabels.shape, testingLabels.shape)

    n_features = trainingVectors.shape[1]

    return trainingVectors, trainingLabels, testingVectors, testingLabels, n_features

# Sequential DNN 
def dnnModel(n_features):
    model = Sequential()
    model.add(Dense(128, activation='relu', kernel_initializer='he_normal'))
    model.add(Dropout(0.4))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(16, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(4, activation='softmax'))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model


# Training model
def trainModel(trainingVectors, trainingLabels, n_epochs, model):
    model.fit(trainingVectors, trainingLabels, epochs=n_epochs, batch_size=32, verbose=1)

    return model

# Evaluating model
def evaluateModel(testingVectors, testingLabels, model):
    loss, accuracy = model.evaluate(testingVectors, testingLabels, verbose=1)
    print('Test Accuracy: %.2f' % accuracy)

    return accuracy

# Making predictions
def makePredictions(testingVectors, model):
    y_pred = model.predict(testingVectors, verbose = 1)

    predictedLabels = []
    for pred in y_pred:
        maxPred = np.max(pred)
        maxIndex = np.where(pred == maxPred)
        maxIndex = int(maxIndex[0])
        predictedLabels.append(maxIndex)

    return predictedLabels

def evaluationMetrics(testingLabels, predictedLabels):
    f1 = f1_score(testingLabels, predictedLabels, average=None)
    precision = precision_score(testingLabels, predictedLabels, average=None)
    recall = recall_score(testingLabels, predictedLabels, average=None)
    #cm = confusion_matrix(testingLabels, predictedLabels, normalize = 'true')

    return f1, precision, recall

def displayCM(testingLabels, predictedLabels):
    moodLabels = ["angry","happy","relaxed","sad"]
    disp = ConfusionMatrixDisplay.from_predictions(testingLabels, predictedLabels, display_labels=moodLabels, normalize = 'true', values_format = '.1g', colorbar = False)

    fig, ax = plt.subplots(figsize=(10, 10))
    plt.rcParams['font.size'] = '20'
    plt.xlabel("Predicted Label",labelpad=40)
    plt.ylabel("True Label",labelpad=12)
    disp.plot(cmap = plt.cm.Greens, ax=ax, colorbar = False)
    
    plt.savefig('confusion_matrix.png')

def main():

    print("loading data")
    trainingVectors, trainingLabels, testingVectors, testingLabels, n_features = loadData()

    model = dnnModel(n_features)

    print("training model")
    trainedModel = trainModel(trainingVectors, trainingLabels, 6, model)

    print("evaluating model")
    accuracy = evaluateModel(testingVectors, testingLabels, trainedModel)

    print("maing predictions")
    predictedLabels = makePredictions(testingVectors, trainedModel)

    print("Evaluation Metrics")
    f1, precision, recall = evaluationMetrics(testingLabels, predictedLabels)

    print("F1-Score: ", f1)
    print("Precision: ", precision)
    print("Recall: ", recall)

    displayCM(testingLabels, predictedLabels)
    

if __name__ == '__main__': sys.exit(main())
