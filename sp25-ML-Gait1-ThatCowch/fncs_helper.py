import numpy as np
from scipy import stats as st
import torch
import torch.nn as nn
import sklearn.metrics as metrics
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# This function loads the signal measusrementa and labels, and splits it into time and values.
def loadTrial_Train(dataFolder,id):
    xt = np.genfromtxt('{}trial{:02d}.x.t.csv'.format(dataFolder,id),delimiter=',')
    xv = np.genfromtxt('{}trial{:02d}.x.v.csv'.format(dataFolder,id),delimiter=',')
    yt = np.genfromtxt('{}trial{:02d}.y.t.csv'.format(dataFolder,id),delimiter=',')
    yv = np.genfromtxt('{}trial{:02d}.y.v.csv'.format(dataFolder,id),delimiter=',')
    yv = yv.astype(int)

    # Returning x measurements and y labels
    return xt, xv, yt, yv

# This function extracts features from the measurements.
def extractFeat(xt,xv,winSz,timeStart,timeEnd,timeStep):
    tList = []
    featList = []

    # Specifying the initial window for extracting features
    t0 = timeStart
    t1 = t0+winSz

    while(t1<=timeEnd):
        # Using the middle time of the window as a reference time
        tList.append((t0+t1)/2)

        # Extracting features
        xWin = xv[(xt>=t0)*(xt<=t1),:]
        f1 = np.mean(xWin,axis=0)
        f2 = np.std(xWin,axis=0)

        # Storing the features
        featList.append(np.concatenate((f1,f2)))

        # Updating the window by shifting it by the step size
        t0 = t0+timeStep
        t1 = t0+winSz

    tList = np.array(tList)
    featList = np.array(featList)

    return tList, featList

# This function returns the mode over a window of data to make it compatible with the features
# extracted.
def extractLabel(yt,yv,winSz,timeStart,timeEnd,timeStep):
    tList = []
    yList = []

    # Specifying the initial window for extracting features
    t0 = timeStart
    t1 = t0+winSz

    while(t1<=timeEnd):
        # Using the middle time of the window as a reference time
        tList.append((t0+t1)/2)

        # Extracting features
        yWin = yv[(yt>=t0)*(yt<=t1)]
        
        # Storing the features
        yList.append(st.mode(yWin).mode)

        # Updating the window by shifting it by the step size
        t0 = t0+timeStep
        t1 = t0+winSz

    tList = np.array(tList)
    yList = np.array(yList)

    return tList, yList

# It loads the data and extracts the features
def loadFeatures(dataFolder,winSz,timeStep,idList):
    for k,id in enumerate(idList):
        # Loading the raw data
        xt, xv, yt, yv = loadTrial_Train(dataFolder,id=id)

        # Extracting the time window for which we have values for the measurements and the response
        timeStart = np.max((np.min(xt),np.min(yt)))
        timeEnd = np.min((np.max(xt),np.max(yt)))

        # Extracting the features
        _, feat = extractFeat(xt,xv,winSz,timeStart,timeEnd,timeStep)
        _, lab = extractLabel(yt,yv,winSz,timeStart,timeEnd,timeStep)

        # Storing the features
        if(k==0):
            featList = feat
            labList = lab
        else:
            featList = np.concatenate((featList,feat),axis=0)
            labList = np.concatenate((labList,lab),axis=0)

    return featList, labList

# Creating a wrapper so we have the same interface for all the methods. This wrapper takes as an
# input an mlp model so we can reuse it with different model architectures.
class NetWrapper:
  def __init__(self,model,device,epochs,weights):
    self.model = model
    self.loss_fn = nn.CrossEntropyLoss(weight = torch.tensor(weights).to(device))
    self.optimizer = torch.optim.Adam(self.model.parameters())
    self.device = device
    self.epochs = epochs

  def fit(self,X,y):
    X = torch.from_numpy(X).float()
    X = X.to(self.device)
    y = torch.from_numpy(y).long()
    y = y.to(self.device)

    for t in range(self.epochs):
      pred = self.model(X)
      loss = self.loss_fn(pred, y)

      self.optimizer.zero_grad()
      loss.backward()
      self.optimizer.step()

      if t%1000 == 999:
        print("[Epoch {t:5d} of {epochs}] loss: {loss:1.6f}".format(
            t=t+1,epochs=self.epochs,loss=loss))

  def predict(self,X):
    X = torch.from_numpy(X).float()
    X = X.to(self.device)

    pred = self.model(X)
    pred = pred.cpu().detach().numpy()
    pred = np.argmax(pred,axis=1)

    return pred


# This function produces a summary of performance metrics including a confusion matrix
def summaryPerf(yTrain,yTrainHat,y,yHat):
    # Plotting confusion matrix for the non-training set:
    cm = metrics.confusion_matrix(y,yHat,normalize='true')
    disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=
                                  ['Walk Hard','Down Stairs','Up Stairs','Walk Soft'])
    disp.plot()

    # Displaying metrics for training and non-training sets
    print('Training:  Acc = {:4.3f}'.format(metrics.accuracy_score(yTrain,yTrainHat)))
    print('Training:  BalAcc = {:4.3f}'.format(metrics.balanced_accuracy_score(yTrain,yTrainHat)))
    print('Validation: Acc = {:4.3f}'.format(metrics.accuracy_score(y,yHat)))
    print('Validation: BalAcc = {:4.3f}'.format(metrics.balanced_accuracy_score(y,yHat)))