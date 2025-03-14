from sklearn.metrics import accuracy_score,recall_score,precision_score
import numpy as np
from sklearn.utils.validation import check_is_fitted

def evaluate(model,xtrain,ytrain,xtest,ytest):
    model.fit(xtrain,ytrain)
    ypred = model.predict(xtest)
    acc = accuracy_score(ypred,ytest)
    recall = recall_score(ypred,ytest)
    prec = precision_score(ypred,ytest)

    return acc,recall,prec

def ensemble_vote(y,thresh = 0.2):
    y = y.reshape(-1,40)
    y = np.sum(y,axis=1)/40
    y = y>thresh #threshold is to be tuned
    return y

def evaluate_with_voting(model,xtrain,ytrain,xtest,ytest,thresh = 0.2):
    model.fit(xtrain,ytrain)
    ypred = model.predict(xtest)
    ypred = ensemble_vote(ypred,thresh)
    ytrue = ensemble_vote(ytest,thresh=0.5)

    acc = accuracy_score(ypred,ytrue)
    recall = recall_score(ypred,ytrue)
    prec = precision_score(ypred,ytrue)

    return acc,recall,prec

def get_predictions(model,x,thresh = 0.2):

    ypred = model.predict(x)
    ypred = ensemble_vote(ypred,thresh)
    ypred = np.array(ypred,dtype=int)
    return ypred
