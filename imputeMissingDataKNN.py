# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 11:21:35 2017

@author: bacro
"""
# %% Load Packages
import numpy as np
import pandas as pd
import random
from sklearn.neighbors import KNeighborsRegressor
import multiprocessing
from joblib import Parallel,delayed

# %% functions

# function to find the nans and not nans training sets for a given column
def splitDfNansNot(ncol, dfexcol):
    nanscol = ncol[ncol.isnull()]
    notnanscol = ncol[~ncol.isnull()]
    nansexdf = dfexcol.ix[ncol.isnull()==True,:]
    notnansexdf = dfexcol.ix[ncol.isnull()==False,:]
    return nanscol, notnanscol,nansexdf,notnansexdf
	
# fcn that builds the training set for a given row
def buildTrainingSet(nrowX,ndf):
    misscols = list(nrowX[~pd.isnull(nrowX)].index)
    notnaArr = np.array(ndf)
    col_mean = np.nanmean(notnaArr,axis=0)
    inds = np.where(np.isnan(notnaArr))
    notnaArr[inds]=np.take(col_mean,inds[1])
    cleandf = pd.DataFrame(notnaArr,index = ndf.index,columns=ndf.columns)
    cleandf = cleandf[misscols]
    return cleandf
	
# fcn that returns the predicted value based using kNN reg	
def kNNRegress(k,X,y,xpred,fitcores):
    neigh = KNeighborsRegressor(n_neighbors=k,n_jobs=fitcores)
    return neigh.fit(X,y).predict(xpred.values.reshape(1,-1))[0]
	
def fillColNans(k,ncol,dfexcol,fitcores):
    nanscol, notnanscol,nansexdf,notnansexdf = splitDfNansNot(ncol,dfexcol)
    ypredLst = []
    ypredLst = [chooseNanFill(k,idx,nansexdf,notnansexdf,notnanscol,fitcores) for idx in list(nanscol.index)]
    ypredcol = pd.DataFrame(ypredLst)
    ypredcol.index = nanscol.index
    newcol = pd.concat([notnanscol,ypredcol])
    return ypredcol
	
def chooseNanFill(k,idx,nansexdf,notnansexdf,notnanscol,fitcores):
    nrowX = nansexdf.loc[idx]
    X = buildTrainingSet(nrowX,notnansexdf)
    if X.shape[1]<k:
        ypred = notnanscol.median
    else:
        ypred = kNNRegress(k,X,notnanscol,nrowX[~pd.isnull(nrowX)],fitcores)
    return ypred
	
def parseDf(n,bigdf):
	ncol = bigdf[n]
	dfexcol = bigdf.iloc[:,bigdf.columns!=n]
	return ncol,dfexcol
	
def imputeMissingDataForCol(n,bigdf,k,fitcores):
    ncol,dfexcol = parseDf(n,bigdf)
    ypredcol = fillColNans(k,ncol,dfexcol,fitcores)
    return ypredcol

def imputeMissingDataKNN(bigdf,k):
	N = bigdf.shape[1]
    ncores = multiprocessing.cpu_count() - 1
    rowcores = fitcores = int(ncores/2)
	predColList = Parallel(n_jobs = rowcores)(delayed(imputeMissingDataForCol)(n,bigdf,k,fitcores) for n in range(N))
	newdf = pd.concat(predColList,axis=1)
	



    

