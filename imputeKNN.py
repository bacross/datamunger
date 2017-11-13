# -*- coding: utf-8 -*-
"""
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

# fcn that applies the knn algorithm to the nans in a column and then creates a new column of the join between nans and notnans	
def fillColNans(k,ncol,dfexcol,fitcores):
    nanscol, notnanscol,nansexdf,notnansexdf = splitDfNansNot(ncol,dfexcol)
    ypredLst = []
    ypredLst = [chooseNanFill(k,idx,nansexdf,notnansexdf,notnanscol,fitcores) for idx in list(nanscol.index)]
    ypredcol = pd.DataFrame(ypredLst)
    ypredcol.index = nanscol.index
    newcol = pd.concat([notnanscol,ypredcol])
    return newcol
	
# helper fcn to fillColNans that decides to use knn or median depending on data	shape
def chooseNanFill(k,idx,nansexdf,notnansexdf,notnanscol,fitcores):
    nrowX = nansexdf.loc[idx]
    X = buildTrainingSet(nrowX,notnansexdf)
    if X.shape[0]<k:
        ypred = notnanscol.median
    else:
        ypred = kNNRegress(k,X,notnanscol,nrowX[~pd.isnull(nrowX)],fitcores)
    return ypred

# fcn that parses DF into column to impute and residual features to model on	
def parseDf(n,bigdf):
    ncol = bigdf[n]
    dfexcol = bigdf.iloc[:,bigdf.columns!=n]
    return ncol,dfexcol

# full fcn that utilizes helper functions to run imputation on a specific column	
def imputeMissingDataForCol(n,bigdf,k,fitcores):
    ncol,dfexcol = parseDf(n,bigdf)
    ypredcol = fillColNans(k,ncol,dfexcol,fitcores)
    return ypredcol

# Parallel implementation of missing data imputation - parallel by column
# Note: included in here is a parameter to tree parallelization if available, but this is not fully functional...
# as it hung on me when I tried to implement it.  For now hardcoded.
def imputeMissingDataKNN(bigdf,k,multicore):
    N = bigdf.shape[1]
    if multicore==True:
        ncores = multiprocessing.cpu_count() - 1
        rowcores = ncores
        fitcores=1 #hardcoded for now because I can't get the multicore tree architecture to work
        predColList = Parallel(n_jobs = rowcores)(delayed(imputeMissingDataForCol)(n,bigdf,k,fitcores) for n in range(N))
    else:
        predColList = [imputeMissingDataForCol(n,bigdf,k,fitcores=1) for n in range(N)]
    newdf = pd.concat(predColList,axis=1)
    newdf.columns=bigdf.columns
    return newdf

#fcn that replaces outliers in a column for a given pctile tolerance with nans
def outlierToNanCol(ncol,lower_lim,upper_lim):
    if ncol.isnull().sum().sum()>0:
        print('Error: There are still nans present. Please remove first.')
    else:
        news=ncol.copy()
        q = news.quantile([lower_lim, upper_lim])
        if len(q) == 2:
            news[news < q.iloc[0]] = np.nan
            news[news > q.iloc[1]] = np.nan
    return news
		
# fcn Parallel implementation of conversion of outliers to nans
def outlierToNanDF(ndf,lower_lim,upper_lim,multicore):
    if multicore==True:
        ncores = multiprocessing.cpu_count()-1
        outColList = Parallel(n_jobs=ncores)(delayed(outlierToNanCol)(ndf[col],lower_lim,upper_lim) for col in ndf.columns)
    else:
        outColList = [outlierToNanCol(ndf[col],lower_lim,upper_lim) for col in ndf.columns]
    newdf = pd.concat(outColList, axis=1)
    newdf.columns=ndf.columns
    return newdf
	
# fcn that finds outliers in a DF, converts them to Nans and then replaces them with imputed values via knn regression
def imputeOutlierKNN(ndf, lower_lim, upper_lim, k, multicore):
    outNanDf = outlierToNanDF(ndf,lower_lim,upper_lim,multicore)
    cleanDF = imputeMissingDataKNN(outNanDf, k,multicore)
    return cleanDF
