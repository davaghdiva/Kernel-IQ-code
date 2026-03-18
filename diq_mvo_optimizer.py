"""
Name    : diq_mvo_optimizer.py
Author  : William Smyth & Layla Abu Khalaf (adapted [in part] from https://github.com/yinsenm/gerber)
Contact : drwss.academy@gmail.com
Date    : 26/12/2025
Desc    : solve MVO for MaxSR
"""

from __future__ import annotations
import json
import math
import matplotlib
matplotlib.use('Agg')
import os
import random
import sys
import warnings
import numpy as np
import numpy.matlib as mt
import pandas as pd
from wiq_cov import wishart_iq_covariance
try:
    from numba import njit, prange
except Exception:
    # fallback when numba is unavailable/incompatible.
    def njit(*args, **kwargs):
        if args and callable(args[0]) and len(args) == 1 and not kwargs:
            return args[0]
        def _wrap(func):
            return func
        return _wrap
    prange = range
from numpy import diag, inf, copy, dot
from numpy.linalg import norm, eigvals
from pathlib import Path
from scipy.optimize import minimize
from sklearn.neighbors import KernelDensity
from typing import Optional

def calc_assets_moments(returns_df: pd.DataFrame,
                        weights=None,
                        cov_function: str = None,
                        freq: str = "monthly") -> tuple:
    factor = 252 if freq == "daily" else 12
    if weights is None:
        std = returns_df.std() * np.sqrt(factor)
        ret = ((1 + returns_df.mean()) ** factor) - 1.0
    else:
        ret = ((1 + returns_df.mul(weights).sum(axis=1).mean()) ** factor) - 1.0
        if cov_function is None:
            std     = returns_df.mul(weights).sum(axis=1).std() * np.sqrt(factor)
        if cov_function is not None:
            cov_mat = returns_df.cov()
            std     = np.sqrt(np.dot(weights.T, np.dot(cov_mat * factor, weights)))
    return ret, std

def calc_monthly_returns(df: pd.DataFrame) -> pd.DataFrame:
    df_start_month     = (df.groupby([df.index.year, df.index.month])).apply(lambda x: x.head(1))
    df_start_month     = df_start_month.droplevel([0, 1])
    df_monthly_returns = df.pct_change().shift(-1).dropna()
    return df_monthly_returns

def set_eps_wgt_to_zeros(in_array, eps=1e-4):
    out_array = np.array(in_array)
    out_array[np.abs(in_array) < eps] = 0
    out_array = np.array(out_array) / np.sum(out_array)
    return out_array

def is_psd_def(cov_mat):
    return np.all(eigvals(cov_mat) > -1e-08)
        
def gerber_cov_stat(rets: np.array, threshold: float):
    n, p    = rets.shape
    sd_vec  = rets.std(axis=0)
    SD      = np.diag(sd_vec)
    cor_mat = np.zeros((p, p))  
    for i in range(p):
        for j in range(i + 1):
            pos, neg, nn = 0, 0, 0
            for k in range(n):
                if ((rets[k, i] >= threshold * sd_vec[i]) and (rets[k, j] >= threshold * sd_vec[j])) or \
                   ((rets[k, i] <= -threshold * sd_vec[i]) and (rets[k, j] <= -threshold * sd_vec[j])):
                    pos += 1
                elif ((rets[k, i] >= threshold * sd_vec[i]) and (rets[k, j] <= -threshold * sd_vec[j])) or \
                     ((rets[k, i] <= -threshold * sd_vec[i]) and (rets[k, j] >= threshold * sd_vec[j])):
                    neg += 1
                elif abs(rets[k, i]) < threshold * sd_vec[i] and abs(rets[k, j]) < threshold * sd_vec[j]:
                    nn += 1
            cor_mat[i, j] = (pos - neg) / (n - nn)
            cor_mat[j, i] = cor_mat[i, j]    
    cov_mat = SD @ cor_mat @ SD
    return cov_mat, cor_mat

# collection of linear and non-linear shrinkage techniques (Ledoit & Wolf)
def cov1Para(Y, k=None):
    N, p = Y.shape
    if k is None or math.isnan(k):
        Y = Y.sub(Y.mean(axis=0), axis=1)
        k = 1
    n = N - k
    sample  = pd.DataFrame(np.matmul(Y.T.to_numpy(), Y.to_numpy())) / n
    diag    = np.diag(sample.to_numpy())
    meanvar = sum(diag) / len(diag)
    target  = meanvar * np.eye(p)
    Y2      = pd.DataFrame(np.multiply(Y.to_numpy(), Y.to_numpy()))
    sample2 = pd.DataFrame(np.matmul(Y2.T.to_numpy(), Y2.to_numpy())) / n
    piMat   = pd.DataFrame(sample2.to_numpy() - np.multiply(sample.to_numpy(), sample.to_numpy()))
    pihat   = sum(piMat.sum())
    gammahat  = np.linalg.norm(sample.to_numpy() - target, ord='fro') ** 2
    rho_diag  = 0
    rho_off   = 0
    rhohat    = rho_diag + rho_off
    kappahat  = (pihat - rhohat) / gammahat
    shrinkage = max(0, min(1, kappahat / n))
    sigmahat  = shrinkage * target + (1 - shrinkage) * sample
    return sigmahat

def cov2Para(Y, k=None):
    N, p = Y.shape
    if k is None or math.isnan(k):
        Y = Y.sub(Y.mean(axis=0), axis=1)
        k = 1
    n = N - k
    sample  = pd.DataFrame(np.matmul(Y.T.to_numpy(), Y.to_numpy())) / n
    diag    = np.diag(sample.to_numpy())
    meanvar = sum(diag) / len(diag)
    meancov = (np.sum(sample.to_numpy()) - np.sum(np.eye(p) * sample.to_numpy())) / (p * (p - 1))
    target  = pd.DataFrame(meanvar * np.eye(p) + meancov * (1 - np.eye(p)))
    Y2      = pd.DataFrame(np.multiply(Y.to_numpy(), Y.to_numpy()))
    sample2 = pd.DataFrame(np.matmul(Y2.T.to_numpy(), Y2.to_numpy())) / n
    piMat   = pd.DataFrame(sample2.to_numpy() - np.multiply(sample.to_numpy(), sample.to_numpy()))
    pihat   = sum(piMat.sum())
    gammahat   = np.linalg.norm(sample.to_numpy() - target, ord='fro') ** 2
    rho_diag   = (sample2.sum().sum() - np.trace(sample.to_numpy()) ** 2) / p
    sum1, sum2 = Y.sum(axis=1), Y2.sum(axis=1)
    temp       = (np.multiply(sum1.to_numpy(), sum1.to_numpy()) - sum2)
    rho_off1   = np.sum(np.multiply(temp, temp)) / (p * n)
    rho_off2   = (sample.sum().sum() - np.trace(sample.to_numpy())) ** 2 / p
    rho_off    = (rho_off1 - rho_off2) / (p - 1)
    rhohat     = rho_diag + rho_off
    kappahat   = (pihat - rhohat) / gammahat
    shrinkage  = max(0, min(1, kappahat / n))
    sigmahat   = shrinkage * target + (1 - shrinkage) * sample
    return sigmahat

def covCor(Y, k=None):
    N, p = Y.shape
    if k is None or math.isnan(k):
        Y = Y.sub(Y.mean(axis=0), axis=1)
        k = 1
    n = N - k
    sample    = pd.DataFrame(np.matmul(Y.T.to_numpy(), Y.to_numpy())) / n
    samplevar = np.diag(sample.to_numpy())
    sqrtvar   = pd.DataFrame(np.sqrt(samplevar))
    rBar      = (np.sum(np.sum(sample.to_numpy() / np.matmul(sqrtvar.to_numpy(), sqrtvar.T.to_numpy()))) - p) / (p * (p - 1))
    target    = pd.DataFrame(rBar * np.matmul(sqrtvar.to_numpy(), sqrtvar.T.to_numpy()))
    target[np.eye(p).astype(bool)] = sample[np.eye(p).astype(bool)]
    Y2        = pd.DataFrame(np.multiply(Y.to_numpy(), Y.to_numpy()))
    sample2   = pd.DataFrame(np.matmul(Y2.T.to_numpy(), Y2.to_numpy())) / n
    piMat     = pd.DataFrame(sample2.to_numpy() - np.multiply(sample.to_numpy(), sample.to_numpy()))
    pihat     = sum(piMat.sum())
    gammahat  = np.linalg.norm(sample.to_numpy() - target, ord='fro') ** 2
    rho_diag  = np.sum(np.diag(piMat))
    term1     = pd.DataFrame(np.matmul((Y ** 3).T.to_numpy(), Y.to_numpy()) / n)
    term2     = pd.DataFrame(np.transpose(mt.repmat(samplevar, p, 1)) * sample)
    thetaMat  = term1 - term2
    thetaMat[np.eye(p).astype(bool)] = 0
    rho_off   = rBar * (np.matmul((1 / sqrtvar).to_numpy(), sqrtvar.T.to_numpy()) * thetaMat).sum().sum()
    rhohat    = rho_diag + rho_off
    kappahat  = (pihat - rhohat) / gammahat
    shrinkage = max(0, min(1, kappahat / n))
    sigmahat  = shrinkage * target + (1 - shrinkage) * sample
    return sigmahat

def covDiag(Y, k=None):
    N, p = Y.shape
    if k is None or math.isnan(k):
        Y = Y.sub(Y.mean(axis=0), axis=1)
        k = 1
    n = N - k
    sample   = pd.DataFrame(np.matmul(Y.T.to_numpy(), Y.to_numpy())) / n
    target   = pd.DataFrame(np.diag(np.diag(sample.to_numpy())))
    Y2       = pd.DataFrame(np.multiply(Y.to_numpy(), Y.to_numpy()))
    sample2  = pd.DataFrame(np.matmul(Y2.T.to_numpy(), Y2.to_numpy())) / n
    piMat    = pd.DataFrame(sample2.to_numpy() - np.multiply(sample.to_numpy(), sample.to_numpy()))
    pihat     = sum(piMat.sum())
    gammahat  = np.linalg.norm(sample.to_numpy() - target, ord='fro') ** 2
    rho_diag  = np.sum(np.diag(piMat))
    rho_off   = 0
    rhohat    = rho_diag + rho_off
    kappahat  = (pihat - rhohat) / gammahat
    shrinkage = max(0, min(1, kappahat / n))
    sigmahat  = shrinkage * target + (1 - shrinkage) * sample
    return sigmahat

def covMarket(Y, k=None):
    N, P = Y.shape
    if k is None or math.isnan(k):
        Y = Y.sub(Y.mean(axis=0), axis=1)
        k = 1
    n = N - k
    sample = pd.DataFrame(np.matmul(Y.T.to_numpy(), Y.to_numpy())) / n
    Ymkt   = Y.mean(axis=1)
    covmkt = pd.DataFrame(np.matmul(Y.T.to_numpy(), Ymkt.to_numpy())) / n
    varmkt = np.matmul(Ymkt.T.to_numpy(), Ymkt.to_numpy()) / n
    target = pd.DataFrame(np.matmul(covmkt.to_numpy(), covmkt.T.to_numpy())) / varmkt
    target[np.eye(P).astype(bool)] = sample[np.eye(P).astype(bool)]
    Y2       = pd.DataFrame(np.multiply(Y.to_numpy(), Y.to_numpy()))
    sample2  = pd.DataFrame(np.matmul(Y2.T.to_numpy(), Y2.to_numpy())) / n
    piMat    = pd.DataFrame(sample2.to_numpy() - np.multiply(sample.to_numpy(), sample.to_numpy()))
    pihat    = sum(piMat.sum())
    gammahat = np.linalg.norm(sample.to_numpy() - target, ord='fro') ** 2
    rho_diag = np.sum(np.diag(piMat))
    temp     = Y * pd.DataFrame([Ymkt for _ in range(P)]).T
    temp     = temp.iloc[:, P:]
    covmktSQ = pd.DataFrame([covmkt[0] for _ in range(P)])
    v1    = pd.DataFrame((1 / n) * np.matmul(Y2.T.to_numpy(), temp.to_numpy()) - np.multiply(covmktSQ.T.to_numpy(), sample.to_numpy()))
    roff1 = (np.sum(np.multiply(v1.to_numpy(), covmktSQ.to_numpy())) - np.sum(np.diag(np.multiply(v1.to_numpy(), covmkt.to_numpy())))) / varmkt
    v3    = pd.DataFrame((1 / n) * np.matmul(temp.T.to_numpy(), temp.to_numpy()) - varmkt * sample)
    roff3 = (np.sum(np.multiply(v3.to_numpy(), np.matmul(covmkt.to_numpy(), covmkt.T.to_numpy()))) - np.sum(np.multiply(np.diag(v3.to_numpy()), covmkt[0] ** 2))) / varmkt ** 2
    rho_off   = 2 * roff1 - roff3
    rhohat    = rho_diag + rho_off
    kappahat  = (pihat - rhohat) / gammahat
    shrinkage = max(0, min(1, kappahat / n))
    sigmahat  = shrinkage * target + (1 - shrinkage) * sample
    return sigmahat

def GIS(Y, k=None):
    N, p = Y.shape
    if k is None or math.isnan(k):
        Y = Y.sub(Y.mean(axis=0), axis=1)
        k = 1
    n = N - k
    c = p / n
    sample    = pd.DataFrame(np.matmul(Y.T.to_numpy(), Y.to_numpy())) / n
    sample    = (sample + sample.T) / 2
    lambda1,u = np.linalg.eigh(sample)
    lambda1   = lambda1.real.clip(min=0)
    dfu       = pd.DataFrame(u, columns=lambda1)
    dfu.sort_index(axis=1, inplace=True)
    lambda1   = dfu.columns
    h         = (min(c ** 2, 1 / c ** 2) ** 0.35) / p ** 0.35
    invlambda = 1 / lambda1[max(1, p - n + 1) - 1 : p]
    dfl     = pd.DataFrame({'lambda': invlambda})
    Lj      = dfl[np.repeat(dfl.columns.values, min(p, n))]
    Lj      = pd.DataFrame(Lj.to_numpy())
    Lj_i    = Lj.subtract(Lj.T)
    theta   = Lj.multiply(Lj_i).div(Lj_i.multiply(Lj_i).add(Lj.multiply(Lj) * h ** 2)).mean(axis=0)
    Htheta  = Lj.multiply(Lj * h).div(Lj_i.multiply(Lj_i).add(Lj.multiply(Lj) * h ** 2)).mean(axis=0)
    Atheta2 = theta ** 2 + Htheta ** 2
    if p <= n:
        deltahat_1 = (1 - c) * invlambda + 2 * c * invlambda * theta
        delta      = 1 / ((1 - c) ** 2 * invlambda + 2 * c * (1 - c) * invlambda * theta + c ** 2 * invlambda * Atheta2)
        delta      = delta.to_numpy()
    else:
        print('p must be <= n for the symmetrized Kullback-Leibler divergence')
        return -1
    temp = pd.DataFrame(deltahat_1)
    x    = min(invlambda)
    temp.loc[temp[0] < x, 0] = x
    deltaLIS_1 = temp[0]
    temp1      = dfu.to_numpy()
    temp2      = np.diag((delta / deltaLIS_1) ** 0.5)
    temp3      = dfu.T.to_numpy().conjugate()
    cov_mat    = np.matmul(np.matmul(temp1, temp2), temp3)
    return cov_mat

def LIS(Y, k=None):
    N, p = Y.shape
    if k is None or math.isnan(k):
        Y = Y.sub(Y.mean(axis=0), axis=1)
        k = 1
    n = N - k
    c = p / n
    sample    = pd.DataFrame(np.matmul(Y.T.to_numpy(), Y.to_numpy())) / n
    sample    = (sample + sample.T) / 2
    lambda1,u = np.linalg.eigh(sample)
    lambda1   = lambda1.real.clip(min=0)
    dfu       = pd.DataFrame(u, columns=lambda1)
    dfu.sort_index(axis=1, inplace=True)
    lambda1   = dfu.columns
    h         = (min(c ** 2, 1 / c ** 2) ** 0.35) / p ** 0.35
    invlambda = 1 / lambda1[max(1, p - n + 1) - 1 : p]
    dfl   = pd.DataFrame({'lambda': invlambda})
    Lj    = dfl[np.repeat(dfl.columns.values, min(p, n))]
    Lj    = pd.DataFrame(Lj.to_numpy())
    Lj_i  = Lj.subtract(Lj.T)
    theta = Lj.multiply(Lj_i).div(Lj_i.multiply(Lj_i).add(Lj.multiply(Lj) * h ** 2)).mean(axis=0)
    if p <= n:
        deltahat_1 = (1 - c) * invlambda + 2 * c * invlambda * theta
    else:
        print("p must be <= n for Stein's loss")
        return -1
    temp = pd.DataFrame(deltahat_1)
    x    = min(invlambda)
    temp.loc[temp[0] < x, 0] = x
    deltaLIS_1 = temp[0]
    temp1      = dfu.to_numpy()
    temp2      = np.diag(1 / deltaLIS_1)
    temp3      = dfu.T.to_numpy().conjugate()
    sigmahat   = np.matmul(np.matmul(temp1, temp2), temp3)
    return sigmahat

def QIS(Y, k=None):
    N, p = Y.shape
    if k is None or math.isnan(k):
        Y = Y.sub(Y.mean(axis=0), axis=1)
        k = 1
    n = N - k
    c = p / n
    sample    = pd.DataFrame(np.matmul(Y.T.to_numpy(), Y.to_numpy())) / n
    sample    = (sample + sample.T) / 2
    lambda1,u = np.linalg.eigh(sample)
    lambda1   = lambda1.real.clip(min=0)
    dfu       = pd.DataFrame(u, columns=lambda1)
    dfu.sort_index(axis=1, inplace=True)
    lambda1   = dfu.columns
    h         = (min(c ** 2, 1 / c ** 2) ** 0.35) / p ** 0.35
    invlambda = 1 / lambda1[max(1, p - n + 1) - 1 : p]
    dfl     = pd.DataFrame({'lambda': invlambda})
    Lj      = dfl[np.repeat(dfl.columns.values, min(p, n))]
    Lj      = pd.DataFrame(Lj.to_numpy())
    Lj_i    = Lj.subtract(Lj.T)
    theta   = Lj.multiply(Lj_i).div(Lj_i.multiply(Lj_i).add(Lj.multiply(Lj) * h ** 2)).mean(axis=0)
    Htheta  = Lj.multiply(Lj * h).div(Lj_i.multiply(Lj_i).add(Lj.multiply(Lj) * h ** 2)).mean(axis=0)
    Atheta2 = theta ** 2 + Htheta ** 2
    if p <= n:
        delta = 1 / ((1 - c) ** 2 * invlambda + 2 * c * (1 - c) * invlambda * theta + c ** 2 * invlambda * Atheta2)
        delta = delta.to_numpy()
    else:
        delta0 = 1 / ((c - 1) * np.mean(invlambda.to_numpy()))
        delta  = np.repeat(delta0, p - n)
        delta  = np.concatenate((delta, 1 / (invlambda * Atheta2)), axis=None)
    deltaQIS = delta * (sum(lambda1) / sum(delta))
    temp1    = dfu.to_numpy()
    temp2    = np.diag(deltaQIS)
    temp3    = dfu.T.to_numpy().conjugate()
    sigmahat = np.matmul(np.matmul(temp1, temp2), temp3)
    return sigmahat

# Random Matrix Theory - adapted from ML for Asset Managers, Lopez de Prado (2020) 
def mpPDF(var, q, pts):
    eMin, eMax = var*(1-(1./q)**.5)**2, var*(1+(1./q)**.5)**2 
    eVal = np.linspace(eMin, eMax, pts) 
    pdf  = q/(2*np.pi*var*eVal)*((eMax-eVal)*(eVal-eMin))**.5 
    pdf  = pd.Series(pdf, index=eVal)
    return pdf

def getPCA(matrix):
    eVal, eVec = np.linalg.eig(matrix) 
    indices    = eVal.argsort()[::-1] 
    eVal,eVec  = eVal[indices],eVec[:,indices]
    eVal       = np.diagflat(eVal) 
    return eVal,eVec
   
def fitKDE(obs, bWidth=.15, kernel='gaussian', x=None):
    if len(obs.shape) == 1: obs = obs.reshape(-1,1)
    kde = KernelDensity(kernel = kernel, bandwidth = bWidth).fit(obs)
    if x is None: x = np.unique(obs).reshape(-1,1)
    if len(x.shape) == 1: x = x.reshape(-1,1)
    logProb = kde.score_samples(x) 
    pdf = pd.Series(np.exp(logProb), index=x.flatten())
    return pdf
    
def cov2corr(cov):
    std  = np.sqrt(np.diag(cov))
    corr = cov/np.outer(std,std)
    corr[corr<-1], corr[corr>1] = -1,1 
    return corr
    
def corr2cov(corr, std):
    cov = corr * np.outer(std, std)
    return cov     
    
def errPDFs(var, eVal, q, bWidth, pts=1000):
    var  = var[0]
    pdf0 = mpPDF(var, q, pts) 
    pdf1 = fitKDE(eVal, bWidth, x=pdf0.index.values) 
    sse  = np.sum((pdf1-pdf0)**2)
    return sse 

def findMaxEval(eVal, q, bWidth):
    out = minimize(lambda *x: errPDFs(*x), x0=np.array(0.5), args=(eVal, q, bWidth), bounds=((1E-5, 1-1E-5),))
    if out['success']: var = out['x'][0]
    else: var=1
    eMax = var*(1+(1./q)**.5)**2
    return eMax, var

def denoisedCorr(eVal, eVec, nFacts):
    eVal_ = np.diag(eVal).copy()
    eVal_[nFacts:] = eVal_[nFacts:].sum()/float(eVal_.shape[0] - nFacts) 
    eVal_ = np.diag(eVal_) 
    corr1 = np.dot(eVec, eVal_).dot(eVec.T) 
    corr1 = cov2corr(corr1) 
    return corr1

def denoisedCorr2(eVal, eVec, nFacts, alpha):
    eValL,eVecL = eVal[:nFacts,:nFacts],eVec[:,:nFacts]
    eValR,eVecR = eVal[nFacts:,nFacts:],eVec[:,nFacts:]
    corr0 = np.dot(eVecL,eValL).dot(eVecL.T)
    corr1 = np.dot(eVecR,eValR).dot(eVecR.T)
    corr2 = corr0 + alpha*corr1 + (1-alpha)*np.diag(np.diag(corr1)) 
    return corr2
        
def RMT1(rets:np.array, q=2, bWidth=0.01):
    cov0  = np.cov(rets, rowvar=0)
    corr0 = cov2corr(cov0)
    eVal0, eVec0 = getPCA(corr0)
    eMax0, var0  = findMaxEval(np.diag(eVal0), q, bWidth)
    nFacts0 = eVal0.shape[0]-np.diag(eVal0)[::-1].searchsorted(eMax0)
    cor_mat = denoisedCorr(eVal0, eVec0, nFacts0) #denoising by constant residual eigenvalue method
    cov_mat = corr2cov(cor_mat, np.diag(cov0)**.5)
    return cov_mat, cor_mat

def RMT2(rets:np.array, q=2, bWidth=0.01, alpha = 0.1):
    cov0  = np.cov(rets, rowvar=0)
    corr0 = cov2corr(cov0)
    eVal0, eVec0 = getPCA(corr0)
    eMax0, var0  = findMaxEval(np.diag(eVal0), q, bWidth)
    nFacts0 = eVal0.shape[0]-np.diag(eVal0)[::-1].searchsorted(eMax0)
    cor_mat = denoisedCorr2(eVal0, eVec0, nFacts0, alpha) #denoising by shrunk residual eigenvalue method
    cov_mat = corr2cov(cor_mat, np.diag(cov0)**.5)
    return cov_mat, cor_mat

class portfolio_optimizer:
    def __init__(self, min_weight: float = 0., max_weight: float = 1.0,
                 cov_function: str = "HC", freq: str = "monthly",
                 lookback_T: int | None = None, wiq_m: int | None = None, wiq_params: dict | None = None,
                 diagnostics_path: str | None = None):
        assert freq in ['daily', 'monthly'], "return series must be daily or monthly"
        assert 1 > min_weight >= 0, "min weight must be [0, 1)"
        assert 1 >= max_weight > 0, "max weight must be (0, 1]"

        self.min_weight   = min_weight
        self.max_weight   = max_weight
        self.factor       = 252 if freq == "daily" else 12  # annual converter
        self.cov_function = cov_function
        self.freq         = freq
        self.init_weights = None
        self.covariance   = None
        self.returns_df   = None
        self.obj_function = None

        # WIQ (Wishart-IQ) configuration (only used if cov_function=="WIQ")
        self.wiq_T = lookback_T
        self.wiq_m = wiq_m
        self.wiq_params = wiq_params or {}

        # optional OOS diagnostics logging (vol ratio, delta_eff, region masses)
        self.diagnostics_path = diagnostics_path
        self._last_diag_key = None

    def _maybe_log_wiq_diagnostics(self, *, dbg: dict, scale_df: pd.DataFrame) -> None:
        """Append one diagnostics row per window, if enabled.

        Uses the window end date (scale_df.index[-1]) to key uniqueness.
        """
        if self.diagnostics_path is None:
            return
        if not isinstance(dbg, dict):
            return
        if not isinstance(dbg.get("delta_eff_summary", None), dict):
            return

        end_date = str(scale_df.index[-1]) if hasattr(scale_df, "index") and len(scale_df.index) > 0 else ""
        key = (end_date, str(self.cov_function))
        if key == self._last_diag_key:
            return
        self._last_diag_key = key

        row = {
            "date": end_date,
            "cov_function": str(self.cov_function),
            "T_live": int(self.wiq_T) if self.wiq_T is not None else None,
            "T_scale": int(self.wiq_m) * int(self.wiq_T) if (self.wiq_T is not None and self.wiq_m is not None) else None,
            "wiq_m": int(self.wiq_m) if self.wiq_m is not None else None,
            "delta_L": float((self.wiq_params or {}).get("delta_L")) if (self.wiq_params or {}).get("delta_L") is not None else None,
            "delta_R": float((self.wiq_params or {}).get("delta_R")) if (self.wiq_params or {}).get("delta_R") is not None else None,
        }
        row.update(dbg.get("delta_eff_summary", {}))
        if isinstance(dbg.get("region_mass_summary", None), dict):
            row.update(dbg.get("region_mass_summary", {}))

        df_row = pd.DataFrame([row])
        write_header = not os.path.exists(self.diagnostics_path)
        df_row.to_csv(self.diagnostics_path, mode=("w" if write_header else "a"), header=write_header, index=False)

    def set_returns(self, returns_df: pd.DataFrame):
        self.returns_df = returns_df.copy(deep=True)

    def optimize(self, obj_function: str, prev_weights: np.array = None,
                 init_weights: np.array = None, cost: float = None) -> np.array:
        
        _, p = self.returns_df.shape

        if init_weights is None:
            self.init_weights = np.array(p * [1. / p])
        else:
            self.init_weights = init_weights

        self.obj_function = obj_function

        if self.cov_function   == "HC":
            self.covariance    = self.returns_df.cov().to_numpy()
        elif self.cov_function == "SM":
            self.covariance, _ = ledoit(self.returns_df.values)
        elif self.cov_function == "GS1":
            self.covariance, _ = gerber_cov_stat(self.returns_df.values, threshold=0.5)
        elif self.cov_function == "GS2":
            self.covariance, _ = gerber_cov_stat(self.returns_df.values, threshold=0.7)
        elif self.cov_function == "GS3":
            self.covariance, _ = gerber_cov_stat(self.returns_df.values, threshold=0.9)
        elif self.cov_function == "LS1":
            self.covariance    = cov1Para(self.returns_df)
        elif self.cov_function == "LS2":
            self.covariance     = cov2Para(self.returns_df)
        elif self.cov_function == "LS3":
            self.covariance    = covCor(self.returns_df)
        elif self.cov_function == "LS4":
            self.covariance    = covDiag(self.returns_df)
        elif self.cov_function == "LS5":
            self.covariance    = covMarket(self.returns_df)
        elif self.cov_function == "NLS6":
            self.covariance    = GIS(self.returns_df)
        elif self.cov_function == "NLS7":
            self.covariance    = LIS(self.returns_df)
        elif self.cov_function == "NLS8":
            self.covariance    = QIS(self.returns_df)
        elif self.cov_function == "CRE":
            self.covariance, _ = RMT1(self.returns_df.values)
        elif self.cov_function == "SRE":
            self.covariance, _ = RMT2(self.returns_df.values) 
        elif self.cov_function in ("WIQ", "WIQ_TRUST"):
            if self.wiq_T is None or self.wiq_m is None:
                raise ValueError("WIQ/WIQ_TRUST requires lookback_T and wiq_m.")

            mT = int(self.wiq_m) * int(self.wiq_T)
            if self.returns_df.shape[0] < mT:
                raise ValueError(f"{self.cov_function} requires at least mT={mT} rows, got {self.returns_df.shape[0]}")

            scale_df = self.returns_df.iloc[-mT:]

            # filter to WIQ-only args
            WIQ_ALLOWED = {
                "eta_L","eta_B","eta_R",
                "delta_L","delta_R",
                "gamma","epsilon","threshold_c",
                "center_method","gamma_mode","gamma_max",
                "a_floor","a_mass_normalize",
                "vol_mode","ewma_halflife_mode","ewma_halflife","ewma_halflife_factor",
                "eta_mode","eta_body_equalize","eta_body_param","eta_delta_max",
                "delta_B","eta_B_pos","eta_B_neg",
            }
            wiq_call = {k: v for k, v in (self.wiq_params or {}).items() if k in WIQ_ALLOWED}

            dbg: dict = {}
            Sigma_df = wishart_iq_covariance(scale_df, int(self.wiq_T), **wiq_call, debug_out=dbg)
            self._maybe_log_wiq_diagnostics(dbg=dbg, scale_df=scale_df)

            if self.cov_function == "WIQ_TRUST":
                from wiq_trust_layer import compute_trust_features, apply_trust_layer
                trust_lambda = float((self.wiq_params or {}).get("trust_lambda", 0.0))
                trust_W = (self.wiq_params or {}).get("trust_W", None)
                feature_set = str((self.wiq_params or {}).get("trust_feature_set", "basic2"))
                tail_c = float((self.wiq_params or {}).get("trust_tail_c", 3.0))

                if trust_lambda > 0.0 and trust_W is not None:
                    F = compute_trust_features(scale_df, int(self.wiq_T), feature_set=feature_set, tail_c=tail_c)
                    Sigma_df = apply_trust_layer(
                        Sigma_df, F, np.array(trust_W, dtype=float),
                        lam=trust_lambda, offdiag_only=True
                    )

            self.covariance = Sigma_df.to_numpy()

        elif self.cov_function == "WIQ_TRUST":
            if self.wiq_T is None or self.wiq_m is None:
                raise ValueError("WIQ_TRUST requires lookback_T and wiq_m to be provided to portfolio_optimizer.")
            mT = int(self.wiq_m) * int(self.wiq_T)
            if self.returns_df.shape[0] < mT:
                raise ValueError(f"WIQ_TRUST requires at least mT={mT} rows, got {self.returns_df.shape[0]}")
            scale_df = self.returns_df.iloc[-mT:]

            WIQ_ALLOWED = {
                "eta_L","eta_B","eta_R",
                "delta_L","delta_R",
                "gamma","epsilon","threshold_c",
                "center_method","gamma_mode","gamma_max",
                "a_floor","a_mass_normalize",
                "vol_mode","ewma_halflife_mode","ewma_halflife","ewma_halflife_factor",
                "eta_mode","eta_body_equalize","eta_body_param","eta_delta_max",
                "delta_B","eta_B_pos","eta_B_neg",
            }
            wiq_call_params = {k: v for k, v in (self.wiq_params or {}).items() if k in WIQ_ALLOWED}

            dbg: dict = {}
            Sigma_df = wishart_iq_covariance(
                scale_df,
                int(self.wiq_T),
                **wiq_call_params,
                debug_out=dbg,
            )
            self._maybe_log_wiq_diagnostics(dbg=dbg, scale_df=scale_df)

            # apply trust layer (off-diagonal only, correlation-space) if parameters are present.
            try:
                from wiq_trust_layer import compute_trust_features, apply_trust_layer
                lam = float((self.wiq_params or {}).get("trust_lambda", 0.0))
                W = (self.wiq_params or {}).get("trust_W", None)
                feature_set = str((self.wiq_params or {}).get("trust_feature_set", "basic2"))
                if lam > 0.0 and W is not None:
                    F = compute_trust_features(scale_df, int(self.wiq_T), feature_set=feature_set)
                    Sigma_df = apply_trust_layer(
                        Sigma_df,
                        F,
                        np.asarray(W, dtype=float),
                        lam=lam,
                        offdiag_only=True,
                    )
            except Exception:
                # if trust-layer code or params are missing, fall back silently to WIQ
                pass

            self.covariance = Sigma_df.to_numpy()

        elif self.cov_function == "GS_opt":
            from pathlib import Path
            here = Path(__file__).resolve().parent
            with open(here / "gs_opt_params.json") as f:
                th = float(json.load(f)["threshold"])
            self.covariance, _ = gerber_cov_stat(self.returns_df.values, threshold=th)
        elif self.cov_function == "SRE_opt":
            from pathlib import Path
            here = Path(__file__).resolve().parent
            with open(here / "sre_opt_params.json") as f:
                alpha = float(json.load(f)["alpha"])
            self.covariance, _ = RMT2(self.returns_df.values, alpha=alpha)

        bounds = tuple((self.min_weight, self.max_weight) for _ in range(p))
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}] 
        
        if prev_weights is not None and cost is not None:  # cost function for optimisation
            cost_fun = lambda weights: self.object_function(weights) + \
                                       np.abs(weights - prev_weights).sum() * cost / 10000.
        else:
            cost_fun = lambda weights: self.object_function(weights)
        opt = minimize(cost_fun, x0=self.init_weights, bounds=bounds, constraints=constraints, method="SLSQP")
        return set_eps_wgt_to_zeros(opt['x'])

    def object_function(self, weights: np.array) -> float:
        if self.obj_function == "maxSharpe":
            return -self.calc_annualized_portfolio_sharpe_ratio(weights)
        else:
            raise ValueError("should be maxSharpe")
            
    def calc_annualized_portfolio_return(self, weights: np.array) -> float:
        return float(np.sum(self.returns_df.mean() * self.factor * weights))
    def calc_annualized_portfolio_std(self, weights: np.array) -> float:
        return np.sqrt(np.dot(weights.T, np.dot(self.covariance * self.factor, weights)))
    def calc_annualized_portfolio_moments(self, weights: np.array) -> tuple:
        return self.calc_annualized_portfolio_return(weights), self.calc_annualized_portfolio_std(weights)
    def calc_annualized_portfolio_sharpe_ratio(self, weights: np.array) -> float:
        return self.calc_annualized_portfolio_return(weights) / self.calc_annualized_portfolio_std(weights)
