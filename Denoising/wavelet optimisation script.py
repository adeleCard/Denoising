# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 16:31:55 2015

@author: cardiologs
"""

import pandas as pd
import utils
import scale_utils as sc
import numpy as np
import pywt
import matplotlib.pyplot as plt

# data load : Y is an array  n*m, where n is the ecg's number and m the number of points per ecg
Y = np.load( "/home/cardiologs/Workspace/denoising/code/data/[DE-IDENTIFIED]_#DICOM#_0000A0A5_0926_000E_0926_286321_635440525514417877.dcm.npz" )
Y = Y["data"]

#ESTIMATION DU BRUIT DES DONNÉES
#v = utils.noise_estimation(Y)
#print v

#bruitage des données
sigma = 0.02
Yb = Y + sigma * np.random.standard_normal(Y.shape)

#création de la table des snr
snrframe = pd.DataFrame(index=['hard', 'soft'])

#débruitage et optimisation du filtre
Yh = np.zeros(Y.shape)
Ys = np.zeros(Y.shape)
Yg = np.zeros(Y.shape)
parameters = np.linspace(1, 5, 600)


#def filt(y, tau):
 #   return utils.gaussian_filter(y, tau*sigma)
for i in range(Y.shape[0]):
    mug = sc.opt(parameters, utils.gaussian_filter, Yb[i], Y[i])
    Yg[i] = utils.gaussian_filter(Yb[i], mug*sigma)
snrframe.loc[:,'gaussian'] = pd.Series(data=[utils.snr(Y, Yg), utils.snr(Y, Yg) ], index=snrframe.index)
snrframe.loc[:,'noisy'] = pd.Series(data=[utils.snr(Y, Yb), utils.snr(Y, Yb) ], index=snrframe.index)

for w in pywt.wavelist('db'):
    print w
    def filt_hard (y, tau):
            return utils.wave_hard_filter(y, sigma, tau, w)
    def filt_soft(y, tau):
        return utils.wave_soft_filter(y, sigma, tau, w)
    for i in range(Y.shape[0]):
        tauhard = sc.opt(parameters, filt_hard, Yb[i], Y[i])
        tausoft = sc.opt(parameters, filt_soft, Yb[i], Y[i])
        Yh[i] = utils.wave_hard_filter(Yb[i], sigma, tauhard, w)
        Ys[i] = utils.wave_soft_filter(Yb[i], sigma, tausoft, w)
    snrframe.loc[:,w] = pd.Series(data=[utils.snr(Y, Yh), utils.snr(Y, Ys) ], index=snrframe.index)

for w in pywt.wavelist('bior'):
    print w
    def filt_hard (y, tau):
            return utils.wave_hard_filter(y, sigma, tau, w)
    def filt_soft(y, tau):
        return utils.wave_soft_filter(y, sigma, tau, w)
    for i in range(Y.shape[0]):
        tauhard = sc.opt(parameters, filt_hard, Yb[i], Y[i])
        tausoft = sc.opt(parameters, filt_soft, Yb[i], Y[i])
        Yh[i] = utils.wave_hard_filter(Yb[i], sigma, tauhard, w)
        Ys[i] = utils.wave_soft_filter(Yb[i], sigma, tausoft, w)
    snrframe.loc[:,w] = pd.Series(data=[utils.snr(Y, Yh), utils.snr(Y, Ys) ], index=snrframe.index)

print "Noisy signal's SNR :" + str(snrframe['noisy'])
print "Snr for a gaussian denoising :" + str(snrframe['gaussian'])    
print "Best wavelet"
sortframe = snrframe.unstack()
sortframe.sort()
print sortframe[-1:]

# results :bruit=0.04 bior6.8 hard : 23.6692
#bruit=0.02 db6 hard 28.358, db8 :28.3827
