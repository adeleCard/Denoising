# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 14:46:00 2015

@author: cardiologs
"""



import utils
import scale_utils as sc
import numpy as np
import matplotlib.pyplot as plt

# data load : Y is an array  n*m, where n is the ecg's number and m the number of points per ecg
Y = np.load( "/home/cardiologs/Workspace/denoising/code/data/[DE-IDENTIFIED]_#DICOM#_0000A0A5_0926_000E_0926_286321_635440525514417877.dcm.npz" )
Y = Y["data"]

#ESTIMATION DU BRUIT DES DONNÉES
#v = utils.noise_estimation(Y)
#print v

#bruitage des données
sigma = 0.01
Yb = Y + sigma * np.random.standard_normal(Y.shape)


#débruitage et optimisation du filtre
Yh = np.zeros(Y.shape)
Ys = np.zeros(Y.shape)
Yg = np.zeros(Y.shape)
parameters = np.linspace(1, 5, 600)

    
i = np.random.randint(0, Y.shape[0])
def filt_hard (y, tau):
    return utils.wave_hard_filter(y, sigma, tau, 'bior6.8')
def filt_soft(y, tau):
    return utils.wave_soft_filter(y, sigma, tau, 'bior6.8')
def filt_in (y, tau):
    return sc.invariant_wave_filter(y, sigma, tau, 'bior6.8')
tauinv = sc.opt(parameters, filt_in, Yb, Y)
tauhard = sc.opt(parameters, filt_hard, Yb[i], Y[i])
tausoft = sc.opt(parameters, filt_soft, Yb[i], Y[i])


Yinv=sc.invariant_wave_filter(Yb, sigma, tauinv, 'bior6.8')
YhardIt = utils.iterative_hard_filter(Yb[i], sigma, tauhard, 'bior6.8')
YsoftIt = utils.iterative_soft_filter(Yb[i], sigma, tausoft, 'bior6.8')
Yhard = utils.wave_hard_filter(Yb[i], sigma, tauhard, 'bior6.8')
Ysoft = utils.wave_soft_filter(Yb[i], sigma, tausoft, 'bior6.8')
mug = sc.opt(parameters, utils.gaussian_filter, Yb[i], Y[i])
Yg[i] = utils.gaussian_filter(Yb[i], mug*sigma)
Yd = sc.butter_bandpass_filter(Yb, 0.08, 50, 500)

plt.figure(figsize=(20, 20))
plt.subplot(811)
plt.plot(Yd[i] )
plt.subplot(812)
plt.plot(Yb[i] - Yhard )
plt.subplot(813)
plt.plot(Yinv[i])
plt.subplot(814)
plt.plot(Yhard )
plt.subplot(815)
plt.plot(Ysoft)
plt.subplot(816)
plt.plot(Yg[i])
plt.subplot(817)
plt.plot(Yb[i])
plt.subplot(818)
plt.plot(Y[i] )

plt.figure()
plt.plot(YhardIt + 1)
plt.plot(Yhard)

