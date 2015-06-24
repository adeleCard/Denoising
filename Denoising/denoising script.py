# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 10:41:22 2015

@author: cardiologs
"""

import scale_utils as sc
import numpy as np
import matplotlib.pyplot as plt
import utils
import pywt

#Signal brut, non-bruité
plt.figure(figsize=(20, 3))
X = np.linspace(0, 10, 5000)
Y = np.load("/home/cardiologs/Workspace/denoising/code/data/[DE-IDENTIFIED]_#DICOM#_0000A0A5_0926_000E_0926_286321_635440525514417877.dcm.npz")
Y = Y["data"]
#plt.plot(X, Y-1)
#plt.title('Clear signal')

#Bruitage du signal par un bruit gaussien de variance sigma
sigma = 0.04
Yb = Y + sigma * np.random.standard_normal(Y.shape)
#plt.plot(X, Yb)
#plt.title('Noisy signal')
print utils.snr(Y, Yb)

##Choix du fltre et débruitage du bruit
w = pywt.Wavelet ('bior6.8')
#Yd = utils.wave_hard_filter(Yb, sigma, 20, w)
#plt.plot(X, Yd+1)
#plt.title("Denoised signal")
#plt.text(0, -1, "SNR :" + str(utils.snr(Y,Yd)))


#optimisation d'un paramètre du filtre
def filt (y, tau):
    return sc.invariant_wave_filter(y, sigma, tau, w)

parameters = np.linspace(1, 5, 600)
tauopt = utils.opt(parameters, filt, Yb, Y)
#
#Yf = utils.wave_hard_filter(Yb, sigma, tauopt, w)
#plt.plot(Yf)
#print ("Snr maximale : "+ str(utils.snr(Y, Yf)))
print( "Tau optimal : " + str(tauopt))
sc.filt_plot(Yb, Y, sc.invariant_wave_filter, params={"sigma": sigma, "tau": tauopt, "w": w, "m": 25})
#def filt_soft (y, t):
#    return utils.wave_soft_filter(y, sigma, t, w)
#plt.figure()
#plt.title('Snr en fonction du parametre du filtre :')
#tausoft = utils.opt(parameters, filt_soft, Yb, Y)
#plt.figure(figsize=(20, 3))
#Yfsoft = utils.wave_soft_filter(Yb, sigma, tausoft, w)
#plt.plot(Yf)
#plt.plot(Yfsoft-1)
#print ("Snr maximale soft : "+ str(utils.snr(Y, Yfsoft)))
#print( "Tau optimal soft : " + str(tausoft))

