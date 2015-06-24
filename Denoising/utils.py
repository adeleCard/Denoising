# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 15:36:48 2015

@author: cardiologs
"""
import numpy as np
import pywt
import pylab as plt
import math


def snr(f0, f):
    """compute the signal noise ratio
    
    Parameter
    ---------
    f0: n d numpy array
        raw signal 
    f: n d numpy array
        denoised signal
    """
    return 20 * np.log10(np.linalg.norm(f0) / np.linalg.norm(f0 - f))


def gaussian_filter (y, mu) :
    """compute the filtration of y by a gaussian discrete filter of variance mu = tau*sigma, where sigma is the noise' variance

    Parameter
    -----------
    y : n d dumpy array
        noisy signal
    mu : int
        variance
    """
    gaus = lambda x, sigma : np.exp(-x**2 / (2 * sigma**2 )) 
    t = np.linspace(-3 * mu, 3 * mu, np.maximum(20 * mu, 20))
    norm = lambda s : gaus(t, s) / np.sum(gaus(t, s))
    return np.convolve(y, norm(mu), 'same')
    
        
def wave_hard_filter(y, sigma, tau, wavelet):
    """return the wavelet transform of y admitting there is a gaussian noise which variance is sigma, for  hard thresholding.
    
    Parameter
    -----------
    y : noisy signal
    sigma : noise variance
    t : threshold for the wave transform
    """
    
    coeffs = pywt.wavedec(y, wavelet)
    threshold = sigma * tau
    hcoeffs = []
    for scale, x in enumerate(coeffs):
        if scale < 5:
            hcoeffs.append(x)
        else:
            hcoeffs.append(pywt.thresholding.hard(x, threshold))
    return pywt.waverec(hcoeffs, wavelet)




def wave_soft_filter(y, sigma, tau, wavelet):
    """return the wavelet transform of y admitting there is a gaussian noise which variance is sigma, for soft thresholding.
    
    Parameter
    -----------
    y : noisy signal
    sigma : noise variance
    tau : threshold for the wave transform
    wavelet : wavelet object used for the transform
    """
    
    coeffs = pywt.wavedec(y, wavelet)
    threshold = sigma * tau
    hcoeffs = []
    for scale, x in enumerate(coeffs):
            hcoeffs.append(pywt.thresholding.soft(x, threshold))
    return pywt.waverec(hcoeffs, wavelet)


def wave_semisoft_filter(y, sigma, tau, w, mu):
    coeffs = pywt.wavedec(y, w)
    threshold = sigma * tau
    hcoeffs = []
    for scale, x in enumerate(coeffs):
            hcoeffs.append(thresholding_semisoft(x, threshold, mu))
    return pywt.waverec(hcoeffs, w)


def wave_stein_filter(y, sigma, tau, w):
    coeffs = pywt.wavedec(y, w)
    threshold = sigma * tau
    hcoeffs = []
    for scale, x in enumerate(coeffs):
            hcoeffs.append(stein_thresholding(x, threshold))
    return pywt.waverec(hcoeffs, w)
    

def thresholding_semisoft(x, threshold, mu):   
    def unit(y, threshold, mu):
        if -mu<y.any()<mu:
                return pywt.thresholding.soft(y, mu*threshold)
        else:
            return pywt.thresholding.hard(x, threshold)
    res = np.zeros(x.shape)
    if len(x.shape) == 2:
        for i in range(x.shape[0]):
           res[i] = unit(x[i], threshold, mu)
    else:
        res = unit(x, threshold, mu)
    return res


def stein_thresholding(x, threshold):
    return  x * np.maximum( 1 - (threshold**2) / (x**2), 0 )
        

def opt(parameters, filt, y, y0):
    """plot the graph of snr, and return the best value of parameters.
    
    Parameter
    ------------
    parameters : parameters for the function filt
    filt : filter
    y : noisy signal
    y0 : original signal
    """
    snrlist = np.zeros(len(parameters))
    for i in range(len(parameters)): 
        snrlist[i] = snr(y0, filt(y, parameters[i]))
    plt.plot(parameters, snrlist)
    i = np.argmax(snrlist)
    return parameters[i]    

    
def iswt(coefficients, wavelet):
    """
      Input parameters: 

        coefficients
          approx and detail coefficients, arranged in level value 
          exactly as output from swt:
          e.g. [(cA1, cD1), (cA2, cD2), ..., (cAn, cDn)]

        wavelet
          Either the name of a wavelet or a Wavelet object

    """
    output = coefficients[0][0].copy() # Avoid modification of input data

    #num_levels, equivalent to the decomposition level, n
    num_levels = len(coefficients)
    for j in range(num_levels,0,-1): 
        step_size = int(math.pow(2, j-1))
        last_index = step_size
        _, cD = coefficients[num_levels - j]
        for first in range(last_index): # 0 to last_index - 1

            # Getting the indices that we will transform 
            indices = np.arange(first, len(cD), step_size)

            # select the even indices
            even_indices = indices[0::2] 
            # select the odd indices
            odd_indices = indices[1::2] 

            # perform the inverse dwt on the selected indices,
            # making sure to use periodic boundary conditions
            x1 = pywt.idwt(output[even_indices], cD[even_indices], wavelet, 'per') 
            x2 = pywt.idwt(output[odd_indices], cD[odd_indices], wavelet, 'per') 

            # perform a circular shift right
            x2 = np.roll(x2, 1)

            # average and insert into the correct indices
            output[indices] = (x1 + x2)/2.  
    return output


def stationary_hard_filter (y, sigma, tau, level=3):
    threshold= tau * sigma    
    coeffs = pywt.swt(y, 'db6', level)
    hcoeffs =[]
    for scale, x in enumerate(coeffs):
        hcoeffs.append(pywt.thresholding.hard(x, threshold))
    return iswt(hcoeffs, 'db6') 