# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 17:40:59 2015

@author: cardiologs
"""
import numpy as np
import pywt
import utils
import matplotlib.pylab as plt

def noise_estimation(y):
    """compute the variance of the noise supposed gaussian in the datas y. Return a list of the variance at each level. 
    
    Parameter
    ----------
    y : array of datas
    """

    return 1.4826 * np.median(np.abs(y-np.median(y)))
   



def multi_wave_filter(y, w, mode='hard'):
    if len(y.shape) == 2:
        N = y[1]
    else:
        N = len(y)
    hcoeffs = []
    coeffs = pywt.wavedec(y, w)
    hcoeffs.append(coeffs[0])    
    for i in range(len(coeffs) - 1):
        t = noise_estimation(coeffs[i + 1])
#        t *= np.sqrt( 2 * np.log10(N) / np.log10(i + 2))
#        print t
        t *= np.sqrt( 3 * np.log(N / 2**(i + 1)))
        if i > 3:
            hcoeffs.append(pywt.thresholding.hard(coeffs[i + 1], t))
        else:
            hcoeffs.append(coeffs[i + 1])
    return pywt.waverec(hcoeffs, w)
    

def invariant_multi_wave_filter(y, w, m):
    fti = np.zeros(y.shape)
    for i in range(1, m):
        y_translate = np.roll(y, i)
        y_translate = multi_wave_filter(y_translate, w)
        y_translate = np.roll(y_translate, y.shape[-1] - i)
        fti += y_translate 
    fti /= float(m)
    return fti
    
    
def wave_hard_filter(y, sigma, tau, w):
    Yh =np.zeros(y.shape)
    if len(y.shape) == 2:
        for i in range(y.shape[0]):
            Yh[i] = utils.wave_hard_filter(y[i], sigma, tau, w)
    else:
        Yh = utils.wave_hard_filter(y, sigma, tau, w)
    return Yh

    
def wave_soft_filter(y, sigma, tau, w):
    Yh =np.zeros(y.shape)
    if len(y.shape) == 2:
        for i in range(y.shape[0]):
            Yh[i] = utils.wave_soft_filter(y[i], sigma, tau, w)
    else:
        Yh = utils.wave_soft_filter(y, sigma, tau, w)
    return Yh


def wave_semisoft_filter(y, sigma, tau, w, mu):
    Yh =np.zeros(y.shape)
    if len(y.shape) == 2:
        for i in range(y.shape[0]):
            Yh[i] = utils.wave_semisoft_filter(y[i], sigma, tau, w, mu)
    else:
        Yh = utils.wave_semisoft_filter(y, sigma, tau, w, mu)
    return Yh


def wave_stein_filter(y, sigma, tau, w):
    if len(y.shape) == 2:
        Yh =np.zeros(y.shape)
        for i in range(y.shape[0]):
            Yh[i] = utils.wave_stein_filter(y[i], sigma, tau, w)
    else: 
        Yh = utils.wave_stein_filter(y, sigma, tau, w)
    return Yh
    

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
        snrlist[i] = utils.snr(y0, filt(y, parameters[i]))
    i = np.argmax(snrlist)
    return parameters[i]
    
    
def invariant_wave_filter(y, sigma, tau, w, m=10, mode = 'hard', mu=2):
    """ compute the denoised signal thanks to an invariant wavelet filter.
    
    Parameters
    --------
    y : noisy signal
    sigma : noise estimation
    tau : threshold = tau*sigam
    w : wavelet
    mode : 'hard, 'soft', 'semisoft', or 'stein'
    """    
    
    fti = np.zeros(y.shape)
    if mode == 'hard':
        for i in range(1, m):
            y_translate = np.roll(y, i)
            y_translate = wave_hard_filter(y_translate, sigma, tau, w)
            y_translate = np.roll(y_translate, y.shape[-1] - i)
            fti += y_translate
    
    if mode == 'soft':
        for i in range(1, m):
            y_translate = np.roll(y, i)
            y_translate = wave_soft_filter(y_translate, sigma, tau, w)
            y_translate = np.roll(y_translate, y.shape[-1] - i)
            fti += y_translate
    
    if mode == 'semisoft':
        for i in range(1, m):
            y_translate = np.roll(y, i)
            y_translate = wave_semisoft_filter(y_translate, sigma, tau, w, mu)
            y_translate = np.roll(y_translate, y.shape[-1] - i)
            fti += y_translate
    
    if mode == 'stein':
        for i in range(1, m):
            y_translate = np.roll(y, i)
            y_translate = wave_stein_filter(y_translate, sigma, tau, w)
            y_translate = np.roll(y_translate, y.shape[-1] - i)
            fti += y_translate
            
    fti /= float(m)
    return fti
    
from scipy.signal import butter, lfilter


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def _wrap(y, filt, params):
    return filt(y, **params)
    

def filt_plot(y, y0, filt, params, title="Denoising"):
    yd = _wrap(y, filt, params)
    splot(y, y0, yd, title)


def splot(y, y0, yd, title="Denoising"):
    """ Plot the denoised signal thanks to the filter filt with its parameters parameters.
    y : noisy signal
    y0 : raw signal
    yd : denoised signal
    mu : gaussian filter parameter
    """
    fig = plt.figure(figsize=(20, 12))
    _y0 = y0[:2000]
    _y = y[:2000]
    _yd = yd[:2000]
    plt.subplot(221)
    plt.plot(_y0)
    plt.title('Raw signal :')
    plt.subplot(222)
    plt.plot(_y)
    plt.title('Noised signal')
#    plt.plot(utils.gaussian_filter(y, mu))
#    plt.title('Result for the gaussian filter - SNR :' + str(utils.snr(y0, utils.gaussian_filter(y, mu))))
    plt.subplot(223)
    plt.plot(_yd, "r")
    plt.plot(_y0, linewidth=2.5, alpha=0.3)
    plt.title('Denoised signal - SNR : %0.2f dB' % utils.snr(y0, yd))
    plt.subplot(224)
    plt.plot(_y0 - _yd)
    plt.title('Differences between raw and denoised signal :')
    fig.suptitle(title, fontsize=30, fontweight="bold")