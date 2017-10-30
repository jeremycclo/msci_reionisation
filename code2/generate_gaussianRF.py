# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 19:51:13 2017

@author: Federico
"""

"""
Generate gaussian random fields with a known power spectrum

"""

import numpy as np
import matplotlib.pyplot as plt


#==============================================================================
# Code below edited from:
# http://andrewwalker.github.io/statefultransitions/post/gaussian-fields/
#==============================================================================

def fftIndgen(n):
    a = range(0, n/2+1)
    b = range(1, n/2)
    b.reverse()
    b = [-i for i in b]
    return a + b

def generate(Pk, size):
    def Pk2(kx, ky):
        if kx == 0 and ky == 0:
            return 0.0
        return np.sqrt(Pk(np.sqrt(kx**2 + ky**2)))
    noise = np.fft.fft2(np.random.normal(size = (size, size)))
    amplitude = np.zeros((size,size))
    for i, kx in enumerate(fftIndgen(size)):
        for j, ky in enumerate(fftIndgen(size)):            
            amplitude[i, j] = Pk2(kx, ky)
    return np.fft.ifft2(noise * amplitude)

#==============================================================================
# 
#==============================================================================

def gaussian(mean, var):   
    return lambda k : np.exp((-(k - mean)**2.)/(2*var))

def power_law(alpha):
    return lambda k : k**alpha

def expn(beta):
    return lambda k : np.exp((-k)/beta)

def binary(f, x_i = 0.5):
    "x_i is the desired total ionized fraction"
    thresh = np.percentile(f,(1 - x_i)*100)
    F = np.zeros(f.shape)
    F[f > thresh] = 1
    return F

def main():
    """
    Test example
    """

    num = 3
    field = generate(Pk = power_law(-1.), size=256)
    fieldR = field.real
    
    for x_i in np.linspace(1./num, 1 - 1./num, num):
        fieldRB = binary(fieldR, x_i)
    
        plt.figure()
        plt.imshow(fieldR, interpolation='none', cmap='hot')
        
        plt.figure()
        plt.imshow(fieldRB, interpolation='none', cmap='Greys')

if __name__ == "__main__":
    main()
