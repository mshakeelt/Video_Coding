import cv2
import numpy as np
import pickle
import scipy.signal as sp
import os
import methods

N = 2
'''Filter Design'''
lpF = np.ones((N, N))/N
pyF = sp.convolve2d(lpF, lpF)/N

xf = open('compressed.txt', 'rb')

try:
    #ctr = 1
    while True:
        dct_y = pickle.load(xf)
        dct_DCb = pickle.load(xf)
        dct_DCr = pickle.load(xf)
        #cv2.imshow('Before iDCT', dct_y)
        
        #Applying Inverse DCT

        Y = 255.0* methods.applyIDCT(dct_y)
        DCb =255.0* methods.applyIDCT(dct_DCb)+128
        DCr = 255.0*methods.applyIDCT(dct_DCr)+128
        #cv2.imshow('After iDCT Y', Y)
        #cv2.imshow('After iDCT DCb', DCb)
        #cv2.imshow('After iDCT DCr', DCr)

        
        #Upsampling
        r, c = Y.shape
        Cb_upsampled = np.zeros((r, c))
        Cb_upsampled[::2, ::2] = DCb
        Cr_upsampled = np.zeros((r, c))
        Cr_upsampled[::2, ::2] = DCr

        #Filtering
        Cb = sp.convolve2d(Cb_upsampled, pyF, mode='same')
        Cr = sp.convolve2d(Cr_upsampled, pyF, mode='same')
        cv2.imshow('After Filtering Cb', Cb)
        cv2.imshow('After Filtering Cr', Cr)
        
        #Gathering YCbCr components
        YCbCr_decoded = np.zeros((r, c, 3)) 
        YCbCr_decoded[:,:,0] = Y
        YCbCr_decoded[:,:,1] = Cb
        YCbCr_decoded[:,:,2] = Cr
        
        #YCbCr to RGB COnversion
        decoded = methods.ycbcr2rgb(YCbCr_decoded)
        cv2.imshow('Final Frame', decoded)
        
        if cv2.waitKey(500) & 0xFF == ord('q'):
            break
except (EOFError):
    pass
