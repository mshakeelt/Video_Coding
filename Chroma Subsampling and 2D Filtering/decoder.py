import cv2
import numpy as np
import pickle
import scipy.signal as sp
import os
import methods

N = 2
'''Filter Design'''
lpF = np.ones((N, N), dtype=np.float32)/N
pyF = sp.convolve2d(lpF, lpF)/N

xi = open('videorecord.txt', 'rb')
xf = open('videorecord_DS_compressed.txt', 'rb')
try:
    ctr = 1
    while True:
        YFrame = pickle.load(xi)
        Y= YFrame [:,:,0]
        
        
        reduced = pickle.load(xf)
        frame = reduced.copy()
        
        #y = frame[:,:,0]
        Cb_down = frame[:,:,0]
        Cr_down = frame[:,:,1]
        
        #Y = methods.app420(y)
        Cb_sub = methods.app420(Cb_down)
        Cr_sub = methods.app420(Cr_down)
        r, c = Y.shape
        Y = Y.astype(np.uint8)
        #cv2.imshow('Y', Y/255.)
        #cv2.imshow('Cb', np.abs(Cb_sub/255.))
        #cv2.imshow('Cr', np.abs(Cr_sub/255.))

        #Y_lpf  = sp.convolve2d(Y, pyF, mode='same')
        Cb_lpf = sp.convolve2d(Cb_sub, pyF, mode='same')
        Cr_lpf = sp.convolve2d(Cr_sub, pyF, mode='same')
        
        frame_lpf = np.zeros((r, c, 3))
        frame_lpf[:,:,0] = Y
        frame_lpf[:,:,1] = Cb_lpf
        frame_lpf[:,:,2] = Cr_lpf
        
        #cv2.imshow('Y', Y_lpf/255.)
        #cv2.imshow('Cb filt', np.abs(Cb_lpf/255.))
        #cv2.imshow('Cr filt', np.abs(Cr_lpf/255.))

        cv2.imshow('Rec', methods.ycbcr2rgb(frame_lpf))

        if cv2.waitKey(500) & 0xFF == ord('q'):
            break
except (EOFError):
    pass
