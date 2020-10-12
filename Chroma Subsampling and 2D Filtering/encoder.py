import cv2
import numpy as np
import pickle
import scipy.signal as sp
import os
import methods

N = 2
'''Filter Design'''
lpF = np.ones((N, N), dtype=np.float32)/2**N
pyF = sp.convolve2d(lpF, lpF)

f_sub = open('videorecord_DS_compressed.txt', 'wb')
f_full = open('videorecord.txt', 'wb')
f_down = open('videorecord_DS.txt', 'wb')
cap = cv2.VideoCapture(0)

while(True):

    ret, frame = cap.read()

    reduced = frame.copy()
    cv2.imshow('original', reduced)
    Y, Cb, Cr = methods.rgb2ycbcr(reduced)
    originalFrame = np.zeros(frame.shape, np.int8)
    originalFrame[:, :, 0] = Y
    originalFrame[:, :, 1] = Cb
    originalFrame[:, :, 2] = Cr
    pickle.dump(originalFrame,  f_full)
    cv2.imshow('Ycbcr', originalFrame/255.)
    #Chroma subsampling with zeros
    Cb_down = np.zeros(Cb.shape)                
    Cb_down[0::2, 0::2] = Cb[0::2,0::2]
    Cb_down[1::2, 1::2] = Cb_down[0::2, 0::2] 
    
    Cr_down = np.zeros(Cr.shape)
    Cr_down[0::2, 0::2] = Cr[0::2,0::2]
    Cr_down[1::2, 1::2] = Cr_down[0::2, 0::2]

    cv2.imshow('Cb Down', np.abs(Cb_down/255.))
    cv2.imshow('Cr Down', np.abs(Cr_down/255.))
    cv2.imshow('Y', Y/255.)

    enc = np.zeros(frame.shape, np.int8)

    enc[:, :, 0] = Y
    enc[:, :, 1] = Cb_down
    enc[:, :, 2] = Cr_down
    
    pickle.dump(enc, f_down)
    
    
    
    #Chroma Subsampling without zeros
    r, c, ch = frame.shape
    
    ds_compressed = np.zeros((r//2,c//2,2), np.int8)
    
    #originalFrame[:, :, 0] = sp.convolve2d(originalFrame[:, :, 0], pyF, mode='same')
    #originalFrame[:, :, 1] = sp.convolve2d(originalFrame[:, :, 1], pyF, mode='same')
    #originalFrame[:, :, 2] = sp.convolve2d(originalFrame[:, :, 2], pyF, mode='same')


    #ds_compressed[:, :, 0]= originalFrame[::2, ::2, 0]
    ds_compressed[:, :, 0]= originalFrame[::2, ::2, 1]
    ds_compressed[:, :, 1]= originalFrame[::2, ::2, 2]
   
    cv2.imshow('cb_compressed', ds_compressed[:, :, 0]/255.)
    cv2.imshow('cr_compressed', ds_compressed[:, :, 1]/255.)

    
    #cv2.imshow('ds_compressed', ds_compressed/255.)

    #Chroma Subsampling without zeros
    pickle.dump(ds_compressed, f_sub)
    

    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

x = f_sub.tell()/1024.0**2
y = f_full.tell()/1024.0**2
z = f_down.tell()/1024.0**2

print ('File size of the 4:2:0 subsampled image data')
print (z , 'Mb')
print ('File size of the downSampled image data')
print (x , 'Mb')
print ('File size of the actual image data')
print (y , 'Mb')
print ('Compression factor')
print (y/x)   
print ('Space saving')
print ((1-(x/y))*100.0 ,  'percent')

cap.release()
cv2.destroyAllWindows()
f_sub.close()
f_full.close()
