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

f_compressed = open('compressed.txt', 'wb')
f_original = open('Original.txt', 'wb')

cap = cv2.VideoCapture(0)
ctr=0
while(True):

    [ret, frame] = cap.read()
    r, c, d = frame.shape
    
    cv2.imshow('Original RGB Frame', frame)
    #RGB to To YCbCr Conversion
    YCbCr = methods.rgb2ycbcr(frame)
    Y = YCbCr[:, :, 0]
    Cb = YCbCr[:, :, 1]
    Cr = YCbCr[:, :, 2]
    
    
    #print("YL ",np.max(Y),np.min(Y))
    #print("Cb:", np.max(Cb), np.min(Cb))
    #print()

    pickle.dump(YCbCr,  f_original)
    
    #Filtering
    filtFrame = np.zeros(frame.shape)
    filtFrame[:, :, 0] = Y
    filtFrame[:, :, 1] = sp.convolve2d(Cb, pyF, mode='same')
    filtFrame[:, :, 2] = sp.convolve2d(Cr, pyF, mode='same')
    
    #Chroma Subsampling
    filtFrame[1::2, 1::2, 1] = filtFrame[0::2, 0::2, 1]
    filtFrame[1::2, 1::2, 2] = filtFrame[0::2, 0::2, 2]
    
    #Downsampling
    DCb = np.zeros((r//2,c//2))
    DCb = filtFrame[0::2, 0::2, 1]
    #Dcb = 
    cv2.imshow('Downsampled Cb Component', DCb)
    DCr = np.zeros((r//2,c//2))
    DCr = filtFrame[0::2, 0::2, 2]
    cv2.imshow('Downsampled Cr Component', DCr)

    #Applying DCT
    dct_y = methods.applyDCT(Y/255.0)
    
    dct_DCb = methods.applyDCT((DCb+127)/255.0)
    dct_DCr = methods.applyDCT((DCr+127)/255.0)
    cv2.imshow('DCT Compressed Y Frame', np.abs(dct_y))
    cv2.imshow('DCT Compressed Cb Frame', np.abs(dct_DCb))
    cv2.imshow('DCT Compressed Cr Frame', np.abs(dct_DCr))
    #print(np.max(dct_DCb))
    #print(np.min(dct_DCb))

    # Zero removed downsampled version
    pickle.dump(dct_y, f_compressed)
    pickle.dump(dct_DCb, f_compressed)
    pickle.dump(dct_DCr, f_compressed)
    ctr+=1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

x = f_compressed.tell()/1024.0**2
y = f_original.tell()/1024.0**2

#.astype(np.float16)

print ('File size of the DCT Compressed Video data')
print (x , 'Mb')
print ('File size of the actual Video data')
print (y , 'Mb')
print ('Compression factor')
print (y/x)   
print ('Space saving')
print ((1-(x/y))*100.0 ,  'percent')

cap.release()
cv2.destroyAllWindows()
f_compressed.close()
f_original.close()
