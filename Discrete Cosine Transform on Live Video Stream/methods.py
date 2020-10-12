import numpy as np
import scipy.signal as sp
import math
import scipy.fftpack as sft

DTCFactor = 4           #The allowed values/ QUALITY FACTORS are 1(1/64),2(1/16) and 4(1/4 frequency data)

RGB = np.matrix([[0.299,     0.587,     0.114],
                 [-0.16864, -0.33107,   0.49970],
                 [0.499813, -0.418531, -0.081282]])

YCbCr = RGB.I

def rgb2ycbcr(frame):

    xframe=np.zeros(frame.shape)

    for i in range(frame.shape[0]):
        xframe[i] = np.dot(RGB, frame[i].T).T

    return xframe

def ycbcr2rgb(frame):

    xframe=np.zeros(frame.shape)

    for i in range(frame.shape[0]):
        xframe[i] = np.dot(YCbCr, frame[i].T).T#/255.

    return xframe


def applyDCT(frame):
    
    r,c = frame.shape
    Mr = np.concatenate([np.ones(DTCFactor), np.zeros(8-DTCFactor)])
    #print(Mr)
    # print "Mr: \n", Mr
    # Mr[int(8/4.0):r]=np.zeros(int(3.0/4.0*8))
    #Mr=Mr[]
    Mc = Mr
    # frame=np.reshape(frame[:,:,1],(-1,8), order='C')
    frame=np.reshape(frame,(-1,8), order='C')
    X=sft.dct(frame,axis=1,norm='ortho')
    #apply row filter to each row by matrix multiplication with Mr as a diagonal matrix from the right:
    X=np.dot(X,np.diag(Mr))
    #shape it back to original shape:
    X=np.reshape(X,(-1,c), order='C')
    #Shape frame with columns of hight 8 by using transposition .T:
    X=np.reshape(X.T,(-1,8), order='C')
    X=sft.dct(X,axis=1,norm='ortho')
    #apply column filter to each row by matrix multiplication with Mc as a diagonal matrix from the right:
    X=np.dot(X,np.diag(Mc))
    #shape it back to original shape:
    X=(np.reshape(X,(-1,r), order='C')).T
    
    #Set to zero the 7/8 highest spacial frequencies in each direction:
    #X=X*M    
    return X


    
def applyIDCT(frame):

    r,c= frame.shape
    X=np.reshape(frame,(-1,8), order='C')
    X=sft.idct(X,axis=1,norm='ortho')
    #shape it back to original shape:
    X=np.reshape(X,(-1,c), order='C')
    #Shape frame with columns of hight 8 (columns: order='F' convention):
    X=np.reshape(X.T,(-1,8), order='C')
    x=sft.idct(X,axis=1,norm='ortho')
    #shape it back to original shape:
    x=(np.reshape(x,(-1,r), order='C')).T

    return x

