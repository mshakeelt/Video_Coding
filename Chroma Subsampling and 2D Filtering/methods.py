import numpy as np
import scipy.signal as sp
import math
def app420(frame):
    """
    frame: 2D component(for example Cb or Cr)
    N:     Upsampling factor for 420 scheme.
           Should be selected based on Downsampling factor
    """
    r, c = frame.shape
    # print frame.shape
    fu = np.zeros((r*2,c*2))
    fu[::2, ::2] = frame
    #fu[1::2,] = fu[::2,]
    #fu[:,1::2] = fu[:,::2]
    return fu

def upsample(frame, N):
    r, c = frame.shape
    # print frame.shape
    fu = np.zeros((r*N,c*N))
    fu[::N, ::N] = frame
    return fu

def rgb2ycbcr(frame):
    R = frame[:,:,0]
    G = frame[:,:,1]
    B = frame[:,:,2]

    #red = frame.copy()
    Y = (0.299*R + 0.587*G + 0.114*B)
    Cb = (-0.16864*R - 0.33107*G + 0.49970*B)
    Cr = (0.499813*R - 0.418531*G - 0.081282*B)
    return Y, Cb, Cr

def ycbcr2rgb(framedec):
    Y = (framedec[:,:,0])/255.
    Cb = (framedec[:,:,1])/255.
    Cr = (framedec[:,:,2])/255.

    '''Compute RGB components'''
    R = (0.771996*Y -0.404257*Cb + 1.4025*Cr)
    G = (1.11613*Y - 0.138425*Cb - 0.7144*Cr)
    B = (1.0*Y + 1.7731*Cb)
    '''Display RGB Components'''
    decfile = np.zeros(framedec.shape)
    decfile[:,:,0] = R
    decfile[:,:,1] = G
    decfile[:,:,2] = B
    return decfile

if __name__ == '__main__':
    # x = np.array([[1,2,3,4,5],[4,5,6,7,8],[7,8,9,10,11]])
    # print "x\n", x
    import cv2
    x = cv2.imread('bb8.jpg')
    # y = app420(x, 2)
    # print 'y\n', y
    r, c, d = x.shape
    # print frame.shape
    xy, xb, xr = rgb2ycbcr(x)

    h = np.ones((2,2))
    xy = sp.convolve2d(xy, h, mode='same')
    xb = sp.convolve2d(xb, h, mode='same')
    xr = sp.convolve2d(xr, h, mode='same')

    xbd=xb[::2,::2]
    xrd=xr[::2,::2]
    print (xb[:4])
    print (xbd[:4])
    # print xd[:1]

    xub = np.zeros((xbd.shape[0]*2 - 1,xbd.shape[1]*2))
    xub[::2, ::2] = xbd
    print (xub[:4])
    zb = sp.convolve2d(xub, h, mode='same')
    print (zb[:4])
    xur = np.zeros((xrd.shape[0]*2-1,xrd.shape[1]*2))
    xur[::2, ::2] = xrd
    zr = sp.convolve2d(xur, h, mode='same')

    # xu[::2, ::2] = xr
    # h = np.ones((8,8))/8
    # zr = sp.convolve2d(xu, h, mode='same')


    # cv2.namedWindow('xy', cv2.WINDOW_NORMAL)
    # cv2.imshow('xy', xy/255.)
    cv2.namedWindow('zr', cv2.WINDOW_NORMAL)
    cv2.imshow('zr', np.abs(zr.astype(np.int8)/255.))
    cv2.namedWindow('zb', cv2.WINDOW_NORMAL)
    cv2.imshow('zb', np.abs(zb.astype(np.int8)/255.))
    cv2.namedWindow('xb', cv2.WINDOW_NORMAL)
    cv2.imshow('xb', np.abs(xb.astype(np.int8)/255.))
    print (xy.shape)
    print (zb.shape)
    print (zr.shape)


    yframe = np.zeros((xy.shape[0],xy.shape[1],3))
    yframe[:,:,0] = xy
    yframe[:,:,1] = zb
    yframe[:,:,2] = zr

    B, G, R = ycbcr2rgb(yframe)
    cframe = np.zeros((xy.shape[0],xy.shape[1],3))
    cframe[:,:,0] = B
    cframe[:,:,1] = G
    cframe[:,:,2] = R
    cv2.namedWindow('winname', cv2.WINDOW_NORMAL)
    cv2.imshow('winname', cframe/255.)
    cv2.namedWindow('Orignal', cv2.WINDOW_NORMAL)
    cv2.imshow('Orignal', x)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
