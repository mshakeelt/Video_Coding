import numpy as np
import cv2
import pickle as pickle
import time

# Color conversion
def rgb2ycbcr(im):
    xform = np.array([[.299, .587, .114], [-.16864, -.33107, .49970], [.499813, -.418531, -.081282]])
    ycbcr = im.dot(xform.T)
    ycbcr[:,:,[1,2]] += 128
    return np.uint8(ycbcr)


def ycbcr2rgb(im):
    xform = np.array([[1, 0, 1.4025], [1, -0.34434, -.7144], [1, 1.7731, 0]])
    rgb = im.astype(np.float)
    rgb[:,:,[1,2]] -= 128
    x=rgb.dot(xform.T)
    return x

cap = cv2.VideoCapture(0)

f1=open('video_raw_data.txt', 'wb')
f2=open('video_raw_data_int8.txt', 'wb')
f3=open('video_raw_data_unint8.txt', 'wb')


count = 0
count+=1
if count%3==0:
    

while(True):
    
    ret, frame = cap.read()
    frame = cv2.resize(frame, (0, 0), None, .8, .8)
    #print(ret)
    converted = rgb2ycbcr (frame)
    converted = cv2.resize(converted, (0, 0), None, .65, .65)
    cv2.imshow ('normal Video', frame / 512)
    cv2.imshow ('Y component', converted [:, :, 0] / 256)
    cv2.imshow ('Cb component', converted [:, :, 1] / 256)
    cv2.imshow ('Cr component', converted [:, :, 2] / 256)
    test = ycbcr2rgb (converted)
    test = cv2.resize(test, (0, 0), None, 1.53846, 1.53846)
    cv2.imshow ('back Video', test / 256)
    
    #print (converted)
    pickle.dump (converted, f1)
    #print('Data Type: %s' % converted.dtype)

    converted2 = np.array (converted[:,:,[1,2]], dtype='int8')
    #print('Data Type: %s' % converted2.dtype)
    pickle.dump (converted2, f2)

    converted3 = np.array (converted[:, :, 0], dtype='uint8')
    #print('Data Type: %s' % converted3.dtype)

    pickle.dump (converted3, f3)
    count+=1
    if (cv2.waitKey (20) & 0xFF == ord ('q')):
        break
cap.release()
cv2.destroyAllWindows ()
