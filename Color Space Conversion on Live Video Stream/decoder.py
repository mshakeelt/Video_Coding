import numpy as np
import cv2
import pickle as pickle
import time

cap = cv2.VideoCapture(0)		


#Color conversion(inverse)
def ycbcr2rgb(im):
    xform = np.array([[1, 0, 1.4025], [1, -0.34434, -.7144], [1, 1.7731, 0]])
    rgb = im.astype(np.float)
    rgb[:,:,[1,2]] -= 128
    x=rgb.dot(xform.T)
    return x


f1 = open ('video_raw_data.txt', 'rb')
f2 = open ('video_raw_data_int8.txt', 'rb')
f3 = open ('video_raw_data_unint8.txt', 'rb')
#print(reduced.shape)

while(True):
    reduced = pickle.load (f1)
    frame =reduced.copy()
    
    converted = ycbcr2rgb (frame)
    #print (max (frame [:, 0, 0]))
    cv2.imshow ('Y', frame [:, :, 0] / 256)
    cv2.imshow ('back', converted / 256)
    # time.sleep(.02)
    if cv2.waitKey(40) & 0xFF == ord('q'):
        break


cv2.destroyAllWindows()