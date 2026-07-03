
import cv2, numpy as np

def blur(img, sigma):
    k=max(3,int(6*sigma+1)|1)
    return cv2.GaussianBlur(img,(k,k),sigma)

def motion_blur(img,length,angle):
    k=np.zeros((length,length),np.float32)
    k[length//2,:]=1
    M=cv2.getRotationMatrix2D((length/2,length/2),angle,1)
    k=cv2.warpAffine(k,M,(length,length))
    k/=k.sum()
    return cv2.filter2D(img,-1,k)

def gaussian_noise(img,std):
    n=np.random.normal(0,std,img.shape)
    x=np.clip(img.astype(np.float32)+n,0,255)
    return x.astype(np.uint8)

def exposure(img,factor):
    x=np.clip(img.astype(np.float32)*factor,0,255)
    return x.astype(np.uint8)

def contrast(img,factor):
    x=img.astype(np.float32)
    m=x.mean(axis=(0,1),keepdims=True)
    y=np.clip((x-m)*factor+m,0,255)
    return y.astype(np.uint8)

def haze(img,beta):
    A=255.
    t=np.exp(-beta)
    y=img.astype(np.float32)*t+A*(1-t)
    return np.clip(y,0,255).astype(np.uint8)

def colorfulness(img,factor):
    hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:,:,1]=np.clip(hsv[:,:,1]*factor,0,255)
    return cv2.cvtColor(hsv.astype(np.uint8),cv2.COLOR_HSV2BGR)
