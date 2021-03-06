_author__ = 'linjx'
import cv2
from PIL import Image
import numpy as np
#from motionblur import motionblur
#img=Image.open('ori10.bmp')
#img=np.array(img)
#img = cv2.imread('ori10.bmp')
def img_random_dis(input_batch, ind = None):
    #if ind is 6:
    #imres = np.random.randint(0,6)
    if ind is None:         
        imres = np.random.randint(0,6)
    else:
        imres = ind
    img_dis=np.zeros((input_batch.shape[0],128+16*imres,96+16*imres,1))
    img_ref=np.zeros((input_batch.shape[0],128+16*imres,96+16*imres,1))
    #downsrate=np.random.randint(0,3)
    #mobkernel= motionblur(4*np.random.randint(0,4)+1,45*np.random.randint(0,4))
    #mobkernel= motionblur(8,45)
    #averagekernel=np.ones(16).reshape(4,4)/16
    #sigma=10*np.random.randint(0,4)
    sigma=30
    for i in range(input_batch.shape[0]):
        img_rgb  = input_batch[i]
        img_rgb = 255*(img_rgb*0.5+0.5) 
        #img_size = img_rgb.shape
        #print img_rgb
        img_rgb =  np.transpose(img_rgb,(1,2,0))
        #print img_rgb.shape
        #img=np.zeros_like(img_rgb)
        #img[:,:,0]=img_rgb[:,:,2]
        #img[:,:,1]=img_rgb[:,:,1]
        #img[:,:,2]=img_rgb[:,:,0]
        #img.astype(float)
        img = np.mean(img_rgb,axis = 2)
        img  = cv2.resize(img,(96+16*imres,128+16*imres),interpolation = cv2.INTER_CUBIC)                                     #multi_scale
        #img1 = cv2.filter2D(img,-1,averagekernel)                                                                             #AVERAGEblur
        img1 = cv2.resize(img,None,fx=1.0/4.0, fy=1.0/4.0, interpolation = cv2.INTER_CUBIC)
        noise= sigma*np.random.randn(img1.shape[0],img1.shape[1])
        img1 = img1+noise                
        img1 = cv2.resize(img1,None,fx=1.0*4.0, fy=1.0*4.0, interpolation = cv2.INTER_NEAREST)                                  #up_sample method

        #img5 = cv2.resize(img2,None,fx=1.0*2**(downsrate), fy=1.0*2**(downsrate), interpolation = cv2.INTER_NEAREST)
    #img6 = cv2.resize(img1,None,fx=1.0*2**(downsrate), fy=1.0*2**(downsrate), interpolation = cv2.INTER_NEAREST)
        img1 = np.expand_dims(img1,axis=2)
    #img1=np.clip(img1,0,255)
        #img1[img1<=0]=0
        #img1[img>=255]=255
        img  = np.expand_dims(img,axis=2)
        img_dis[i]=img1
        img_ref[i]=img
        #image2 = Image.fromarray(img)
        #image2.show()
        #cv2.imshow('Original', img3)
        #cv2.waitKey(0)
    
    img_dis = np.transpose(img_dis,(0,3,1,2))
    img_ref = np.transpose(img_ref,(0,3,1,2))
    #print img_ref.shape
    return (img_dis/255-0.5)*2,(img_ref/255-0.5)*2
