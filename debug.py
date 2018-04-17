import cv2
import numpy as np
import torchvision.utils as vutils
import torch
img = cv2.imread('000012.jpg')
print(img.shape)
# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# img = np.expand_dims(img, axis = 2)
img = np.mean(img, axis = 2)

imres = 6
sigma = 30
img = cv2.resize(img, (96 + 16*imres, 128+16*imres), interpolation =
        cv2.INTER_CUBIC)
img1 = cv2.resize(img,None,fx=1.0/4.0, fy=1.0/4.0, interpolation = cv2.INTER_CUBIC)
noise= sigma*np.random.randn(img1.shape[0],img1.shape[1])
img1 = img1+noise                
img1 = cv2.resize(img1,None,fx=1.0*4.0, fy=1.0*4.0, interpolation = cv2.INTER_NEAREST)                                  #up_sample method

img1 = np.expand_dims(img1,axis=2)
img = np.expand_dims(img, axis=2)
cv2.imwrite('test_change_1.png', img)
cv2.imwrite('test_change_1_noise.png',img1)
img = np.transpose(img,(2,0,1))
img1 = np.transpose(img1,(2,0,1))
#img = np.expand_dims(img, axis = 0)
#out of img_random_...
img = (img/255-0.5)*2
img1 = (img1/255-0.5)*2
img = torch.from_numpy(img)
img1 = torch.from_numpy(img1)
img = img.float()
img1 = img1.float()
vutils.save_image(img1,'test_change_2_noise.png')
vutils.save_image(img,'test_change_2.png')



'''another part'''
import os
from data_loader import Dataset
b_data_set = Dataset("data", [218,178,3],"test",False)
b_data_loader = torch.utils.data.DataLoader(dataset=b_data_set,
        batch_size=1,
        shuffle=False,
        num_workers=2)
b_data_loader.shape = b_data_set.shape
B_loader = iter(b_data_loader)
valid_x_B = B_loader.next()
vutils.save_image(valid_x_B*0.5+0.5,"test_change_3.png")
