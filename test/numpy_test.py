import numpy as np
img=np.zeros((32,32,3))
img[:,:,0]=1
img[:,:,1]=10
img[:,:,2]=20
mean=np.array([100,200,300])
# print(img)
img=img*mean
print(img[1,1,0])