import PIL.Image
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms
trans=torchvision.transforms.ToPILImage()
mean = np.array([0.4380, 0.4440, 0.4730])
std = np.array([0.1751, 0.1771, 0.1744])
#make sure img looks like 32*32*3
img=np.array(dataset_dict['test']['ood_dsets'][0][0]['x_lb'].permute(1,2,0))*std+mean
plt.imshow(img)
plt.show()
