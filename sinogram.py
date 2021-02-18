import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import imageio
import scipy
from scipy import fftpack
from skimage.transform import rotate
import radontea
from skimage.transform import iradon

sinogram = np.array(Image.open("sinogram.png"))

red = sinogram[...,0]
green = sinogram[...,1]
blue = sinogram[...,2]

recon = np.dstack((red,green,blue))

red_fft = fftpack.rfft(red,axis=1)
green_fft = fftpack.rfft(green,axis=1)
blue_fft = fftpack.rfft(blue,axis=1)

recon_red = fftpack.irfft(red_fft,axis=1)
recon_green = fftpack.irfft(green_fft,axis=1)
recon_blue = fftpack.irfft(blue_fft,axis=1)


recon_image = np.zeros((recon_red.shape[0],recon_red.shape[1]))
theta = 180.0/recon_red.shape[0]
for i in range(recon_red.shape[0]):
    var = np.tile(recon_red[i],(recon_red.shape[0],1))
    var = rotate(var,theta*i)
    recon_image += var
    
recon_image2 = np.zeros((recon_red.shape[0],recon_red.shape[1]))
theta2 = 180.0/recon_red.shape[0]
for i2 in range(recon_red.shape[0]):
    var2 = np.tile(recon_red[i],(recon_red.shape[0],1))
    var2 = rotate(var2,theta2*i2)
    recon_image2 += var2
    
recon_image3 = np.zeros((recon_red.shape[0],recon_red.shape[1]))
theta3 = 180.0/recon_red.shape[0]
for i3 in range(recon_red.shape[0]):
    var3 = np.tile(recon_red[i],(recon_red.shape[0],1))
    var3 = rotate(var3,theta3*i3)
    recon_image3 += var3

    
econ = np.dstack((recon_image,recon_image2,recon_image3))

#final_im = iradon(recon)
plt.figure()
plt.imshow(recon_image)

