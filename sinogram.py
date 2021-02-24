import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import PIL
import imageio
import scipy
from scipy import fftpack
from skimage.transform import rotate
from skimage.transform import iradon

sinogram = np.array(Image.open("sinogram.png"))

print(sinogram.shape)

red = sinogram[...,0]
green = sinogram[...,1]
blue = sinogram[...,2]

#recon = np.dstack((red,green,blue))

red_fft = fftpack.rfft(red,axis=1)
green_fft = fftpack.rfft(green,axis=1)
blue_fft = fftpack.rfft(blue,axis=1)

ramp = np.floor(np.arange(0.5, red_fft.shape[1]//2 + 0.1,0.5))
window = np.hamming(red_fft.shape[1])
red_fft = red_fft*ramp*window
green_fft = green_fft*ramp*window
blue_fft = blue_fft*ramp*window

recon_red = fftpack.irfft(red_fft,axis=1)
recon_green = fftpack.irfft(green_fft,axis=1)
recon_blue = fftpack.irfft(blue_fft,axis=1)




def reconstruct(image):
    recon_image = np.zeros((image.shape[1],image.shape[1]))
    steps = 360
    dtheta = -180/steps
    
    for i in range(steps):
        temp = np.tile(image[i,:],(image.shape[1],1))
        temp = rotate(temp, dtheta*i)
        #temp = np.pad(temp,((0,0),(149,149)),'constant')
        recon_image += temp

    return recon_image

zero_image = np.zeros((red.shape[1],red.shape[1]))

r = reconstruct(recon_red)
g = reconstruct(recon_green)
b = reconstruct(recon_blue)

def convert_to_8bit(ch):
    chi, clo = ch.max(),ch.min()
    chnorm = 255*(ch-clo)/(chi-clo)
    ch8bit = np.floor(chnorm).astype('uint8')
    return ch8bit


r = convert_to_8bit(r)
g = convert_to_8bit(g)
b = convert_to_8bit(b)

final_image = np.dstack((r,g,b))


plt.figure()
#plt.imshow(b,cmap=plt.cm.gray)
plt.imshow(final_image)
plt.axis("off")

