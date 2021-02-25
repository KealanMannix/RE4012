import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy import fftpack
from skimage.transform import rotate

#Load in sinogram
sinogram = np.array(Image.open("sinogram.png"))

#split sinogram into 3 color channel
red = sinogram[...,0]
green = sinogram[...,1]
blue = sinogram[...,2]


#Find the width of the image to be reconstructed by counting leading zeros in 
#sinogram image 
#assumes aspect ratio is 1:1
def get_im_size(sinogram):
    length = sinogram.shape[1]
    zeros = True
    i=0
    num_zeros = 0
    while (zeros == True):
        if (sinogram[1,i] == 0):
            num_zeros += 1
        else:
            zeros = False
        i+=1
    length = length - (num_zeros*2)
    return length

#Get fft of an image
def get_fft(image):
    image = fftpack.rfft(image,axis=1)
    return image

#Mulyiply a frequency domain image by ramp filter
def ramp_filter(fft):
    ramp = np.floor(np.arange(0.5, fft.shape[1]//2 + 0.1,0.5))
    return ramp*fft

#Mulyiply a frequency domain image by hamming window
def window(fft):
    window = np.hamming(fft.shape[1])
    return window*fft

#Convert images to 8 bit values (between 0 and 255)
def convert_to_8bit(ch):
    chi, clo = ch.max(),ch.min()
    chnorm = 255*(ch-clo)/(chi-clo)
    ch8bit = np.floor(chnorm).astype('uint8')
    return ch8bit

#Reconstruct the image by backprojects (length is size of image to be reconstructed)
def reconstruct(fft,length):
    image = fftpack.irfft(fft,axis=1)
    
    recon_image = np.zeros((image.shape[1],image.shape[1]))
    steps = 360
    dtheta = 180/steps
    
    for i in range(steps):
        temp = np.tile(image[i,:],(image.shape[1],1))
        temp = rotate(temp, dtheta*i)
        recon_image += temp
        
    padding_width = (image.shape[1]-length)/2
    left = int(padding_width)
    right = int(padding_width+length)
    
    crop_image = recon_image[left:right, left:right]
    
    crop_image = convert_to_8bit(crop_image)

    return crop_image


#Build RGB image from 3 frequency domain channels
def build_image(r,g,b,length):
    red_ch = reconstruct(r, length)
    green_ch = reconstruct(g, length)
    blue_ch = reconstruct(b, length)
    
    image = np.dstack((red_ch,green_ch,blue_ch))
    return image

length = get_im_size(red)

##Reconstruct with no filtering
no_filter = build_image(red,green,blue,length)

##Reconstruct with Ramp filter
red_ramp = ramp_filter(get_fft(red))
green_ramp = ramp_filter(get_fft(green))
blue_ramp = ramp_filter(get_fft(blue))

ramp_filter = build_image(red_ramp,green_ramp,blue_ramp,length)

##Reconstruct with ramp filter and window
red_win = window(red_ramp)
green_win = window(green_ramp)
blue_win = window(blue_ramp)

window_ramp = build_image(red_win,green_win,blue_win,length)



plt.figure(1)
plt.imshow(no_filter,vmin=0,vmax=255)
plt.axis("off")
plt.title('Image Reconstructed Without Ramp Filtering or Windowing')
plt.show()

plt.figure(2)
plt.imshow(ramp_filter,vmin=0,vmax=255)
plt.axis("off")
plt.title('Image Reconstructed With Ramp Filtering and No Windowing')
plt.show()

plt.figure(3)
plt.imshow(window_ramp,vmin=0,vmax=255)
plt.axis("off")
plt.title('Image Reconstructed With Ramp Filtering and Hamming Windowing')
plt.show()
