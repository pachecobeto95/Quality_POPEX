import torch
import torch.nn as nn
import numpy as np
import sys, time, math
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision
from torchvision import datasets, transforms
from torchvision.utils import save_image
import os 
import cv2
from PIL import Image, ImageEnhance
from torchvision import transforms, utils, datasets
from torch.utils.data import Dataset, DataLoader, random_split, SubsetRandomSampler
from load_datasets import LoadDataset

class QualityConverter():
  def __init__(self, dataLoader, dataset_name, savePath):
    """
    Converts a dataset into distorted dataset with different distortion types and levels.

    Arguments are

    * dataLoader:                         contains the dataset and the classes of each image
    * dataset_name:                       the dataset name
    * savePath                            path to save the distorted images
    """

    self.dataLoader = dataLoader
    self.dataset_name = dataset_name
    self.savePath = savePath

  def __applyGaussianBlur(self, img, blur_std):
    """
    this method adds gaussian blur into the images
    img (tensor):      image in tensor format, whose shape is (batch_size, channel, width, height)
    blur_std (int):    stadard deviation used to add blur into the image
    return:
    img_pil (PIL Image):    image in PIL object
    """
    img_pil = transforms.ToPILImage()(img[0]) # img[0] removes batch_size-related component and transforms in PIL Image
    blur = cv2.GaussianBlur(np.array(img_pil), (4*blur_std+1, 4*blur_std+1),blur_std, None, blur_std, cv2.BORDER_CONSTANT)
    img_pil = Image.fromarray(np.uint8(blur),  'RGB')
    return img_pil    
    #return cv2.GaussianBlur(img,(4*blur_std+1, 4*blur_std+1),blur_std, None, blur_std, cv2.BORDER_CONSTANT)

  def __applyGaussianNoise(self, img, blur_noise):
    # this method adds Additive White Gaussian Noise into the images

    img_np = img.numpy()[0]
    distorted_img = img_np + np.random.normal(0, blur_noise, (img_np.shape[0], img_np.shape[1], img_np.shape[2]))
    img_pil = Image.fromarray(np.uint8(distorted_img), "RGB")
    return img_pil

  def __applyBrightness(self, img, brightness_lvl):
    # this method changes the brightness of the image. In summary, this increases the pixels values of the image. 
    img = Image.fromarray(np.uint8(img*255))
    enhancer = ImageEnhance.Brightness(img)
    distorted_img = enhancer.enhance(brightness_lvl)
    return np.array(distorted_img)


  def generate_random_lines(self, imshape, slant, drop_length):
    drops=[]    
    for i in range(1500): ## If You want heavy rain, try increasing this        
      if slant<0:            
        x= np.random.randint(slant, imshape[-1])        
      else:            
        x= np.random.randint(0, imshape[-1]-slant)        

      y= np.random.randint(0, imshape[-1]-drop_length)        
      drops.append((x,y))    
    return drops            

  def __applyFakeRain(self, img, rain_lvl):
    imshape = img.shape
    slant_extreme = 10
    slant = np.random.randint(-slant_extreme, slant_extreme)
    drop_length = 20    
    drop_width = 4 
    drop_color = (200,200,200) ## a shade of gray    
      
    rain_drops = self.generate_random_lines(imshape, slant, drop_length)        
    img = np.swapaxes(img, -1, 0)
    for rain_drop in rain_drops:
      
      cv2.line(img, (rain_drop[0], rain_drop[1]),(rain_drop[0]+slant,rain_drop[1]+drop_length),drop_color,drop_width)    
      img = cv2.blur(img,(7,7)) ## rainy view are blurry        
      brightness_coefficient = 0.7 ## rainy days are usually shady     
      image_HLS = cv2.cvtColor(img, cv2.COLOR_RGB2HLS) ## Conversion to HLS    
      image_HLS[:,:,1] = image_HLS[:,:,1]*brightness_coefficient ## scale pixel values down for channel 1(Lightness)    
      image_RGB = cv2.cvtColor(image_HLS, cv2.COLOR_HLS2RGB) ## Conversion to RGB    
    
    return np.swapaxes(image_RGB, -1, 0)

  def __applyRainGlass(self, img, rain_lvl):
    img = np.swapaxes(img, -1, 0)
    img_PIL = Image.fromarray(np.uint8(img*255))
    img2 = Image.open("./rainglass.png").convert("RGBA")
    img_PIL.paste(img2, (0, 0), img2)
    return img_PIL

  def gausianBlur(self, distortion_lvl):
    self.distortion_type = "gaussian_blur"
    self.distortion_lvl = distortion_lvl
    distortion = self.__applyGaussianBlur
    self.__generate_distorted_dataset(distortion)

  def whiteGaussianNoise(self, distortion_lvl):
    self.distortion_type = "awgn"
    self.distortion_lvl = distortion_lvl
    distortion = self.__applyGaussianNoise
    self.__generate_distorted_dataset(distortion)

  def brightness(self, distortion_lvl):
    '''
    Brightness: This class can be used to control the brightness of an image. An enhancement
    factor of 0.0 gives a black image. A factor of 1.0 gives the original image.
    
    Argument is

    * distortion_lvl:     list or int containing the brighness level                                                             
    '''
    self.distortion_type = "brightness"
    self.distortion_lvl = distortion_lvl
    distortion = self.__applyBrightness
    self.__generate_distorted_dataset(distortion)

  def rain(self, distortion_lvl):
    '''
    This method adds a fake rain effect in the images

    Argument is

    * distortion_lvl:     list or int containing the rain level                                                             
    '''
    self.distortion_type = "rain"
    self.distortion_lvl = distortion_lvl
    distortion = self.__applyFakeRain
    self.__generate_distorted_dataset(distortion)

  def rainGlass(self):
    '''
    This method adds a fake rain effect in the images in the camera glass.

    Argument is

    * distortion_lvl:     list or int containing the rain level                                                             
    '''
    self.distortion_type = "rain_glass"
    self.distortion_lvl = distortion_lvl
    distortion = self.__applyRainGlass
    self.__generate_distorted_dataset(distortion)


  def __generate_distorted_dataset(self, distortion, log_interval=100):
    """
    This method 
    Arguments are

    * distortion:         a function that adds distortion in the images 
    """
    # if the distortion_level parameter is a int, this converts this integer to a list.
    if (isinstance(self.distortion_lvl, int)):
      self.distortion_lvl = list([self.distortion_lvl])

    # this converts the dataset for different levels of a distortion type. 
    for dist_lvl in self.distortion_lvl:
      print("Blur Level: %s"%(dist_lvl))
      print("Converting . . .")
      for i, (img, label) in enumerate(self.dataLoader):
        print(i)
        finalSavePath = os.path.join(self.savePath, self.dataset_name, self.distortion_type, str(dist_lvl), str(label.item()))
        if (not os.path.exists(finalSavePath)):
          os.makedirs(finalSavePath)
        distorted_img = distortion(img, dist_lvl)
        distorted_img.save(os.path.join(finalSavePath, "%s.jpg"%(i)))
        if (i%log_interval == 0):
          print("Saving image: %s"%(i))


input_img = 224
batch_size_train = 64
batch_size_test = 1
dataset_name = "imageNet"
savePath = "./distorted_dataset/"
imagenet_root_path = os.path.join(".", "undistorted_dataset", "ImageNet", "val")
dataset = LoadDataset(input_img, batch_size_train, batch_size_test)
_, testLoader = dataset.imageNet(imagenet_root_path)
blur_lvl_list = [1, 2, 3, 4, 5, 6]
awgn_lvl_list = [10, 20, 40, 60, 80, 100] 
#rain_lvl_list = [1000, 1200, 1400, 1600]
quality_converter = QualityConverter(testLoader, dataset_name, savePath)
quality_converter.gausianBlur(blur_lvl_list)
quality_converter.whiteGaussianNoise(awgn_lvl_list)

