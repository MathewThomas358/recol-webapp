import cv2
import matplotlib.pyplot as plt
import pywt
import glob
import numpy as np
import pickle
import urllib
import os

from skimage import color
from keras.preprocessing import image

def recol_fn(input_image):

    def fusion(image1, image2):
        def fuseCoeff(cooef1, cooef2, method):

            cooef = (cooef1 + cooef2) / 2
            return cooef

        FUSION_METHOD = 'mean'


        image1 = color.rgb2grey(image1)
        image2 = color.rgb2grey(image2)


        I1 = image1
        I2 = image2
        I2 = cv2.resize(I2,I1.shape)

        wavelet = 'db1'
        cooef1 = pywt.wavedec2(I1[:,:], wavelet)
        cooef2 = pywt.wavedec2(I2[:,:], wavelet)

        fusedCooef = []
        for i in range(len(cooef1)-1):

            if(i == 0):
                fusedCooef.append(fuseCoeff(cooef1[0],cooef2[0],FUSION_METHOD))
            else:
                c1 = fuseCoeff(cooef1[i][0], cooef2[i][0], FUSION_METHOD)
                c2 = fuseCoeff(cooef1[i][1], cooef2[i][1], FUSION_METHOD)
                c3 = fuseCoeff(cooef1[i][2], cooef2[i][2], FUSION_METHOD)

                fusedCooef.append((c1,c2,c3))

        fusedImage = pywt.waverec2(fusedCooef, wavelet)
        fusedImage = np.multiply(np.divide(fusedImage - np.min(fusedImage),(np.max(fusedImage) - np.min(fusedImage))),255)
        fusedImage = fusedImage.astype(np.uint8)

        fusedImage = color.grey2rgb(fusedImage)

        return fusedImage

    def illum(img):
        sumImg = [0,0,0]
        for i in range(len(img)):
            for j in range(len(img[i])):
                sumImg[0]+=img[i][j][0]
                sumImg[1]+=img[i][j][1]
                sumImg[2]+=img[i][j][2]


        illum=[sumImg[0]/((len(img))*(len(img[0]))),sumImg[1]/((len(img))*(len(img[0]))),sumImg[2]/((len(img))*(len(img[0])))]
        b,g,r = cv2.split(img)
        scale=(illum[0]+illum[1]+illum[2])/3

        r=r*scale/illum[2];
        g=g*scale/illum[1];
        b=b*scale/illum[0];

        rgb = cv2.merge([b,g,r])
        return rgb

    def make_square(img):
        try:

            height, width, channels = img.shape

            x = height if height > width else width
            y = height if height > width else width
            square= np.zeros((x,y,3), np.uint8)
            square[(y-height)//2:y-(y-height)//2, (x-width)//2:x-(x-width)//2] = img

            return square

        except:
            try:
                height, width, channels = img.shape
                x = height if height > width else width
                y = height if height > width else width
                square= np.zeros((x,y,3), np.uint8)
                a = (y-height)//2
                b = y-(y-height)//2
                c = (x-width)//2
                d = x-(x-width)//2

                square[a:b-1, c:d] = img

                return square

            except:
                try:
                    height, width, channels = img.shape
                    x = height if height > width else width
                    y = height if height > width else width
                    square= np.zeros((x,y,3), np.uint8)
                    a = (y-height)//2
                    b = y-(y-height)//2
                    c = (x-width)//2
                    d = x-(x-width)//2

                    square[a:b-1, c:d-1] = img

                    return square
                except:
                    height, width, channels = img.shape
                    x = height if height > width else width
                    y = height if height > width else width
                    square= np.zeros((x,y,3), np.uint8)
                    a = (y-height)//2
                    b = y-(y-height)//2
                    c = (x-width)//2
                    d = x-(x-width)//2

                    square[a:b, c:d-1] = img

                    return square

    def surfn(fus):
        surf = cv2.xfeatures2d.SURF_create(1500)
        keypoints, descriptors = surf.detectAndCompute(fus, None)
        #fus_surf = cv2.drawKeypoints(fus, keypoints, None)
        keypoints = len(keypoints)
        return keypoints

    def main_fn(imag):
        img = make_square(imag)
        b,g,r = cv2.split(img)
        bgr = cv2.merge((b,g,r))
        blur = cv2.GaussianBlur(img,(5,5),0)
        im = illum(img)
        fus1 = fusion(bgr, blur)
        fus = fusion(fus1,im)
        kp = surfn(fus)

        return kp

    kp = main_fn(input_image)

    return kp
