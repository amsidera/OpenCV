# -*- coding: utf-8 -*-
import cv2 
import numpy as np 
import sys 
import scipy.stats as st
import os 
import math 

def readimage(name):
    if len(sys.argv)<2:
        img = cv2.imread(name)
    return img


def capturing():
    cap = cv2.VideoCapture(0)
    retval,image=cap.read()
    return image


def reload(nb):
    if nb!=0 and os.path.isfile(nb):
        imagen = readimage(nb)
    elif len(nb) > 1 and os.path.isfile(nb)==False:
        print ('There is no image with that name. Try again.')
        imagen=0
        main()
    elif len(nb) < 1:
        imagen=capturing()
    cv2.imshow('CS_512',imagen)
    return imagen


def writing(imagen):
    cv2.imwrite("out.jpg",imagen)


def converting(image):
    imagen_new = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imshow('CS_512',imagen_new)
    return imagen_new 


def convertinggray(image):
    imagen_new = np.zeros((image.shape[0], image.shape[1]), dtype = image.dtype)
    for (x, y), v in np.ndenumerate(imagen_new):
        value = image[x, y, 0] * 0.299 + image[x, y, 1] * 0.587 + image[x ,y, 2] * 0.114
        imagen_new[x, y] = value
    cv2.imshow('CS_512',imagen_new)
    return imagen_new    


def cycle(nb,counter):
    imagen_new = reload(nb)
    if (counter == 0):
        imagen_new[:,:,1] = 0
        imagen_new[:,:,2] = 0
        counter += 1
    elif (counter == 1):
        imagen_new[:,:,0] = 0
        imagen_new[:,:,2] = 0
        counter += 1
    elif (counter == 2):
        imagen_new[:,:,0] = 0
        imagen_new[:,:,1] = 0
        counter = 0 
    cv2.imshow('CS_512', imagen_new)
    return imagen_new, counter


def downsample():
    imagen_new = cv2.pyrDown(reload(nb))
    cv2.imshow('CS_512', imagen_new)
    return imagen_new


def resize(): 
    imagen_new = cv2.resize(reload(nb),None,fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC)
    cv2.imshow('CS_512', imagen_new)   
    return imagen_new

def sliderHandler(self):
    global imagen_new
    if self != 0:
        n=self
        imagen = cv2.cvtColor(reload(nb), cv2.COLOR_BGR2GRAY)
        kernel = np.ones((n,n),np.float32)/(n*n)
        imagen_new = cv2.filter2D(imagen,-1,kernel)
    cv2.imshow('CS_512_1',imagen_new)
    

def grayandsmooth():
    imagen_new = cv2.cvtColor(reload(nb), cv2.COLOR_BGR2GRAY)
    cv2.imshow('CS_512_1', imagen_new)
    cv2.createTrackbar('S','CS_512_1', 0, 255, sliderHandler)   

def sliderHandler_2(self):
    global imagen_new
    if self != 0:
        sigma = 7 
        size = self 
        result = cv2.cvtColor(reload(nb), cv2.COLOR_BGR2GRAY)
        interval = (2*sigma+1.)/(size)
        x = np.linspace(-sigma-interval/2., sigma+interval/2., size+1)
        kern1d = np.diff(st.norm.cdf(x))
        kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
        kernel = kernel_raw/kernel_raw.sum()
        result = cv2.filter2D(result, -1, kernel)
        imagen_new = result
    cv2.imshow('CS_512_1',imagen_new)
  
def gaussianfilter():
    imagen_new =cv2.cvtColor(reload(nb), cv2.COLOR_BGR2GRAY)
    cv2.imshow('CS_512_1', imagen_new)
    cv2.createTrackbar('S','CS_512_1', 0, 255, sliderHandler_2)  
    result = sliderHandler_2 
    return result

def gradientx():
    imagen_new = cv2.cvtColor(reload(nb), cv2.COLOR_BGR2GRAY)
    imagen_new = cv2.Sobel(imagen_new,cv2.CV_64F, 1, 0,ksize = 7)
    imagen_new = cv2.normalize(imagen_new,imagen_new,alpha = 0, beta = 1,norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_64F)
    cv2.imshow('CS_512', imagen_new)
    return imagen_new


def gradienty():
    imagen = cv2.cvtColor(reload(nb), cv2.COLOR_BGR2GRAY)
    imagen_new = cv2.Sobel(imagen,cv2.CV_64F, 0, 1,ksize = 7)
    imagen_new = cv2.normalize(imagen_new, imagen_new, alpha = 0, beta = 1,norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_64F)
    cv2.imshow('CS_512', imagen_new)
    return imagen_new


def magnitud(): 
    sobelX = cv2.Sobel(reload(nb),cv2.CV_64F, 1, 0,ksize = 7)
    sobelY = cv2.Sobel(reload(nb),cv2.CV_64F, 0, 1,ksize = 7)
    imagen = cv2.magnitude(sobelX, sobelY)
    imagen = cv2.normalize(imagen,imagen, alpha = 0, beta = 1,norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_64F)
    cv2.imshow('CS_512', imagen)
    return imagen


def sliderHandler_3(self):
    global imagen_new
    n=self
    imagen = cv2.cvtColor(reload(nb), cv2.COLOR_BGR2GRAY)
    rot = cv2.getRotationMatrix2D((imagen.shape[1] / 2, imagen.shape[0] / 2), n, 1)
    imagen = cv2.warpAffine(imagen, rot,(imagen.shape[1], imagen.shape[0]))
    imagen_new = imagen
    cv2.imshow('CS_512_1', imagen)
    

def rotate():
    cv2.imshow('CS_512_1', reload(nb))    
    cv2.createTrackbar('S','CS_512_1', 0, 360, sliderHandler_3)  


def sliderHandler_4(self):
    global imagen_new
    img = cv2.cvtColor(reload(nb), cv2.COLOR_BGR2GRAY)
    if self != 0:
        n=self
        sobelX = cv2.Sobel(img,cv2.CV_64F, 0, 1,ksize = 7)
        sobelY = cv2.Sobel(img,cv2.CV_64F, 1, 0, ksize = 7)
        for x in range(0, img.shape[0], n):
            for y in range(0, img.shape[1], n):
                gradientAngle = math.atan2(sobelY[x, y], sobelX[x, y])
                dstX = int(x + n * math.cos(gradientAngle))
                dstY = int(y + n * math.sin(gradientAngle))
                cv2.arrowedLine(img, (y, x), (dstY, dstX), (0, 0, 0))
    imagen_new = img
    cv2.imshow('CS_512_1', img)


def vectorfield():
    img = cv2.cvtColor(reload(nb), cv2.COLOR_BGR2GRAY)
    cv2.imshow('CS_512_1', img)
    cv2.createTrackbar('S','CS_512_1', 0, 255, sliderHandler_4)

def main():
    global nb
    nb = input('Choose a file or press enter if want a camera image: ')
    counter = 0 
    global imagen_new
    imagen_new=reload(nb)

    while True:      
        key=cv2.waitKey()
        if key == ord('i'):
            imagen_new = reload(nb)
        elif key == ord('w'):
            writing(imagen_new)
        elif key == ord('g'):
            imagen_new = converting(reload(nb))
        elif key == ord('G'):
            imagen_new = convertinggray(reload(nb))
        elif key == ord('c'):
            imagen_new, counter = cycle(nb,counter)
        elif key == ord('s'): 
            grayandsmooth()
        elif key == ord('S'):
            imagen_new = gaussianfilter()
        elif key == ord('D'):
            imagen_new = downsample()
        elif key == ord('d'):
            imagen_new = resize()
        elif key == ord('x'):
            imagen_new = gradientx()
        elif key == ord('y'):
            imagen_new = gradienty()
        elif key == ord('m'):
            imagen_new = magnitud()
        elif key == ord('p'):
            vectorfield()
        elif key == ord('r'):
            rotate()
        elif key == ord('h'):
            print("i - reload the image\nw - save the image in out.jpg\ng - grayscale the image\nG - grayscale the image with my filter\nc - circle the image between 3 channels\ns -grayscale the image and smooth it\nS - grayscale the image and smooth it with my filter\nd - downsample the image without smooth\nD - downsample the image with smoot\nx - x derivative\ny - y derivative\nm - gradient magnitude\np - vector field\nr - rotate the image\nh - help\n")
        elif key==27: 
            cv2.destroyAllWindows()
            break  
    
if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        try:
            cv2.destroyAllWindows()
        except SystemExit:
            os._exit(0)