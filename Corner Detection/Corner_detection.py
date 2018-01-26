# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 10:22:17 2017

@author: AnaMaria
"""

import cv2
import numpy as np
import os 
global imagen_final
def sliderHandler_1(self):
    if self != 0:
        update()
        
def sliderHandler_2(self):
    update()

def sliderHandler_3(self):
    if self >=40 and self <= 60:
        update()
        

def sliderHandler_4(self):
    if self != 0:
        update()
          
def update():
    global name1
    global name2
    neighborhood= cv2.getTrackbarPos('Neighborhood','CS_512_imagen6')
    threshold = cv2.getTrackbarPos('Threshold','CS_512_imagen6')/100
    image1, corner1, r_max1 = findcorners(name1)
    image2, corner2, r_max2 = findcorners(name2)
    imagen_final = features(name1, name2)
    while corner1:
        x=corner1.pop()
        if x[2]> (threshold * r_max1):
            cv2.rectangle(imagen_final,(x[0]-neighborhood,x[1]+neighborhood),(x[0]+neighborhood,x[1]-neighborhood),(0,0,255),1)
    while corner2:
        x=corner2.pop()
        if x[2]> (threshold * r_max2):
            cv2.rectangle(imagen_final,(x[0]-neighborhood+image1.shape[1],x[1]+neighborhood),(x[0]+neighborhood+image1.shape[1],x[1]-neighborhood),(0,255,0),1)
    while imagen_final.shape[0] > 1200 or imagen_final.shape[1] > 1200:
        imagen_final = cv2.resize(imagen_final,(int(imagen_final.shape[1]/2), int(imagen_final.shape[0]/2)))
    cv2.imshow('CS_512_imagen6',imagen_final)

def grayscale(image):
    imagen = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    return imagen 

def findcorners(name):
    neighborhood= cv2.getTrackbarPos('Neighborhood','CS_512_imagen6')
    if cv2.getTrackbarPos('Gaussian','CS_512_imagen6')> 0:
        gaussian = cv2.getTrackbarPos('Gaussian','CS_512_imagen6')
    else: 
        gaussian = 1 
    kernel = np.ones((gaussian,gaussian),np.float32)/(gaussian*gaussian)
    if cv2.getTrackbarPos('Weight','CS_512_imagen6')>=40 and cv2.getTrackbarPos('Weight','CS_512_imagen6')<=60:
        weight = cv2.getTrackbarPos('Weight','CS_512_imagen6')/1000
    else: 
         weight = 0.04
    imagen_new = cv2.imread(name)
    imagen_new_1 = grayscale(imagen_new)
    imagen_new_1 = cv2.filter2D(imagen_new_1,-1,kernel) 
    imagen_new_2 = imagen_new_1 
    imagenxx = cv2.Sobel(imagen_new_1,cv2.CV_64F,1,0,ksize=5)
    Ixx = cv2.normalize(imagenxx, imagen_new_1, alpha = 0, beta = 1,norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_64F)
    Ixx = Ixx**2
    imagenyy = cv2.Sobel(imagen_new_2,cv2.CV_64F,1,0,ksize=5)
    Iyy = cv2.normalize(imagenyy, imagen_new_2, alpha = 0, beta = 1,norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_64F)
    Iyy = Iyy**2
    Ixy = Iyy*Ixx
    height = Ixx.shape[0]
    width = Ixx.shape[1]
    r_max = 0
    cornerList = []
    print ("Finding Corners...")
    for y in range(neighborhood, height-neighborhood):
        for x in range(neighborhood, width-neighborhood):
            windowIxx = Ixx[y-neighborhood:y+neighborhood+1, x-neighborhood:x+neighborhood+1]
            windowIxy = Ixy[y-neighborhood:y+neighborhood+1, x-neighborhood:x+neighborhood+1]
            windowIyy = Iyy[y-neighborhood:y+neighborhood+1, x-neighborhood:x+neighborhood+1]
            Sxx = windowIxx.sum()
            Sxy = windowIxy.sum()
            Syy = windowIyy.sum()
            M = np.matrix([[Sxx,Sxy],[Sxy,Syy]])
            det = np.linalg.det(M)
            tr = np.trace(M)
            r = det - weight*tr**2
            cornerList.append([x, y, r])
            if r > r_max:
                r_max = r
    return imagen_new, cornerList, r_max

def combinar(image1, image2):
    image_combine = np.concatenate((image1, image2), axis=1)
    return image_combine
       
def featurevectores(imagen):
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(imagen,None)
    return kp1, des1 

def features(name1,name2):
    threshold = ((100-cv2.getTrackbarPos('Threshold','CS_512_imagen6'))/100)+0.3
    img1 = cv2.imread(name1)
    img2 = cv2.imread(name2)
    imagen_final = img1
    gaussian = cv2.getTrackbarPos('Gaussian','CS_512_imagen6')
    kernel = np.ones((gaussian,gaussian),np.float32)/(gaussian*gaussian)
    img1 = cv2.filter2D(img1,-1,kernel) 
    img2 = cv2.filter2D(img2,-1,kernel) 
    img1= grayscale(img1)
    img2= grayscale(img2)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    feature_x1,feature_y1 = featurevectores(img1)
    feature_x2,feature_y2 = featurevectores(img2)
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(feature_y1,feature_y2, k=2)
    good = []
    for m,n in matches:
        if m.distance < threshold*n.distance:
            good.append([m])
    print ("Features Corners...")
    imagen_final = cv2.drawMatchesKnn(img1,feature_x1,img2,feature_x2,good, imagen_final,flags=2)
    return imagen_final
       
        


def proceso_principal(name1, name2):
    imagen1 = cv2.imread(name1)
    imagen2 = cv2.imread(name2)
    imagen1 = grayscale(imagen1)
    imagen2 = grayscale(imagen2)
    imagen_final = combinar(imagen1, imagen2)
    while imagen_final.shape[0] > 1200 or imagen_final.shape[1] > 1400:
        imagen_final = cv2.resize(imagen_final,(int(imagen_final.shape[1]/2), int(imagen_final.shape[0]/2)))
    cv2.imshow('CS_512_imagen6', imagen_final)
    cv2.createTrackbar('Gaussian','CS_512_imagen6', 1, 255, sliderHandler_1)
    cv2.createTrackbar('Neighborhood','CS_512_imagen6', 1, 10, sliderHandler_2)
    cv2.createTrackbar('Weight','CS_512_imagen6', 40, 60, sliderHandler_3)
    cv2.createTrackbar('Threshold','CS_512_imagen6', 50, 100, sliderHandler_4)
    update()
    while True:      
        key=cv2.waitKey()
        if key == ord('h'):
            print('Move the Gaussian trackbar (1-255) to smooth the images and get a different result from the corners and features.\nMove the neighbor trackbar to increase or decrease the large (1-10) which is the window in the Harris algorithm. \nMove the threshold trackbar to increase or decrease the percentage (1-100) of corners and features that you want displayed. \nMoves the trackbar the weight to increase or decrease between (0.04-0.06) the parameter that we pass to the algorithm of Harris. ')
        elif key==27: 
            cv2.destroyAllWindows()
            break  

def main():
    global name1
    name1 ='img1.png'
    global name2
    name2 ='img2.png'
    proceso_principal(name1, name2)
    if KeyboardInterrupt:
        try:
            cv2.destroyAllWindows()
        except SystemExit:
            os._exit(0)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        try:
            cv2.destroyAllWindows()
        except SystemExit:
            os._exit(0)
