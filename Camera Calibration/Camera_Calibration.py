# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 22:34:19 2017

@author: AnaMaria
"""
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob


        
    
def main():
   
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    

    objp = np.zeros((6*7,3), np.float32)
    objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)
    

    objpoints = [] 
    imgpoints = []
    images = glob.glob('*.jpg')
    for name in images: 
        imagen_final = cv2.imread(name)
        while imagen_final.shape[0] > 1200 or imagen_final.shape[1] > 1400:
            imagen_final = cv2.resize(imagen_final,(int(imagen_final.shape[1]/2), int(imagen_final.shape[0]/2))) 
        cv2.imshow('img',imagen_final)
        gray = cv2.cvtColor(imagen_final,cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (7,6),None)
        fo = open("corners.txt", "w")
        if ret == True:
            objpoints.append(objp)
        
            corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
            imagen_final = cv2.drawChessboardCorners(imagen_final, (7,6), corners2,ret)
            
            imgpoints.append(corners2)
            corners2 = corners2.tolist()
#    print(objpoints)
    imgpointsx = []
    imgpointsy = []
    names = []
    for i in range(0, len(imgpoints[0])):
        imgpointsx.append(imgpoints[0][i][0][0])
        imgpointsy.append(imgpoints[0][i][0][1])
        names.append(i)
    
    cv2.imshow('img',imagen_final)
    for item in corners2:
        fo.write("%s\n" %item)
    fo.close()
    fig, ax = plt.subplots()
    ax.scatter(imgpointsx, imgpointsy)
    
    for i, txt in enumerate(names):
        ax.annotate(int(txt), (imgpointsx[i],imgpointsy[i]))

    while(1):
        k = cv2.waitKey(1) & 0xFF
        if k == ord('s'):
            cv2.destroyAllWindows()
            break

if __name__ == "__main__":
    main()

