# -*- coding: utf-8 -*-
import pylab as py
import numpy as np
from scipy import linalg
import cv2


ix,iy = -1,-1
lista1 = []
lista2 = []
pts1 = []
pts2 = []
point1 = []
point2 = []
F = []
    
def compute_fundamental(x1,x2):
   
    n = x1.shape[1]
    if x2.shape[1] != n:
        raise ValueError("Number of points don't match.")
    A = np.zeros((n,9))
    for i in range(n):
        A[i] = [x1[0,i]*x2[0,i], x1[0,i]*x2[1,i], x1[0,i]*x2[2,i],
                x1[1,i]*x2[0,i], x1[1,i]*x2[1,i], x1[1,i]*x2[2,i],
                x1[2,i]*x2[0,i], x1[2,i]*x2[1,i], x1[2,i]*x2[2,i] ]
    U,S,V = np.linalg.svd(A)
    F = V[-1].reshape(3,3)
    U,S,V = np.linalg.svd(F)
    S[2] = 0
    F = np.dot(U,np.dot(np.diag(S),V))
    
    return F/F[2,2]


def compute_epipole_right(F):
    U,S,V = np.linalg.svd(F)
    e = V[-1]
    return e/e[2]

def compute_epipole_left(F):
    U,S,V = np.linalg.svd(F.T)
    e = V[-1]
    return e/e[2]
    
    
def plot_epipolar_line(im,F,x,epipole,number):
    global point1, point2 
    m,n = im.shape[:2]
    print(m)
    print(n)
    line = np.dot(F,x)
    t = np.linspace(0,n,100)
    lt = np.array([(line[2]+line[0]*tt)/(-line[1]) for tt in t])
    maximo = 0 
    otro = 0 
    for i in range(0,100):
        if m> lt[i]:
            maximo = lt[i]
        if n > t[i]:
            otro = t[i]
    if number == 1:
        cv2.line(im,(int(point2[0]),int(point2[1])),(int(maximo),int(otro)),(0,255,0),2)
        return im
    if number == 2:
        cv2.line(im,(int(point1[0]),int(point1[1])),(int(maximo),int(otro)),(0,255,0),2)
        return im
    

def drawlines(img1,img2,pts1,pts2):
    r,c,d = img1.shape
    for p in pts1:
        img1 = cv2.circle(img1,(int(p[0]),int(p[1])),2,(0,0,255), -1)
    for p in pts2:
        img2 = cv2.circle(img2,(int(p[0]),int(p[1])),2,(0,0,255), -1)
    return img1,img2

def combinar(image1, image2):
    image_combine = np.concatenate((image1, image2), axis=1)
    return image_combine

def draw_circle(event,x,y,flags,param):
    global lista1, lista2, pts1, pts2, img1
    flag = 0 
    if event == cv2.EVENT_LBUTTONDOWN:
        if x < img1.shape[0]:
            k = 0 
            for i,j in pts1:
                k += 1 
                if abs(i-x)<5 and abs(j-y)<5 and flag == 0: 
                    flag = 1 
                    lista1.append((i,j,1))
                    lista2.append((pts2[k][0],pts2[k][1], 1))
                    k = 0 
            flag = 0 
                    
        elif x > img1.shape[0]:
            k = 0 
            for i,j in pts2:
                k += 1 
                if abs(x-img1.shape[0]-i)<5 and abs(j-y)<5 and flag == 0:
                    flag = 1
                    lista2.append((i,j,1))
                    lista1.append((pts1[k][0],pts1[k][1], 1))
                    k = 0
            flag = 0
        print("Number of points selected:")
        print(len(lista1))
        print("At least 8.")

def point2_draw(event,x,y,flags,param):
    global img1,pts2,pts1, point2, point1
    flag = 0 
    if event == cv2.EVENT_LBUTTONDOWN:
        if x > img1.shape[0]:
            k = 0 
            for i,j in pts2:
                k+=1
                if abs(x-img1.shape[0]-i)<5 and abs(j-y)<5 and flag == 0:
                    flag = 1 
                    point2 = (i,j,1)
                    point2 = np.asarray(point2)
                    point1 = (pts1[k][0],pts1[k][1], 1)
                    point1 = np.asarray(point1)
                    k = 0 
        print("Point selected:")
        print(point2)
        flag = 0 

def point1_draw(event,x,y,flags,param):
    global img1,pts1,pts2, point1
    flag = 0 
    if event == cv2.EVENT_LBUTTONDOWN:
        if x < img1.shape[0]:
            k = 0 
            for i,j in pts1:
                k+=1
                if abs(i-x)<5 and abs(j-y)<5 and flag == 0: 
                    flag = 1 
                    point1 = (i,j,1)
                    point1 = np.asarray(point1)
                    point2 = (pts2[k][0],pts2[k][1], 1)
                    point2 = np.asarray(point2)
                    k = 0 
        print("Point selected:")
        print(point1)
        flag = 0 
        

def main():
    global pts1, pts2, img1, lista1, lista2, point1, point2, er, el, F 

    img1 = cv2.imread('corridor-r_1.jpg')
    img2 = cv2.imread('corridor-l_1.jpg')
    orb = cv2.ORB_create()
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    kp1, des1 = orb.detectAndCompute(img1,None)
    kp2, des2 = orb.detectAndCompute(img2,None)
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2, k=2)
    for i,(m,n) in enumerate(matches):
        pts2.append(kp2[m.trainIdx].pt)
        pts1.append(kp1[m.queryIdx].pt)
    img1, img2 = drawlines(img1,img2,pts1,pts2)
    imagen_final = combinar(img1, img2)
    cv2.imshow('CS_512_imagen6', imagen_final)
    cv2.setMouseCallback('CS_512_imagen6',draw_circle)
    while True:      
        key=cv2.waitKey()
        if key == ord('f'):
            if len(lista1) > 8 and len(lista2) > 8 and len(lista1) ==len(lista2):
                lista1 = np.asarray(lista1)
                lista2 = np.asarray(lista2)
                F = compute_fundamental(lista1,lista2)
                print("The fundamental matrix is:")
                print(F)
        if key == ord('e'):
            print("The epipole right is: ")
            er = compute_epipole_right(F)
            print(er)
        if key == ord('t'):
            print("The epipole left is: ")
            el = compute_epipole_left(F)
            print(el)
        if key == ord('r'):
            print("Listening... give me a point in the right image: ")
            cv2.setMouseCallback('CS_512_imagen6',point2_draw)
        if key==ord('m'):
            img1 = plot_epipolar_line(img1,F,point2,er,1)
            imagen_final = combinar(img1, img2)
            cv2.imshow('CS_512_imagen6', imagen_final)
        if key == ord('l'):
            print("Listening... give me a point in the left image: ")
            cv2.setMouseCallback('CS_512_imagen6',point1_draw)
        if key==ord('n'):
            img2 = plot_epipolar_line(img2,F,point1,el,2)
            imagen_final = combinar(img1, img2)
            cv2.imshow('CS_512_imagen6', imagen_final)
        if key == ord('h'):
            print('To use the program you must first mark at least 8 points in one of the two images. Then press F to get the matrix F. Then e or t to see the coordinates of the e = epipole right and t to see the coordinates of the epipole left. Now it\'s time to draw the epipolar lines and first you have to press r and wait for a button to click on the image on the right. You can press all the points you want but you will only be left with the last one. And to see the epipolar line of that point you must press m below. All this process can be done in the opposite image using the letters l to be able to press the point in image 1 and n to see the new epipolar line.')
        elif key==27: 
            cv2.destroyAllWindows()
            break  
    
if __name__ == "__main__":
    main()