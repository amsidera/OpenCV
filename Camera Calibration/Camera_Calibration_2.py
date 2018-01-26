# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 00:28:43 2017

@author: AnaMaria
"""
import numpy as np
from matplotlib import pyplot as plt
from sklearn import linear_model
import scipy # use numpy if scipy unavailable
import scipy.linalg # use numpy if scipy unavailable
import configparser 
import re

def estimate_2dpoints(matrix,worldpoints): 
    image_pointsestimated = []
    image_points2x = []
    image_points2y = []
    for i in range(0,len(worldpoints)):
        result = np.dot(matrix, worldpoints[i]).tolist()       
        image_points2x.append(result[0][0]/result[0][2])
        image_points2y.append(result[0][1]/result[0][2])
        image_pointsestimated.append([result[0][0]/result[0][2], result[0][1]/result[0][2]])
       
    image_pointsestimated = np.asarray(image_pointsestimated)
    image_points2x = np.asarray(image_points2x)
    image_points2y = np.asarray(image_points2y)
    return image_pointsestimated, image_points2x, image_points2y 


def calculateparameters(worldpoints, image_points):
    
    Q = np.zeros(shape=(len(worldpoints)*2,12))
    j = 0 
    for i in range(0,len(worldpoints)):
            
            Q[j]= [worldpoints[i][0],worldpoints[i][1],worldpoints[i][2],1, 0,0,0,0, -image_points[i][0] * worldpoints[i][0], -image_points[i][0] * worldpoints[i][1], -image_points[i][0] * worldpoints[i][2], -image_points[i][0]]
            Q[j+1] =  [0,0,0,0, worldpoints[i][0],worldpoints[i][1],worldpoints[i][2], 1, -image_points[i][1] * worldpoints[i][0], -image_points[i][1] * worldpoints[i][1],-image_points[i][1] * worldpoints[i][2], -image_points[i][1]]
            j += 2
    
    U, s, V = np.linalg.svd(Q, full_matrices=True)
    
    M = np.zeros(shape=(3,4))
    i = 0
    
    for j in range(0,3):
        for k in range(0,4):
            M[j][k]= V[i][11]
            i += 1
    rho = 1/np.linalg.norm(M[2,0:3])
    a1 = M[0,0:3]   
    a2 = M[1,0:3]
    a3 = M[2,0:3]
    b = M[0:3, 3].reshape(3,1)  

    u0 = np.abs(rho)**2*np.dot(a1, a3)
    v0 = np.abs(rho)**2*np.dot(a2, a3)
 
    alphav = np.sqrt(np.abs(rho)**2*(np.dot(a2, a2))-v0**2)
    s = (np.abs(rho)**4)/alphav*np.dot(np.cross(a1, a3),np.cross(a2, a3) )
    alphau = np.sqrt(np.abs(rho)**2*np.dot(a1, a1)-s**2-u0**2)

    r1 = np.cross(a2, a3)  / np.linalg.norm(np.cross(a2, a3) )
    r3 = a3
    r2 = np.cross(r3, r1)

    K = np.matrix([[alphau, s, u0],[0,   alphav, v0],[0, 0, 1]])
    invK = np.linalg.inv(K)
    sigma = np.sign(b[2])
    b = np.array([b[0][0],b[1][0],b[2][0]]).reshape(1,3)
    t = sigma*rho*np.dot(invK,b[0]).tolist()
    extrinsicMatrix = np.matrix([[r1[0], r2[0], r3[0], t[0][0]],[r1[1], r2[1], r3[1], t[0][1]],[r1[2], r2[2], r3[2], t[0][2]]])
    return K, extrinsicMatrix


def ransac(worldpoints,image_points,n,k,t,debug=False):
    iterations = 0
    besterr = np.inf

    while iterations < k:
        maybe_idxs_world, test_idxs_world = random_partition(n,worldpoints.shape[0])
        maybe_idxs_image, test_idxs_image = random_partition(n,image_points.shape[0])
     
        maybeinliers_world = worldpoints[maybe_idxs_world,:]
        maybeinliers_image = image_points[maybe_idxs_image,:]
        
        K, extrinsicMatrix = calculateparameters(maybeinliers_world, maybeinliers_image)
        matrix = np.dot(K, extrinsicMatrix)
        image_pointsestimated, image_pointsestimatedx, image_pointsestimatedy  = estimate_2dpoints(matrix,maybeinliers_world)
        msex = (np.sqrt((image_pointsestimated - maybeinliers_image)**2))
        distance = np.zeros(shape=(len(msex),1))

        for i in range(0,len(msex)):
            distance[i] = msex[i][0]+msex[i][1]
        mean_distance = np.mean(distance)
        j = 0 
        k = 0 
        for i in range(0,len(msex)):
            if distance[i]< t*mean_distance:
                j +=1
            else: 
                k += 1

        inliers_world = np.zeros(shape=(j,4))
        inliers_image = np.zeros(shape=(j,3))
        j = 0 
        for i in range(0,len(msex)):
            if distance[i]< mean_distance:
                inliers_world[j][0] = maybeinliers_world[i][0]
                inliers_world[j][1] = maybeinliers_world[i][1]
                inliers_world[j][2] = maybeinliers_world[i][2]
                inliers_world[j][3] = maybeinliers_world[i][3]
                inliers_image[j][0] = maybeinliers_image[i][0]
                inliers_image[j][1] = maybeinliers_image[i][1]
                inliers_image[j][2] = maybeinliers_image[i][2]
                j +=1
        K, extrinsicMatrix = calculateparameters(inliers_world, inliers_image)
        matrix = np.dot(K, extrinsicMatrix)
        image_pointsestimated,image_pointsestimatedx, image_pointsestimatedy  = estimate_2dpoints(matrix,inliers_world)
        msex = np.mean((image_pointsestimated - inliers_image)**2)
        if msex < besterr:
            bestlenght = len(inliers_world)
            besterr = msex
            iterations+=1


    return besterr, bestlenght
    
def random_partition(n,n_data):
    """return n random rows of data (and also the other len(data)-n rows)"""
    all_idxs = np.arange( n_data )
    np.random.shuffle(all_idxs)
    idxs1 = all_idxs[:n]
    idxs2 = all_idxs[n:]
    return idxs1, idxs2    

        
def read_txt3D(name):
    
    file = open(name, "r")
    result = np.zeros(shape=(268,4))
    resultx = np.zeros(shape=(268,1))
    resulty = np.zeros(shape=(268,1))
    resultz = np.zeros(shape=(268,1))
    i = 0 
    for line in file:
        a = [x for x in line.split()]
        for j in range(0,3):
            result[i][j] = float(a[j])
        result[i][3] = 1
        resultx[i][0] = float(a[0])
        resulty[i][0] = float(a[1])
        resultz[i][0] = float(a[2])
        i+=1
    file.close()
    return result, resultx, resulty, resultz

def read_txt2D(name):
    
    file = open(name, "r")
    result = np.zeros(shape=(268,2))
    resultx = np.zeros(shape=(268,1))
    resulty = np.zeros(shape=(268,1))
    i = 0 
    for line in file:
        a = [x for x in line.split()]
        result[i][0] = float(a[0])
        result[i][1] = float(a[1])
        resultx[i][0] = float(a[0])
        resulty[i][0] = float(a[1])
        i+=1
    file.close()
    return result, resultx, resulty

def ConfigSectionMap():
    parser = configparser.ConfigParser()
    parser.read_file(open('ransac.config'))
    n = int(parser['data']['n'])
    k = int(parser['data']['k'])
    t = int(parser['data']['t'])

    return n,k,t

def estimate_2dpoints(matrix,worldpoints): 
    image_pointsestimated = np.zeros(shape=(len(worldpoints),3))
    image_points2x = np.zeros(shape=(len(worldpoints),1))
    image_points2y = np.zeros(shape=(len(worldpoints),1))
    for i in range(0,len(worldpoints)):
        result = np.dot(matrix, worldpoints[i]).tolist()   
        image_points2x[i][0] = result[0][0]/result[0][2]
        image_points2y[i][0] = result[0][1]/result[0][2]
        image_pointsestimated[i][0] = result[0][0]/result[0][2]
        image_pointsestimated[i][1] = result[0][1]/result[0][2]
        image_pointsestimated[i][2] = 1
    return image_pointsestimated, image_points2x, image_points2y 
        
def test():
    n1,k1,t= ConfigSectionMap()
    name3d = ["3D_noise1.txt","3D_noise2.txt"]
    name2d = ["2D_noise1.txt","2D_noise2.txt"]
    name = ["noise1.txt","noise2.txt"]
    for i in range(0,len(name3d)):
        worldpoints, worldpointsx,worldpointsy, worldpointsz = read_txt3D(name3d[i])
        worldpoints1 = np.ones(shape=(len(worldpoints),1))
        imagepoints1 = np.ones(shape=(len(worldpoints),1))
        image_points, image_pointsx , image_pointsy   = read_txt2D(name2d[i])

        all_data1 = np.hstack( (worldpointsx,worldpointsy,worldpointsz,worldpoints1) )
        all_data2 = np.hstack( (image_pointsx,image_pointsy, imagepoints1) )
        debug = False
        
        msex, n = ransac(all_data1, all_data2,n1,k1,t,debug=debug)
        res = np.power((-1*(10**(np.log10(0.01)/k1)-1)), np.float64(1/n))
        print(name[i])
        print("Number of points at each evaluation")
        print(n1)
        print("Number of trials")
        print(k1)
        print("Treshold to determine is one point is an inlier")
        print(t)
        print("Probability that a point is an inlier:")
        print(res)
        print("Best model MSE:")
        print(msex)

               
if __name__ == "__main__":
    test()