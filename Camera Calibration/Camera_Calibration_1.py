# -*- coding: utf-8 -*-
"""
@author: AnaMaria
"""
import numpy as np
from matplotlib import pyplot as plt

def calculateparameters(worldpoints, image_points):
    
    Q = np.zeros(shape=(len(worldpoints)*2,12))
    j = 0 
    for i in range(0,len(worldpoints)):
            
            Q[j]= [worldpoints[i][0],worldpoints[i][1],worldpoints[i][2],1, 0,0,0,0, -image_points[i][0] * worldpoints[i][0], -image_points[i][0] * worldpoints[i][1], -image_points[i][0] * worldpoints[i][2], -image_points[i][0]]
            Q[j+1] =  [0,0,0,0, worldpoints[i][0],worldpoints[i][1],worldpoints[i][2], 1, -image_points[i][1] * worldpoints[i][0], -image_points[i][1] * worldpoints[i][1],-image_points[i][1] * worldpoints[i][2], -image_points[i][1]]
            j += 2
#            print(Q)
#            break
#    Q = np.dot(Q.T, Q)
    U, s, V = np.linalg.svd(Q, full_matrices=True)
    
    M = np.zeros(shape=(3,4))
    i = 0
    
    for j in range(0,3):
        for k in range(0,4):
            M[j][k]= V[i][11]
            i += 1
    print(M)
    print(M[2,0:3])
    rho = 1/np.linalg.norm(M[2,0:3])
#    M = rho * M
    a1 = M[0,0:3]   
    a2 = M[1,0:3]
    a3 = M[2,0:3]
    b = M[0:3, 3].reshape(3,1)
    print("a1")
    print(a1)
    print("a2")
    print(a2)
    print("a3")
    print(a3)
    print("b")
    print(b)
    

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
    print("a1,a2,a3")
    print(a1,a2,a3)
    print("rho")
    print(rho)
    print("b")
    print(b)
    print("u0,v0")
    print(u0,v0)
    print("alphav, s ,alphau")
    print(alphav, s ,alphau)
    print("r1,r2,r3")
    print(r1,r2,r3)
    print(t)
    return K, extrinsicMatrix
    
def estimate_2dpoints(matrix,worldpoints, name): 
    image_pointsestimated = []
    image_points2x = []
    image_points2y = []
    file = open(name, "w")
    for i in range(0,len(worldpoints)):
        result = np.dot(matrix, worldpoints[i]).tolist()       
        image_points2x.append(result[0][0]/result[0][2])
        image_points2y.append(result[0][1]/result[0][2])
        image_pointsestimated.append([result[0][0]/result[0][2], result[0][1]/result[0][2],1])
        file.write("%s %s\n" %(result[0][0]/result[0][2], result[0][1]/result[0][2])) 
    image_pointsestimated = np.asarray(image_pointsestimated)
    image_points2x = np.asarray(image_points2x)
    image_points2y = np.asarray(image_points2y)
    file.close()
    return image_pointsestimated, image_points2x, image_points2y 
    
def read_txt(name,number):
    result = [] 
    resultx = []
    resulty = []
    file = open(name, "r")
    for line in file:
        a = [x for x in line.split()]
        if number == 1:
            result.append([float(a[0]), float(a[1]), float(a[2]), 1])
        elif number == 0: 
            result.append([float(a[0]), float(a[1])])
            resultx.append([float(a[0])])
            resulty.append([float(a[1])])
    file.close()
    resultx = np.asarray(resultx)
    resulty = np.asarray(resulty)
    result = np.asarray(result)
    return result, resultx, resulty


def main():

    
    name3d = "3D_withoutnoise.txt"
    worldpoints, worldpointsx,worldpointsy = read_txt(name3d,1)
    name2d = "2D_withoutnoise.txt"
    image_points, image_pointsx , image_pointsy   = read_txt(name2d,0)
    
    K, extrinsicMatrix = calculateparameters(worldpoints, image_points )
    matrix = np.dot(K, extrinsicMatrix)
    image_pointsestimated,image_pointsestimatedx, image_pointsestimatedy  = estimate_2dpoints(matrix,worldpoints, "2D_new.txt")

    msex = np.mean((image_pointsestimatedx - image_pointsx)**2)
    msey = np.mean((image_pointsestimatedy - image_pointsy)**2)
    msetotal = msex + msey
    print("MSE")
    print(msetotal)
    plt.plot(image_pointsestimatedx, image_pointsestimatedy, 'ro')
    plt.show()
    image_pointsestimatedx = image_pointsestimatedx.reshape(-1, 1)
    image_pointsestimatedy = image_pointsestimatedy.reshape(-1, 1)
               
if __name__ == "__main__":
    main()