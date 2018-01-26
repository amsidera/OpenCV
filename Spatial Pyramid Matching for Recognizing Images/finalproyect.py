
import pickle
import Descriptor
import os 
import numpy as np
import cv2
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
        
def train_test(trainDataPath, trainLabelPath, testDataPath, testLabelPath):
    file = open(trainDataPath, 'rb')  
    trainData = np.array(pickle.load(file, encoding='latin1'))
    file2 = open(trainLabelPath, 'rb')
    trainLabels = pickle.load(file2, encoding='latin1')  
    file3 = open(testDataPath, 'rb')
    testData = np.array(pickle.load(file3, encoding='latin1'))
    file4 = open(testLabelPath, 'rb')
    testLabels = pickle.load(file4, encoding='latin1')
    return trainData, trainLabels, testData, testLabels

def SVM(trainDataPath, trainLabelPath, testDataPath, testLabelPath, kernelType):
    train_Data, train_Labels, test_Data, test_Labels = train_test(trainDataPath, trainLabelPath, testDataPath, testLabelPath)
    clf = SVC(kernel = kernelType)
    clf.fit(train_Data, train_Labels)
    results = clf.predict(test_Data)

    accuracy = sum(1.0 * (results == test_Labels)) / len(test_Labels)
    print("The accuracy for this test is:")
    print (accuracy)

def KNN(trainDataPath, trainLabelPath, testDataPath, testLabelPath):
    train_Data, train_Labels, test_Data, test_Labels = train_test(trainDataPath, trainLabelPath, testDataPath, testLabelPath)

    KNN = KNeighborsClassifier()
    KNN.fit(train_Data, train_Labels)
    KNN_t = KNN.predict(test_Data)

    accuracy = sum(1.0 * (KNN_t == test_Labels)) / len(test_Labels)
    print("The accuracy for this test is:")
    print (accuracy)
    
    
def readimages(folder, number):
    images = []
    i =0 
    name = ['faces','airplanes_side','motorbikes_side']
    for x in range(0,number):
        path = folder +"/"+ name[x]
        label = "/"+ name[x]
        for imags in os.listdir(path):
            newpath = path +"/"+ imags
            img = cv2.imread(newpath)
            height = img.shape[0]
            width = img.shape[1]
#            im = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            scale = 16/3.0
            x,y = np.meshgrid(range(8,width,8),range(8,height,8))
            xx,yy = x.flatten(),y.flatten()
            frame = np.array([xx,yy,scale * np.ones(xx.shape[0]), np.zeros(xx.shape[0])])
            np.savetxt('tmp.frame',frame.T,fmt='%03.3f')    
            os.system("sift "+str(i)+" --output=temp.sift --read-frames=tmp.frame")
            f = np.loadtxt('temp.sift')
            k = f[:,:4]
            m = f[:,4:]
            descriptors = []
            i+=1
            for i in range(k.shape[0]):
                descriptors.append(Descriptor.sift(k[i][0], k[i][1], m[i]))
            images.append(Descriptor.Descriptors(descriptors, label, width, height))
            

    return images


      
        
def Histogram(path, level, number):
    featurehis = []
    labels_image = []

    file = open("Data/voc.pkl", 'rb')
    voc = pickle.load(file, encoding='latin1')
    trainData = readimages("images/"+path, number)
    for image in trainData:

        featurehis.append(voc.buildHistogramForEachImageAtDifferentLevels(image, level))
        labels_image.append(image.label)

    fo = open("Data/"+path+"Level" +str(level)+ ".pkl", "wb")
    pickle.dump(featurehis, fo)
    fo.close()
    
    fo1 = open("Data/"+path+"la.pkl", "wb")
    pickle.dump(labels_image, fo1)
    fo1.close()




def main():
    print("This program classifies the images using spatial images matching.\n")   
    categories = input("Number of different categories that you want to classify in the process: 2, 3.\n")
    if int(categories) < 2 or int(categories) > 3:
        print("Not a possible number of categories. The default is 2.\n")  
        categories = '2'
    nb = input('What level do you want to use: 0, 1, 2 or 3?\n')
    if nb == '0':
        print("Developing a histogram with level 0.\n")
        Histogram("testing", 0, int(categories))
        Histogram("training", 0, int(categories))
    elif nb == '1':
        print("Developing a histogram with level 0 and level 1.\n")
        Histogram("testing", 1, int(categories))
        Histogram("training", 1, int(categories))
    elif nb == '2':
        print("Developing a histogram with level 0, level 1 and level 2.\n")
        Histogram("testing", 2, int(categories))
        Histogram("training", 2, int(categories))
    elif nb == '3':
        print("Developing a histogram with level 0, level 1, level 2 and level 3.\n")
        Histogram("testing", 3, int(categories))
        Histogram("training", 3, int(categories))
    else: 
        print("Not a level in this program. The default is 2.\n")
        Histogram("testing", 2, int(categories))
        Histogram("training", 2, int(categories))
    print("Next step is ")
    classifier = input("Classifier you want to use: Linear, Polynomial, Radial basis function, Logistic or KNN\n")
    if classifier == "Linear":
        kernel = "linear"
        SVM("Data/trainingLevel"+nb+".pkl", "Data/trainingla.pkl", "Data/testingLevel"+nb+".pkl", "Data/testingla.pkl", kernel)
    elif classifier == "Polynomial":
        kernel = "poly"
        SVM("Data/trainingLevel"+nb+".pkl", "Data/trainingla.pkl", "Data/testingLevel"+nb+".pkl", "Data/testingla.pkl", kernel)
    elif classifier == "Radial basis function":
        kernel = "rbf"
        SVM("Data/trainingLevel"+nb+".pkl", "Data/trainingla.pkl", "Data/testingLevel"+nb+".pkl", "Data/testingla.pkl", kernel)
    elif classifier == "Logistic":
        kernel = "sigmoid"
        SVM("Data/trainingLevel"+nb+".pkl", "Data/trainingla.pkl", "Data/testingLevel"+nb+".pkl", "Data/testingla.pkl", kernel)
    elif classifier == "KNN":
        KNN("Data/trainingLevel"+nb+".pkl", "Data/trainingla.pkl", "Data/testingLevel"+nb+".pkl", "Data/testingla.pkl")
    else: 
        print("Not a kernel in this program. The default is linear.\n")
        kernel = "linear"
        SVM("Data/trainingLevel"+nb+".pkl", "Data/trainingla.pkl", "Data/testingLevel"+nb+".pkl", "Data/testingla.pkl", kernel)
if __name__ == "__main__":

    main()




