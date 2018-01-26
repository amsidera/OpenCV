
from sklearn.cluster import MiniBatchKMeans
import numpy as np
from scipy.cluster.vq import * 

class Vocabulary:

    def __init__(self, stackOfDescriptors, k,  subSampling = 10):
        kmeans_t = MiniBatchKMeans(init='k-means++', n_clusters=k, n_init=10)
        kmeans_t.fit(stackOfDescriptors)

        self.vocabulary = kmeans_t.cluster_centers_
        self.size = self.vocabulary.shape[0]

    def buildHistogramForEachImageAtDifferentLevels(self, descriptorsOfImage, level):

        width = int(descriptorsOfImage.width/ 8)
        height = int(descriptorsOfImage.height/ 8)

        descriptors = descriptorsOfImage.descriptors


        histogramOfLevelThree = np.zeros((64, self.size))
        for descriptor in descriptors:
            x = descriptor.x
            y = descriptor.y
            indice = int(x / width)  + int(y / height) *8

            feature = descriptor.descriptor
            shape = feature.shape[0]
            feature = feature.reshape(1, shape)

            co, nothing = vq(feature, self.vocabulary)
            
            if indice <64:
                histogramOfLevelThree[indice][co[0]-1] += 1
            else: 
                histogramOfLevelThree[63][co[0]-1] += 1
                
        histogramOfLevelTwo = np.zeros((16, self.size))
        histogramOfLevelTwo[0] = histogramOfLevelThree[0] + histogramOfLevelThree[1] + histogramOfLevelThree[8] + histogramOfLevelThree[9]
        histogramOfLevelTwo[1] = histogramOfLevelThree[2] + histogramOfLevelThree[3] + histogramOfLevelThree[10] + histogramOfLevelThree[11]
        histogramOfLevelTwo[2] = histogramOfLevelThree[4] + histogramOfLevelThree[5] + histogramOfLevelThree[12] + histogramOfLevelThree[13]
        histogramOfLevelTwo[3] = histogramOfLevelThree[6] + histogramOfLevelThree[7] + histogramOfLevelThree[14] + histogramOfLevelThree[15]
        histogramOfLevelTwo[4] = histogramOfLevelThree[16] + histogramOfLevelThree[17] + histogramOfLevelThree[24] + histogramOfLevelThree[25]
        histogramOfLevelTwo[5] = histogramOfLevelThree[18] + histogramOfLevelThree[19] + histogramOfLevelThree[26] + histogramOfLevelThree[27]
        histogramOfLevelTwo[6] = histogramOfLevelThree[20] + histogramOfLevelThree[21] + histogramOfLevelThree[28] + histogramOfLevelThree[29]
        histogramOfLevelTwo[7] = histogramOfLevelThree[22] + histogramOfLevelThree[23] + histogramOfLevelThree[30] + histogramOfLevelThree[31]
        histogramOfLevelTwo[8] = histogramOfLevelThree[32] + histogramOfLevelThree[33] + histogramOfLevelThree[40] + histogramOfLevelThree[41]
        histogramOfLevelTwo[9] = histogramOfLevelThree[34] + histogramOfLevelThree[35] + histogramOfLevelThree[42] + histogramOfLevelThree[43]
        histogramOfLevelTwo[10] = histogramOfLevelThree[36] + histogramOfLevelThree[37] + histogramOfLevelThree[44] + histogramOfLevelThree[45]
        histogramOfLevelTwo[11] = histogramOfLevelThree[38] + histogramOfLevelThree[39] + histogramOfLevelThree[46] + histogramOfLevelThree[47]
        histogramOfLevelTwo[12] = histogramOfLevelThree[48] + histogramOfLevelThree[49] + histogramOfLevelThree[56] + histogramOfLevelThree[57]
        histogramOfLevelTwo[13] = histogramOfLevelThree[50] + histogramOfLevelThree[51] + histogramOfLevelThree[58] + histogramOfLevelThree[59]
        histogramOfLevelTwo[14] = histogramOfLevelThree[52] + histogramOfLevelThree[53] + histogramOfLevelThree[60] + histogramOfLevelThree[61]
        histogramOfLevelTwo[15] = histogramOfLevelThree[54] + histogramOfLevelThree[55] + histogramOfLevelThree[62] + histogramOfLevelThree[63]
        
        
        histogramOfLevelOne = np.zeros((4, self.size))
        histogramOfLevelOne[0] = histogramOfLevelTwo[0] + histogramOfLevelTwo[1] + histogramOfLevelTwo[4] + histogramOfLevelTwo[5]
        histogramOfLevelOne[1] = histogramOfLevelTwo[2] + histogramOfLevelTwo[3] + histogramOfLevelTwo[6] + histogramOfLevelTwo[7]
        histogramOfLevelOne[2] = histogramOfLevelTwo[8] + histogramOfLevelTwo[9] + histogramOfLevelTwo[12] + histogramOfLevelTwo[13]
        histogramOfLevelOne[3] = histogramOfLevelTwo[10] + histogramOfLevelTwo[11] + histogramOfLevelTwo[14] + histogramOfLevelTwo[15]

        histogramOfLevelZero = histogramOfLevelOne[0] + histogramOfLevelOne[1] + histogramOfLevelOne[2] + histogramOfLevelOne[3]


        if level == 0:
            return histogramOfLevelZero

        elif level == 1:
            tempZero = histogramOfLevelZero.flatten() * 0.5
            tempOne = histogramOfLevelOne.flatten()
            result = np.concatenate((tempZero, tempOne))
            return result

        elif level == 2:

            tempZero = histogramOfLevelZero.flatten()*0.25
            tempOne = histogramOfLevelOne.flatten() * 0.5
            tempTwo = histogramOfLevelTwo.flatten() 
            result = np.concatenate((tempZero, tempOne, tempTwo))
            return result
        
        elif level == 3:

            tempZero = histogramOfLevelZero.flatten() * 0.125
            tempOne = histogramOfLevelOne.flatten() * 0.25
            tempTwo = histogramOfLevelTwo.flatten() * 0.5
            tempThree = histogramOfLevelThree.flatten()
            result = np.concatenate((tempZero, tempOne, tempTwo, tempThree))
            return result