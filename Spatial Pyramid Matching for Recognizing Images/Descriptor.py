
import numpy as np

class sift:

    def __init__(self, x, y, descrip):
        self.x = x
        self.y = y
        self.descriptor = self.normSIFT(descrip)

    def normSIFT(self, descrip):
        descriptor = np.array(descrip)
        norm = np.linalg.norm(descrip)

        if norm > 1.0:
            descriptor /= float(norm)

        return descriptor


class Descriptors:

    def __init__(self, descriptors, label, width, height):
        self.descriptors = descriptors
        self.label = label
        self.width = width
        self.height = height
