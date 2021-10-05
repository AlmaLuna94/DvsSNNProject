import scipy.io


class matToArrays:

    def __init__(self, filePath = None) -> None:
        self.filePath = filePath
        if filePath != None:
            self.mat = scipy.io.loadmat(self.filePath)

    def setFilePath(self, filePath):
        self.filePath = filePath
        self.mat = scipy.io.loadmat(self.filePath)

    def getMetaData(self):
        #print(mat["__header__"]) #b'MATLAB 5.0 MAT-file, Platform: GLNXA64, Created on: Thu May 16 13:55:20 2019'
        #print(mat["__version__"]) #1.0
        #print(mat["__globals__"]) #[] Empty
        return [self.mat["__header__"], self.mat["__version__"], self.mat["__globals__"]]

    def getXYZPOSjoints(self):
        #print(len(mat["XYZPOS"][0][0])) # 13 Joints So 13 separate arrays of XYZ positions
        return self.mat["XYZPOS"][0][0]


# How to use Class example.
mtA = matToArrays()
mtA.setFilePath('/home/julia/Documents/PythonProjectsGit/DvsSNNProject/mat/S1_1_1.mat')
print(mtA.getMetaData())
print(mtA.getXYZPOSjoints())

'''
mat = scipy.io.loadmat('F:/PythonProjects/pythonProjectImage/testDHP19Data/S1_1_1.mat')

print(len(mat))
print(mat["__header__"]) #b'MATLAB 5.0 MAT-file, Platform: GLNXA64, Created on: Thu May 16 13:55:20 2019'
print(mat["__version__"]) #1.0
print(mat["__globals__"]) #[] Empty

print(len(mat["XYZPOS"])) # One array of length one containing all data of xyz positions, Flatten it?
print(len(mat["XYZPOS"][0])) # One array of length one containing all data of xyz positions, Flatten it?
print(len(mat["XYZPOS"][0][0])) # 13 Joints So 13 separate arrays of XYZ positions
print(len(mat["XYZPOS"][0][0][0])) #2379 XYZ Positions ordered after one another in time
print(mat["XYZPOS"][0].dtype) #[('head', 'O'), ('shoulderR', 'O'), ('shoulderL', 'O'), ('elbowR', 'O'), ('elbowL', 'O'), ('hipL', 'O'), ('hipR', 'O'), 
                              #('handR', 'O'), ('handL', 'O'), ('kneeR', 'O'), ('kneeL', 'O'), ('footR', 'O'), ('footL', 'O')]  Joint names.

'''