import torch
import numpy as np
import matplotlib.pyplot as plt
import norse
from norse.torch import li_step, LICell, LIState, LIParameters, LIF
from norse.torch import lif_step, LIFCell, LIFState, LIFParameters
from norse.torch.module import leaky_integrator as li
from norse.torch.module import LICell as lic
from norse.torch.module import LIFCell as lifc
import ProjectMethods as pm
from scipy.signal import convolve2d
from torch.nn import Conv2d as conv2
import ae4ToNumpy as atn
from time import perf_counter_ns as pc

class layer1:
        

    #The test Input data should be to start with: A dvs video of gesture
    #In the form of a numpy array of events
    #The results should be lines that fit the image
    def getTestData(self):
        a = atn.ae4ToNumpy("/home/julia/Documents/PythonProjectsGit/DvsSNNProject/ae4/mov1.aedat4")
        return a.getEventsAndTriggers()[0]


    #The training input should be lines of differing angles and sizes
    #Being still / Moving across the scren
    #With outputs being lines that should match, angle, position and length
    def getTrainingData():
        pass

    #Checks for spikes in layer 1.1 and sends those spikes as inputs 
    def checkForSpikes():
        pass

        #1 neuron per x*x pixels 
    def createL11Neurons(self):

        a = atn.ae4ToNumpy("/home/julia/Documents/PythonProjectsGit/DvsSNNProject/ae4/mov1.aedat4")
        w = a.getMetaData()[0]["width"]
        h = a.getMetaData()[0]["height"]

        p2 = LIFParameters(tau_syn_inv = torch.as_tensor(1 / 5e-3), 
        tau_mem_inv = torch.as_tensor(1 / 1e-2), v_leak = torch.as_tensor(-5), 
        v_th = torch.as_tensor(5), v_reset=torch.tensor([0]))
        
        self.cell = lifc(p=p2)

        # Creating a matrix where each index is a state for a cell
        self.cellStateMatrix = []
        for x in range(w):
            cellRow = []
            for y in range(h):

                output, state = self.cell(torch.ones(1))
                cellRow.append(state)
            self.cellStateMatrix.append(cellRow) 

        #No need really
        #return self.cellStateMatrix


    #Right now, makes no difference for polarity
    def sendInputToL11(self,x,y, input = 1):
        
        #input into specific neuron, get state back to neuron
        spike, self.cellStateMatrix[x][y] = self.cell(input_tensor=torch.tensor([input]), state=self.cellStateMatrix[x][y])
       
        #Take the spikes and run spike function
        if torch.eq(torch.tensor([1]), spike):
            self.spikeFunction(x,y)

    #When a neuron spikes, send small input to nearby neurons, like 50 % to 8 closest
    # and 25 percent to 24 closest
    # And also, send full input to layer 1.2 neurons that connect to it. 
    def spikeFunction(self, x, y):


        #send full input to layer 1.2 neurons that connect to it. 
        #sendInputToL12(x,y) 

        #And also,  send small input to nearby neurons, like 50 % to 8 closest
        # and 25 percent to 24 closest
        for x1 in range(x-2,x+2):
            for y1 in range(y-2,y+2):
                if x1==x and y1==y:
                    continue
                if (x1<x-2 or x1>x+2 or y1<y-2 or y1>y+2):
                     self.sendInputToL11(x,y,input=0.25)
                else:
                    self.sendInputToL11(x,y,input=0.5)


        #1 neuron per combination of angle, position
    def createL12Neurons():
        pass

        #Create a matrix that connects each pixel in L11 to multiple L12 neurons. It is a many to many correlation. <
    def createL11to12CorrMatrix():
        pass

l1 = layer1()
l1.createL11Neurons() # Works i think, but slow
list = l1.getTestData()
print(len(list[125]))
print(list[120].shape)
print(list[120][120])
print(list[121][120])
print(list[120][120]["x"])

time_start = pc()
for z in range(1000):
    event = list[0][z]
    l1.sendInputToL11(event["x"],event["y"])
print((pc()-time_start)*1e-6)
#for array in list:
#    for event in array:
#        l1.sendInputToL11(event["x"],event["y"])

 
#l1.sendInputToL11(10,10) #Works i think
#cellMatrix = layer1.createL11Neurons(self) #346 width, 260 height
    #print(len(cellMatrix[0]))
    #Below