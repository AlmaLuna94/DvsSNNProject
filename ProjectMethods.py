import torch
import numpy as np
import matplotlib.pyplot as plt
import norse
import random as rand
from norse.torch import lif_step, lif_feed_forward_step, lif_current_encoder, LIFParameters
from norse.torch import li_step, LICell, LIState, LIParameters
from norse.torch.module import leaky_integrator as li
import random
from norse.torch.functional import lif as lif

#Maybe create one that simulates an image. 
def datasetCreate():
    zr = np.zeros([100,80])
    on = np.ones([100,20])
    rand = np.concatenate((zr,on),1)
    #print(rand)
    rand = np.ndarray.flatten(rand)
    np.random.shuffle(rand)
    rand = rand.reshape([100,100])
    #print(rand.shape)
    #print(rand[10,12])
    #print(np.sum(rand))

    tens = rand
    for x in range(75,85,1):
        for y in range(10,90,1):
            if random.uniform(1,10) > 6:
                tens[y,x] = 1

    tens = torch.from_numpy(tens)
    print(tens.shape[1])
    print(tens.type)
    return tens

def randPixel(size_x, size_y):
    return torch.tensor([torch.randint(size_x,(1,)), torch.randint(size_y,(1,))])



   #0 - Loop function
def loopthrough(self, times):
    #Testloop

    loop1 = loop(100,100)

    data_gen = loop1.data_generator()

    for x in range(times):
        time_start = pc() #util.nanosecond_to_milisecond(pc())
        list = data_gen.__next__()
        input_tensor = loop1.sparse_data_to_tensor(list)
        loop1.input_to_neurons(input=input_tensor)
        angle = loop1.calculate_angle()
        loop1.angle_to_motors(angle)
        print("Time to run one step = {} milliseconds".format(util.nanosecond_to_milisecond(pc()-time_start)))

def randPixel_around_pos(size_x, size_y, position):

    def min(x):
        if x < 3:
            return 0
        else:
            return x - 3
        
    def max(x, size):
        if x > (size - 4):
            return size - 1
        else:
            return x + 3

    return torch.tensor([torch.randint(min(position[0]), max(position[0], size_x),(1,)), torch.randint(min(position[1]), max(position[1], size_y),(1,))])

def add_pixel_to_array(array, pixel):
    for item in array:
        if pixel[0] == item[0]:
            if pixel[1] == item[1]:
                return array
    else:
        return array.append(pixel)

def get_one_t_array(size_x , size_y, activity_position, rand_act=1, tru_act=1) :

    def sort_order(position):
      return position[0]*size_y+position[1]

    list = []

    #Adding Random activities
    for y in range(10*rand_act):
        add_pixel_to_array(list, randPixel(size_x, size_y))
    

    #Adding true activities around activity position
    for y in range(5):
        add_pixel_to_array(list, randPixel_around_pos(size_x, size_y, activity_position,))

    #Sorting the so that activities are ordered based on position like data from cameras
    list.sort(key=sort_order)

    return list

def create_sparse_data(length, size_x, size_y):

    def sort_order(position):
        return position[0]*size_y+position[1]
    #length, time length of data array
    #size, size of 2D image that the array values correspond to, size = [x_size, y_size]


    #Data should be in list form with each list item being a variable size array 
    # These arrays should be ints, with each int corresponding to a spot in a 2D image. 
    # The int being in the array corresponds to an activity in that spot

    activity_position = randPixel(size_x, size_y)
    print("Activity in position  [{} , {}] ".format(activity_position[0], activity_position[1]))

    Data = []


    for x in range(length):
        array = []
        #two or more random activities
        for y in range(20):
            add_pixel_to_array(array, randPixel(size_x, size_y))

        #Add true activities around activity position
        for y in range(5):
            add_pixel_to_array(array, randPixel_around_pos(size_x, size_y, activity_position,))
        array.sort(key=sort_order)
        Data.append(array)
        

    return Data

def sparse_data_to_sparse_matrix(data, size):
    smatrix = torch.zeros(size)
    for count, array in enumerate(data, start=0):
        for val in array:
            smatrix[count, val[0], val[1]] = 1

            
    return smatrix

def plotNeurons(voltages, N):
    plt.figure()
    i=0
    for n in range(N):
        if n % 5 != 0:
            continue
        plt.subplot((N//10),2,i+1)
        plt.plot(voltages[n])
        plt.ylabel("v, Neuron {}".format(n))
        plt.xlabel("time [ms]")
        i+=1
    plt.show()

def neuron_nr_to_coord(neuron_nr):
    #x = (neuron_nr // 10 ) * 10
    x = torch.div(neuron_nr, 10, rounding_mode='floor') * 10
    y = (neuron_nr % 10 ) * 10
    #print(x)
    #print(y)
    return torch.tensor([x,y])

def rel_angle_to_abs_angle(id_x, angle):
    #Check where motors are right now, and use that to calculate rel to abs angle

    #curang = Checkpos(id_x)
    # new_ang = curang + x 
    # if new_ang > max_ang
        # print("Can not move motor to angle "new_ang" so instead moving to maximumn angle "max_ang" ")
        # new_ang = max_ang
    # if new_ang < min_ang
        # print("Can not move motor to angle "new_ang" so instead moving to minimum angle "min_ang" ")
        # new_ang = min_ang
        #
    # move_motor(id_x, angle )

    print("Function \"rel_angle_to_abs_angle\" not done")

class decode():

    def __init__(self, p):
        (self.max, self.min) = self.get_max_min(p)

    def decode_to_angle(self, x,adVal=1):
        return self.valToAngle(self.voltToDec(x,adVal))

    def train_val_to_angle_adjust(self, x, true_angle, adVal):
        output = self.valToAngle(self.voltToDec(x,adVal))
        new_adVal = adVal + (true_angle - output ) * 0.001
        return new_adVal

    def decode_max(self, x):
        return torch.argmax(torch.max(x,1)[0])

    def voltToDec(self,x,adVal):
        return ((x-self.min)/self.max) * adVal
            
    def valToAngle(self, x):
        return (x*700+1800).int()

    def get_max_min(self, p):
        #Initial state of test cell
        stateCell_max = li.LIState(v = torch.zeros(1),
                    i = torch.zeros(1))
        stateCell_min = li.LIState(v = torch.zeros(1),
                    i = torch.zeros(1))
        for y in range(200):
            max, stateCell_max = li.li_feed_forward_step(1, state=stateCell_max, p=p, dt = 0.001)
            min, stateCell_min = li.li_feed_forward_step(0, state=stateCell_min, p=p, dt = 0.001)
        return(max,min)
    
    def print_max_min(self):
        print("Max: {}".format(self.max))
        print("Min: {}".format(self.min))
    
 




