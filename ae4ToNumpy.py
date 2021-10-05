import aedat


class ae4ToNumpy:
    def __init__(self, filePath = None) -> None:
        self.filePath = filePath
        if filePath != None:
            self.decoder = aedat.Decoder(filePath) 

    def setFilePath(self, filePath):
        self.filePath = filePath
        self.decoder = aedat.Decoder(filePath) 

    def getMetaData(self):
        return self.decoder.id_to_stream()  #{3: {'type': 'triggers'}, 2: {'type': 'imus'}, 1: {'type': 'frame', 
                                            #'width': 346, 'height': 260}, 0: {'type': 'events', 'width': 346, 'height': 260}}

    def getEventsAndTriggers(self):
        self.eventArray = []
        self.triggerArray = []
        for packet in self.decoder:
            #print(packet['stream_id'], end=': ')
            if 'events' in packet:
                #print('{} polarity events'.format(len(packet['events'])))
                self.eventArray.append(packet['events'])
            elif 'frame' in packet:
                print('{} x {} frame'.format(packet['frame']['width'], packet['frame']['height']))
            elif 'imus' in packet:
                print('{} IMU samples'.format(len(packet['imus'])))
            elif 'triggers' in packet:
                #print('{} trigger events'.format(len(packet['triggers'])))
                self.triggerArray.append(packet['triggers'])


        return [self.eventArray, self.triggerArray]

# Example of how to use
#atN = ae4ToNumpy("/home/julia/Documents/PythonProjectsGit/DvsSNNProject/ae4/mov1.aedat4")
#print(atN.getMetaData()[0]["width"]) #Metadata about what id means
#print(atN.getEventsAndTriggers()[1]) #Triggers
#print(atN.getEventsAndTriggers()[0]) # Events

#Decodes ae4 data to numpy array. Events are in format described below:
"""
    packet['events'] is a structured numpy array with the following dtype:
        [
            ('t', '<u8'),
            ('x', '<u2'),
            ('y', '<u2'),
            ('on', '?'),
        ]
"""

"""
        packet['frame'] is a dictionary with the following structure:
            {
                't': <int>,
                'begin_t': <int>,
                'end_t': <int>,
                'exposure_begin_t': <int>,
                'exposure_end_t': <int>,
                'format': <str>,
                'width': <int>,
                'height': <int>,
                'offset_x': <int>,
                'offset_y': <int>,
                'pixels': <numpy.array(shape=(height, width), dtype=uint8)>,
            }
        format is one of 'Gray', 'BGR', 'BGRA'
        """

"""
    packet['imus'] is a structured numpy array with the following dtype:
        [
            ('t', '<u8'),
            ('temperature', '<f4'),
            ('accelerometer_x', '<f4'),
            ('accelerometer_y', '<f4'),
            ('accelerometer_z', '<f4'),
            ('gyroscope_x', '<f4'),
            ('gyroscope_y', '<f4'),
            ('gyroscope_z', '<f4'),
            ('magnetometer_x', '<f4'),
            ('magnetometer_y', '<f4'),
            ('magnetometer_z', '<f4'),
        ]
"""
"""
        packet['triggers'] is a structured numpy array with the following dtype:
            [
                ('t', '<u8'),
                ('source', 'u1'),
            ]
        the source value has the following meaning:
            0: timestamp reset
            1: external signal rising edge
            2: external signal falling edge
            3: external signal pulse
            4: external generator rising edge
            5: external generator falling edge
            6: frame begin
            7: frame end
            8: exposure begin
            9: exposure end
        """