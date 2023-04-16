"""
Created on Wed Feb 15 12:27:04 2023

@author: seanj
"""

import os
import numpy as np
from ultralytics import YOLO
from PIL import ImageGrab
import cv2
import time


class ObjectRecognition:
    def __init__(self, model, bounding_box, show_results, conf_threshold):
        self.model = model
        self.bounding_box = bounding_box
        self.curr_screen = None
        self.show_results = show_results
        self.conf_threshold = conf_threshold

    def get_screen_data(self):
        last_time = time.time()
        self.curr_screen = np.array(ImageGrab.grab(bbox=self.bounding_box))
        results = self.model.predict(cv2.cvtColor(self.curr_screen, cv2.COLOR_BGR2RGB), conf=self.conf_threshold,
                                show=self.show_results)
        print('Time taken for OR model to get data from screen: ', str(time.time() - last_time))
        return results[0].boxes.boxes


# Set current working directory
#HOME = os.getcwd()
#model = YOLO(f'{HOME}/weights/best.pt')

# objrecognition = ObjectRecognition(model, (0, 40, 640, 480), True, 0.2)
# objrecognition.get_screen_data()
