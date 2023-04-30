"""
Created on Wed Feb 15 12:27:04 2023

@author: seanj
"""

import os
import numpy as np
from PIL import ImageGrab
import cv2
import time


def extract_objects(tensor):
    # Loops through each detected object stored by the output tensor and returns them as a list
    objects = [obj for obj in tensor]  # This single line is put into a function for my own readability.
    return objects


def get_overlaps(objects):
    # This could be declared outside the scope of this function to check specific overlaps during runtime.
    found_overlaps = []

    # Loop through each object that hasn't been compared yet and check for overlap with other objects
    compared = set()
    for i, obj1 in enumerate(objects):
        for j, obj2 in enumerate(objects):
            if i >= j:
                continue
            if (i, j) in compared:
                continue
            # Check if the bounding boxes intersect
            x1_min, y1_min = obj1[0], obj1[1]
            x1_max, y1_max = obj1[2], obj1[3]
            x2_min, y2_min = obj2[0], obj2[1]
            x2_max, y2_max = obj2[2], obj2[3]
            if (x1_min <= x2_max and x1_max >= x2_min and
                    y1_min <= y2_max and y1_max >= y2_min):
                found_overlaps.append((i, j))

    # Return whether any overlaps have been found.
    if len(found_overlaps) > 0:
        return True
    else:
        return False


class ObjectRecognition:
    def __init__(self, model, bounding_box, show_results, conf_threshold):
        self.model = model
        self.bounding_box = bounding_box
        self.curr_screen = None
        self.DEBUG_MODE = False
        self.show_results = show_results
        self.conf_threshold = conf_threshold

    def get_screen_data(self):
        last_time = time.time()
        self.curr_screen = np.array(ImageGrab.grab(bbox=self.bounding_box))
        results = self.model.predict(cv2.cvtColor(self.curr_screen, cv2.COLOR_BGR2RGB), conf=self.conf_threshold,
                                     show=self.show_results)
        if self.DEBUG_MODE:
            print('Time taken for OR model to get data from screen: ', str(time.time() - last_time))
        return results[0].boxes.boxes
