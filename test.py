import os
from ultralytics import YOLO
import object_recognition as OR
import numpy as np


def extract_objects(tensor):
    # Loops through each detected object stored by the output tensor and returns them as a list
    objects = [obj for obj in tensor]  # This single line is put into a function for my own readability.
    return objects


def get_overlaps(objects):
    found_overlaps = []  # This could be declared outside the scope of this function to check specific overlaps during runtime.

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

class BossEnvironment:
    def __init__(self):
        self.player_label = 1
        self.boss_label = 2
        self.player_position = None
        self.boss_position = None

    def reset(self):
        self.player_position = None
        self.boss_position = None

    def observe(self, obj_recognition_output):
        # Create dictionaries for keeping track of highest accuracy player / boss. initializing to -1 for sorting purposes.
        highest_accuracy = {self.player_label: -1, self.boss_label: -1}
        highest_accuracy_objdata = {self.player_label: None, self.boss_label: None}

        # Loop through objects detected by the object recognition model
        for obj in obj_recognition_output:
            x, y, w, h, accuracy, label = obj.tolist()

            # Update positions/accuracy for obj and find most accurate for ea label.
            if label == self.player_label and accuracy > highest_accuracy[label]:
                highest_accuracy[label] = accuracy
                highest_accuracy_objdata[label] = np.array([x, y, w, h, accuracy])
            elif label == self.boss_label and accuracy > highest_accuracy[label]:
                highest_accuracy[label] = accuracy
                highest_accuracy_objdata[label] = np.array([x, y, w, h, accuracy])

        # Update the player and boss positions with the highest accuracy objects
        self.player_position = highest_accuracy_objdata[self.player_label]
        self.boss_position = highest_accuracy_objdata[self.boss_label]

    def reward(self):
        if self.player_position is None or self.boss_position is None:
            # No data from state, neutral reward
            return 0

        if get_overlaps([self.player_position, self.boss_position]):
            # Player and boss have collided, provide negative reward
            return -1
        else:
            # Player and boss haven't collided, player is alive so reward the agent
            return 1

    def get_state(self):

        print("player pos:", self.player_position)
        print("boss pos:", self.boss_position)

        if self.player_position is None or self.boss_position is None or len(self.player_position) == 0 or len(self.boss_position) == 0:
            print('null or empty game data')
            return None
        # Return state in a numpy array
        return np.concatenate([self.player_position, self.boss_position])


# Set current working directory and load model
HOME = os.getcwd()
model = YOLO(f'{HOME}/weights/best.pt')
BE = BossEnvironment()

while True:
    # Test object recognition by getting data from screen
    obj_recognition = OR.ObjectRecognition(model, (0, 40, 640, 480), True, 0.2)
    result = obj_recognition.get_screen_data()
    print(result)

    # Test state building using envioronment class
    BE.observe(result)
    curr_state = BE.get_state()

# Test overlap detection
# all_objects = [[-3, -3, -2, -2], [2, 2, 4, 5]]
# overlaps = get_overlaps(all_objects)
# print("Test Case 1: ([-3, -3, -2, -2], [2, 2, 4, 5])", " - Result : ", overlaps)
#
# all_objects = [[3, 3, 2, 2], [2, 2, 4, 5]]
# overlaps = get_overlaps(all_objects)
# print("Test Case 2: ([3, 3, 2, 2], [2, 2, 4, 5])", " - Result : ", overlaps)
#
# all_objects = [[2, 2, 4, 5], [3, 3, 2, 2]]
# overlaps = get_overlaps(all_objects)
# print("Test Case 3: ([2, 2, 4, 5], [3, 3, 2, 2])", " - Result : ", overlaps)


