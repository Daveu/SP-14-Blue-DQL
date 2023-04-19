import numpy as np
import object_recognition as o_r


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

        if o_r.get_overlaps([self.player_position, self.boss_position]):
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