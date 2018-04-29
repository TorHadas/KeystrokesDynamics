import os

class Data():
    members_data = []
    def __init__(self, directory_in_str):
        
        #directory = os.listdir(directory_in_str)

        for file in os.listdir(directory_in_str):
            with open(directory_in_str + "\\" + file, "r") as file:
                first_row = True
                vector = []
                for line in file.readlines():
                    vector.append(int(line))
                self.members_data.append(vector)
