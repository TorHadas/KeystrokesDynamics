import os

class Data():

    def __init__(self, directory_in_str):
        members_data = []
        directory = os.fsencode(directory_in_str)

        for file in os.listdir(directory):
            filename = os.fsdecode(file)
            if filename.endswith(".txt"):
                with open(filename, "r") as file:
                    first_row = True
                    vector = []
                    for line in file.readlines():
                        words = line.split()
                        if(first_row):
                            member = words
                            first_row = False
                        else:
                            vector.append(words)
                    members_data.append((member, vector))
            else:
                continue
