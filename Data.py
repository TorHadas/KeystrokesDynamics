import os
import numpy as np
import re
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

def create_person_data(directory_of_file):
    password = ''
    timing_array = []
    with open(directory_of_file, 'r') as file:
        first_Row = True
        lines = file.readlines()
        last = lines[-1]
        for line in lines:
            if(first_Row):
                password = line.split('\n')
                password = password[0]
                first_Row = False
            else:
                if (line is not '\n'):
                    new_row = re.split('; |,|\*|\n',line)
                    if line is not last:
                        new_row = new_row[0:-1]
                    corrected_row = np.array(new_row)
                    corrected_row = np.asfarray(corrected_row, float)
                    timing_array.append(corrected_row)

    return password, timing_array



#file_dir = "C:\\Users\\T8497069\\Desktop\\bloch 1.txt"
#person1_data = create_person_data(file_dir)
#print(len(person1_data))
#print(len(person1_data[1]))
#print(person1_data[1])
#for i in person1_data[1]:
#    print(len(i))