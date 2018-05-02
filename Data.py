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

    return [password, timing_array]

def create_multiple_members(folder_directory):
    members = []
    for file in os.listdir(folder_directory):
        inserted = False
        file_path = os.path.relpath("logs\\" + file)
        print(file_path)
        if(file_path == "logs\passwords.txt"):
            continue
        new_member = create_person_data(file_path)
        for i in range(len(members)):
            if new_member[0] == members[i][0]:
                inserted = True
                members[i][1] += new_member[1]
        if not inserted:
            members.append(new_member)
    return convert_members(members)


def convert_members(members):
    converted_members = []
    for i in range(len(members)):
        for j in range(len(members[i][1])):
            converted_members.append((i, members[i][1][j]))
    return converted_members

#file_dir = "C:\\Users\\T8497069\\Desktop\\bloch 1.txt"
#person1_data = create_person_data(file_dir)
#print(len(person1_data))
#print(len(person1_data[1]))
#print(person1_data[1])
#for i in person1_data[1]:
#    print(len(i))



folder_dir = 'C:\\Users\\T8497069\\Desktop\\Smop\\KeystrokesDynamics\\logs'
