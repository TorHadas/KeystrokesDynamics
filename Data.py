import os
import math
import numpy as np
import re

def get_numbers_data():
    FILe = os.getcwd() + "\\data.txt"
    inputs = []
    answers = []
    ret = []
    with open(FILe, 'r') as file:
        lines = file.readlines()
        for line in lines:
            if (line is not '\n'):
                answer = line.split(";")[0]
                data = line.split(";")[1].split(",")
                data = np.array(data)
                data = np.asfarray(data, float)
                #print("DATA ---- \t" + str(data))
                inputs.append(data)
                answers.append(answer)
                ret.append((int(answer), data))
    return ret

def get_fake_data():
    data = []
    directory = os.getcwd() + "\\data"
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".txt"):
            with open(directory + "\\" + filename, "r") as file:
                first_row = True
                vector = []
                for line in file.readlines():
                    if(first_row):
                        member = int(line)
                        first_row = False
                    else:
                        vector.append(int(line))
                vector = np.array(vector)
                v = vector
                v[v > 0] -= 95
                v[v == 0] = -10
                data.append((member, vector))


    return data

def create_person_data(directory_of_file):
    password = ''
    inputs = []
    with open(directory_of_file, 'r') as file:
        lines = file.readlines()
        password = (lines[0].split('\n'))[0]
        lines = lines[1:]
        for line in lines:
            if (line is not '\n'):
                new_row = re.split('; |,|\*|\n',line)
                line = line.split("\n")[0].split(",")
                data = np.array(line)
                data = np.asfarray(data, float)
                # MEAN
                mean = sum(data) / len(data)
                data -= mean
                data /= mean/20
                # PRESS MEAN
                press_mean = sum([data[2*i] for i in range(math.floor((len(data) + 1)/2))]) / ((len(data) + 1)/2)
                latency_mean = sum([data[2*i + 1] for i in range(math.floor((len(data))/2))]) / (len(data)/2)
                data = np.concatenate((np.array([mean, press_mean, latency_mean]), data))
                #print("DATA ---- \t" + str(data))
                inputs.append(data)


    return [password, inputs]

def create_multiple_members(folder_directory):
    members = []
    for file in os.listdir(folder_directory):
        inserted = False
        file_path = os.path.relpath("logs\\" + file)
        if(file_path == "logs\passwords.txt"):
            continue
        new_member = create_person_data(file_path)
        #for i in range(len(members)):
        #    if new_member[0] == members[i][0]:
        #        inserted = True
        #        members[i][1] += new_member[1]
        if not inserted:
            members.append(new_member)
    return convert_members(members)


def convert_members(members):
    converted_members = []
    for i in range(2):
        #for i in range(len(members)):
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
