import os
import math
import numpy as np
import re

def get_pictures():
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

#return the password and the times for a file
def readfile(filename):
    password = ''
    vectors = []
    with open(filename, 'r') as file:
        lines = file.readlines()
        password = (lines[0].split('\n'))[0]
        for line in lines[1:]:
            if (line is '\n'):
                continue
            vec = line.split("\n")[0].split(",")
            if(len(vec) < 15):
                print("OMG")
                print(filename)
            vectors.append(normalize(vec))
    return [password, vectors]

#change the initial vector to a normilized vector
def normalize(line):
    data = np.array(line)
    data = np.asfarray(data, float)
    # MEAN
    mean = sum(data) / len(data)
    data -= mean
    data /= mean/20
    # PRESS MEAN
    press_mean = sum([data[2*i] for i in range(math.floor((len(data) + 1)/2))]) / ((len(data) + 1)/2)
    latency_mean = sum([data[2*i + 1] for i in range(math.floor((len(data))/2))]) / (len(data)/2)
    return np.concatenate((np.array([mean, press_mean, latency_mean]), data))

def get_data():
    return get_users(str(os.getcwd())+"\\logs\\")

def get_users(folder):
    users = []
    passwords = []
    for file in os.listdir(folder):
        inserted = False
        file_path = "logs\\" + file
        if(file_path == "logs\\passwords.txt" or file_path == "logs\\fake"):
            continue
        user = readfile(file_path)
        user = [user[0], user[1], []]
        if(user[0] in passwords):
            for i in range(len(users)):
                if(user[0] == users[i]):
                    users[i][2] += user[1]
                    break
            continue
        passwords.append(user[0])
        users.append(user)
    users = get_fake_users(folder + "fake", users)
    #users = [to_tuples(users[i]) for i in range(len(users))]
    return users

def get_fake_users(folder, users):
    for file in os.listdir(folder):
        file_path = "logs\\fake\\" + file
        imposter = readfile(file_path)
        for i in range(len(users)):
            if imposter[0] == users[i][0]:
                users[i][2] += imposter[1]
    return users


# def to_tuples(users):
#     converted_users = []
#     for i in range(2):
#         for j in range(len(users[i][1])):
#             converted_users.append((i, users[i][1][j]))
#     return converted_users

def to_tuples(user):
    data = []
    for i in range(2):
        for j in range(len(user[i + 1])):
            data.append((i, user[i + 1][j]))
    return data

#file_dir = "C:\\Users\\T8497069\\Desktop\\bloch 1.txt"
#person1_data = readfile(file_dir)
#print(len(person1_data))
#print(len(person1_data[1]))
#print(person1_data[1])
#for i in person1_data[1]:
#    print(len(i))
