import numpy as np
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import faiss                   # make faiss available
import time
import sys

def trans_data(distances):
    new_distances = []
    for row in distances:
        new_row = []
        # if (len(row) > 120):
        #     flag = "long"
        # else:
        #     flag = "short"
        for idx in range(120):
            if row[idx] == '' or row[idx] == ' ' or row[idx] == '\n':
                for it in range(120-idx):
                    new_row.append(0)
                break
            else:
                new_row.append(float(row[idx]))
        new_distances.append(new_row)
    return new_distances

def open_file(filename):
    matrix=[]
    with open(filename, 'r') as file_object:
        lines = file_object.readlines()
        for line in lines:
            matrix.append(list(line.split(' ')))
    return matrix


def trans_value(value):
    new_value=[]
    for row in value:
        new_row = []
        for item in row:
            if item == '' or item == '\n':
                continue
            else:
                new_row.append(item)

        # row.remove('\n')

        new_value.append(new_row)
    return new_value

if __name__ == "__main__":
    folderpath= '../dataset/'
    filename1 = 'result.txt'
    filename2 = 'embedding20.txt'
    filename3 = 'embedding30.txt'
    filename4 = 'embedding60.txt'

    # filename2 = 'valueX.txt'
    # filename3 = 'valueY.txt'
    # filename4 = 'valueZ.txt'


    distances = open_file(folderpath+filename1)
    new_distances = trans_data(distances)

    #
    # valueX = open_file(filename2)
    # valueY = open_file(filename3)
    # valueZ = open_file(filename4)
    key20 = trans_value(open_file(folderpath+filename2))
    key30 = trans_value(open_file(folderpath+filename3))
    key60 = trans_value(open_file(folderpath+filename4))


    key_table = np.unique(np.array(new_distances)).reshape(-1,1).astype('float32')


    key_train=key_table[:round(len(key_table)*0.9)]
    key_test=key_table[round(len(key_table)*0.9):]

    quantizer = faiss.IndexFlatL2(len(key_train[0]))   # build the index
    index = faiss.IndexIVFFlat(quantizer, len(key_train[0]), 50)



    index.train(key_table)
    print(index.is_trained)
    index.add(key_table)
    print(index.ntotal)
    k = 1  # we want to see 4 nearest neighbors

    start = time.time()
    D, I = index.search(key_table, k)  # sanity check
    end = time.time()
    #
    # print("the index of the sanity check is: ",I)
    # print("the distance of the sanity check is: ",D)
    print("time: ",end-start)





















