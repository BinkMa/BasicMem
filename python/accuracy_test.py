import faiss                   # make faiss available
import time
import numpy as np
import threading
from sklearn.metrics import mean_squared_error


def timeit(func):
    def timed(*args, **kw):
        ts = time.time()
        result = func(*args, **kw)
        te = time.time()
        print('func:%r took: %2.4f sec' % \
          (func.__name__, te-ts))
        return result
    return timed

@timeit
def trans_data(distances):
    new_distances = []
    for row in distances:
        new_row = []
        if len(row)<120:
            new_row.extend(row)
            for i in range(120-len(row)):
                new_row.append(0)
        else:
            for i in range (120):
                new_row.append(row[i])
        new_distances.append(new_row)
    return np.array(new_distances)

@timeit
def open_file(filename):
    matrix=[]
    with open(filename, 'r') as file_object:
        lines = file_object.readlines()
        for line in lines:
            # line=line.strip('\n')
            line=list(line.split(' '))
            line.pop()
            matrix.append(line)
    return matrix

@timeit
def open_key(filename):
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


def MSE2_loss(predX,predY,predZ,realX,realY,realZ):
    sum_error=0
    sum_norm=0

    for i in range(np.shape(predX)[0]):

        sum_error+= np.linalg.norm(predX[i]-realX[i])+np.linalg.norm(predY[i]-realY[i])+np.linalg.norm(predZ[i]-realZ[i])
        sum_norm+=np.linalg.norm(realX[i])+np.linalg.norm(realY[i])+np.linalg.norm(realZ[i])

    return sum_error/sum_norm

@timeit
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

    k = 1  # we want to see 4 nearest neighbors


    folderpath= '../dataset/'
    filename1 = 'result.txt'
    filename2 = 'valueX.txt'
    filename3 = 'valueY.txt'
    filename4 = 'valueZ.txt'

    keyname1='embedding20.txt'
    keyname2='embedding30.txt'
    keyname3='embedding60.txt'


    distances = open_file(folderpath+filename1)
    new_distances = trans_data(distances)
    key20 = trans_value(open_key(folderpath+keyname1))
    key30 = trans_value(open_key(folderpath+keyname2))
    key60 = trans_value(open_key(folderpath+keyname3))
    valueX = open_file(folderpath+filename2)
    valueY = open_file(folderpath+filename3)
    valueZ = open_file(folderpath+filename4)

    lengthX = max(map(len, valueX))
    lengthY = max(map(len, valueY))
    lengthZ = max(map(len, valueZ))

    valueXd = np.array([xi + [0] * (lengthX - len(xi)) for xi in valueX])
    valueYd = np.array([yi + [0] * (lengthY - len(yi)) for yi in valueY])
    valueZd = np.array([zi + [0] * (lengthZ - len(zi)) for zi in valueZ])

    # valueXd=np.load('valueXd.npy').astype(float)
    # valueYd=np.load('valueYd.npy').astype(float)
    # valueZd=np.load('valueZd.npy').astype(float)


    print("loaded data")

    # new_distances = np.array(new_distances).astype('float32')
    key_table = np.array(new_distances).astype('float32')

    cutratio=0.95

    cutoff= round(len(key_table)*cutratio)
    key_train=key_table[:cutoff]
    key_test=key_table[cutoff:]

    #
    quantizer = faiss.IndexFlatL2(len(key_train[0]))   # build the index
    index = faiss.IndexIVFFlat(quantizer, len(key_train[0]), 50)
    index.train(key_train)
    print("index is trained or not : ", index.is_trained)
    index.add(key_train)
    print(index.ntotal)



    start = time.time()
    D, I = index.search(key_test, k)  # sanity check
    end = time.time()

    print("search time : ", end-start)

    resX=valueXd[I].reshape(round(cutoff/(cutratio/(1-cutratio))),-1).astype(float)
    resY=valueYd[I].reshape(round(cutoff/(cutratio/(1-cutratio))),-1).astype(float)
    resZ=valueZd[I].reshape(round(cutoff/(cutratio/(1-cutratio))),-1).astype(float)

    # print(np.shape(resX))
    # print(np.shape(valueXd[cutoff:]))
    # print(resX[:5])

    # print("MSE2 loss of X : ", MSE2_loss(resX, valueXd[cutoff:]))
    # print("MSE2 loss of Y : ", MSE2_loss(resY, valueYd[cutoff:]))
    # print("MSE2 loss of Z : ", MSE2_loss(resZ, valueZd[cutoff:]))
    print("MSE2 loss of X : ", MSE2_loss(resX,resY,resZ, valueXd[cutoff:],valueYd[cutoff:],valueZd[cutoff:]))






