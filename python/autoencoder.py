# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
import torch
import torch.nn as nn
import time
import torch.nn.functional as F

class mlp(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(mlp, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x

class autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim[0]),
            nn.ReLU(True),
            nn.Linear(hidden_dim[0], hidden_dim[1]),
            nn.ReLU(True),

            nn.Linear( hidden_dim[1], output_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(output_dim, hidden_dim[1]),
            nn.ReLU(True),
            nn.Linear(hidden_dim[1], hidden_dim[0]),
            nn.ReLU(True),
            nn.Linear(hidden_dim[0], input_dim),
        )

    def forward(self, x):
        e = self.encoder(x)
        x = self.decoder(e)
        return e,x


class DeepAutoencoder(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super(DeepAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim[0]),
            nn.ReLU(True),
            nn.Linear(hidden_dim[0], hidden_dim[1]),
            nn.ReLU(True),
            nn.Linear(hidden_dim[1], hidden_dim[2]),
            nn.ReLU(True),
            nn.Linear(hidden_dim[2], output_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(output_dim, hidden_dim[2]),
            nn.ReLU(True),
            nn.Linear(hidden_dim[2], hidden_dim[1]),
            nn.ReLU(True),
            nn.Linear(hidden_dim[1], hidden_dim[0]),
            nn.ReLU(True),
            nn.Linear(hidden_dim[0], input_dim),
        )

    def forward(self, x):
        e = self.encoder(x)
        x = self.decoder(e)
        return e,x

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

class train_encoder():
    def __init__(self,epoch=150):
        self.epoch = epoch
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    def train(self,X):
        N,dim_in= X.shape

        model= autoencoder(dim_in,[10,5],1).to(self.device)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
        for epoch in range(self.epoch):
            optimizer.zero_grad()
            embedding, output = model(X)
            loss = criterion(output, X)
            loss.backward()
            optimizer.step()
            print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, self.epoch, loss.item()))

        start= time.time()
        embedding, output = model(X)
        end= time.time()
        print("time: ",end-start)
        return embedding.data.numpy(),model


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    folderpath= '../dataset/'
    filename1 = 'result.txt'
    filename2 = 'valueX.txt'
    filename3 = 'valueY.txt'
    filename4 = 'valueZ.txt'

    distances= open_file(folderpath+filename1)
    new_distances=trans_data(distances)

    # valueX= open_file(filename2)
    # valueY= open_file(filename3)
    # valueZ= open_file(filename4)

    input = np.array(new_distances).reshape(-1, 90)
    input = input
    input = torch.from_numpy(input)
    input = input.to(torch.float32)

    embedding,model= train_encoder().train(input)


    print(embedding.shape)
    embedding=embedding.reshape(-1,2)
    with open("embedding90_2.txt",'w') as f:
        for row in embedding:
            for item in row:
                f.write(str(item))
                f.write(' ')
            f.write('\n')

    # print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
