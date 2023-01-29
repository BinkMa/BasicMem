
import torch
import torch.nn as nn
import numpy as np
import time

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
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, output_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x):
        e = self.encoder(x)
        x = self.decoder(e)
        return e,x

class autoencoder2(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(autoencoder2, self).__init__()
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

class train_encoder():
    def __init__(self,epoch=100):
        self.epoch = epoch
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    def train(self,X):
        N,dim_in= X.shape
        # model= autoencoder2(dim_in,[100,50],10).to(self.device)
        model= autoencoder(dim_in,10,1).to(self.device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-5)
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


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        print('%r (%r, %r) %2.2f sec' % \
              (method.__name__, args, kw, te-ts))
        return result
    return timed

@timeit
def open_file(filename):
    matrix=[]
    with open(filename, 'r') as file_object:
        lines = file_object.readlines()
        for line in lines:
            matrix.append(list(line.split(' ')))
    return matrix

# def trans_data(distances):
#     new_distances = []
#     for row in distances:
#         new_row = []
#         for idx in range(120):
#             if row[idx] == '' or row[idx] == ' ' or row[idx] == '\n':
#                 for it in range(120-idx):
#                     new_row.append(0)
#                 break
#             else:
#                 new_row.append(float(row[idx]))
#         new_distances.append(new_row)
#     return new_distances

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


def trans_key(distances):

    new_distances = []
    for row in distances:
        new_row = []
        for idx in range(120):
            if row[idx] == '' or row[idx] == '\n':
                for it in range(120-idx):
                    new_row.append(0)
                break
            else:
                new_row.append(float(row[idx]))
        new_distances.append(new_row)
    return np.array(new_distances)


if __name__ == "__main__":
    folderpath= '../dataset/'

    key1file=open_file(folderpath+'key1.txt')
    key2file=open_file(folderpath+'key2.txt')
    key3file=open_file(folderpath+'key3.txt')
    key1=trans_key(key1file)
    key2=trans_key(key2file)
    key3=trans_key(key3file)

    Key_long = np.concatenate((key1, key2, key3), axis=1)
    input =Key_long.reshape(-1, 3)
    input = torch.from_numpy(input)
    input = input.to(torch.float32)

    embedding,model= train_encoder().train(input)

    print(embedding.shape)
    embedding=embedding.reshape(-1,120)
    with open("embedding120.txt",'w') as f:
        for row in embedding:
            for item in row:
                f.write(str(item))
                f.write(' ')
            f.write('\n')


