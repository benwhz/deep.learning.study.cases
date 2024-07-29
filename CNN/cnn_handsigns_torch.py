import numpy as np
import h5py
import matplotlib.pyplot as plt
import os
import torch
from torch import nn
import torch.utils.data as td

from cnn_utils import *

np.random.seed(1)
X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset('signs')
len, h, w, ch = X_train_orig.shape[0], X_train_orig.shape[1], X_train_orig.shape[2], X_train_orig.shape[3]
X_train_orig = X_train_orig.reshape(len, ch, h, w)
X_test_orig = X_test_orig.reshape(X_test_orig.shape[0], ch, h, w)

index = 10
#print(X_train_orig[0, 0, :, :]/255)
#print(Y_train_orig[:, 0])
plt.imshow(X_test_orig[index].reshape(h, w, ch))
plt.show()
#print ("y = " + str(np.squeeze(Y_train_orig[:, index])))

X_train = X_train_orig/255.
X_test = X_test_orig/255.

Y_train = Y_train_orig.reshape(Y_train_orig.shape[1])
Y_test = Y_test_orig.reshape(Y_test_orig.shape[1])

#Y_train = convert_to_one_hot(Y_train_orig, 6).T
#Y_test = convert_to_one_hot(Y_test_orig, 6).T
print ("number of training examples = " + str(X_train.shape[0]))
print ("number of test examples = " + str(X_test.shape[0]))
print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(Y_train.shape))
print ("X_test shape: " + str(X_test.shape))
print ("Y_test shape: " + str(Y_test.shape))

net = nn.Sequential(
        nn.Conv2d(3, 8, kernel_size=5, padding=2), nn.BatchNorm2d(8), nn.ReLU(),
        nn.MaxPool2d(kernel_size=8, stride=4),
        nn.Conv2d(8, 16, kernel_size=2), nn.BatchNorm2d(16), nn.ReLU(),
        nn.MaxPool2d(kernel_size=4, stride=4),
        nn.Flatten(),
        nn.Linear(144, 72), nn.BatchNorm1d(72), nn.ReLU(),
        nn.Linear(72, 6)
    )

X = torch.rand(size=(10, 3, 64, 64), dtype=torch.float32)
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__,'output shape: \t',X.shape)

def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        nn.init.xavier_uniform_(m.weight)

net.apply(init_weights)
    
def load_array(batch_size):
    print(torch.from_numpy(X_train).shape, torch.from_numpy(Y_train).shape)
    train_dataset = td.TensorDataset(torch.from_numpy(X_train).to(torch.float32), torch.from_numpy(Y_train))
    test_dataset = td.TensorDataset(torch.from_numpy(X_test).to(torch.float32), torch.from_numpy(Y_test))
    return (td.DataLoader(train_dataset, batch_size, shuffle=True), td.DataLoader(test_dataset, batch_size, shuffle=False))

def accuracy(y_hat, y):  #@save
    """计算预测正确的数量"""
    if y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())

lr = 0.08
num_epochs = 10
batch_size = 60
train_iter, test_iter = load_array(batch_size=batch_size)

optimizer = torch.optim.SGD(net.parameters(), lr=lr)
loss = nn.CrossEntropyLoss()

data = np.zeros((2, num_epochs))

for epoch in range(num_epochs):
    net.train()
    
    minibatch_cost = 0.
    accuracy_count = 0
    batch_count = 0
    
    for i, (X, y) in enumerate(train_iter):
        optimizer.zero_grad()
        y_hat = net(X)
        #print(f'y_hat = {y_hat.shape}, y = {y.shape}')
        l = loss(y_hat, y)
        l.backward()
        batch_count += 1
        with torch.no_grad():        
            minibatch_cost += l
        accuracy_count += accuracy(y_hat, y)
        optimizer.step()
    
    with torch.no_grad():        
        #print(minibatch_cost, accuracy_count)
        data[0, epoch] = minibatch_cost/batch_count
        data[1, epoch] = accuracy_count/(y.numel()*batch_count)
        print(f'no. {epoch} epoch training ... loss = {data[0, epoch]}, accuracy = ', data[1, epoch])
    
plt.plot(np.arange(num_epochs), data[0])
plt.plot(np.arange(num_epochs), data[1])
plt.show()    
    
with torch.no_grad():
    x = torch.from_numpy(X_test).to(torch.float32)[index:index+2,:,:,:]
    y = net(x)        
    print(x.shape, y.shape)
    print(y, Y_test[index:index+2], torch.softmax(y[0,:], 0))