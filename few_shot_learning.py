# Hyperparameters
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as dset
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import os	
import time
import pickle
from siamese_nets import Siamese_Network
from omniglot_dset import OmniglotTrain, OmniglotTest
import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt
from torch.autograd import Variable


num_epochs = 200
batch_size = 100
learning_rate = 0.001

MODEL_STORE_PATH = './model'

trans = transforms.Compose([
    transforms.RandomAffine(15),
    transforms.ToTensor()
])

train_path = 'images_background'
test_path = 'images_evaluation'
train_dataset = dset.ImageFolder(root=train_path)
test_dataset = dset.ImageFolder(root=test_path)



def weight_init(shape, m):
    """Initialize weights as specified in the paper"""
    values_w = rng.normal(loc=0,scale=1e-2,size=shape)
    values_b=rng.normal(loc=0.5,scale=1e-2,size=shape)
    m.weight.data.normal_(values_w)
    m.bias.data.fill_(values_b)


N = 20

dataSet = OmniglotTrain(train_dataset, transform=trans)
testSet = OmniglotTest(test_dataset, transform=transforms.ToTensor(), times = num_epochs, way = N)

testLoader = DataLoader(testSet, batch_size=N, shuffle=False, num_workers=16)

dataLoader = DataLoader(dataSet, batch_size=128,\
                        shuffle=False, num_workers=16)

#Loss specified in the paper
loss_value = torch.nn.BCEWithLogitsLoss(size_average=True)

learning_rate = 0.0001

model = Siamese_Network()

training_loss = []
model.train()
optimizer = torch.optim.Adam(model .parameters(),lr = learning_rate )
optimizer.zero_grad()

show_every = 10
save_every = 100
test_every = 100
train_loss = []
loss_val = 0
max_iter = 90000


for batch_id, (img1, img2, label) in enumerate(dataLoader, 1):
	if batch_id > max_iter:
		break
	batch_start = time.time()
	img1, img2, label = Variable(img1), Variable(img2), Variable(label)
	optimizer.zero_grad()
	output = model.forward(img1, img2)
	loss = loss_fn(output, label)
	loss_value += loss.data[0]
	loss.backward()
	optimizer.step()
	if batch_id % show_every == 0 :
		print('[%d]\t : Loss:\t%.5f\tTook\t%.2f s'%(batch_id, loss_value/show_every, (time.time() - batch_start)*show_every))
		loss_value = 0
	if batch_id % save_every == 0:
		torch.save(net.state_dict(), './model/model-batch-%d.pth'%(batch_id+1,))
	if batch_id % test_every == 0:
		right, error = 0, 0

	for _, (test1, test2) in enumerate(testLoader, 1):

		test1, test2 = Variable(test1), Variable(test2)
		output = net.forward(test1, test2).data.cpu().numpy()
		pred = np.argmax(output)
		if pred == 0:
			right += 1
		else: error += 1
		print('*'*70)
		print('[%d]\tright:\t%d\terror:\t%d\tprecision:\t%f'%(batch_id, right, error, right*1.0/(right+error)))
		print('*'*70)
	train_loss.append(loss_value)
#  learning_rate = learning_rate * 0.95

with open('train_loss', 'wb') as f:
    pickle.dump(train_loss, f)

