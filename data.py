#Loading train data and val data
from torch.autograd import Variable
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision
import torch
def train_val_data(batch_size, shuffle, num_workers, valid_size):
	normalize = transforms.Normalize((0.1307,), (0.3081,))
	transform = transforms.Compose([transforms.ToTensor(), normalize,])
	dataset = torchvision.datasets .MNIST(root = './data', download = 1, transform = transform) 
	dataloader = torch.utils.data.DataLoader(dataset, batch_size = batch_size, shuffle = shuffle, num_workers = num_workers)
	num_train = len(dataset)
	indices = list(range(num_train))
	split = int(np.floor(valid_size * num_train))
	if shuffle:
		np.random.shuffle(indices)

	train_idx, valid_idx = indices[split:], indices[:split]

	train_sampler = SubsetRandomSampler(train_idx)
	valid_sampler = SubsetRandomSampler(valid_idx)

	train_loader = torch.utils.data.DataLoader(
	    dataset, batch_size=batch_size, sampler=train_sampler,
	    num_workers=num_workers,
	)

	valid_loader = torch.utils.data.DataLoader(
	    dataset, batch_size=batch_size, sampler=valid_sampler,
	    num_workers=num_workers,
	)
	return train_loader, valid_loader



