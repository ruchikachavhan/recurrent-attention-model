from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from torch.distributions import Normal
from modules import GlimpseNet, RNN, actionNet, locationNet, basenet
from main_model import RecurrentAttention
import torch.optim as optim
import cv2
import imageio
import matplotlib
import matplotlib.pyplot as plt

def denormalize(location, dim):
	x = (location[0]+1)*dim*0.5
	y = (location[1]+1)*dim*0.5
	loc = torch.tensor([x,y])
	return loc

def give_images(image, l_t, window_size):
	scenes = []
	for b in range(0, image.shape[0]):
		save_image = torch.reshape(image[b].detach(), (28,28,1)).numpy()
		save_image = save_image *255
		img = cv2.cvtColor(save_image, cv2.COLOR_GRAY2RGB)
		l_t[b] = denormalize(l_t[b], 28)
		cv2.rectangle(img,(l_t[b][0], l_t[b][1]), (l_t[b][0]+ window_size, l_t[b][1]+window_size), (0,255,0),1)
		scenes.append(img)
	return scenes

def one_hot(num_classes, labels):
	out = torch.zeros((labels.shape[0], 1, num_classes))
	for i in range(0, labels.shape[0]):
		out [i][0][labels[i]]= 1
	return out
class Trainer(object):
	def __init__(self, config, data_loader):
		self.config = config
		self.window_size = config.window_size
		self. num_glimpses = config.num_glimpses
		self.num_patches = config.num_patches
		self.h_image = config.h_image
		self.h_loc = config.h_loc 
		self.h_hidden = config.h_hidden
		self.num_classes = config.num_classes
		self.std_dev = config.std_dev
		self.M = config.M 
		self.epochs = config.epochs
		self.momentum = config.momentum
		self.lr = config.init_lr
		self.data = data_loader
		self.batch_size = config.batch_size
		self.RAMmodel = RecurrentAttention(self.window_size, self.num_glimpses, self.h_image, self.h_loc, self.h_hidden, self.num_classes, self.std_dev)
		self.optimizer = optim.Adam(self.RAMmodel.parameters(), lr=self.lr)
	def reset(self):
		h_t = torch.zeros(self.batch_size, self.h_hidden)
		h_t = Variable(h_t)
		l_t = torch.Tensor(self.batch_size, 2).uniform_(-1, 1)
		l_t = Variable(l_t)

		return h_t, l_t
	def train_epoch(self, epoch):
		loss_epoch = 0
		loss_list = []
		for iter, (image, label) in  enumerate(self.data):
			h_t, l_t = self.reset()
			log_probas = 0
			y = one_hot(self.num_classes, label).long()
			scenes = []
			for num in range(0, self.num_patches):
				locs = []
				log_pi = []
				baselines = []
				h_t, l_t, b_t, log_probas, p = self.RAMmodel.forward(image.detach(), l_t.detach(), h_t.detach(), last=True)
				log_pi.append(p)
				baselines.append(b_t)
				locs.append(l_t)
				# convert list to tensors and reshape
				baselines = torch.stack(baselines)
				log_pi_ = torch.stack(log_pi)
				# calculate reward
				predicted = torch.max(log_probas, 1)[1]
				R = (predicted.detach() == label).float()
				

				loss_action = F.nll_loss(log_probas,label)
				loss_baseline = F.mse_loss(baselines, R)

				# compute reinforce loss
				# summed over timesteps and averaged across batch
				adjusted_reward = R - baselines.detach()
				loss_reinforce = torch.sum(-log_pi_*adjusted_reward, dim=1)
				loss_reinforce = torch.mean(loss_reinforce, dim=0)

				# sum up into a hybrid loss
				loss = loss_action + loss_baseline + loss_reinforce

				print("LOSS", loss)
				loss.backward()
				self.optimizer.step()
				result_images = give_images(image, l_t, self.window_size)
				for rr in range(0, len(result_images)):
					scenes.append(result_images[rr])
			name = "vid"+ str(epoch) + "_" + str(iter)+ ".gif"
			imageio.mimsave(name, scenes)
		return loss_epoch/(iter+1)
	def train(self):
		for epoch in range(0, self.epochs):
			loss = self.train_epoch(epoch)
			#Printing the loss
			print("epoch", epoch)
			print("loss", loss)

			

					




