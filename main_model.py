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


class RecurrentAttention(nn.Module):
	def __init__(self, window_size, num_glimpses, h_image, h_loc, h_hidden,num_classes, std_dev):
		super(RecurrentAttention, self).__init__() 
		self.window_size = window_size
		self.num_glimpses = num_glimpses
		self.h_image = h_image
		self.h_loc = h_loc
		self.h_hidden = h_hidden
		self.num_classes = num_classes
		self.std_dev = std_dev
		self.glimpse_net = GlimpseNet(self.h_image, self.h_loc, self.window_size*self.window_size, self.num_glimpses)
		self.rnn = RNN(self.h_image+self.h_loc, self.h_hidden)
		self.action_net = actionNet(self.h_hidden, self.num_classes)
		self.loc_net = locationNet(self.h_hidden, self.std_dev)
		self.base_net = basenet(self.h_hidden)
	def forward(self, batch_images, location, hidden_prev, last):
		out_glimpse = self.glimpse_net(batch_images, location)
		hidden = self.rnn(out_glimpse, hidden_prev)
		mu, l_t = self.loc_net(hidden)
		b_t = self.base_net(hidden).squeeze()

		log_pi = Normal(mu, self.std_dev).log_prob(l_t)
		log_pi = torch.sum(log_pi, dim=1)

		if last:
		    log_probas = self.action_net(hidden)
		    return hidden, l_t, b_t, log_probas, log_pi

		return hidden, l_t, b_t, log_pi



