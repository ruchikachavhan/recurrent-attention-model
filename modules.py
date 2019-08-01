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

#The network will always return points in the image as tanh function will be applied
def denormalize(location, dim):
	x = (location[0]+1)*dim*0.5
	y = (location[1]+1)*dim*0.5
	loc = torch.tensor([x,y])
	return loc


#Defining the retina
class Retina(object):
	def __init__(self, location, window_size, num_glimpes):
		self.location = location #top left corner
		self.window_size = window_size
		self.num_glimpes = num_glimpes
	def return_patches(self, batch_images, location):
		#image is a tensor of dimensions (batch_size, 1, 28, 28)
		(batch_size, channels, height, width) = batch_images.shape
		out = []
		for b in range(0, batch_size):
			image = batch_images[b][0]
			location[b] = denormalize(location[b], height)
			res = torch.zeros((self.num_glimpes, self.window_size, self.window_size))
			for n in range(0, self.num_glimpes):
				for x in range(0, self.window_size):
					x_coord = location[b][0]+x
					if(x_coord < width-1):
						for y in range(0, self.window_size):
							y_coord = location[b][1]+y
							if(y_coord < height-1):
								res[n][x][y] = image[int(location[b][0]+x)][int(location[b][1]+y)]
			out.append(res)
		out = torch.stack(out, dim = 0)
		return out


class GlimpseNet(nn.Module):
	def __init__(self, h_image, h_loc, shape_, num_glimpses): 
		super(GlimpseNet, self).__init__() 
		self.eye = Retina(h_loc, shape_, num_glimpses)
		self.image_model = nn.Sequential( 
			nn.Linear(12288, h_image),
			nn.ReLU(),
			)
		self.location_model = nn.Sequential(
			nn.Linear(2, h_loc),
			nn.ReLU(),
			)
		self.fc1 = nn.Linear(h_image, h_image + h_loc)
		self.fc2 = nn.Linear(h_loc, h_image + h_loc)
	def forward(self, batch_images,location):
		self.data = self.eye.return_patches(batch_images, location)
		input_ = self.data
		input_ = input_.view(input_.shape[0], -1)
		out_i = self.image_model(input_)
		out_l = self.location_model(location)
		out_i = self.fc1(out_i)
		out_l = self.fc2(out_l)
		out = F.relu(out_i+out_l)
		return out
	
class RNN(nn.Module):
	def __init__(self, image_size, h_hidden):
		super(RNN, self).__init__()
		self.image2hidden = nn.Linear(image_size, h_hidden)
		self.hidden2hidden = nn.Linear(h_hidden, h_hidden)
	def forward(self, image, hidden):
		out_i = self.image2hidden(image)
		out_h = self.hidden2hidden(hidden)
		out = F.relu(out_h + out_i)
		return out

class actionNet(nn.Module):
	def __init__(self, h_hidden, num_classes):
		super(actionNet, self).__init__()
		self.fc1 = nn.Linear(h_hidden, num_classes)
	def forward(self, x):
		out = F.log_softmax(self.fc1(x))
		return out

class locationNet(nn.Module):
	def __init__(self, h_hidden,std_dev):
		super(locationNet, self).__init__()
		self.std_dev = std_dev
		self.fc1 = nn.Linear(h_hidden,2)
	def forward(self, x):
		out = F.tanh(self.fc1(x.detach()))
		noise = torch.zeros_like(out)
		noise.data.normal_(std=self.std_dev)
		loc = out + noise
		loc = F.tanh(loc)
		return out, loc 

class basenet(nn.Module):
	def __init__(self, h_hidden):
		super(basenet, self).__init__()
		self.fc1 = nn.Linear(h_hidden,1)
	def forward(self, x):
		out = F.relu(self.fc1(x.detach()))
		return out



		
	

