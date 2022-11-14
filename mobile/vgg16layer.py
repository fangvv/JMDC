import math
import torch
import torch.nn as nn
from torch.autograd import Variable

# num_classes=10
#
# __all__ = ['vgg']
#
# defaultcfg = {
# 	11 : [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
# 	13 : [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
# 	16 : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512],
# 	19 : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512],
# }
#
# class vgg(nn.Module):
# 	def __init__(self, dataset='cifar10', depth=16, init_weights=True, cfg=None):
# 		super(vgg, self).__init__()
# 		if cfg is None:
# 			cfg = defaultcfg[depth]
#
# 		self.cfg = cfg
#
# 		self.feature = self.make_layers(cfg, True)
#
# 		if dataset == 'cifar10':
# 			num_classes = 10
# 		elif dataset == 'cifar100':
# 			num_classes = 100
# 		self.classifier = nn.Sequential(
# 			  nn.Linear(cfg[-1], 512),
# 			  nn.BatchNorm1d(512),
# 			  nn.ReLU(inplace=True),
# 			  nn.Linear(512, num_classes)
# 			)
# 		if init_weights:
# 			self._initialize_weights()
#
# 	# def make_layers(self, cfg, batch_norm=False):
# 	def make_layers(self, cfg, batch_norm=True):
# 		layers = []
# 		in_channels = 3
# 		for v in cfg:
# 			if v == 'M':
# 				layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
# 			else:
# 				conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1, bias=False)
# 				if batch_norm:
# 					layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
# 				else:
# 					layers += [conv2d, nn.ReLU(inplace=True)]
# 				in_channels = v
# 		return nn.Sequential(*layers)
NUM_CLASSES = 10
class VGG16Layer(nn.Module):
	def __init__(self, num_classes=NUM_CLASSES):
		super(VGG16Layer, self).__init__()
		self.features = nn.Sequential(
			nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
			nn.BatchNorm2d(64),
			nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
			nn.BatchNorm2d(64),
			nn.MaxPool2d(kernel_size=2, stride=2),
			nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
			nn.BatchNorm2d(128),
			nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
			nn.BatchNorm2d(128),
			nn.MaxPool2d(kernel_size=2, stride=2),
			nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
			nn.BatchNorm2d(256),
			nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
			nn.BatchNorm2d(256),
			nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
			nn.BatchNorm2d(256),
			nn.MaxPool2d(kernel_size=2, stride=2),
			nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
			nn.BatchNorm2d(512),
			nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
			nn.BatchNorm2d(512),
			nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
			nn.BatchNorm2d(512),
			nn.MaxPool2d(kernel_size=2, stride=2),
			nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
			nn.BatchNorm2d(512),
			nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
			nn.BatchNorm2d(512),
			nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
			nn.BatchNorm2d(512),
			nn.MaxPool2d(kernel_size=2, stride=2),
		)
		self.classifier = nn.Sequential(
			nn.Linear(512, 4096),
			nn.Linear(4096, 4096),
			nn.Linear(4096, NUM_CLASSES),
		)
	def forward(self, x):
		x = self.features(x)
		x = x.view(x.size(0), 2*2*128)
		x = self.classifier(x)
		return x

	def forward(self, x, startLayer, endLayer, isTrain):
		if isTrain:
			x = self.features(x)
			x = x.view(x.size(0), 512)
			x = self.classifier(x)
		else:
			if startLayer==endLayer:
				if startLayer==31:
					x = x.view(x.size(0), 512)
					x = self.classifier[startLayer-31](x)
				elif startLayer<31:
					x = self.features[startLayer](x)
				else:
					x = self.classifier[startLayer-31](x)
			else:
				for i in range(startLayer, endLayer+1):
					if i<31:
						x = self.features[i](x)
					elif i==31:
						x = x.view(x.size(0), 512)
						x = self.classifier[i-31](x)
					else:
						x = self.classifier[i-31](x)
		return x
model=VGG16Layer()
print(model)