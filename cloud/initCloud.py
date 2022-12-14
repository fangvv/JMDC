import numpy as np
import argparse
import torch
import torch.nn as nn
from torchvision import transforms
import torch.utils.data as Data
from data import get_data_set
import socket
import threading
import pickle
import io
import sys
import time

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
# parser.add_argument('--arch', '-a', metavar='ARCH', default='vgg19',
#                     choices=model_names,
#                     help='model architecture: ' + ' | '.join(model_names) +
#                     ' (default: vgg19)')
parser.add_argument('--dataset', type=str, default='cifar10',
                    help='training dataset (default: cifar100)')
parser.add_argument('--depth', default=16, type=int,
                    help='depth of the neural network')
parser.add_argument('--refine', default='', type=str, metavar='PATH',
                    help='需要加载的预训练好的模型的路径')
args = parser.parse_args()

_IMG_SIZE = 32
_NUM_CHANNELS = 3
_BATCH_SIZE = 128
_CLASS_SIZE = 10

ALEXNET_MODEL_PATH="model/alexnetlayermodel.pkl"
VGG16_MODEL_PATH="model/vgg16layermodel.pkl"

IP="192.168.123.10"
PORT=8081

class Data(object):

	def __init__(self, inputData, startLayer, endLayer):
		self.inputData=inputData
		self.startLayer=startLayer
		self.endLayer=endLayer

def run(model, inputData, startLayer, endLayer):
	print("云端运行%d到%d层" % (startLayer, endLayer))
	outputs = model(inputData, startLayer, endLayer, False)
	return outputs

def sendData(server, inputData, startLayer, endLayer):
	data=Data(inputData, startLayer, endLayer)
	str=pickle.dumps(data)
	server.send(len(str).to_bytes(length=6, byteorder='big'))
	server.send(str)

def receiveData(server, model):
	while True:
		conn,addr=server.accept()
		while True:
			lengthData=conn.recv(6) #接收一个信息，并指定接收的大小为6字节
			length=int.from_bytes(lengthData, byteorder='big')
			b=bytes()
			if length==0:
				continue
			count=0
			while True:
				value=conn.recv(length)
				b=b+value
				count+=len(value)
				if count>=length:
					break
			data=pickle.loads(b)
			outputs=run(model, data.inputData, data.startLayer, data.endLayer)
			sendData(conn, outputs, data.endLayer+1, 1)

if __name__=="__main__":
	# model=torch.load(ALEXNET_MODEL_PATH, map_location='cpu')
	checkpoint = torch.load(args.refine)
	model = models.__dict__[args.arch](dataset=args.dataset, depth=args.depth, cfg=checkpoint['cfg'], map_location='cpu')
	model.load_state_dict(checkpoint['state_dict'])
	device = torch.device("cpu")
	torch.set_num_threads(3)
	test_x,test_y,test_l=get_data_set("test")
	test_x=torch.from_numpy(test_x[0:10]).float()
	test_y=torch.from_numpy(test_y[0:10]).long()
	print("模型加载成功")
	server=socket.socket(socket.AF_INET, socket.SOCK_STREAM)
	server.setblocking(1)
	server.bind((IP, PORT))
	print("云端启动，准备接受任务")
	server.listen(1)
	receiveData(server, model)




