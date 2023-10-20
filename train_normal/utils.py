import pandas as pd
import torch
import json
from easydict import EasyDict as edict
import os
import numpy as np
import random
from torch import distributed as dist

class Recorder:
	def __init__(self, name):
		self.metrics = {}
		self.name = name
	def record(self, metrics_name, index, metrics):
		if metrics_name not in self.metrics:
			self.metrics[metrics_name] = ([index,], [metrics,])
		else:
			self.metrics[metrics_name][0].append(index)
			self.metrics[metrics_name][1].append(metrics)
	def save(self, metrics_names, dataset, distribution):
		save_dir = '../res/convergence/{}_{}_{}.csv'.format(self.name, dataset, distribution)
		data = pd.DataFrame(index=self.metrics[metrics_names[0]][0])
		for metrics_name in metrics_names:
			data[metrics_name] = self.metrics[metrics_name][1]
		data.to_csv(save_dir)
	def clear(self):
		self.metrics = {}
	def save_test(self, metrics_name, dataset, metrics, distribution):
		data = pd.read_csv('../res/res_{}_{}.csv'.format(metrics_name, distribution), index_col=0).copy()
		data[dataset][self.name] = metrics
		data.to_csv('../res/res_{}_{}.csv'.format(metrics_name, distribution), index=True)
	def save_model(self, method, model, dataset, c_type):
		if not os.path.exists('../trained_models/{}/{}'.format(method, c_type)):
			os.makedirs('../trained_models/{}/{}'.format(method, c_type))
		torch.save(model, '../trained_models/{}/{}/{}.pth'.format(method, c_type, dataset))
#		with open('../trained_models/{}/{}_{}/{}.json'.format(dataset, metrics_name, metrics, method), 'w') as f:
#			json.dump(hyperparam, f)
	def read_model(self, method, model, dataset, c_type):
		model = torch.load('../trained_models/{}/{}/{}.pth'.format(method, c_type, dataset))
		return model
			

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def load_param(method, dataset, TPR, FPR):
    if TPR == 1:
        metric = 'OPAUC'
    else:
        metric = 'TPAUC'
    with open('../trained_models/{}/{}_{}/{}.json'.format(dataset, metric, FPR, method), 'r') as f:
        args = json.load(f)
        return args

def load_json(dataset=None, model=None):
	with open('../configs/base_config.json', 'r') as f:
		args = json.load(f)
		args = edict(args)
	if dataset is not None:
		with open('../configs/datasets/%s.json'%dataset, 'r') as f:
			args.dataset.update(edict(json.load(f)))
	if model is not None:
		with open('../configs/models/%s.json'%model, 'r') as f:
			args.model.update(edict(json.load(f)))
	return args