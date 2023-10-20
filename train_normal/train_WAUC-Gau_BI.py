import sys
import os
import copy
sys.path.append(os.pardir)
import numpy as np
from losses.WAUCCOST import WAUCGau
from optimizer.BIOPT import BIOPT
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
from dataloaders import get_datasets
from dataloaders import get_data_loaders
from models import generate_net
from metrics.COSTAUC import COSTAUCMetric
from utils import Recorder, load_json, set_seed
import json
set_seed(11)

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

method='WAUC-Gau'
c_type='Normal'

p_pos = {
	'cifar-10-long-tail-1': 0.144, 'cifar-10-long-tail-2': 0.241, 'cifar-10-long-tail-3': 0.086, 
	'cifar-100-long-tail-1': 0.127, 'cifar-100-long-tail-2': 0.058, 'cifar-100-long-tail-3': 0.077, 
	'tiny-imagenet-200-1': 0.03, 'tiny-imagenet-200-2': 0.02, 'tiny-imagenet-200-3': 0.06
}

# hyper parameters
hyper_param = {
	'mini-batch':    256,
	'weight_decay':  1e-5,
	'init_lr':       0.005,
	'C':             50,
	'M':             2048,
	'M_':            1024,
	'kappa':         10
}


rec = Recorder(method) # record the metrics during training
sigmoid = nn.Sigmoid() # Limit the output score between 0 and 1

for dataset in ['cifar-10-long-tail-1','cifar-10-long-tail-2','cifar-10-long-tail-3',
			'cifar-100-long-tail-1','cifar-100-long-tail-2','cifar-100-long-tail-3',]:
	print(dataset)
	
	# load data and dataloader
	args = load_json(dataset)
	train_set, val_set, test_set = get_datasets(args.dataset)
	train_loader, val_loader, test_loader, data_num = get_data_loaders(
	  train_set,
	  val_set,
	  test_set,
	  hyper_param['mini-batch'],
	  hyper_param['mini-batch']
	)

     # load model (train model from the scratch, using model: resnet18)
	args = load_json(dataset, 'resnet18')
	args.model['pretrained'] = None
	model = generate_net(args.model).cuda()
	model = nn.DataParallel(model)

	# define loss and optimizer
	c = torch.clip(torch.normal(0.5, 1, (1, hyper_param['C'])).cuda(), 0, 1)
	losses = WAUCGau(hyper_param['C'], p_pos[dataset], c)
	criterion = COSTAUCMetric(hyper_param['C'], p_pos[dataset], c.cpu().detach().numpy())
	optim_outter = BIOPT([{'params': model.parameters(), 'clip':(-1, 1)}], hyper_param=hyper_param)
	optim_inner = torch.optim.SGD([{'params': losses.tau}], lr=0.001)
	# some basic variable
	best_model = model.state_dict()
	best_perf = 1
	switch_type = 'inner_gradient'
	T = 5
	# train 10 epoch
	for epoch in range(10):
		model.train()
		for i, (img, lbl, idx) in enumerate(train_loader):
			optim_outter.zero_grad()	
			optim_inner.zero_grad()		
			img = img.cuda()
			lbl = lbl.cuda().float()
			out = sigmoid(model(img))
			if switch_type == 'inner_gradient':
				inner_loss = losses.inner_loss(out, lbl)
				inner_loss.backward()
				optim_inner.step()
				with torch.no_grad():
					losses.tau.clip_(0, 1)
				T -= 1
				if T == 0:
					switch_type = 'outter_gradient'
					optim_outter.zero_grad()
					optim_inner.zero_grad()
					out = sigmoid(model(img))
					inner_loss = losses.inner_loss(out, lbl)
					grad_g_tau = torch.autograd.grad(inner_loss, losses.tau, 
						        retain_graph=True,  create_graph=True)[0]
					grad_g_tau_tau = torch.autograd.grad(grad_g_tau, losses.tau, 
						        retain_graph=True,
								grad_outputs=torch.ones_like(losses.tau))[0]
					
					for p in list(model.parameters()):
						if p.grad is not None:
							optim_outter.state[p]["grad_g_p_tau"] = torch.autograd.grad(
                                grad_g_tau, p, retain_graph=True,
                                grad_outputs=torch.ones_like(losses.tau)
                            )[0]

							optim_outter.state[p]["grad_g_tau_tau"] = grad_g_tau_tau
			else:
				outter_loss = losses.outter_loss(out, lbl)
				outter_loss.backward()
				# losses.tau.grad.data.zero_()
				grad_f_tau = losses.tau.grad
				optim_outter.step(
					t=epoch, grad_f_tau=grad_f_tau
				)
				switch_type = 'inner_gradient'
				T = 5
				
		# record instances' prediction and label of val set
		model.eval()
		val_pred = np.array([])
		val_label = np.array([])
		for i, (img, lbl, idx) in enumerate(val_loader):
			img = img.cuda()
			lbl = lbl.cuda().float()
			out = sigmoid(model(img))
			label = lbl.cpu().detach().numpy().reshape((-1, ))
			pred = out.cpu().detach().numpy().reshape((-1, ))
			val_pred = np.hstack([val_pred, pred])
			val_label = np.hstack([val_label, label])
			
		# calculate the metric of model
		criterion.solve_tau(val_pred, val_label)
#		criterion.tau = losses.tau.cpu().detach().numpy()
		wauc = criterion.wauc(val_pred, val_label)
		cost = criterion.cost(val_pred, val_label)
		auc = roc_auc_score(val_label, val_pred)
		
		rec.record('WAUC', epoch, wauc)
		rec.record('COST', epoch, cost)
		rec.record('AUC', epoch, auc)
		
		print('epoch:{} val wauc:{}, cost:{}, auc:{}'.format(epoch, wauc, cost, auc))

		# choose the best model and apply the parameters
		if cost < best_perf:
			best_perf = cost
			best_model = copy.deepcopy(model.state_dict())

			
	# calculate parial auc on testset 
	rec.save(['WAUC', 'COST', 'AUC'], dataset, c_type)
	rec.clear()
	
	# record instances' prediction and label of test set
	model.load_state_dict(best_model)
	model.eval()
	test_pred = np.array([])
	test_label = np.array([])
	for i, (img, lbl, idx) in enumerate(test_loader):
		img = img.cuda()
		lbl = lbl.cuda()
		out = sigmoid(model(img))
		label = lbl.cpu().detach().numpy().reshape((-1, ))
		pred = out.cpu().detach().numpy().reshape((-1, ))
		test_pred = np.hstack([test_pred, pred])
		test_label = np.hstack([test_label, label])
		
	criterion.tau = losses.tau.cpu().detach().numpy()
	# calculate the metric of model
	criterion.solve_tau(val_pred, val_label)
	wauc = criterion.wauc(val_pred, val_label)
	cost = criterion.cost(val_pred, val_label)
	auc = roc_auc_score(val_label, val_pred)
	print('test wauc:{}, cost:{}, auc:{}'.format(wauc, cost, auc))
	rec.save_test('WAUC', dataset, wauc, c_type)
	rec.save_test('COST', dataset, cost, c_type)
	rec.save_test('AUC', dataset, auc, c_type)
	rec.save_model(method, best_model, dataset, c_type)