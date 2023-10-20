import torch
import torch.nn as nn

class WAUCGau(nn.Module):
    def __init__(self, K, pi, c):
        super(WAUCGau,self).__init__()
        self.device = torch.device('cuda:0')
        
        self.beta = torch.tensor(7.0).cuda()
        self.h = torch.tensor(0.2).cuda()
        self.pi = torch.tensor(pi).cuda()
        self.c = c
        self.constant = torch.sqrt(torch.tensor(2*torch.pi))
        
        # optimzation variable
        self.tau = torch.zeros((1, K), device=self.device).cuda()
        self.tau.requires_grad = True
        
        self.per_tau = torch.tensor(0.5, device=self.device).cuda()
        self.per_tau.requires_grad = True

    def outter_loss(self, pred, target):
        pred_p = pred[target.eq(1)].reshape((-1, 1))
        pred_n = pred[target.ne(1)].reshape((-1, 1))

        fpr_pdf = torch.mean(torch.exp(-torch.square((pred_n-self.tau)/self.h))/self.constant, dim=0)/self.h
        TPR = torch.mean(1/(1+torch.exp(-self.beta*(pred_p-self.tau))), dim=0)
        
        res = torch.mean(fpr_pdf*TPR)
        return 1-res
    def inner_loss(self, pred, target):
        pred_p = pred[target.eq(1)].reshape((-1, 1))
        pred_n = pred[target.ne(1)].reshape((-1, 1))

        pos = torch.mean(self.c * self.pi*torch.mean(1-1/(1+torch.exp(-self.beta*(pred_p-self.tau))),dim=0))
        neg = torch.mean((1-self.c) * (1-self.pi)*torch.mean(1/(1+torch.exp(-self.beta*(pred_n-self.tau))),dim=0))
        return pos + neg
    def per_inner_loss(self, pred, target, c):
        target = target.reshape((-1, ))
        pred_p = pred[target.eq(1)]
        pred_n = pred[target.ne(1)]
        c_p = c[target.eq(1)]
        c_n = c[target.ne(1)]
        pos = self.pi*torch.mean((1-1/(1+torch.exp(-self.beta*(pred_p-self.per_tau))))*c_p)
        neg = (1-self.pi)*torch.mean(1/(1+torch.exp(-self.beta*(pred_n-self.per_tau)))*(1-c_n))
        return pos + neg

    def per_outer_loss(self, pred, target):
        target = target.reshape((-1, ))
        pred_p = pred[target.eq(1)]
        pred_n = pred[target.ne(1)]

        fpr_pdf = torch.mean(torch.exp(-torch.square((pred_n-self.per_tau)/self.h))/self.constant, dim=0)/self.h
        TPR = torch.mean(1/(1+torch.exp(-self.beta*(pred_p-self.per_tau))), dim=0)
        
        res = torch.mean(fpr_pdf*TPR)
        return 1-res
class WAUCLog(nn.Module):
    def __init__(self, K, pi, c):
        super(WAUCLog,self).__init__()
        self.device = torch.device('cuda:0')
        
        self.beta = torch.tensor(7.0).cuda()
        self.h = torch.tensor(0.2).cuda()
        self.pi = torch.tensor(pi).cuda()
        self.c = c
        
        # optimzation variable
        self.tau = torch.zeros((1, K), device=self.device).cuda()
        self.tau.requires_grad = True
        
    def outter_loss(self, pred, target):
        pred_p = pred[target.eq(1)].reshape((-1, 1))
        pred_n = pred[target.ne(1)].reshape((-1, 1))

        fpr_pdf = torch.mean(1/(2+torch.exp((pred_n-self.tau)/self.h)+torch.exp((self.tau-pred_n)/self.h)), dim=0)/self.h
        TPR = torch.mean(1/(1+torch.exp(-self.beta*(pred_p-self.tau))), dim=0)
        
        res = torch.mean(fpr_pdf*TPR)
        return 1-res
    def inner_loss(self, pred, target):
        pred_p = pred[target.eq(1)].reshape((-1, 1))
        pred_n = pred[target.ne(1)].reshape((-1, 1))

        pos = torch.mean(self.c * self.pi*torch.mean(1-1/(1+torch.exp(-self.beta*(pred_p-self.tau))),dim=0))
        neg = torch.mean((1-self.c) * (1-self.pi)*torch.mean(1/(1+torch.exp(-self.beta*(pred_n-self.tau))),dim=0))
        return pos + neg + self.outter_loss(pred, target)