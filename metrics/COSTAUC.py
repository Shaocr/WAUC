import numpy as np

class COSTAUCMetric:
    def __init__(self, K, pi, c):
        self.beta = 7.0
        self.h = 0.2
        self.pi = pi
        self.c = c
        self.K = K
        
        # optimzation variable
        self.tau = np.zeros((1, K)) + 0.5
        self.constant = np.sqrt(2*np.pi)
    
    def wauc_(self, pred, target):
        pred_p = pred[target==1].reshape((-1, 1))
        pred_n = pred[target!=1].reshape((-1, 1))

        fpr_pdf = np.mean(np.exp(-np.square((pred_n-self.tau)/self.h))/self.constant, axis=0)/self.h
        TPR = np.mean((pred_p-self.tau)>0, axis=0)
        
        res = fpr_pdf*TPR
        return res

    def wauc(self, pred, target):
        pred_p = pred[target==1].reshape((-1, 1))
        pred_n = pred[target!=1].reshape((-1, 1))

        fpr_pdf = np.mean(np.exp(-np.square((pred_n-self.tau)/self.h))/self.constant, axis=0)/self.h
        TPR = np.mean((pred_p-self.tau)>0, axis=0)
        
        res = np.mean(fpr_pdf*TPR)
        return res
    def cost(self, pred, target):
        pred_p = pred[target==1].reshape((-1, 1))
        pred_n = pred[target!=1].reshape((-1, 1))

        pos = np.mean(self.c * self.pi*np.mean((self.tau-pred_p)>0,axis=0))
        neg = np.mean((1-self.c) * (1-self.pi)*np.mean((pred_n-self.tau)>0,axis=0))
        return pos + neg
    def solve_tau(self, pred, target):
        pred_p = pred[target==1].reshape((-1, 1))
        pred_n = pred[target!=1].reshape((-1, 1))
        pred = pred.reshape((1, -1))
        for k in range(self.K):
            pos = self.c[0, k] * self.pi*np.mean((pred-pred_p)>0,axis=0)
            neg = (1-self.c[0, k]) * (1-self.pi)*np.mean((pred_n-pred)>0,axis=0)
            op_k = np.argmin(pos+neg)
            self.tau[0, k] = pred[0, op_k]
    def cost_all(self, pred, target):
        pred_p = pred[target==1].reshape((-1, 1))
        pred_n = pred[target!=1].reshape((-1, 1))

        pos = self.c * self.pi*np.mean((self.tau-pred_p)>0,axis=0)
        neg = (1-self.c) * (1-self.pi)*np.mean((pred_n-self.tau)>0,axis=0)
        return pos + neg