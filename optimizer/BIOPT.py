import torch
from torch.optim import SGD
import math

class BIOPT(SGD):
    def __init__(self, params=None, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, hyper_param=None):
        super(BIOPT, self).__init__(params, momentum, dampening,
                                     weight_decay, nesterov)
        self.device = torch.device('cuda:0')
        self.init_lr = torch.tensor(hyper_param['init_lr']).cuda()
        self.Lg1 = torch.tensor(hyper_param['M'] * 
                                3*100*math.exp(2*10)/(math.exp(10)+1)**4 *
                                hyper_param['M_']).cuda()
    @torch.no_grad()
    def step(self, closure=None, t=0, grad_f_tau=None):
        """Performs a single optimization step.

        Args:
          closure (callable, optional): A closure that reevaluates the model
              and returns the loss.
        """
        eta = self.init_lr/math.sqrt(t+1)
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            params_with_grad = []
            d_p_list = []
            momentum_buffer_list = []
            for p in group['params']:
                if p.grad is not None:
                    state = self.state[p]
                    params_with_grad.append(p) 
                    # add_grad = torch.matmul(
                    #     torch.ones_like(state["grad_g_tau_tau"]) - 1 / (state["grad_g_tau_tau"] * self.Lg1), grad_f_tau.T
                    # ) * self.Lg1

                    add_grad = torch.matmul(state["grad_g_tau_tau"], grad_f_tau.T) * self.Lg1
                    if 'clip' in group.keys():
                        p_grad = torch.clip(p.grad, group['clip'][0], group['clip'][1])
                        add_grad = torch.clip(add_grad * state["grad_g_p_tau"], 
                                          group['clip'][0], group['clip'][1])
                    else:
                        p_grad = p.grad
                        add_grad *= state["grad_g_p_tau"]
                    add_grad = -1 * add_grad.reshape(p_grad.size())
                    p_grad.add_(add_grad)
                    d_p_list.append(p_grad)
                    if 'momentum_buffer' not in state:
                        momentum_buffer_list.append(None)
                    else:
                        momentum_buffer_list.append(state['momentum_buffer'])
            sgd(params_with_grad,
                d_p_list,
                momentum_buffer_list,
                eta,
                weight_decay=group['weight_decay'],
                momentum=group['momentum'],
                dampening=group['dampening'],
                nesterov=group['nesterov'])
                # update momentum_buffers in state
            for p, momentum_buffer in zip(params_with_grad, momentum_buffer_list):
                    state = self.state[p]
                    state['momentum_buffer'] = momentum_buffer
        return loss


def sgd(params,
        d_p_list,
        momentum_buffer_list,
        lr,
        weight_decay,
        momentum,
        dampening,
        nesterov):
    r"""Functional API that performs SGD algorithm computation.

    See :class:`~torch.optim.SGD` for details.
    """

    for i, param in enumerate(params):

        d_p = d_p_list[i]
        if weight_decay != 0:
            d_p = d_p.add(param, alpha=weight_decay)

        if momentum != 0:
            buf = momentum_buffer_list[i]

            if buf is None:
                buf = torch.clone(d_p).detach()
                momentum_buffer_list[i] = buf
            else:
                buf.mul_(momentum).add_(d_p, alpha=1 - dampening)

            if nesterov:
                d_p = d_p.add(buf, alpha=momentum)
            else:
                d_p = buf

        param.add_(d_p, alpha=-lr)
