
from torch import tensor
import torch.nn as nn
import torch
import inspect

class BPDA(nn.Module):

    def __init__(self, ori_f, sub_f):
        """[summary]

        Parameters
        ----------
        ori_f : [type]
            Currently BPDA not supports ori_f with * and ** parameter;
            Only support ori_f like defense/time_domain/QT_Non_Diff
        sub_f : [type]
            Should accept the same number of input and return the same number of outputs as ori_f
        """
        super().__init__()
        self.f = self.get_diff_func(ori_f, sub_f)
        ori_f_args = inspect.getfullargspec(ori_f).args
        self.ori_f_defaults = inspect.getfullargspec(ori_f).defaults # maybe None --> no default parameters
        self.ori_f_num_required = len(ori_f_args) - len(self.ori_f_defaults) if self.ori_f_defaults else len(ori_f_args)
        self.ori_f_option_parameters = ori_f_args[-len(self.ori_f_defaults):] if self.ori_f_defaults else []

    def get_diff_func(self, ori_f, sub_f):

        class differ_func(torch.autograd.Function):

            @staticmethod
            def forward(ctx, *args):
                ctx.args = args
                return ori_f(*args)
            
            @staticmethod
            @torch.enable_grad()
            def backward(ctx, *grad_output):
                inputs = ctx.args
                inputs_all = [] 
                inputs_need_grad = []
                for input_ in inputs:
                    if torch.is_tensor(input_): # TO DO: change to float or double tensor 
                        input_ = input_.detach().clone().requires_grad_()
                        inputs_need_grad.append(input_)
                    inputs_all.append(input_)
                outputs = sub_f(*inputs_all)
                num_output_ori = len(grad_output)
                num_output_sub = len(outputs) if isinstance(outputs, (tuple, list)) else 1
                assert num_output_ori == num_output_sub, 'The number of outputs of sub_f mismatches with ori_f'
                return torch.autograd.grad(outputs, inputs_need_grad, *grad_output) + tuple([None] * (len(inputs_all) - len(inputs_need_grad)))
        
        return differ_func
    
    def forward(self, *args, **kwargs):
        if len(list(kwargs.keys())) > 0:
            args = list(args)
            start = len(args) - self.ori_f_num_required
            for k, v in zip(self.ori_f_option_parameters[start:], self.ori_f_defaults[start:]):
                if k in kwargs.keys():
                    args.append(kwargs[k])
                else:
                    args.append(v)
            args = tuple(args)
        return self.f.apply(*args)