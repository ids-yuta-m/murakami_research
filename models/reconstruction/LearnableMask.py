import torch
import torch.nn as nn

class BinarizeHadamardFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight):
        ctx.save_for_backward(input, weight)
        tmp_zero = torch.zeros(weight.shape).to(weight.device)
        tmp_one = torch.ones(weight.shape).to(weight.device)
        weight_b = torch.where(weight>0, tmp_one, tmp_zero)
        output = input * weight_b
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        tmp_zero = torch.zeros(weight.shape).to(weight.device)
        tmp_one = torch.ones(weight.shape).to(weight.device)
        weight_b = torch.where(weight>0, tmp_one, tmp_zero)
        grad_input = grad_weight = None
        if ctx.needs_input_grad[0]:
            grad_input = grad_output * weight_b
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output * input
        return grad_input, grad_weight

class LearnableMask(nn.Module):
    def __init__(self, t=16, s=8):
        super().__init__()
        self.t = t
        self.s = s
        self.weight = nn.Parameter(torch.Tensor(t, s, s))
        self.reset_parameters()

    def reset_parameters(self):
        self.stdv = torch.sqrt(torch.tensor(1.5 / (self.s * self.s * self.t)))
        self.weight.data.uniform_(-self.stdv, self.stdv)

    def forward(self, input):
        return BinarizeHadamardFunction.apply(input, self.weight)

    def get_binary_mask(self):
        with torch.no_grad():
            return torch.where(self.weight > 0, 
                             torch.ones_like(self.weight), 
                             torch.zeros_like(self.weight))