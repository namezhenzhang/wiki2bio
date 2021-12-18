import jittor
import jittor.nn as nn
from jittor import init


class dualAttentionWrapper(nn.Module):
    def __init__(self, hidden_size, input_size, field_size, scope_name):
        
        super(dualAttentionWrapper, self).__init__()

        self.hidden_size = hidden_size
        self.input_size = input_size
        self.field_size = field_size
        self.scope_name = scope_name

        self.linear_h = nn.Linear(input_size,hidden_size)
        self.linear_s = nn.Linear(input_size,hidden_size)
        self.linear_o = nn.Linear(2*input_size,hidden_size)
        self.linear_f = nn.Linear(field_size,hidden_size)
        self.linear_r = nn.Linear(input_size,hidden_size)

    def add_inputs(self, hs, fds):

        self.hs = jittor.transpose(hs, [1,0,2])  # input_len * batch * input_size
        self.fds = jittor.transpose(fds, [1,0,2])
        
    def execute(self, x, coverage = None, finished = None):

        phi_hs = jittor.tanh(self.linear_h(self.hs))
        phi_fds = jittor.tanh(self.linear_f(self.fds))

        gamma_h = jittor.tanh(self.linear_s(x))  # batch * hidden_size
        alpha_h = jittor.tanh(self.linear_r(x))

        fd_weights = jittor.sum(phi_fds * alpha_h, dim=2, keepdims=True)
        fd_weights = jittor.exp(fd_weights - jittor.max(fd_weights, dim=0, keepdims=True))
        fd_weights = jittor.divide(fd_weights, (1e-6 + jittor.sum(fd_weights, dim=0, keepdims=True)))
        
        weights = jittor.sum(phi_hs * gamma_h, dim=2, keepdims=True)  # input_len * batch
        weights = jittor.exp(weights - jittor.max(weights, dim=0, keepdims=True))
        weights = jittor.divide(weights, (1e-6 + jittor.sum(weights, dim=0, keepdims=True)))
        weights = jittor.divide(weights * fd_weights, (1e-6 + jittor.sum(weights * fd_weights, dim=0, keepdims=True)))

        context = jittor.sum(self.hs * weights, dim=0)  # batch * input_size
        out = jittor.tanh(self.linear_o(jittor.concat([context, x], -1)))

        if finished is not None:
            condition = jittor.array(finished,dtype=bool)
            out[condition] = jittor.zeros_like(out)[condition]
            
        return out, weights