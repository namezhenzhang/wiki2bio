import jittor
import jittor.nn as nn
from jittor import init

class AttentionWrapper(nn.Module):
    def __init__(self, hidden_size, input_size, scope_name):
        super(AttentionWrapper, self).__init__()

        self.hidden_size = hidden_size
        self.input_size = input_size
        self.scope_name = scope_name

        self.linear_h = nn.Linear(input_size,hidden_size)
        self.linear_s = nn.Linear(input_size,hidden_size)
        self.linear_o = nn.Linear(2*input_size,hidden_size)

    def add_inputs(self, hs):
        self.hs = jittor.transpose(hs, [1, 0, 2])
        
    def execute(self, x, finished=None):

        hs2d = jittor.reshape(self.hs, [-1, self.input_size])
        phi_hs2d = jittor.tanh(self.linear_h(hs2d))
        phi_hs = jittor.reshape(phi_hs2d, self.hs.shape)

        gamma_h = jittor.tanh(self.linear_s(x))

        weights = jittor.sum(phi_hs*gamma_h, dim=2, keepdims=True)
        weights = jittor.exp(weights - jittor.max(weights, dim=0, keepdims=True))
        weights = jittor.divide(weights, (1e-6 + jittor.sum(weights, dim=0, keepdims=True)))
        context = jittor.sum(self.hs*weights, dim=0)

        out = jittor.tanh(self.linear_o(jittor.concat([context, x], -1)))
        
        if finished is not None:
            # out = jittor.ternary(finished, jittor.zeros_like(out), out)
            condition = jittor.array(finished,dtype=bool)
            out[condition] = jittor.zeros_like(out)[condition]
        return out, weights