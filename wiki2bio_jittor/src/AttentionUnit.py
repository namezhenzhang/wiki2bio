import jittor
import pickle
import jittor.nn as nn
from jittor import init

class AttentionWrapper(nn.Module):
    def __init__(self, hidden_size, input_size, hs, scope_name):
        self.hs = jittor.transpose(hs, [1, 0, 2])
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.scope_name = scope_name
        self.params = {}
        

        self.linear_h = nn.Linear(input_size,hidden_size)
        self.linear_s = nn.Linear(input_size,hidden_size)
        self.linear_o = nn.Linear(2*input_size,hidden_size)

        hs2d = jittor.reshape(self.hs, [-1, input_size])
        phi_hs2d = jittor.tanh(self.linear_h(hs2d))
        self.phi_hs = jittor.reshape(phi_hs2d, self.hs.shape)
        
    def execute(self, x, finished=None):
        gamma_h = jittor.tanh(self.linear_s(x))
        weights = jittor.sum(self.phi_hs*gamma_h, dim=2, keep_dims=True) # jittor要换成reduce和sum
        weight = weights
        weights = jittor.exp(weights - jittor.max(weights, dim=0, keep_dims=True))
        weights = jittor.divide(weights, (1e-6 + jittor.sum(weights, dim=0, keep_dims=True)))
        context = jittor.sum(self.hs*weights, dim=0)
        # print wrt.get_shape().as_list()
        out = jittor.tanh(jittor.nn.Linear(jittor.concat([context, x], -1), self.Wo, self.bo))
        
        if finished is not None:
            condition = jittor.array(finished,dtype=bool)
            out[condition] = jittor.zeros_like(out)[condition]
        return out, weights