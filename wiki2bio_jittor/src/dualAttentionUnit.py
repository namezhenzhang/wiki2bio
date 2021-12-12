import jittor
import jittor.nn as nn
from jittor import init


class dualAttentionWrapper(nn.Module):
    def __init__(self, hidden_size, input_size, field_size, scope_name):
        
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.field_size = field_size
        self.scope_name = scope_name
        self.params = {}

        # with jittor.variable_scope(scope_name):
        #     self.Wh = jittor.get_variable('Wh', [input_size, hidden_size])
        #     self.bh = jittor.get_variable('bh', [hidden_size])
        #     self.Ws = jittor.get_variable('Ws', [input_size, hidden_size])
        #     self.bs = jittor.get_variable('bs', [hidden_size])
        #     self.Wo = jittor.get_variable('Wo', [2*input_size, hidden_size])
        #     self.bo = jittor.get_variable('bo', [hidden_size])
        #     self.Wf = jittor.get_variable('Wf', [field_size, hidden_size])
        #     self.bf = jittor.get_variable('bf', [hidden_size])
        #     self.Wr = jittor.get_variable('Wr', [input_size, hidden_size])
        #     self.br = jittor.get_variable('br', [hidden_size])

        # self.params.update({'Wh': self.Wh, 'Ws': self.Ws, 'Wo': self.Wo,
        #                     'bh': self.bh, 'bs': self.bs, 'bo': self.bo,
        #                     'Wf': self.Wf, 'Wr': self.Wr, 
        #                     'bf': self.bf, 'br': self.br})

        self.linear_h = nn.Linear(input_size,hidden_size)
        self.linear_s = nn.Linear(input_size,hidden_size)
        self.linear_o = nn.Linear(2*input_size,hidden_size)
        self.linear_f = nn.Linear(field_size,hidden_size)
        self.linear_r = nn.Linear(input_size,hidden_size)

    def add_inputs(self, hs, fds):

        self.hs = jittor.transpose(hs, [1,0,2])  # input_len * batch * input_size
        self.fds = jittor.transpose(fds, [1,0,2])
        # self.hs_shape = self.hs.shape
        # self.hs2d = jittor.reshape(self.hs, [-1, self.input_size])
        # self.fds2d = jittor.reshape(fds, [-1, self.field_size])        

    def execute(self, x, coverage = None, finished = None):


        
        phi_hs = jittor.tanh(self.linear_h(self.hs))
        # phi_hs = jittor.reshape(phi_hs2d, self.hs_shape)

        phi_fds = jittor.tanh(self.linear_f(self.fds))
        # phi_fds = jittor.reshape(phi_fds2d, self.hs_shape)

        gamma_h = jittor.tanh(self.linear_s(x))  # batch * hidden_size
        alpha_h = jittor.tanh(self.linear_r(x))
        # print((phi_fds * alpha_h).shape)
        fd_weights = jittor.sum(phi_fds * alpha_h, dim=2, keepdims=True)
        fd_weights = jittor.exp(fd_weights - jittor.max(fd_weights, dim=0, keepdims=True))
        fd_weights = jittor.divide(fd_weights, (1e-6 + jittor.sum(fd_weights, dim=0, keepdims=True)))
        
        
        weights = jittor.sum(phi_hs * gamma_h, dim=2, keepdims=True)  # input_len * batch
        weights = jittor.exp(weights - jittor.max(weights, dim=0, keepdims=True))
        weights = jittor.divide(weights, (1e-6 + jittor.sum(weights, dim=0, keepdims=True)))
        try:
            weights = jittor.divide(weights * fd_weights, (1e-6 + jittor.sum(weights * fd_weights, dim=0, keepdims=True)))
        except:
            raise NotImplementedError()
        
        context = jittor.sum(self.hs * weights, dim=0)  # batch * input_size
        out = jittor.tanh(self.linear_o(jittor.concat([context, x], -1)))

        if finished is not None:
            condition = jittor.array(finished,dtype=bool)
            out[condition] = jittor.zeros_like(out)[condition]
        return out, weights