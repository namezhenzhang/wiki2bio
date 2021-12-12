import jittor
import jittor.nn as nn
from jittor import init



class fgateLstmUnit(nn.Module):
    def __init__(self, hidden_size, input_size, field_size, scope_name):
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.field_size = field_size
        self.scope_name = scope_name
        # self.params = {}

        # with jittor.variable_scope(scope_name):
        #     self.W = jittor.get_variable('W', [self.input_size+self.hidden_size, 4*self.hidden_size])
        #     self.b = jittor.get_variable('b', [4*self.hidden_size], initializer=jittor.zeros_initializer(), dtype=jittor.float32)
        #     self.W1 = jittor.get_variable('W1', [self.field_size, 2*self.hidden_size])
        #     self.b1 = jittor.get_variable('b1', [2*hidden_size], initializer=jittor.zeros_initializer(), dtype=jittor.float32)
        # self.params.update({'W':self.W, 'b':self.b, 'W1':self.W1, 'b1':self.b1})

        self.linear = nn.Linear(self.input_size+self.hidden_size,4*self.hidden_size)
        self.linear1 = nn.Linear(self.field_size,2*self.hidden_size)

    def execute(self, x, fd, s, finished = None):
        """
        :param x: batch * input
        :param s: (h,s,d)
        :param finished:
        :return:
        """
        h_prev, c_prev = s  # batch * hidden_size
        # print('output\n',[x.shape, h_prev.shape],'\nend')
        x = jittor.concat([x, h_prev], dim=1)
        # fd = jittor.concat([fd, h_prev], 1)
        l_x = self.linear(x)
        i, j, f, o = jittor.split(l_x, l_x.shape[1]//4, 1)
        l_fd = self.linear1(fd)
        r, d = jittor.split(l_fd, l_fd.shape[1]//2, 1)
        # Final Memory cell
        c = jittor.sigmoid(f+1.0) * c_prev + jittor.sigmoid(i) * jittor.tanh(j) + jittor.sigmoid(r) * jittor.tanh(d)  # batch * hidden_size
        h = jittor.sigmoid(o) * jittor.tanh(c)

        out, state = h, (h, c)
        if finished is not None:
            # print(h.shape,c.shape)
            condition = jittor.array(finished,dtype=bool)
            logical_not_condition = jittor.logical_not(condition)
            out = jittor.zeros_like(out)
            out[logical_not_condition] = h[logical_not_condition]
            # out = jittor.where(finished, jittor.zeros_like(h), h)
            # print('\nc\n',c)
            try:
                state = jittor.zeros_like(h), jittor.zeros_like(c)
            except:
                print('\nh\n',h,'\nc\n',c)
                raise NotImplementedError()
            state[0][condition] = h_prev[condition]
            state[0][logical_not_condition] = h[logical_not_condition]
            state[1][condition] = c_prev[condition]
            state[1][logical_not_condition] = c[logical_not_condition]
            # state = (jittor.where(finished, h_prev, h), jittor.where(finished, c_prev, c))
            # out = jittor.multiply(1 - finished, h)
            # state = (jittor.multiply(1 - finished, h) + jittor.multiply(finished, h_prev),
            #          jittor.multiply(1 - finished, c) + jittor.multiply(finished, c_prev))

        return out, state

    # def save(self, path):
    #     param_values = {}
    #     for param in self.params:
    #         param_values[param] = self.params[param].eval()
    #     with open(path, 'wb') as f:
    #         pickle.dump(param_values, f, True)

    # def load(self, path):
    #     param_values = pickle.load(open(path, 'rb'))
    #     for param in param_values:
    #         self.params[param].load(param_values[param])