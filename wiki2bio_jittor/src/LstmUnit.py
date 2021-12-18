import jittor
import jittor.nn as nn
from jittor import init, logical_or


class LstmUnit(nn.Module):
    def __init__(self, hidden_size, input_size, scope_name):

        super(LstmUnit, self).__init__()

        self.hidden_size = hidden_size
        self.input_size = input_size
        self.scope_name = scope_name
    
        self.linear = nn.Linear(self.input_size+self.hidden_size,4*self.hidden_size)

    def execute(self, x, s, finished = None):

        h_prev, c_prev = s

        x = jittor.concat([x, h_prev], 1)
        l_x = self.linear(x)
        i, j, f, o = jittor.split(l_x, l_x.shape[1]//4, 1)


        #TODO sigmoid为什么要加1.0？
        # Final Memory cell
        c = jittor.sigmoid(f+1.0) * c_prev + jittor.sigmoid(i) * jittor.tanh(j)
        h = jittor.sigmoid(o) * jittor.tanh(c)

        out, state = h, (h, c)
        if finished is not None:

            condition = jittor.array(finished,dtype=bool)
            logical_not_condition = jittor.logical_not(condition)
            out = jittor.zeros_like(out)
            out[logical_not_condition] = h[logical_not_condition]

            state = jittor.zeros_like(h), jittor.zeros_like(c)
            state[0][condition] = h_prev[condition]
            state[0][logical_not_condition] = h[logical_not_condition]
            state[1][condition] = c_prev[condition]
            state[1][logical_not_condition] = c[logical_not_condition]

        return out, state
        