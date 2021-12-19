import jittor
import jittor.nn as nn
from jittor import init



class fgateLstmUnit(nn.Module):
    def __init__(self, hidden_size, input_size, field_size, scope_name):

        super(fgateLstmUnit, self).__init__()

        self.hidden_size = hidden_size
        self.input_size = input_size
        self.field_size = field_size
        self.scope_name = scope_name
       
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

        x = jittor.concat([x, h_prev], dim=1)

        l_x = self.linear(x)
        i, j, f, o = jittor.split(l_x, l_x.shape[1]//4, 1)
        l_fd = self.linear1(fd)
        r, d = jittor.split(l_fd, l_fd.shape[1]//2, 1)

        #TODO sigmoid为什么要加1.0？
        # Final Memory cell
        c = jittor.sigmoid(f+1.0) * c_prev + jittor.sigmoid(i) * jittor.tanh(j) + jittor.sigmoid(r) * jittor.tanh(d)  # batch * hidden_size
        h = jittor.sigmoid(o) * jittor.tanh(c)

        out, state = h, (h, c)
        if finished is not None:
            # out = jittor.ternary(finished, jittor.zeros_like(h), h)
            # state = (jittor.ternary(finished, h_prev, h), jittor.ternary(finished, c_prev, c))
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