import jittor
import jittor.nn as nn
from jittor import init


class OutputUnit(nn.Module):
    def __init__(self, input_size, output_size, scope_name):

        super(OutputUnit, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.scope_name = scope_name

        self.linear = nn.Linear(input_size,output_size)

    def execute(self, x, finished = None):
        
        out = self.linear(x)

        if finished is not None:
            # out = jittor.ternary(finished, jittor.zeros_like(out), out)
            condition = jittor.array(finished,dtype=bool)
            out[condition] = jittor.zeros_like(out)[condition]

        return out

