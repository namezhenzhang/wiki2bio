import jittor
import jittor.nn as nn
from jittor import init


class OutputUnit(nn.Module):
    def __init__(self, input_size, output_size, scope_name):
        self.input_size = input_size
        self.output_size = output_size
        self.scope_name = scope_name
        # self.params = {}

        # with tf.variable_scope(scope_name):
        #     self.W = tf.get_variable('W', [input_size, output_size])
        #     self.b = tf.get_variable('b', [output_size], initializer=tf.zeros_initializer(), dtype=tf.float32)

        # self.params.update({'W': self.W, 'b': self.b})

        self.linear = nn.Linear(input_size,output_size)

    def execute(self, x, finished = None):
        out = self.linear(x)

        if finished is not None:
            condition = jittor.array(finished,dtype=bool)
            out[condition] = jittor.zeros_like(out)[condition]
            # out = tf.where(finished, tf.zeros_like(out), out)
            #out = tf.multiply(1 - finished, out)
        return out

