import jittor
import pickle
import logger
from AttentionUnit import AttentionWrapper
from jittor import init
from jittor import nn
from dualAttentionUnit import dualAttentionWrapper
from LstmUnit import LstmUnit
from fgateLstmUnit import fgateLstmUnit
from OutputUnit import OutputUnit

log = logger.get_logger(__name__)


class SeqUnit(nn.Module):
    def __init__(self, batch_size, hidden_size, emb_size, field_size, pos_size, source_vocab, field_vocab,
                 position_vocab, target_vocab, field_concat, position_concat, fgate_enc, dual_att,
                 encoder_add_pos, decoder_add_pos, learning_rate, scope_name, name, start_token=2, stop_token=2, max_length=150):
        '''
        batch_size, hidden_size, emb_size, field_size, pos_size: size of batch; hidden layer; word/field/position embedding
        source_vocab, target_vocab, field_vocab, position_vocab: vocabulary size of encoder words; decoder words; field types; position
        field_concat, position_concat: bool values, whether concat field/position embedding to word embedding for encoder inputs or not
        fgate_enc, dual_att: bool values, whether use field-gating / dual attention or not
        encoder_add_pos, decoder_add_pos: bool values, whether add position embedding to field-gating encoder / decoder with dual attention or not
        '''

        super(SeqUnit, self).__init__()

        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.emb_size = emb_size
        self.field_size = field_size
        self.pos_size = pos_size
        self.uni_size = emb_size if not field_concat else emb_size+field_size
        self.uni_size = self.uni_size if not position_concat else self.uni_size+2*pos_size
        self.field_encoder_size = field_size if not encoder_add_pos else field_size+2*pos_size
        self.field_attention_size = field_size if not decoder_add_pos else field_size+2*pos_size
        self.source_vocab = source_vocab
        self.target_vocab = target_vocab
        self.field_vocab = field_vocab
        self.position_vocab = position_vocab
        self.grad_clip = 5.0
        self.start_token = start_token
        self.stop_token = stop_token
        self.max_length = max_length
        self.scope_name = scope_name
        self.name = name
        self.field_concat = field_concat
        self.position_concat = position_concat
        self.fgate_enc = fgate_enc
        self.dual_att = dual_att
        self.encoder_add_pos = encoder_add_pos
        self.decoder_add_pos = decoder_add_pos

        self.wrong_output = 0
        self.units = {}


        self.cs_loss = nn.CrossEntropyLoss()

        if self.fgate_enc:
            log.info('field-gated encoder LSTM')
            self.enc_lstm = fgateLstmUnit(self.hidden_size, self.uni_size, self.field_encoder_size, 'encoder_select')
        else:
            log.info('normal encoder LSTM')
            self.enc_lstm = LstmUnit(self.hidden_size, self.uni_size, 'encoder_lstm')
        self.dec_lstm = LstmUnit(self.hidden_size, self.emb_size, 'decoder_lstm')
        self.dec_out = OutputUnit(self.hidden_size, self.target_vocab, 'decoder_output')

        if self.dual_att:
            log.info('dual attention mechanism used')
            self.att_layer = dualAttentionWrapper(self.hidden_size, self.hidden_size, self.field_attention_size, "attention")
        else:
            log.info("normal attention used")
            self.att_layer = AttentionWrapper(self.hidden_size, self.hidden_size, "attention")


        # TODO 将子模型加入到module列表中，add_module



        self.embedding = nn.Embedding(self.source_vocab,self.emb_size)
        if self.field_concat or self.fgate_enc or self.encoder_add_pos or self.decoder_add_pos:
            self.fembedding = nn.Embedding(self.field_vocab,self.field_size)
        if self.position_concat or self.encoder_add_pos or self.decoder_add_pos:
            self.pembedding = nn.Embedding(self.position_vocab,self.pos_size)
            self.rembedding = nn.Embedding(self.position_vocab,self.pos_size)

    def step(self,type_,encoder_input,decoder_input,encoder_len,decoder_len,decoder_output,encoder_field=None,encoder_pos=None,encoder_rpos=None,beam_size=5):

        assert ((self.field_concat or self.fgate_enc or self.encoder_add_pos or self.decoder_add_pos) == (encoder_field is not None))
        assert (self.position_concat or self.encoder_add_pos or self.decoder_add_pos) == (encoder_pos is not None) == (encoder_rpos is not None)
        
        # ======================================== embeddings ======================================== #
        encoder_embed = self.embedding(encoder_input)
        decoder_embed = self.embedding(decoder_input)
        if self.field_concat or self.fgate_enc or self.encoder_add_pos or self.decoder_add_pos:
            field_embed = self.fembedding(encoder_field)
            field_pos_embed = field_embed
            if self.field_concat:
                encoder_embed = jittor.concat([encoder_embed, field_embed], 2)
        if self.position_concat or self.encoder_add_pos or self.decoder_add_pos:
            pos_embed = self.pembedding(encoder_pos)
            rpos_embed = self.rembedding(encoder_rpos)
            if self.position_concat:
                encoder_embed = jittor.concat([encoder_embed, pos_embed, rpos_embed], 2)
                field_pos_embed = jittor.concat([field_embed, pos_embed, rpos_embed], 2)
            elif self.encoder_add_pos or self.decoder_add_pos:
                field_pos_embed = jittor.concat([field_embed, pos_embed, rpos_embed], 2)
    

        # ======================================== encoder ======================================== #
        if self.fgate_enc:
            en_outputs, en_state = self.fgate_encoder(encoder_embed, field_pos_embed, encoder_len)
        else:
            en_outputs, en_state = self.encoder(encoder_embed, encoder_len)

        # ======================================== decoder ======================================== #
        if self.dual_att:
            self.att_layer.add_inputs(en_outputs, field_pos_embed)

        else:
            self.att_layer.add_inputs(en_outputs)

        if type_=='training':
            # decoder for training
            return self.decoder_t(en_state, decoder_embed, decoder_len,en_outputs)
        if type_=='testing':
            # decoder for testing
            return self.decoder_g(en_state, en_outputs)
        elif type_=='beam':
            # raise NotImplementedError()
            beam_seqs, beam_probs, cand_seqs, cand_probs, atts = self.decoder_beam(en_state, beam_size)
            return beam_seqs, beam_probs, cand_seqs, cand_probs, atts
    def execute(self,encoder_input,decoder_input,encoder_len,decoder_len,decoder_output,encoder_field=None,encoder_pos=None,encoder_rpos=None):
        
        de_outputs, de_state = self.step('training',encoder_input,decoder_input,encoder_len,decoder_len,decoder_output,encoder_field,encoder_pos,encoder_rpos)

        losses = self.cs_loss(de_outputs.reshape((-1,de_outputs.shape[-1])), decoder_output.reshape((-1,)))
        # print(decoder_output.dtype)
        mask = jittor.nn.sign(jittor.float32(decoder_output))
        losses = mask * losses
        self.mean_loss = jittor.mean(losses)

        return self.mean_loss

    def encoder(self, inputs, inputs_len):

        batch_size = inputs.shape[0]
        max_time = inputs.shape[1]
        hidden_size = self.hidden_size

        # time = jittor.array(0,dtype=jittor.int32)
        time = 0
        h0 = (jittor.zeros([batch_size, hidden_size]),
              jittor.zeros([batch_size, hidden_size]))
        f0 = jittor.zeros([batch_size]).bool()
        inputs = jittor.transpose(inputs, [1,0,2])
        emit_ta = []

        t, x_t, s_t, finished = time, inputs[time], h0, f0
        while jittor.logical_not(jittor.all(finished)):
            o_t, s_t = self.enc_lstm(x_t, s_t, finished)
            emit_ta.append(o_t)
            finished = jittor.greater_equal(t+1, inputs_len)
            if jittor.all(finished):
                x_t = jittor.zeros([batch_size, self.uni_size], dtype=jittor.float32)
            else:
                x_t = inputs[t+1]
            t+=1
        # assert len(emit_ta)==max_time
        outputs = jittor.stack(emit_ta).transpose([1,0,2])
        state = s_t

        return outputs, state

    def fgate_encoder(self, inputs, fields, inputs_len):

        batch_size = inputs.shape[0]
        max_time = inputs.shape[1]
        hidden_size = self.hidden_size
        # time = jittor.array(0,dtype=jittor.int32)
        time = 0
        h0 = (jittor.zeros([batch_size, hidden_size]),
              jittor.zeros([batch_size, hidden_size]))
        f0 = jittor.zeros([batch_size]).bool()

        inputs = jittor.transpose(inputs, [1,0,2])
        fields = jittor.transpose(fields, [1,0,2])
        emit_ta = []

        t, x_t, d_t, s_t, finished = time, inputs[time],fields[time], h0, f0

        while jittor.logical_not(jittor.all(finished)):
            o_t, s_t = self.enc_lstm(x_t, d_t, s_t, finished)
            emit_ta.append(o_t)
            finished = jittor.greater_equal(t+1, inputs_len)
            if jittor.all(finished):
                x_t = jittor.zeros([batch_size, self.uni_size], dtype=jittor.float32)
                d_t = jittor.zeros([batch_size, self.field_attention_size], dtype=jittor.float32)
            else:
                x_t = inputs[t+1]
                d_t = fields[t+1]
            t+=1

        outputs = jittor.stack(emit_ta).transpose([1,0,2])
        state = s_t

        return outputs, state

    def decoder_t(self, initial_state, inputs, inputs_len, en_outputs):

        batch_size = inputs.shape[0]
        max_time = inputs.shape[1]

        # time = jittor.array(0,dtype=jittor.int32)
        time = 0
        h0 = initial_state
        f0 = jittor.zeros([batch_size]).bool()
        x0 = self.embedding(jittor.array([self.start_token]*batch_size))
        inputs = jittor.transpose(inputs, [1,0,2])
        emit_ta = []

        t, x_t, s_t, finished = time, x0, h0, f0
        while jittor.logical_not(jittor.all(finished)):
            o_t, s_t = self.dec_lstm(x_t, s_t, finished)
            o_t, _ = self.att_layer(o_t)
            o_t = self.dec_out(o_t, finished)
            emit_ta.append(o_t)
            finished = jittor.greater_equal(t, inputs_len)
            if jittor.all(finished):
                x_t = jittor.zeros([batch_size, self.emb_size], dtype=jittor.float32)
            else:
                x_t = inputs[t]
            t+=1

        outputs = jittor.stack(emit_ta).transpose([1,0,2])
        state = s_t
        return outputs, state

    def decoder_g(self, initial_state, en_outputs):

        batch_size = initial_state[0].shape[0]
        # time = jittor.array(0,dtype=jittor.int32)
        time = 0
        h0 = initial_state
        f0 = jittor.zeros([batch_size]).bool()
        x0 = self.embedding(jittor.array([self.start_token]*batch_size))
        emit_ta = []
        att_ta = []

        t, x_t, s_t, finished = time, x0, h0, f0
        tag = jittor.logical_not(jittor.all(finished)).item()
        while tag:
            o_t, s_t = self.dec_lstm(x_t, s_t, finished)
            o_t, w_t = self.att_layer(o_t)
            o_t = self.dec_out(o_t, finished)
            emit_ta.append(o_t)
            att_ta.append(w_t)

            next_token = jittor.argmax(o_t, 1)[0]
            x_t = self.embedding(next_token)
            finished = jittor.logical_or(finished, jittor.equal(next_token, self.stop_token))
            finished = jittor.logical_or(finished, jittor.greater_equal(t, self.max_length))
            t+=1
            tag = jittor.logical_not(jittor.all(finished)).item()

        outputs = jittor.stack(emit_ta).transpose([1,0,2])
        pred_tokens = jittor.argmax(outputs, 2)[0]
        atts = jittor.stack(att_ta)
        return pred_tokens, atts
    
    def decoder_beam(self, initial_state, beam_size):
        batch_size = initial_state[0].shape[0]
        # print(batch_size)
        att_ta = []
        def beam_init():
            # return beam_seqs_1 beam_probs_1 cand_seqs_1 cand_prob_1 next_states time


            time_1 = 1
            beam_seqs_0 = jittor.array([[self.start_token]] * beam_size)
            beam_probs_0 = jittor.array([0.] * beam_size)
            cand_seqs_0 = jittor.array([[self.start_token]])
            cand_probs_0 = jittor.array([-3e38])

            # beam_seqs_0._shape = tf.TensorShape((None, None))
            # beam_probs_0._shape = tf.TensorShape((None,))
            # cand_seqs_0._shape = tf.TensorShape((None, None))
            # cand_probs_0._shape = tf.TensorShape((None,))
            
            inputs = jittor.array([self.start_token])
            x_t = self.embedding(inputs)
            # print(x_t.get_shape().as_list())
            o_t, s_nt = self.dec_lstm(x_t, initial_state)
            o_t, w_t = self.att_layer(o_t)
            o_t = self.dec_out(o_t)
            att_ta.append(w_t)
            # print(s_nt[0].get_shape().as_list())
            # initial_state = tf.reshape(initial_state, [1,-1])
            # print(o_t.shape)
            # print (nn.log_softmax(o_t,dim=-1).shape)
            # print(jittor.reshape(beam_probs_0, [-1, 1]).shape)
            logprobs2d = nn.log_softmax(o_t,dim=-1) + jittor.reshape(beam_probs_0, [-1, 1])
            # print(logprobs2d.shape)
            total_probs = logprobs2d #+ beam_probs_0.reshape(-1, 1)

            total_probs_noEOS = jittor.concat([total_probs[0:1,0:self.stop_token],
                               jittor.repeat(jittor.array([[-3e38]]), [1, 1]),
                               total_probs[0:1,self.stop_token + 1:self.target_vocab]], dim=1)
            flat_total_probs = jittor.reshape(total_probs_noEOS, [-1])
            # print (flat_total_probs.get_shape().as_list())
            # assert len(jittor.size(flat_total_probs))==2
            beam_k = jittor.minimum(flat_total_probs.shape[0], beam_size)
            # print(flat_total_probs.shape,type(beam_k))
            next_beam_probs, top_indices = jittor.topk(flat_total_probs, k=int(beam_k))

            next_bases = jittor.floor_divide(top_indices, self.target_vocab)#?
            next_mods = jittor.mod(top_indices, self.target_vocab)

            # print(beam_seqs_0.shape,next_bases,jittor.reshape(next_mods, [-1, 1]).shape)
            next_beam_seqs = jittor.concat([jittor.gather(beam_seqs_0, 0,next_bases),
                                        jittor.reshape(next_mods, [-1, 1])], dim=1)#?

            cand_seqs_pad = jittor.nn.pad(cand_seqs_0, (0,1,0,0))
            beam_seqs_EOS = jittor.nn.pad(beam_seqs_0, (0,1,0,0))
            new_cand_seqs = jittor.concat([cand_seqs_pad, beam_seqs_EOS], 0)
            # print (new_cand_seqs.get_shape().as_list())

            EOS_probs = total_probs[0:beam_size,self.stop_token:self.stop_token+1]
            new_cand_probs = jittor.concat([cand_probs_0, jittor.reshape(EOS_probs, [-1])], 0)
            assert len(jittor.size(new_cand_probs))==1
            cand_k = jittor.minimum(new_cand_probs.shape[0], beam_size)
            next_cand_probs, next_cand_indices = jittor.topk(new_cand_probs, k=int(cand_k))
            next_cand_seqs = jittor.gather(new_cand_seqs, 0,next_cand_indices)#?

            part_state_0 = jittor.reshape(jittor.stack([s_nt[0]]*beam_size), [beam_size, self.hidden_size])
            part_state_1 = jittor.reshape(jittor.stack([s_nt[1]]*beam_size), [beam_size, self.hidden_size])
            # part_state_0._shape = tf.TensorShape((None, None))
            # part_state_1._shape = tf.TensorShape((None, None))
            next_states = (part_state_0, part_state_1)
            # print (next_states[0].get_shape().as_list())
            return next_beam_seqs, next_beam_probs, next_cand_seqs, next_cand_probs, next_states, time_1

        beam_seqs_1, beam_probs_1, cand_seqs_1, cand_probs_1, states_1, time_1 = beam_init()
        # beam_seqs_1._shape = tf.TensorShape((None, None))
        # beam_probs_1._shape = tf.TensorShape((None,))
        # cand_seqs_1._shape = tf.TensorShape((None, None))
        # cand_probs_1._shape = tf.TensorShape((None,))
        # states_1._shape = tf.TensorShape((2, None, self.hidden_size))
        def beam_step(beam_seqs, beam_probs, cand_seqs, cand_probs, states, time):
            '''
            beam_seqs : [beam_size, time]
            beam_probs: [beam_size, ]
            cand_seqs : [beam_size, time]
            cand_probs: [beam_size, ]
            states : [beam_size * hidden_size, beam_size * hidden_size]
            '''
            inputs = jittor.reshape(beam_seqs[0:beam_size,time:time+1], [beam_size])
            # print inputs.get_shape().as_list()
            x_t = self.embedding(inputs)
            # print(x_t.get_shape().as_list())
            o_t, s_nt = self.dec_lstm(x_t, states)
            o_t, w_t = self.att_layer(o_t)
            o_t = self.dec_out(o_t)
            att_ta.append(w_t)
            logprobs2d = jittor.nn.log_softmax(o_t)
            # print (logprobs2d.get_shape().as_list())
            total_probs = logprobs2d + jittor.reshape(beam_probs, [-1, 1])
            # print (total_probs.get_shape().as_list())
            # total_probs[0:beam_size,self.stop_token + 1:self.target_vocab]
            total_probs_noEOS = jittor.concat([total_probs[0:beam_size,0:self.stop_token],
                                           jittor.repeat(jittor.array([[-3e38]]), [beam_size, 1]),
                                           total_probs[0:beam_size,self.stop_token + 1:self.target_vocab]], 1)
            # print (total_probs_noEOS.get_shape().as_list())
            flat_total_probs = jittor.reshape(total_probs_noEOS, [-1])
            # print (flat_total_probs.get_shape().as_list())

            beam_k = jittor.minimum(flat_total_probs.shape[0], beam_size)
            next_beam_probs, top_indices = jittor.topk(flat_total_probs, k=int(beam_k))
            # print (next_beam_probs.get_shape().as_list())

            next_bases = jittor.floor_divide(top_indices, self.target_vocab)
            next_mods = jittor.mod(top_indices, self.target_vocab)
            # print (next_mods.get_shape().as_list())

            next_beam_seqs = jittor.concat([jittor.gather(beam_seqs, 0,next_bases),#?
                                        jittor.reshape(next_mods, [-1, 1])], 1)
            next_states = (jittor.gather(s_nt[0], 0,next_bases), jittor.gather(s_nt[1], 0,next_bases))#?
            # print (next_beam_seqs.get_shape().as_list())

            cand_seqs_pad = jittor.nn.pad(cand_seqs, (0,1,0,0))
            beam_seqs_EOS = jittor.nn.pad(beam_seqs, (0,1,0,0))
            new_cand_seqs = jittor.concat([cand_seqs_pad, beam_seqs_EOS], 0) 
            # print (new_cand_seqs.get_shape().as_list())
            EOS_probs = total_probs[0:beam_size,self.stop_token:self.stop_token+1]
            new_cand_probs = jittor.concat([cand_probs, jittor.reshape(EOS_probs, [-1])], 0)
            assert len(new_cand_probs.shape)==1
            cand_k = jittor.minimum(new_cand_probs.shape[0], beam_size)
            next_cand_probs, next_cand_indices = jittor.topk(new_cand_probs, k=int(cand_k))
            next_cand_seqs = jittor.gather(new_cand_seqs, 0,next_cand_indices)

            return next_beam_seqs, next_beam_probs, next_cand_seqs, next_cand_probs, next_states, time+1
        
        def beam_cond(beam_probs, beam_seqs, cand_probs, cand_seqs, state, time):
            length =  (jittor.max(beam_probs) >= jittor.min(cand_probs))
            return jittor.logical_and(length, jittor.less(time, 60) )
            # return tf.less(time, 18)

        # loop_vars = [beam_seqs_1, beam_probs_1, cand_seqs_1, cand_probs_1, states_1, time_1]
        while beam_cond(beam_seqs_1, beam_probs_1, cand_seqs_1, cand_probs_1, states_1, time_1):
            beam_seqs_1, beam_probs_1, cand_seqs_1, cand_probs_1, states_1, time_1 = beam_step(beam_seqs_1, beam_probs_1, cand_seqs_1, cand_probs_1, states_1, time_1)
        # ret_vars = tf.while_loop(cond=beam_cond, body=beam_step, loop_vars=loop_vars, back_prop=False)
        # beam_seqs_all, beam_probs_all, cand_seqs_all, cand_probs_all, _, time_all = ret_vars
        atts = jittor.stack(att_ta)
        return beam_seqs_1, beam_probs_1, cand_seqs_1, cand_probs_1,atts
        # return beam_seqs_all, beam_probs_all, cand_seqs_all, cand_probs_all

    def generate(self, encoder_input,decoder_input,encoder_len,decoder_len,decoder_output,encoder_field=None,encoder_pos=None,encoder_rpos=None):
        g_tokens, atts = self.step('testing',encoder_input,decoder_input,encoder_len,decoder_len,decoder_output,encoder_field,encoder_pos,encoder_rpos)
        return g_tokens, atts

    def generate_beam(self, encoder_input,decoder_input,encoder_len,decoder_len,decoder_output,encoder_field=None,encoder_pos=None,encoder_rpos=None):
        beam_seqs_all, beam_probs_all, cand_seqs_all, cand_probs_all, atts = self.step('beam',encoder_input,decoder_input,encoder_len,decoder_len,decoder_output,encoder_field,encoder_pos,encoder_rpos)
        return beam_seqs_all, beam_probs_all, cand_seqs_all, cand_probs_all, atts