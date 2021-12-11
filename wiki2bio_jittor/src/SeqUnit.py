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


class SeqUnit(jittor.module):
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

        self.units = {}
        self.params = {}

        self.cs_loss = nn.CrossEntropyLoss()

        if self.fgate_enc:
            log.info('field-gated encoder LSTM')
            self.enc_lstm = fgateLstmUnit(self.hidden_size, self.uni_size, self.field_encoder_size, 'encoder_select')
        else:
            log.info('normal encoder LSTM')
            self.enc_lstm = LstmUnit(self.hidden_size, self.uni_size, 'encoder_lstm')
        self.dec_lstm = LstmUnit(self.hidden_size, self.emb_size, 'decoder_lstm')
        self.dec_out = OutputUnit(self.hidden_size, self.target_vocab, 'decoder_output')

        if self.fgate_enc:
            log.info('field gated encoder used')
        else:
            log.info('normal encoder used')

        if self.dual_att:
            log.info('dual attention mechanism used')
            self.att_layer = dualAttentionWrapper(self.hidden_size, self.hidden_size, self.field_attention_size,
                                                        en_outputs, self.field_pos_embed, "attention")
            # self.units.update({'attention': self.att_layer})
        else:
            log.info("normal attention used")
            self.att_layer = AttentionWrapper(self.hidden_size, self.hidden_size, en_outputs, "attention")
            # self.units.update({'attention': self.att_layer})

        # TODO 将子模型加入到module列表中，add_module
        # self.units.update({'encoder_lstm': self.enc_lstm,'decoder_lstm': self.dec_lstm,
        #                    'decoder_output': self.dec_out})


        self.embedding = nn.Embedding(self.source_vocab,self.emb_size)
        if self.field_concat or self.fgate_enc or self.encoder_add_pos or self.decoder_add_pos:
            self.fembedding = nn.Embedding(self.field_vocab,self.field_size)
        if self.position_concat or self.encoder_add_pos or self.decoder_add_pos:
            self.pembedding = nn.Embedding(self.position_vocab,self.pos_size)
            self.rembedding = nn.Embedding(self.position_vocab,self.pos_size)

    def execute(self,encoder_input,decoder_input,encoder_len,encoder_field=None,encoder_pos=None,encoder_rpos=None):

        assert (self.field_concat or self.fgate_enc or self.encoder_add_pos or self.decoder_add_pos) == (encoder_field!=None)
        assert (self.position_concat or self.encoder_add_pos or self.decoder_add_pos) == (encoder_pos!=None) == (encoder_rpos!=None)
        # ======================================== embeddings ======================================== #
        self.encoder_embed = self.embedding(encoder_input)
        self.decoder_embed = self.embedding(decoder_input)
        if self.field_concat or self.fgate_enc or self.encoder_add_pos or self.decoder_add_pos:
            self.field_embed = self.fembedding(encoder_field)
            self.field_pos_embed = self.field_embed
            if self.field_concat:
                self.encoder_embed = jittor.concat([self.encoder_embed, self.field_embed], 2)
        if self.position_concat or self.encoder_add_pos or self.decoder_add_pos:
            self.pos_embed = self.pembedding(encoder_pos)
            self.rpos_embed = self.rembedding(encoder_rpos)
            if self.position_concat:
                self.encoder_embed = jittor.concat([self.encoder_embed, self.pos_embed, self.rpos_embed], 2)
                self.field_pos_embed = jittor.concat([self.field_embed, self.pos_embed, self.rpos_embed], 2)
            elif self.encoder_add_pos or self.decoder_add_pos:
                self.field_pos_embed = jittor.concat([self.field_embed, self.pos_embed, self.rpos_embed], 2)
        
        # if self.field_concat or self.fgate_enc:
        #     self.params.update({'fembedding': self.fembedding})
        # if self.position_concat or self.encoder_add_pos or self.decoder_add_pos:
        #     self.params.update({'pembedding': self.pembedding})
        #     self.params.update({'rembedding': self.rembedding})
        # self.params.update({'embedding': self.embedding})

        # ======================================== encoder ======================================== #
        if self.fgate_enc:
            en_outputs, en_state = self.fgate_encoder(self.encoder_embed, self.field_pos_embed, encoder_len)
        else:
            en_outputs, en_state = self.encoder(self.encoder_embed, encoder_len)

        # ======================================== decoder ======================================== #
        # decoder for training
        de_outputs, de_state = self.decoder_t(en_state, self.decoder_embed, self.decoder_len)
        # decoder for testing
        self.g_tokens, self.atts = self.decoder_g(en_state)
        # self.beam_seqs, self.beam_probs, self.cand_seqs, self.cand_probs = self.decoder_beam(en_state, beam_size)

        losses = self.cs_loss(de_outputs, self.decoder_output)
        mask = jittor.sign(jittor.to_float(self.decoder_output))
        losses = mask * losses
        self.mean_loss = jittor.reduce_mean(losses)

        return self.mean_loss

    def encoder(self, inputs, inputs_len):
        batch_size = inputs.shape[0]
        max_time = inputs.shape[1]
        hidden_size = self.hidden_size

        time = jittor.array(0,dtype=jittor.int32)
        h0 = (jittor.zeros([batch_size, hidden_size]),
              jittor.zeros([batch_size, hidden_size]))
        f0 = jittor.zeros([batch_size]).to_bool()

        t, x_t, s_t, emit_ta, finished = time, inputs_ta.read(0), h0, emit_ta, f0

        while

