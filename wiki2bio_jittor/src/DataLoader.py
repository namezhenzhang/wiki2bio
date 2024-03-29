import time
import logger
import numpy as np
import jittor
from jittor.dataset import Dataset

log = logger.get_logger(__name__)


class DataLoader(Dataset):
    def __init__(self, data_dir, limits, split):
        super().__init__()
        self.num_error = 0
        
        self.data_path = {
            'train': [data_dir + '/train/train.summary.id', data_dir + '/train/train.box.val.id',
                      data_dir + '/train/train.box.lab.id', data_dir + '/train/train.box.pos',
                      data_dir + '/train/train.box.rpos'],
            "test": [data_dir + '/test/test.summary.id', data_dir + '/test/test.box.val.id',
                     data_dir + '/test/test.box.lab.id', data_dir + '/test/test.box.pos',
                     data_dir + '/test/test.box.rpos'],
            "dev": [data_dir + '/valid/valid.summary.id', data_dir + '/valid/valid.box.val.id',
                    data_dir + '/valid/valid.box.lab.id', data_dir + '/valid/valid.box.pos',
                    data_dir + '/valid/valid.box.rpos']
        }
        self.limits = limits
        self.man_text_len = 100
        start_time = time.time()

        log.info(f'Reading {split} dataset ...')
        self.data_set = self.load_data(self.data_path[split])
        assert len(self.data_set[0]) == len(self.data_set[1]) == len(self.data_set[2]) == len(self.data_set[3]) == len(self.data_set[4])
        
        log.info(f'Reading {split} dataset comsumes %.3f seconds. Total: {len(self.data_set[0])}.' % (time.time() - start_time))

        self.set_attrs(total_len=len(self.data_set[0]))

    def load_data(self, path):
        summary_path, text_path, field_path, pos_path, rpos_path = path
        summaries = open(summary_path, 'r').read().strip().split('\n')
        texts = open(text_path, 'r').read().strip().split('\n')
        fields = open(field_path, 'r').read().strip().split('\n')
        poses = open(pos_path, 'r').read().strip().split('\n')
        rposes = open(rpos_path, 'r').read().strip().split('\n')
        if self.limits > 0:
            summaries = summaries[:self.limits]
            texts = texts[:self.limits]
            fields = fields[:self.limits]
            poses = poses[:self.limits]
            rposes = rposes[:self.limits]
        log.debug(summaries[0].strip().split(' '))
        summaries = [list(map(int, summary.strip().split(' ')))
                     for summary in summaries]
        texts = [list(map(int, text.strip().split(' '))) for text in texts]
        fields = [list(map(int, field.strip().split(' '))) for field in fields]
        poses = [list(map(int, pos.strip().split(' '))) for pos in poses]
        rposes = [list(map(int, rpos.strip().split(' '))) for rpos in rposes]

        return summaries, texts, fields, poses, rposes

        # length = 10
        # return summaries[:length], texts[:length], fields[:length], poses[:length], rposes[:length]

    def __getitem__(self, idx):
        summaries, texts, fields, poses, rposes = self.data_set
        return  summaries[idx], texts[idx], fields[idx], poses[idx], rposes[idx]


    def collate_batch(self, batch):
        batch_size = len(batch)
        batch = [[batch[j][i] for j in range(batch_size)] for i in range(5)]

        summaries, texts, fields, poses, rposes = batch
        
        max_summary_len = max([len(sample) for sample in summaries])
        max_text_len = max([len(sample) for sample in texts])

        batch_data = {'encoder_input':[], 'encoder_field':[], 'encoder_pos':[], 'encoder_rpos':[], 'encoder_len':[],
                          'decoder_input':[], 'decoder_len':[], 'decoder_output':[]}
        for summary, text, field, pos, rpos in zip(*batch):
            summary_len = len(summary)
            text_len = len(text)
            pos_len = len(pos)
            rpos_len = len(rpos)
            if text_len != len(field):
                self.num_error+=1
                continue
            assert text_len == len(field)
            assert pos_len == len(field)
            assert rpos_len == pos_len
            gold = summary + [2] + [0] * (max_summary_len - summary_len)
            summary = summary + [0] * (max_summary_len - summary_len)
            text = text + [0] * (max_text_len - text_len)
            field = field + [0] * (max_text_len - text_len)
            pos = pos + [0] * (max_text_len - text_len)
            rpos = rpos + [0] * (max_text_len - text_len)
            if max_text_len > self.man_text_len:
                text = text[:self.man_text_len]
                field = field[:self.man_text_len]
                pos = pos[:self.man_text_len]
                rpos = rpos[:self.man_text_len]
                text_len = min(text_len, self.man_text_len)

            batch_data['encoder_input'].append(text)
            batch_data['encoder_len'].append(text_len)
            batch_data['encoder_field'].append(field)
            batch_data['encoder_pos'].append(pos)
            batch_data['encoder_rpos'].append(rpos)
            batch_data['decoder_input'].append(summary)
            batch_data['decoder_len'].append(summary_len)
            batch_data['decoder_output'].append(gold)

        batch_data = { key:jittor.array(batch_data[key]) for key in batch_data}
        
        return batch_data
