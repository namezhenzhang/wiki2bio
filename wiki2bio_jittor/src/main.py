
import logger
from configs import get_args_parser,  get_args, save_args
import random
import numpy  as np
import jittor
import time, os, sys, shutil
# sys.path.append('./')
from tqdm import tqdm
from SeqUnit import *
from preprocess import *
from PythonROUGE import PythonROUGE
from DataLoader import DataLoader
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

log = logger.get_logger(__name__)

def set_seed(seed):
    # random.seed(seed)
    # np.random.seed(seed)
    jittor.misc.set_global_seed(seed)
    # torch.manual_seed(seed)
    # if args.n_gpu > 0:
    #     torch.cuda.manual_seed_all(seed)

def do_before_running():
    args = get_args_parser()
    logger.setup_applevel_logger(file_name = args.logger_file_name)
    log.info(f"output dir: '{args.output_dir}'")
    save_args()
    
    log.info(f"set seed {args.seed}")
    set_seed(args.seed)
    return args

def train(model, train_dataloader,dev_dataloader,test_dataloader):
    global_step = 0
    optimizer = nn.Adam(model.parameters(), args.learning_rate, eps=1e-8, betas=(0.9, 0.999))
    total_loss, start_time = 0.0, time.time()
    loss = 0.0
    for num_eopch in range(args.epoch):
        with tqdm(total=len(train_dataloader)) as Tqdm:
            for  x in train_dataloader:
                model.train()
                Tqdm.update(1)
                Tqdm.set_description(f"epoch {num_eopch}")

                now_loss = model(**x)
                loss = now_loss + loss
                total_loss += now_loss.item()
                    
                global_step+=1
                if global_step % args.accumulation==0:
                    loss = loss/args.accumulation
                    Tqdm.set_postfix(loss = loss.item() )
                    optimizer.step(loss)
                    loss = 0.0
                
                
                if (global_step % args.report == 0):
                    cost_time = time.time() - start_time
                    log.info("%d : loss = %.3f, time = %.3f " % (global_step // args.report, total_loss/args.report, cost_time))
                    total_loss, start_time = 0.0, time.time()
                    #TODO保存模型
                    if global_step // args.report >= 1: 
                        ksave_dir = save_model(model, save_dir, global_step // args.report)
                        log.info(evaluate(model, dev_dataloader, ksave_dir, 'valid'))

def test(model, dataloader):
    log.info(evaluate(model, dataloader, save_dir, 'test'))

def save_model(model, save_dir, cnt):
    new_dir = save_dir + 'checkpoints' + '/' 
    if not os.path.exists(new_dir):
        os.mkdir(new_dir)
    nnew_dir = new_dir + str(cnt) + '/'
    if not os.path.exists(nnew_dir):
        os.mkdir(nnew_dir)
    model.save(nnew_dir+'model.ckpt')
    return nnew_dir

def write_word(pred_list, save_dir, name):
    ss = open(save_dir + name, "w+")
    for item in pred_list:
        ss.write(" ".join(item) + '\n')
@jittor.no_grad()
def evaluate(model, dataloader, ksave_dir, mode='valid'):
    if mode == 'valid':
        # texts_path = "original_data/valid.summary"
        texts_path = os.path.join(args.root_dir,"processed_data/valid/valid.box.val")
        gold_path = os.path.join(args.root_dir,'processed_data/valid/valid_split_for_rouge/gold_summary_')
        # evalset = dataloader.dev_set
    else:
        # texts_path = "original_data/test.summary"
        texts_path = os.path.join(args.root_dir,"processed_data/test/test.box.val")
        gold_path = os.path.join(args.root_dir,'processed_data/test/test_split_for_rouge/gold_summary_')
        # evalset = dataloader.test_set
    
    # for copy words from the infoboxes
    texts = open(texts_path, 'r').read().strip().split('\n')
    texts = [list(t.strip().split()) for t in texts]
    v = Vocab()

    # with copy
    pred_list, pred_list_copy, gold_list = [], [], []
    pred_unk, pred_mask = [], []
    
    k = 0
    with tqdm(total=len(dataloader)) as Tqdm:
        for x in dataloader:
            model.eval()
            Tqdm.update(1)
            predictions, atts = model.generate(**x)

            atts = np.squeeze(np.array(atts),axis=-1)
            idx = 0
            for summary in np.array(predictions):
                with open(pred_path + str(k), 'w') as sw:
                    summary = list(summary)
                    if 2 in summary:
                        summary = summary[:summary.index(2)] if summary[0] != 2 else [2]
                    real_sum, unk_sum, mask_sum = [], [], []
                    for tk, tid in enumerate(summary):
                        if tid == 3:
                            sub = texts[k][np.argmax(atts[tk,: len(texts[k]),idx])]
                            real_sum.append(sub)
                            mask_sum.append("**" + str(sub) + "**")
                        else:
                            real_sum.append(v.id2word(tid))
                            mask_sum.append(v.id2word(tid))
                        unk_sum.append(v.id2word(tid))
                    sw.write(" ".join([str(x) for x in real_sum]) + '\n')
                    pred_list.append([str(x) for x in real_sum])
                    pred_unk.append([str(x) for x in unk_sum])
                    pred_mask.append([str(x) for x in mask_sum])
                    k += 1
                    idx += 1

    write_word(pred_mask, ksave_dir, mode + "_summary_copy.txt")
    write_word(pred_unk, ksave_dir, mode + "_summary_unk.txt")


    for tk in range(k):
        with open(gold_path + str(tk), 'r') as g:
            gold_list.append([g.read().strip().split()])

    gold_set = [[gold_path + str(i)] for i in range(k)]
    pred_set = [pred_path + str(i) for i in range(k)]

    recall, precision, F_measure = PythonROUGE(pred_set, gold_set, ngram_order=4)
    bleu = corpus_bleu(gold_list, pred_list)
    copy_result = "with copy F_measure: %s Recall: %s Precision: %s BLEU: %s\n" % \
    (str(F_measure), str(recall), str(precision), str(bleu))
    # print copy_result

    for tk in range(k):
        with open(pred_path + str(tk), 'w') as sw:
            sw.write(" ".join(pred_unk[tk]) + '\n')

    recall, precision, F_measure = PythonROUGE(pred_set, gold_set, ngram_order=4)
    bleu = corpus_bleu(gold_list, pred_unk)
    nocopy_result = "without copy F_measure: %s Recall: %s Precision: %s BLEU: %s\n" % \
    (str(F_measure), str(recall), str(precision), str(bleu))
    # print nocopy_result
    result = copy_result + nocopy_result 
    # print result
    if mode == 'valid':
        log.info(result)

    return result

def copy_file(dst, src=os.path.dirname(os.path.abspath(__file__))):
    files = os.listdir(src)
    saved_files = []
    for file in files:
        file_ext = file.split('.')[-1]
        if file_ext=='py':
            saved_files.append(file)
            shutil.copy(os.path.join(src,file), dst)
    log.info(f'saved files {saved_files} to {dst}')

# def test_main():
#     print(os.path.join(args.root_dir,args.dir))
#     train_dataloader = DataLoader(os.path.join(args.root_dir,args.dir), args.limits, 'dev').set_attrs(batch_size=32, shuffle=True)
#     with tqdm(total=len(train_dataloader)) as Tqdm:
#         for i in train_dataloader:
#             Tqdm.update(1)
#             pass
#     print(train_dataloader.num_error)



def main():
    copy_file(save_file_dir)
    if args.mode == 'train':
        train_dataloader = DataLoader(os.path.join(args.root_dir,args.dir), args.limits, 'train').set_attrs(batch_size=args.batch_size, shuffle=True)
        dev_dataloader = DataLoader(os.path.join(args.root_dir,args.dir), args.limits, 'dev').set_attrs(batch_size=args.batch_size_valid, shuffle=False)
    test_dataloader = DataLoader(os.path.join(args.root_dir,args.dir), args.limits, 'test').set_attrs(batch_size=args.batch_size_valid, shuffle=False)
    if args.mode == 'train':
        log.info(f'erro_data: train {train_dataloader.num_error}, dev {dev_dataloader.num_error}, test {test_dataloader.num_error}')
    else:
        log.info(f'erro_data: test {test_dataloader.num_error}')

    model = SeqUnit(batch_size=args.batch_size, hidden_size=args.hidden_size, emb_size=args.emb_size,
                        field_size=args.field_size, pos_size=args.pos_size, field_vocab=args.field_vocab,
                        source_vocab=args.source_vocab, position_vocab=args.position_vocab,
                        target_vocab=args.target_vocab, scope_name="seq2seq", name="seq2seq",
                        field_concat=args.field, position_concat=args.position,
                        fgate_enc=args.fgate_encoder, dual_att=args.dual_attention, decoder_add_pos=args.decoder_pos,
                        encoder_add_pos=args.encoder_pos, learning_rate=args.learning_rate)
    if args.load != '0':
        model.load(save_dir)
    if args.mode == 'train':
        train(model,train_dataloader,dev_dataloader,test_dataloader )
    else:
        test(model,test_dataloader)

if __name__=='__main__':
    args = do_before_running()

    os.chdir(args.root_dir)
    last_best = 0.0
    gold_path_test = 'processed_data/test/test_split_for_rouge/gold_summary_'
    gold_path_valid = 'processed_data/valid/valid_split_for_rouge/gold_summary_'
    # test phase
    if args.load != "0":
        save_dir = os.path.dirname(args.output_dir)+ f'/{args.load}' + '/res/'
        save_file_dir = save_dir + '/files/'
        pred_dir = os.path.dirname(args.output_dir)+ f'/{args.load}' + '/evaluation/'
        if not os.path.exists(pred_dir):
            os.mkdir(pred_dir)
        if not os.path.exists(save_file_dir):
            os.mkdir(save_file_dir)
        pred_path = pred_dir + 'pred_summary_'
        pred_beam_path = pred_dir + 'beam_summary_'
    # train phase
    else:
        save_dir = args.output_dir + '/res/'
        save_file_dir = save_dir + 'files/'
        pred_dir = args.output_dir + '/evaluation/'
        os.mkdir(save_dir)
        if not os.path.exists(pred_dir):
            os.mkdir(pred_dir)
        if not os.path.exists(save_file_dir):
            os.mkdir(save_file_dir)
        pred_path = pred_dir + 'pred_summary_'
        pred_beam_path = pred_dir + 'beam_summary_'

    log_file = args.logger_file_name

    if jittor.compiler.has_cuda and args.use_cuda:
        jittor.flags.use_cuda = 1
        log.info('use cuda!')
    else:
        log.info('use cpu!')

    main()

    log.info('all finished')


