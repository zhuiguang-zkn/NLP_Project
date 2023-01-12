#coding=utf8
import sys, os, time, gc
from torch.optim import Adam
from math import sqrt
import pickle
from xpinyin import Pinyin
from datetime import datetime

install_path = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(install_path)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

from utils.args import init_args
from utils.initialization import *
from utils.example import Example
from utils.batch import from_example_list
from utils.vocab import PAD
from model.slu_baseline_tagging import SLUTagging


def preperation(args):

    # load configuration
    start_time = time.time()
    train_path = os.path.join(args.dataroot, 'train.json')
    dev_path = os.path.join(args.dataroot, 'development.json')
    ontology_path = os.path.join(args.dataroot, 'ontology.json')
    word2vec_path = args.word2vec_path
    if args.trainset_spoken_language_select == "both":
        args.trainset_spoken_language_select = ['asr_1best', 'manual_transcript']
    if args.trainset_augmentation:
        aug_path = os.path.join(args.dataroot, 'augmentation.json')
        train_path = [train_path, aug_path]
    else:
        train_path = train_path
    Example.configuration(vocab_path=train_path, 
                            ontology_path=ontology_path, 
                            word2vec_path=word2vec_path,
                            spoken_language_select=args.trainset_spoken_language_select,
                            word_embedding = args.word_embedding)
    
    # load dataset and preprocessing
    # train_dataset = Example.load_dataset(train_path)
    train_dataset = Example.load_dataset(train_path, spoken_language_select=args.trainset_spoken_language_select)
    dev_dataset = Example.load_dataset(dev_path, spoken_language_select='asr_1best')
    print("Load dataset and database finished, cost %.2f s ..." % (time.time() - start_time))
    print("Dataset size: train -> %d ; dev -> %d" % (len(train_dataset), len(dev_dataset)))

    # update some parameters based on corpus
    args.vocab_size = Example.word_vocab.vocab_size
    args.pad_idx = Example.word_vocab[PAD]
    args.num_tags = Example.label_vocab.num_tags
    args.tag_pad_idx = Example.label_vocab.convert_tag_to_idx(PAD)
        # changes will be stored in `args`

    # model
    device = set_torch_device(args.device)
    print("Use cuda:%s" % (args.device) if args.device >= 0 else "Use CPU as target torch device")

    return Example, train_dataset, dev_dataset, device


def set_optimizer(model, args):
    params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    grouped_params = [{'params': list(set([p for n, p in params]))}]
    optimizer = Adam(grouped_params, lr=args.lr)
    return optimizer

def decode(model, dataset, device, args):
    # assert choice in ['train', 'dev']
    model.eval()
    # dataset = train_dataset if choice == 'train' else dev_dataset
    predictions, labels = [], []
    total_loss, count = 0, 0
    with torch.no_grad():
        for i in range(0, len(dataset), args.batch_size):
            cur_dataset = dataset[i: i + args.batch_size]
            current_batch = from_example_list(args, cur_dataset, device, train=True)
            pred, label, loss = model.decode(Example.label_vocab, current_batch)
            predictions.extend(pred)
            labels.extend(label)
            total_loss += loss
            count += 1
        if args.anti_noise == True:
            predictions = anti_noise_prediction(predictions)
        metrics = Example.evaluator.acc(predictions, labels)
    torch.cuda.empty_cache()
        # RuntimeError: CUDA error: out of memory
        # CUDA kernel errors might be asynchronously reported at some other API call,so the stacktrace below might be incorrect.
    gc.collect()
    return metrics, total_loss / count


'''
先查找听到的内容是否与ontology中的slot相对应，如果没有就开始修改，并获取最相近的slot。

'''
def anti_noise_prediction(preds):
    p = Pinyin()
    pos = {'poi名称', 'poi修饰', 'poi目标', '起点名称', '起点修饰', '起点目标', '终点名称', '终点修饰', '终点目标', '途经点名称'}
    extra = {'请求类型' : [Example.label_vocab.request_map_dic, Example.label_vocab.request_pinyin_set],
             '出行方式' : [Example.label_vocab.travel_map_dic, Example.label_vocab.travel_pinyin_set],
             '路线偏好' : [Example.label_vocab.route_map_dic, Example.label_vocab.route_pinyin_set],
             '对象' : [Example.label_vocab.object_map_dic, Example.label_vocab.object_pinyin_set],
             '页码' : [Example.label_vocab.page_map_dic, Example.label_vocab.page_pinyin_set],
             '操作' : [Example.label_vocab.opera_map_dic, Example.label_vocab.opera_pinyin_set],
             '序列号' : [Example.label_vocab.ordinal_map_dic, Example.label_vocab.ordinal_pinyin_set] }
    for i, pred in enumerate(preds):
        if len(pred) == 0 :
            continue
        for j in range(len(pred)):
            sp = pred[j].split('-')
            pronun = p.get_pinyin(sp[2], ' ')
            if sp[1] != 'value':
                if sp[1] in pos:
                    map_dic, pinyin_set = Example.label_vocab.poi_map_dic, Example.label_vocab.poi_pinyin_set
                else:
                    [map_dic, pinyin_set] = extra[sp[1]]
                preds[i][j] = sp[0] + '-' + sp[1] + '-' + subst(map_dic, pinyin_set, pronun)                  
    return  preds            

def subst(map_dic, pinyin_set, noise) :
    if noise in pinyin_set :
        return map_dic[noise]
    cnt, prn = 0, ''
    for std_py in iter(pinyin_set) :
        simval = len(set(std_py.split(' ')) & set(noise.split(' '))) / (len(set(std_py.split(' '))) + len(set(noise.split(' '))))
        if simval > cnt :
            cnt = simval
            prn = std_py
    if cnt == 0 : 
        return '无' 
    return map_dic[prn]
        


if __name__=='__main__':
    args = init_args(sys.argv[1:])
    set_random_seed(args.seed)
    print("Initialization finished ...")
    Example, train_dataset, dev_dataset, device = preperation(args)

    model = SLUTagging(args).to(device)
    Example.word2vec.load_embeddings(model.word_embed, Example.word_vocab, device=device)
    if not args.testing:
        num_training_steps = ((len(train_dataset) + args.batch_size - 1) // args.batch_size) * args.max_epoch
        print('Total training steps: %d' % (num_training_steps))
        optimizer = set_optimizer(model, args)
        nsamples, best_result = len(train_dataset), {'dev_acc': 0., 'dev_f1': 0.}
        train_index, step_size = np.arange(nsamples), args.batch_size
        print('Start training ......')
        for i in range(args.max_epoch):
            start_time = time.time()
            epoch_loss = 0
            np.random.shuffle(train_index)
            model.train()
            count = 0
            for j in range(0, nsamples, step_size):
                cur_dataset = [train_dataset[k] for k in train_index[j: j + step_size]]
                current_batch = from_example_list(args, cur_dataset, device, train=True)
                output, loss = model(current_batch)
                epoch_loss += loss.item()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                count += 1
            print('Training: \tEpoch: %d\tTime: %.4f\tTraining Loss: %.4f' % (i, time.time() - start_time, epoch_loss / count))
            torch.cuda.empty_cache()
            gc.collect() # Run the garbage collector

            start_time = time.time()
            metrics, dev_loss = decode('dev')
            dev_acc, dev_fscore = metrics['acc'], metrics['fscore']
            print('Evaluation: \tEpoch: %d\tTime: %.4f\tDev acc: %.2f\tDev fscore(p/r/f): (%.2f/%.2f/%.2f)' % (i, time.time() - start_time, dev_acc, dev_fscore['precision'], dev_fscore['recall'], dev_fscore['fscore']))
            if dev_acc > best_result['dev_acc']:
                best_result['dev_loss'], best_result['dev_acc'], best_result['dev_f1'], best_result['iter'] = dev_loss, dev_acc, dev_fscore, i
                torch.save({
                    'epoch': i, 'model': model.state_dict(),
                    'optim': optimizer.state_dict(),
                }, open('RNN_model.bin', 'wb'))
                print('NEW BEST MODEL: \tEpoch: %d\tDev loss: %.4f\tDev acc: %.2f\tDev fscore(p/r/f): (%.2f/%.2f/%.2f)' % (i, dev_loss, dev_acc, dev_fscore['precision'], dev_fscore['recall'], dev_fscore['fscore']))

        print('FINAL BEST RESULT: \tEpoch: %d\tDev loss: %.4f\tDev acc: %.4f\tDev fscore(p/r/f): (%.4f/%.4f/%.4f)' % (best_result['iter'], best_result['dev_loss'], best_result['dev_acc'], best_result['dev_f1']['precision'], best_result['dev_f1']['recall'], best_result['dev_f1']['fscore']))
    else:
        start_time = time.time()
        metrics, dev_loss = decode('dev')
        dev_acc, dev_fscore = metrics['acc'], metrics['fscore']
        print("Evaluation costs %.2fs ; Dev loss: %.4f\tDev acc: %.2f\tDev fscore(p/r/f): (%.2f/%.2f/%.2f)" % (time.time() - start_time, dev_loss, dev_acc, dev_fscore['precision'], dev_fscore['recall'], dev_fscore['fscore']))
