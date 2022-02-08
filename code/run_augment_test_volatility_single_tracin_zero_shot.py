import os
import warnings

warnings.filterwarnings('ignore')

from E_MDCA_transformer import *
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils import data
# from sklearn.model_selection import train_test_split
# from torchtext import data, datasets, vocab
# import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.special import softmax
import seaborn as sns;

sns.set_theme()
import random, tqdm, sys, math, gzip
from sklearn.metrics import classification_report, accuracy_score

from argparse import ArgumentParser
import zipfile
import pdb
import uuid
import pickle
from collections import Counter
import pif as ptif
from pif.influence_functions.influence_functions import calc_all_grad, calc_all_grad_then_test
import copy

parser = ArgumentParser(description='PyTorch Hierachical Transformer for New Forecaster')
parser.add_argument("--num_epochs", type=int, default=5)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--lr", type=float, default=2e-5)
parser.add_argument("--embedding_size", type=int, default=768)
parser.add_argument("--vocab_size", type=int, default=50000)
parser.add_argument("--max_length", type=int, default=500)
parser.add_argument("--depth", type=int, default=2)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--lr_warmup", type=int, default=500)
# parser.add_argument("--paras_times", type=int, default=1,
#                     help='paragraphs rotation times, 1 means no paragraphs rotation')
# parser.add_argument("--sents_times", type=int, default=1,
#                     help='sentences rotation times, 1 means no sentences rotation')
parser.add_argument("--gradient_clipping", type=float, default=1.0)
parser.add_argument("--target_folder", type=str, default="../data/")  # ../generate_sent_emb/volatility/
parser.add_argument("--target_saved_model_folder", type=str, default="../saved_model/")
parser.add_argument("--train_pkl", type=str, default="train_sample.pkl")
# parser.add_argument("--train_pkl", type=str, default="train_opening_eca_aligned_res_volatility_each_sentence.json.pkl")
parser.add_argument("--dev_pkl", type=str, default="dev_sample.pkl")
parser.add_argument("--test_pkl", type=str, default="test_sample.pkl")
parser.add_argument("--alpha", type=float, default=0.5)
parser.add_argument("--task", type=str, default="cls")
parser.add_argument("--label_size", type=int, default=3)
parser.add_argument("--round", type=int, default=5)
parser.add_argument("--idx", type=int, default=3,
                    help="Idx means different volatility indicator, 0~9 represents r3_daily, r7_daily, r15_daily, r30_daily, r90_daily, r3_mean, r7_mean, r15_mean,r30_mean, r90_mean respectively")
parser.add_argument("--max_pool", action='store_true')
parser.add_argument("--factor", type=int, default=10)
parser.add_argument("--d_model", type=int, default=512)
parser.add_argument("--n_heads", type=int, default=8)
parser.add_argument("--e_layers", type=int, default=3)
parser.add_argument("--d_layers", type=int, default=2)
parser.add_argument("--dropout", type=int, default=0.3)
parser.add_argument("--d_ff", type=int, default=512)
parser.add_argument("--cuda", action='store_true', default=True)
parser.add_argument("--mode", action='store_true', default='finetune')
parser.add_argument("--model", type=str, default='cosformer')
parser.add_argument("--aug_type", type=str, default='zero_shot', help='pn')
parser.add_argument("--content", type=str, default='op')
parser.add_argument("--kl", type=int, default=0)
parser.add_argument("--dcg", type=bool, default=True)
parser.add_argument("--num_topics", type=int, default=1)

arg = parser.parse_args()

uid = str(uuid.uuid4())[:4]

arg.saved_model_path = arg.target_saved_model_folder + '{5}_model{8}_b{0}_el{1}_f{2}_h{3}_dp{9}_md{10}_{4}_idx{6}_kl{11}_uuid{7}.out'.format(
    arg.batch_size, arg.e_layers, arg.factor, arg.n_heads, arg.lr, arg.task, arg.idx, uid, arg.model, arg.dropout,
    arg.d_model, arg.kl)

assert arg.mode == 'finetune' or arg.mode == 'retrain'
np.random.seed(arg.seed)
torch.manual_seed(arg.seed)

# if torch.cuda.is_available():
#     if not arg.cuda:
#         print('WARNING: You have a CUDA device, so you should probably run with --cuda')
#     else:
#         torch.cuda.manual_seed_all(arg.seed)

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
# device = 'cpu'

# all_considered_sents = pickle.load(open('top20_topic_related_EC_sentence_using_bm25.pkl', 'rb'))


model_dict = {
    'E_MDCA': Informer_simple(enc_in=arg.embedding_size, dec_in=arg.embedding_size, c_out=arg.embedding_size,
                                 seq_len=arg.max_length, label_len=arg.label_size, out_len=0,
                                 factor=arg.factor, d_model=arg.d_model, n_heads=arg.n_heads, e_layers=arg.e_layers,
                                 d_ff=arg.d_ff,
                                 dropout=arg.dropout, attn='prob', embed='fixed', freq='h', activation='gelu',
                                 output_attention=True, distil=True, mix=True, device=device),
    'MDRM': MDRM(dropout=arg.dropout, emb_size=768, d_model=arg.d_model, label_num=arg.label_size),
    'BOW': BOW(num_labels=arg.label_size, emb_size=768, d_model=100, vocab_size=arg.vocab_size),
    'HTML': RRTransformer(emb=arg.embedding_size, heads=arg.n_heads, depth=arg.depth,
                          seq_length=arg.max_length, num_tokens=arg.vocab_size, label_size=arg.label_size,
                          max_pool=arg.max_pool, dropout=arg.dropout),
    'Informer': Informer(enc_in=arg.embedding_size, dec_in=arg.embedding_size, c_out=arg.embedding_size,
                         seq_len=arg.max_length, label_len=arg.label_size, out_len=0,
                         factor=arg.factor, d_model=arg.d_model, n_heads=arg.n_heads, e_layers=arg.e_layers,
                         d_layers=arg.d_layers, d_ff=arg.d_ff,
                         dropout=arg.dropout, attn='prob', embed='fixed', freq='h', activation='gelu',
                         output_attention=True, distil=True, mix=True,
                         device=device)
    }

assert arg.model in model_dict
assert arg.aug_type in ['zero_shot', 'pn']


class Dataset(data.Dataset):
    def __init__(self, docs, val, label):
        'Initialization'
        self.docs = docs
        self.val = val
        self.label = label

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.label)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        if torch.is_tensor(index):
            index = index.tolist()

        # Load data and get label
        X = self.docs[index, :, :]
        y = self.val[index]
        y_b = self.label[index, :]
        return X, y, y_b


class neg_Dataset(data.Dataset):
    def __init__(self, docs):
        'Initialization'
        self.docs = docs

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.docs)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        if torch.is_tensor(index):
            index = index.tolist()

        # Load data and get label
        X = self.docs[index, :, :]
        return X


def draw_heat_map(heat_map_matrix, cnt, num_b=0, num_head=0):
    matrix = heat_map_matrix.cpu().detach().numpy()

    matrix = softmax(matrix[num_b][num_head], axis=1)
    matrix = matrix.tolist()

    #     for num, matrix in enumerate(heat_map_matrix):
    print('drawing the {0}-th batch and the {1}-th head attention !'.format(num_b, num_head))

    fig, ax = plt.subplots(figsize=(20, 16))
    ticks = [str(i) for i in range(len(matrix))]
    sns.heatmap(matrix, vmin=0, vmax=1, cmap="Blues",  # annot_kws={"size": 25},
                xticklabels=ticks, yticklabels=ticks, ax=ax)
    #     plt.xticks(rotation=15)
    #     plt.yticks(rotation=30)
    #         ax.tick_params(labelsize=25)
    # ax.figure.axes[-1].yaxis.label.set_size(25)
    #         cax = plt.gcf().axes[-1]
    #         cax.tick_params(labelsize=24)
    plt.savefig('figures/attention_{}_test.png'.format(cnt))
    print('finish drawing the {0}-th one!'.format(cnt))


def retrieve_top_K_important_sentences(docs_considered, attns, K=5):
    attns = attns.cpu().detach().numpy()
    #     print('attns: ', attns.shape, len(docs_considered), len(docs_considered[0]))
    #     print(docs_considered[0])
    results = []
    for idx, attn in enumerate(attns):
        ind = np.argsort(attn[:len(docs_considered[idx])])[::-1][:K]
        results.append([docs_considered[idx][i] for i in ind])
    return results


def check_range_idx(val, idx=0):
    if idx == 0:
        label = check_range(val, -4.842167985588413, -3.8491269844660705)
    elif idx == 1:
        label = check_range(val, -4.9267306708110485, -3.929482919485536)
    elif idx == 2:
        label = check_range(val, -4.985104242587564, -3.9460967548642953)
    elif idx == 3:
        label = check_range(val, -4.926683464074086, -3.8857175818820613)
    elif idx == 4:
        label = check_range(val, -4.853248520031541, -3.811075067677807)
    elif idx == 5:
        label = check_range(val, -4.879759482001096, -3.901871160228038)
    elif idx == 6:
        label = check_range(val, -4.257784041477206, -3.620774670138319)
    elif idx == 7:
        label = check_range(val, -4.183695440167521, -3.600704745838628)
    elif idx == 8:
        label = check_range(val, -4.128396917743695, -3.555616323172549)
    elif idx == 9:
        label = check_range(val, -4.052031703095732, -3.4527679278407106)
    return label


def check_range(val, val_a, val_b):
    if val < val_a:
        label = 0
    elif val > val_b:
        label = 2
    else:
        label = 1
    return label


def check_sign(val):
    return 0 if val < 0 else 1


def padding(docs_emb):
    # Padding
    #     b = np.zeros([len(docs_emb), len(max(docs_emb, key=lambda x: len(x))), arg.embedding_size])
    #     b = np.zeros([len(docs_emb), len(max(docs_emb, key=lambda x: len(x))), 221])
    b = np.zeros([len(docs_emb), arg.max_length, arg.embedding_size])
    cnt = 0
    # pdb.set_trace()
    for i, j in enumerate(docs_emb):
        try:
            b[i][0:len(j), :] = j
        except Exception as err:
            # pdb.set_trace()
            cnt += 1
            # print(err)
    #     print("null cnt {0}".format(cnt))
    return b


def flatten(emb):
    new_emb = []
    for doc in emb:
        sents = []
        for sent in doc:
            sents.extend(sent)
        new_emb.append(sents)
    return new_emb


def get_X_n_y_vec(data_pkl, idx=0, mode=None):
    # idx = 0 : r3_daily
    # idx = 1 : r7_daily
    # idx = 2 : r15_daily
    # idx = 3 : r30_daily
    # idx = 4 : r90_daily
    # idx = 5 : r3_mean
    # idx = 6 : r7_mean
    # idx = 7 : r15_mean
    # idx = 8 : r30_mean
    # idx = 9 : r90_mean
    X_vec = []
    y_vec = []
    y_vec_b = []
    X_docs = []

    aug_embeds = defaultdict(list)
    aug_texts = defaultdict(list)
    unique_topic = []
    cnt = 0
    for i, data in enumerate(data_pkl):
        #         print(data.keys())
        if data["y_emb"][idx] is not None:

            y_vec_b.append([check_range_idx(data["y_emb"][idx], idx)])

            X_vec.append(data["docs_emb"])
            y_vec.append(data["y_emb"][idx])

            if arg.content == 'op':
                assert 'opening' in data
                if mode == 'train':
                    for topic, instance in data['op_info'].items():
                        aug_embeds[cnt].append((topic, instance["op_embeds"]))
                        #                    print(instance["op_embeds"][0])
                        #                    print('==========================================')
                        #                    print('\n')
                        aug_texts[cnt].append((topic, instance["op_texts"]))
                        unique_topic += [topic]
                    cnt += 1

            elif arg.content == 'qa':
                # assert 'qa' in data
                if 'qa' not in data: continue

                if mode == 'train':
                    for topic, instance in data['qa_info'].items():
                        aug_embeds[cnt].append((topic, instance["qa_embeds"]))
                        aug_texts[cnt].append((topic, instance["qa_texts"]))
                        unique_topic += [topic]
                    cnt += 1

            else:
                if mode == 'train':
                    for topic, instance in data['op_info'].items():
                        aug_embeds[cnt].append((topic, instance["op_embeds"]))
                        aug_texts[cnt].append((topic, instance["op_texts"]))
                        unique_topic += [topic]

                    if 'qa' not in data: continue
                    for topic, instance in data['qa_info'].items():
                        aug_embeds[cnt].append((topic, instance["qa_embeds"]))
                        aug_texts[cnt].append((topic, instance["qa_texts"]))
                        unique_topic += [topic]
                    cnt += 1

    #                    if cnt > 10:
    #                        break

    X_docs = []  ## replace
    return X_vec, y_vec, y_vec_b, X_docs, aug_embeds, aug_texts, set(unique_topic)


def compute_kl_loss(p, q, pad_mask=None):
    p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='none')
    q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction='none')

    # pad_mask is for seq-level tasks
    if pad_mask is not None:
        p_loss.masked_fill_(pad_mask, 0.)
        q_loss.masked_fill_(pad_mask, 0.)

    # You can choose whether to use function "sum" and "mean" depending on your task
    p_loss = p_loss.sum()
    q_loss = q_loss.sum()

    loss = (p_loss + q_loss) / 2
    return loss


def go(arg):
    print("Processing data to get sentence embedding ...")
    # get sentence embedding for training data
    #     print(os.path.join(arg.target_folder, arg.train_pkl))

    docs_train = pickle.load(open(os.path.join(arg.target_folder, arg.train_pkl), 'rb'))

    X_train_emb, y_train_emb, y_train_b_emb, X_train_docs, aug_embs, aug_txts, unique_topic = get_X_n_y_vec(docs_train,
                                                                                                            idx=arg.idx,
                                                                                                            mode='train')

    # get sentence embedding for dev data
    docs_dev = pickle.load(open(os.path.join(arg.target_folder, arg.dev_pkl), 'rb'))

    X_dev_emb, y_dev_emb, y_dev_b_emb, X_dev_docs, _, _, _ = get_X_n_y_vec(docs_dev, idx=arg.idx)
    # get sentence embedding for test data
    docs_test = pickle.load(open(os.path.join(arg.target_folder, arg.test_pkl), 'rb'))

    X_test_emb, y_test_emb, y_test_b_emb, X_test_docs, _, _, _ = get_X_n_y_vec(docs_test, idx=arg.idx)
    print('Embedding processing DONE!')

    # pdb.set_trace()
    # prepare the dev and test datasets
    X_train_embs = padding(flatten(X_train_emb))
    X_dev_embs = padding(flatten(X_dev_emb))
    X_test_embs = padding(flatten(X_test_emb))

    train_set = Dataset(X_train_embs, np.array(y_train_emb), np.array(y_train_b_emb))
    dev_set = Dataset(X_dev_embs, np.array(y_dev_emb), np.array(y_dev_b_emb))
    test_set = Dataset(X_test_embs, np.array(y_test_emb), np.array(y_test_b_emb))

    trainloader = torch.utils.data.DataLoader(train_set, batch_size=arg.batch_size, shuffle=False, num_workers=1)
    devloader = torch.utils.data.DataLoader(dev_set, batch_size=1, shuffle=False, num_workers=1)
    testloader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=1)

    if arg.mode == 'finetune':

        model = model_dict[arg.model]

        if torch.cuda.device_count() > 1:
            print("DataParallel to train")
            print("Using", torch.cuda.device_count(), torch.cuda.get_device_name(device), "GPUs to train!")
            para_model = nn.DataParallel(model).to(device)
        else:
            #         print("Using", torch.cuda.device_count(), torch.cuda.get_device_name(device), "GPUs to train!")
            para_model = model.to(device)
        #         para_model = model.to(device)

        opt = torch.optim.Adam(lr=arg.lr, params=para_model.parameters())

    # training loop
    neg_flag = False
    best_model = None

    for a_e in tqdm.tqdm(range(arg.round)):

        if arg.mode == 'finetune':

            if best_model is not None:
                para_model = torch.load(arg.saved_model_path).to(device)
                opt = torch.optim.Adam(lr=arg.lr, params=para_model.parameters())

        if arg.mode == 'retrain':

            model = model_dict[arg.model]

            if torch.cuda.device_count() > 1:
                print("DataParallel to train")
                print("Using", torch.cuda.device_count(), torch.cuda.get_device_name(device), "GPUs to train!")
                para_model = nn.DataParallel(model).to(device)
            else:
                #         print("Using", torch.cuda.device_count(), torch.cuda.get_device_name(device), "GPUs to train!")
                para_model = model.to(device)
            #             para_model = model.to(device)

            opt = torch.optim.Adam(lr=arg.lr, params=para_model.parameters())

        seen = 0

        best_acc_epoch = 0
        best_report = 0
        best_epoch = 0
        #         print("paragraphs rotation:{0}, sentences rotation:{1}, augmentation num:{2}".format(arg.paras_times,
        #                                                                                      arg.sents_times, a_e+1))

        #        new_aug_X_data, new_aug_y_data, new_aug_y_data_b, aug_neg_X_data = training_data_augmentation_zero_shot(X_train_emb, y_train_emb, y_train_b_emb, para_model, aug_embs, aug_txts, unique_topic, a_e)

        if a_e > 0:
            print(len(X_train_emb), len(new_aug_X_data))
            X_train_embs = np.concatenate([padding(flatten(X_train_emb)), new_aug_X_data], axis=0)
            train_set = Dataset(X_train_embs, np.array(y_train_emb + new_aug_y_data),
                                np.array(y_train_b_emb + new_aug_y_data_b))
            trainloader = torch.utils.data.DataLoader(train_set, batch_size=arg.batch_size, shuffle=False,
                                                      num_workers=1)

            if neg_flag:
                X_train_embs = aug_neg_X_data
                neg_train_set = neg_Dataset(X_train_embs)
                neg_trainloader = torch.utils.data.DataLoader(neg_train_set, batch_size=1, shuffle=False, num_workers=1)
                neg_train_x = [x for i, x in enumerate(neg_trainloader)]

        for e in tqdm.tqdm(range(arg.num_epochs)):
            # for e in tqdm.tqdm(range(1)):
            train_loss_tol = 0.0
            para_model.train(True)
            train_times = 0
            train_prediction_labels_list, train_true_labels_list = [], []
            print('Starting Training !!')
            for i, data in enumerate(trainloader):
                if arg.lr_warmup > 0 and seen < arg.lr_warmup:
                    lr = max((arg.lr / arg.lr_warmup) * seen, 1e-10)
                    opt.lr = lr

                opt.zero_grad()

                inputs, values, labels = data
                #                 print(type(inputs))

                if arg.cuda:
                    inputs = Variable(inputs.type(torch.FloatTensor)).cuda()
                    values = torch.tensor(values, dtype=torch.float32).cuda()
                    labels = torch.tensor(labels, dtype=torch.long).cuda()
                else:
                    inputs = Variable(inputs.type(torch.FloatTensor))
                    values = torch.tensor(values, dtype=torch.float32)
                    labels = torch.tensor(labels, dtype=torch.long)

                #                     docs_considered = X_train_docs[start_doc_num: start_doc_num + arg.batch_size]
                #                 start_doc_num = start_doc_num + arg.batch_size

                if inputs.size(1) > arg.max_length:
                    inputs = inputs[:, :arg.max_length, :]

                if arg.task == 'cls':

                    if not arg.kl:

                        if arg.model not in ['E_MDCA', 'Informer']:
                            out_b = para_model(inputs)
                        else:
                            out_b, attns = para_model(inputs, inputs, inputs, inputs)
                        loss = F.cross_entropy(out_b.view(-1, arg.label_size), labels.view(-1))
                    #                         results = retrieve_top_K_important_sentences(docs_considered, attns, K=3)
                    #                         print(results)
                    #                         print('start drawing !')
                    else:

                        if arg.model not in ['E_MDCA', 'Informer']:
                            out_b = para_model(inputs, inputs, inputs, inputs)
                            loss = F.cross_entropy(out_b.view(-1, arg.label_size), labels.view(-1))
                        else:
                            out_b, attns = para_model(inputs, inputs, inputs, inputs)
                            out_b1, _ = para_model(inputs, inputs, inputs, inputs)

                            # cross entropy loss for classifier
                            ce_loss = 0.5 * (F.cross_entropy(out_b.view(-1, arg.label_size),
                                                             labels.view(-1)) + F.cross_entropy(
                                out_b1.view(-1, arg.label_size), labels.view(-1)))
                            kl_loss = compute_kl_loss(out_b, out_b1)

                            # carefully choose hyper-parameters
                            loss = ce_loss + 0.3 * kl_loss

                    train_loss_tol += loss

                    label_logits = torch.argmax(F.log_softmax(out_b, dim=1), dim=1)
                    pred_labels = label_logits.detach().cpu().numpy()
                    label_ids = labels.to('cpu').numpy()
                    true_label = [item[0] for item in label_ids.tolist()]
                    train_prediction_labels_list.extend(pred_labels)
                    train_true_labels_list.extend(true_label)



                else:
                    ValueError("Not support task type, please select from, 'reg', 'cls', 'both'.")

                loss.backward()

                # clip gradients
                # - If the total gradient vector has a length > 1, we clip it back down to 1.
                if arg.gradient_clipping > 0.0:
                    nn.utils.clip_grad_norm_(model.parameters(), arg.gradient_clipping)

                opt.step()

                seen += inputs.size(0)

                train_times += 1

                ## negative samples
                if neg_flag and (i <= len(neg_train_x) - 1):

                    inputs = neg_train_x[i]

                    if arg.cuda:
                        inputs = Variable(inputs.type(torch.FloatTensor)).cuda()
                        labels = torch.tensor([1 / 3, 1 / 3, 1 / 3]).view(1, 3).float().cuda()
                    else:
                        inputs = Variable(inputs.type(torch.FloatTensor))
                        labels = torch.tensor([1 / 3, 1 / 3, 1 / 3]).view(1, 3).float()

                    if inputs.size(1) > arg.max_length:
                        inputs = inputs[:, :arg.max_length, :]

                    if arg.task == 'cls':

                        if arg.model not in ['E_MDCA', 'Informer']:
                            out_b = para_model(inputs)
                            loss = compute_kl_loss(out_b, labels)
                        else:
                            out_b, attns = para_model(inputs, inputs, inputs, inputs)
                            out_b1, _ = para_model(inputs, inputs, inputs, inputs)

                            loss = 0.5 * (compute_kl_loss(out_b, labels) + compute_kl_loss(out_b1, labels))
                            kl_loss = compute_kl_loss(out_b, out_b1)
                            loss = loss + 0.3 * kl_loss

                        train_loss_tol += loss

                        loss.backward()

                    else:
                        ValueError("Not support task type, please select from, 'reg', 'cls', 'both'.")

                    # clip gradients
                    # - If the total gradient vector has a length > 1, we clip it back down to 1.
                    if arg.gradient_clipping > 0.0:
                        nn.utils.clip_grad_norm_(model.parameters(), arg.gradient_clipping)

                    opt.step()

                    seen += inputs.size(0)

                    train_times += 1
            train_acc_label = accuracy_score(train_true_labels_list, train_prediction_labels_list)
            print("epoch: {0}, train loss: {1}, acc in train: {2}".format(e, train_loss_tol, train_acc_label.item()))

            if e == 0:
                print('training examples', len(train_set))
                print('dev examples', len(dev_set))
                print('test examples', len(test_set))
            train_loss_tol = train_loss_tol / train_times  # avg loss

            with torch.no_grad():
                para_model.train(False)

                ######### for dev data #############################
                total_epoch_loss = 0.0
                dev_prediction_labels_list, dev_true_labels_list = [], []
                start_doc_num = 0

                for i, data in enumerate(devloader):
                    inputs, values, labels = data
                    #                 docs_considered = X_dev_docs[start_doc_num: start_doc_num + 1]
                    start_doc_num = start_doc_num + 1

                    if arg.cuda:
                        inputs = torch.tensor(inputs, dtype=torch.float32).cuda()
                        values = torch.tensor(values, dtype=torch.float32).cuda()
                        labels = torch.tensor(labels, dtype=torch.long).cuda()
                    else:
                        inputs = torch.tensor(inputs, dtype=torch.float32)
                        values = torch.tensor(values, dtype=torch.float32)
                        labels = torch.tensor(labels, dtype=torch.long)

                    if inputs.size(1) > arg.max_length:
                        inputs = inputs[:, :arg.max_length, :]

                    if arg.task == 'cls':

                        if arg.model not in ['E_MDCA', 'Informer']:
                            out_b = para_model(inputs)
                        else:
                            #                     out_b = para_model(inputs)
                            #                     print(inputs.size())
                            out_b, attns = para_model(inputs, inputs, inputs, inputs)
                        #                     results = retrieve_top_K_important_sentences(docs_considered, attns, K=3)
                        #                     print(results)

                        loss = F.cross_entropy(out_b.view(-1, arg.label_size), labels.view(-1))
                        total_epoch_loss += loss.item()

                        label_logits = torch.argmax(F.log_softmax(out_b, dim=1), dim=1)
                        pred_labels = label_logits.detach().cpu().numpy()
                        label_ids = labels.to('cpu').numpy()
                        true_label = [item[0] for item in label_ids.tolist()]
                        dev_prediction_labels_list.extend(pred_labels)
                        dev_true_labels_list.extend(true_label)

            if arg.task == 'cls':
                dev_acc_label = accuracy_score(dev_true_labels_list, dev_prediction_labels_list)
                total_epoch_loss = total_epoch_loss / len(devloader)
                print("in dev data")
                print("pred counter:", Counter(dev_prediction_labels_list))
                print("true counter:", Counter(dev_true_labels_list))

            if arg.task == 'cls':
                print("epoch: {0}, train loss: {1}, acc loss in dev: {2}, acc classifier in dev: {3}".format(e,
                                                                                                             train_loss_tol,
                                                                                                             total_epoch_loss,
                                                                                                             dev_acc_label.item()))
                if best_acc_epoch < dev_acc_label.item():
                    best_acc_epoch = dev_acc_label.item()
                    best_epoch = e
                    best_report = classification_report([str(s) for s in dev_true_labels_list],
                                                        [str(s) for s in dev_prediction_labels_list])
                    torch.save(para_model, arg.saved_model_path)

        if arg.task == 'cls':
            print("best classification report {0}".format(best_report))
            print("best acc is: {0} in epoch {1}".format(best_acc_epoch, best_epoch))

        best_model = torch.load(arg.saved_model_path)

        with torch.no_grad():
            best_model.train(False)

            ######### for test data #############################

            test_prediction_labels_list, test_true_labels_list = [], []
            start_doc_num = 0
            for i, data in enumerate(testloader):
                inputs, values, labels = data
                #             docs_considered = X_test_docs[start_doc_num: start_doc_num + arg.batch_size]
                start_doc_num = start_doc_num + arg.batch_size

                if arg.cuda:
                    inputs = torch.tensor(inputs, dtype=torch.float32).cuda()
                    values = torch.tensor(values, dtype=torch.float32).cuda()
                    labels = torch.tensor(labels, dtype=torch.long).cuda()
                else:
                    inputs = torch.tensor(inputs, dtype=torch.float32)
                    values = torch.tensor(values, dtype=torch.float32)
                    labels = torch.tensor(labels, dtype=torch.long)

                if inputs.size(1) > arg.max_length:
                    inputs = inputs[:, :arg.max_length, :]

                if arg.task == 'cls':
                    #                 out_b = para_model(inputs)
                    #                     out_b, attns = para_model(inputs, inputs, inputs, inputs)
                    #                 results = retrieve_top_K_important_sentences(docs_considered, attns, K=3)
                    #                 print(results)
                    #                 if i == 0:
                    #                     draw_heat_map(attns[0], i)

                    if arg.model not in ['E_MDCA', 'Informer']:
                        out_b = best_model(inputs)
                    else:
                        out_b, attns = best_model(inputs, inputs, inputs, inputs)

                    label_logits = torch.argmax(F.log_softmax(out_b, dim=1), dim=1)
                    pred_labels = label_logits.detach().cpu().numpy()
                    label_ids = labels.to('cpu').numpy()
                    true_label = [item[0] for item in label_ids.tolist()]
                    test_prediction_labels_list.extend(pred_labels)
                    test_true_labels_list.extend(true_label)

        # if arg.task == 'cls':
        #    test_acc_label = accuracy_score(test_true_labels_list, test_prediction_labels_list)
        #    print("acc in test dataset is: {0}".format(test_acc_label))
        #    print(classification_report([str(s) for s in test_true_labels_list],
        #                                [str(s) for s in test_prediction_labels_list]))

        if a_e != 4:
            new_aug_X_data, new_aug_y_data, new_aug_y_data_b, aug_neg_X_data = training_data_augmentation_zero_shot(
                X_train_emb, y_train_emb, y_train_b_emb, para_model, aug_embs, aug_txts, unique_topic, a_e)
            #            new_aug_X_data, new_aug_y_data, new_aug_y_data_b, aug_neg_X_data = [], [], [], []
            neg_flag = False if aug_neg_X_data == [] else True


def dcg(influence, aug_topics, num_topics=1):
    influence_topic = sorted(list(zip(influence[0], aug_topics, list(range(len(influence[0]))))), reverse=True)
    result = defaultdict(float)
    pos_ind = []
    neg_ind = []

    for ind, (infl, topic, _) in enumerate(influence_topic):
        result[topic] += infl / math.log2(ind + 2)

    ranked_topic = [item[0] for item in sorted(result.items(), key=lambda x: x[1], reverse=True)]
    pos_topic = ranked_topic[:num_topics]
    neg_topic = ranked_topic[-num_topics:]
    for (_, topic, ind) in influence_topic:
        if topic in pos_topic:
            pos_ind.append(ind)
        elif topic in neg_topic:
            neg_ind.append(ind)
    # print(pos_ind, neg_ind)
    return pos_ind, neg_ind


def training_data_augmentation_zero_shot(X_train_emb, y_train_emb, y_train_b_emb, para_model, aug_embs, aug_txts,
                                         unique_topic, rnd):
    config = {
        "outdir": '../code/IF/roberta-large-model_5_train_w_qa/',
    # the dir should combine dataset and method, e.g. IF-model_1_val_w_qa
        "seed": 42,
        "gpu": 0,
        "recursion_depth": 10,  # set recursion to use entire training data
        "r_averaging": 1,
        "scale": 25,
        "damp": 0.01,
        "num_classes": 3,
        #            "test_sample_num": args.test_sample_num,
        "test_start_index": 0,
        "test_end_index": 0,
        "test_hessian_start_index": 0,
        "test_hessian_end_index": 10,
        "train_start_index": 0,
        "train_end_index": 0,
        "log_filename": None,
    }

    ptif.init_logging(config["log_filename"])

    new_aug_X_data = []
    new_aug_y_data = []
    new_aug_y_data_b = []
    pos_X_text = []
    neg_X_data = []
    neg_X_text = []

    all_selected_indices = np.random.choice(range(len(X_train_emb)), size=len(X_train_emb), replace=False)

    for idx in all_selected_indices:

        if aug_embs[idx] == []:
            continue

        # print(len(aug_embs[idx]))

        X_train_embs = padding(flatten([X_train_emb[idx]]))
        # print(X_train_emb[idx])
        y_train_embs = [y_train_emb[idx]]
        y_train_b_embs = [y_train_b_emb[idx]]

        train_set = Dataset(X_train_embs, np.array(y_train_embs), np.array(y_train_b_embs))
        trainloader = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=False, num_workers=1)

        ind_check_pos = defaultdict(list)
        cnt = 0
        # print('total len: ', len(aug_embs[idx]))
        aug_pos_emb = [aug_embs[idx][i][1] for i in range(len(aug_embs[idx]))]
        aug_topics = [aug_embs[idx][i][0] for i in range(len(aug_embs[idx]))]
        aug_pos_text = [(aug_txts[idx][i][0], aug_txts[idx][i][1][0], aug_txts[idx][i][1][1]) for i in
                        range(len(aug_embs[idx]))]

        aug_neg_emb = [aug_embs[idx][i][1] for i in range(len(aug_embs[idx]))]
        aug_neg_text = [(aug_txts[idx][i][0], aug_txts[idx][i][1][0], aug_txts[idx][i][1][2]) for i in
                        range(len(aug_embs[idx]))]

        X_aug_embs = padding(flatten(aug_pos_emb))
        # print(X_aug_embs.shape)
        y_vec = [0 for _ in range(len(aug_pos_emb))]
        y_vec_b = [[0] for _ in range(len(aug_pos_emb))]

        #     print(len(X_train_embs), len(y_vec), len(y_vec_b))
        aug_set = Dataset(X_aug_embs, np.array(y_vec), np.array(y_vec_b))
        augloader = torch.utils.data.DataLoader(aug_set, batch_size=1, shuffle=False, num_workers=1)
        # print(len(trainloader), len(augloader))
        influence_results = calc_all_grad_then_test(config, arg.model, para_model, trainloader, augloader,
                                                    calculate_if=False)
        #     print(len(influence_results["influences"]), len(influence_results["influences"][0]))

        if arg.aug_type == 'zero_shot':

            if arg.dcg:
                pos_ind, neg_ind = dcg(influence_results["influences"], aug_topics, num_topics=arg.num_topics)
                pos_cand = [X_aug_embs[ind] for ind in pos_ind]
                neg_cand = [X_aug_embs[ind] for ind in neg_ind]
                l1, l2, l3 = len(pos_cand), X_aug_embs.shape[1], X_aug_embs.shape[2]
                if l1 == 1:
                    pos_x = pos_cand[0]
                else:
                    pos_x = copy.deepcopy(X_train_embs)
                    for i in range(l1):
                        pos_x = np.where(pos_x != pos_cand[i], pos_cand[i], pos_x)

                l1, l2, l3 = len(neg_cand), X_aug_embs.shape[1], X_aug_embs.shape[2]
                if l1 == 1:
                    neg_x = neg_cand[0]
                else:
                    neg_x = copy.deepcopy(X_train_embs)
                    for i in range(l1):
                        neg_x = np.where(neg_x != neg_cand[i], neg_cand[i], neg_x)

            else:
                influence = influence_results["influences"]
                # print('inf: ', influence)
                # print('rank: ', np.argmax(influence))
                x = aug_pos_emb[np.argmax(influence)]

            new_aug_X_data.append(pos_x)
            neg_X_data.append(neg_x)

            if not arg.dcg:
                x = padding(flatten([x]))
            x = torch.tensor(pos_x, dtype=torch.float32).cuda()
            pos_txt = [aug_pos_text[i] for i in pos_ind]
            neg_txt = [aug_neg_text[i] for i in neg_ind]
            # print(x.size())
            if arg.model in ['E_MDCA', 'Informer']:
                logits, _ = para_model(x, x, x, x)
            else:
                logits = para_model(x)

            label = torch.argmax(F.log_softmax(logits, dim=1), dim=1)
            label = label.detach().cpu().item()

            # print('predicted label: ', label)

            pos_X_text.append(pos_txt)
            neg_X_text.append(neg_txt)
            new_aug_y_data.append(label)
            new_aug_y_data_b.append([label])

        else:

            for i, influence in enumerate(influence_results["influences"]):

                ind = selected_indices[i]

                if ind in ind_check_pos:
                    indices = np.array(ind_check_pos[ind])
                    influence = np.array(influence)[indices]

                    new_aug_X_data += [aug_sample[np.argmax(influence)]]

                    new_aug_y_data += [y_train_emb[i]]

                    new_aug_y_data_b += [y_train_b_emb[i]]

    # save selected augmentation texts

    #     pickle.dump(pos_X_text, open('../results/model_{0}_{1}_{2}_pos.pkl'.format(arg.model, arg.train_pkl[12:-4], arg.idx), 'wb'))
    #     pickle.dump(neg_X_text, open('../results/model_{0}_{1}_{2}_neg.pkl'.format(arg.model, arg.train_pkl[12:-4], arg.idx), 'wb'))

    #     arg.saved_model_path = arg.target_saved_model_folder + '{5}_model{8}_b{0}_el{1}_f{2}_h{3}_dp{9}_md{10}_{4}_idx{6}_kl{11}_uuid{7}.out'.format(arg.batch_size, arg.e_layers, arg.factor, arg.n_heads, arg.lr, arg.task, arg.idx, uid, arg.model, arg.dropout, arg.d_model, arg.kl)

    print('Finished Augmentation !')

    return new_aug_X_data, new_aug_y_data, new_aug_y_data_b, neg_X_data


if __name__ == "__main__":
    print(arg)
    evaluation = go(arg)
    print(arg)