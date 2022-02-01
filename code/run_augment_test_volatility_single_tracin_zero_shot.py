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
import seaborn as sns; sns.set_theme()
import random, tqdm, sys, math, gzip
from sklearn.metrics import classification_report, accuracy_score

from argparse import ArgumentParser
import zipfile
import pdb
import uuid
import pickle
from collections import Counter
import pif as ptif
from pif.influence_functions.influence_functions import calc_all_grad_then_test
import copy

parser = ArgumentParser(description='PyTorch Hierachical Transformer for New Forecaster')
parser.add_argument("--num_epochs", type=int, default=5)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--lr", type=float, default=1e-5)
parser.add_argument("--d_model",type=int,default=512)
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
parser.add_argument("--target_folder", type=str, default="data/")
parser.add_argument("--target_saved_model_folder", type=str, default="saved_model/")
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
parser.add_argument("--factor",type=int,default=10)
parser.add_argument("--n_heads",type=int,default=8)
parser.add_argument("--e_layers",type=int,default=3)
parser.add_argument("--d_layers",type=int,default=2)
parser.add_argument("--dropout",type=int,default=0.2)
parser.add_argument("--input_size", type=int, default=768)
parser.add_argument("--d_ff",type=int,default=512)
parser.add_argument("--cuda", action='store_true', default=True)
parser.add_argument("--mode", action='store_true', default='finetune')
parser.add_argument("--model", type=str, default='E_MDCA')
parser.add_argument("--aug_type",type=str,default='zero_shot',help='pn')
parser.add_argument("--alpha", type=float,default=0.3, help="KL hyper-parameter")

parser.add_argument("--kl",type=int,default=0)

arg = parser.parse_args()

uid = str(uuid.uuid4())[:4]

arg.saved_model_path = arg.target_saved_model_folder + '{5}_model_{8}_b{0}_el{1}_f{2}_h{3}_dp{9}_md{10}_{4}_idx{6}_kl{11}_uuid{7}.out'.format(arg.batch_size, arg.e_layers, arg.factor, arg.n_heads, arg.lr, arg.task, arg.idx, uid, arg.model, arg.dropout, arg.d_model, arg.kl)

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


model_dict = {'E_MDCA': EC_Encoder(enc_in=arg.input_size, dec_in=arg.input_size, c_out=arg.input_size,
                                 seq_len=arg.max_length, label_len=arg.label_size, out_len=0,
                                factor=arg.factor, d_model=arg.d_model, n_heads=arg.n_heads, e_layers=arg.e_layers, d_ff=arg.d_ff,
                                dropout=arg.dropout, attn='prob', embed='fixed', freq='h', activation='gelu',
                                output_attention = True, distil=True, mix=True, device=device),
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

    fig, ax = plt.subplots(figsize=(20,16))
    ticks = [str(i) for i in range(len(matrix))]
    sns.heatmap(matrix, vmin=0, vmax=1, cmap="Blues", #annot_kws={"size": 25},
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
    b = np.zeros([len(docs_emb), arg.max_length, arg.input_size])
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

    pos_checklist = defaultdict(list)
    neg_checklist = defaultdict(list)
    aug_embs = [[] for _ in range(5)]
    aug_pos_txts = [[] for _ in range(5)]
    aug_neg_txts = [[] for _ in range(5)]
    cnt_i = 0
    cnt_j = 0
    neg_cnt = 0

    for data in data_pkl:
#         print(data.keys())
        if data["y_emb"][idx] is not None:

            X_vec.append(data["docs_emb"])
            y_vec.append(data["y_emb"][idx])

#             print('docs_emb: ', len(data["docs_emb"]), data["docs_emb"][0].shape)

            if mode=='train':

                pos_checklist[cnt_i] = [[] for _ in range(5)]
                neg_checklist[cnt_i] = [[] for _ in range(5)]

                for rnd in range(5):
#                     if 'pos_augmented_data' not in data or 'neg_augmented_data' not in data or \
#                     'pos_augmented_texts' not in data or 'neg_augmented_texts' not in data:
#                         continue
                    try:
                        if data['pos_augmented_data'][rnd] != []:

        #                     print("found positive")

        #                     for emb in data['pos_augmented_data']:
                            for aug_num, (topic, emb) in enumerate(data['pos_augmented_data'][rnd]):

                                pos_checklist[cnt_i][rnd].append(cnt_j)
                                aug_embs[rnd].append(emb)
                                cnt_j += 1

                            for aug_num, (topic, text) in enumerate(data['pos_augmented_text'][rnd]):
                                aug_pos_txts[rnd].append([topic, text])
    #                             print(text)
    #                             print('==========================================================')
                    except:
                        pass


    #                         print('pos_emb: ', len(emb), emb[0].shape)
                    try:
                        if data['neg_augmented_data'][rnd] != []:

        #                     print('found neg')

                            for aug_num, (topic, emb) in enumerate(data['neg_augmented_data'][rnd]):

                                neg_checklist[cnt_i][rnd].append(emb)
                                neg_cnt += 1

                            for aug_num, (topic, text) in enumerate(data['neg_augmented_text'][rnd]):
                                aug_neg_txts[rnd].append([topic, text])
                    except:
                        pass

            y_vec_b.append([check_range_idx(data["y_emb"][idx], idx)])
            cnt_i += 1
#     print(cnt_j, neg_cnt)

    X_docs = [] ## replace

    return X_vec, y_vec, y_vec_b, X_docs, pos_checklist, neg_checklist, aug_embs, aug_pos_txts, aug_neg_txts

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

    X_train_emb, y_train_emb, y_train_b_emb, X_train_docs, pos_checklist, neg_checklist, aug_embs, aug_pos_txts, aug_neg_txts = get_X_n_y_vec(docs_train, idx=arg.idx, mode='train')


    # get sentence embedding for dev data
    docs_dev = pickle.load(open(os.path.join(arg.target_folder, arg.dev_pkl), 'rb'))

    X_dev_emb, y_dev_emb, y_dev_b_emb, X_dev_docs, _, _, _, _, _ = get_X_n_y_vec(docs_dev, idx=arg.idx)
    # get sentence embedding for test data
    docs_test = pickle.load(open(os.path.join(arg.target_folder, arg.test_pkl), 'rb'))

    X_test_emb, y_test_emb, y_test_b_emb, X_test_docs, _, _, _, _, _ = get_X_n_y_vec(docs_test, idx=arg.idx)
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


        if a_e > 0:
            print(len(X_train_emb), len(new_aug_X_data))
            X_train_embs = padding(flatten(X_train_emb + new_aug_X_data))
            train_set = Dataset(X_train_embs, np.array(y_train_emb + new_aug_y_data), np.array(y_train_b_emb + new_aug_y_data_b))
            trainloader = torch.utils.data.DataLoader(train_set, batch_size=arg.batch_size, shuffle=False, num_workers=1)

            if neg_flag:
                X_train_embs = padding(flatten(aug_neg_X_data))
                neg_train_set = neg_Dataset(X_train_embs)
                neg_trainloader = torch.utils.data.DataLoader(neg_train_set, batch_size=1, shuffle=False, num_workers=1)
                neg_train_x = [x for i, x in enumerate(neg_trainloader)]

        for e in tqdm.tqdm(range(arg.num_epochs)):
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
                    labels = torch.tensor(labels, dtype=torch.long).cuda()
                else:
                    inputs = Variable(inputs.type(torch.FloatTensor))
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

                            #cross entropy loss for classifier
                            ce_loss = 0.5 * (F.cross_entropy(out_b.view(-1, arg.label_size), labels.view(-1)) + F.cross_entropy(out_b1.view(-1, arg.label_size), labels.view(-1)))
                            kl_loss = compute_kl_loss(out_b, out_b1)

                            # carefully choose hyper-parameters
                            loss = ce_loss + arg.alpha * kl_loss

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
                        labels = torch.tensor([1/3, 1/3, 1/3]).view(1, 3).float().cuda()
                    else:
                        inputs = Variable(inputs.type(torch.FloatTensor))
                        labels = torch.tensor([1/3, 1/3, 1/3]).view(1, 3).float()

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
                            loss = loss + arg.alpha * kl_loss

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
                    start_doc_num = start_doc_num + 1


                    if arg.cuda:
                        inputs = torch.tensor(inputs, dtype=torch.float32).cuda()
                        labels = torch.tensor(labels, dtype=torch.long).cuda()
                    else:
                        inputs = torch.tensor(inputs, dtype=torch.float32)
                        labels = torch.tensor(labels, dtype=torch.long)

                    if inputs.size(1) > arg.max_length:
                        inputs = inputs[:, :arg.max_length, :]

                    if arg.task == 'cls':

                        if arg.model not in ['E_MDCA', 'Informer']:
                            out_b = para_model(inputs)
                        else:
                            out_b, attns = para_model(inputs, inputs, inputs, inputs)

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
                    labels = torch.tensor(labels, dtype=torch.long).cuda()
                else:
                    inputs = torch.tensor(inputs, dtype=torch.float32)
                    labels = torch.tensor(labels, dtype=torch.long)

                if inputs.size(1) > arg.max_length:
                    inputs = inputs[:, :arg.max_length, :]

                if arg.task == 'cls':

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

        if arg.task == 'cls':
            test_acc_label = accuracy_score(test_true_labels_list, test_prediction_labels_list)
            print("acc in test dataset is: {0}".format(test_acc_label))
            print(classification_report([str(s) for s in test_true_labels_list],
                                        [str(s) for s in test_prediction_labels_list]))

        if a_e != 4:

            new_aug_X_data, new_aug_y_data, new_aug_y_data_b, aug_neg_X_data = training_data_augmentation_zero_shot(X_train_emb, pos_checklist, neg_checklist, y_train_emb, y_train_b_emb, para_model, aug_embs, aug_pos_txts, aug_neg_txts, a_e)
            neg_flag = False if aug_neg_X_data == [] else True


def training_data_augmentation_zero_shot(X_train_emb, pos_checklist, neg_checklist, y_train_emb, y_train_b_emb, para_model, aug_embs, aug_pos_txts, aug_neg_txts, rnd):

    config = {
        "outdir": 'code/IF/roberta-large-model_5_train_w_qa/', # the dir should combine dataset and method, e.g. IF-model_1_val_w_qa
        "seed": 42,
        "gpu": 0,
        "recursion_depth": 10, # set recursion to use entire training data
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


    for st_ind in range(200):

        if st_ind * 100 >= len(X_train_emb):
            break

        selected_indices = all_selected_indices[st_ind * 100:  (st_ind+1) * 100]

        X_train_embs = padding(flatten([X_train_emb[ind] for ind in selected_indices]))
        y_train_embs = [y_train_emb[ind] for ind in selected_indices]
        y_train_b_embs = [y_train_b_emb[ind] for ind in selected_indices]

        train_set = Dataset(X_train_embs, np.array(y_train_embs), np.array(y_train_b_embs))
        trainloader = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=False, num_workers=1)


        aug_sample = []
        aug_sample_text = []
        ind_check_pos = defaultdict(list)
        cnt = 0
        for ind in selected_indices:
            if ind in pos_checklist:
                aug_sample += [aug_embs[rnd][i] for i in pos_checklist[ind][rnd]]
                aug_sample_text += [aug_pos_txts[rnd][i] for i in pos_checklist[ind][rnd]]
                for _ in pos_checklist[ind][rnd]:

                    ind_check_pos[ind].append(cnt)
                    cnt += 1

                assert len(aug_sample) == len(aug_sample_text)

            if ind in neg_checklist:

                for emb_id, emb in enumerate(neg_checklist[ind][rnd]):

                    neg_X_data.append(emb)
                neg_X_text.append(aug_neg_txts[rnd])


        X_train_embs = padding(flatten(aug_sample))
        y_vec = [0 for _ in range(len(aug_sample))]
        y_vec_b = [[0] for _ in range(len(aug_sample))]

    #     print(len(X_train_embs), len(y_vec), len(y_vec_b))
        aug_set = Dataset(X_train_embs, np.array(y_vec), np.array(y_vec_b))
        augloader = torch.utils.data.DataLoader(aug_set, batch_size=1, shuffle=False, num_workers=1)


        influence_results = calc_all_grad_then_test(config, arg.model, para_model, trainloader, augloader, calculate_if=False)
    #     print(len(influence_results["influences"]), len(influence_results["influences"][0]))

        if arg.aug_type == 'zero_shot':

            for i, influence in enumerate(influence_results["influences"]):

                ind = selected_indices[i]

                if ind in ind_check_pos:

                    indices = np.array(ind_check_pos[ind])
                    influence = np.array(influence)[indices]


                    x = aug_sample[np.argmax(influence)]
                    new_aug_X_data.append(x)

                    topic_txt = aug_sample_text[np.argmax(influence)][:2]

                    x = padding(flatten([x]))
                    x = torch.tensor(x, dtype=torch.float32).cuda()
                    if arg.model in ['E_MDCA', 'Informer']:
                        logits, _ = para_model(x, x, x, x)
                    else:
                        logits = para_model(x)

                    label = torch.argmax(F.log_softmax(logits, dim=1), dim=1)
                    label = label.detach().cpu().item()

#                     print(len(topic_txt), len(x), label)

                    topic_txt.append(label)
                    pos_X_text.append(topic_txt)

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

#     pickle.dump(pos_X_text, open('results/model_{0}_{1}_{2}_pos.pkl'.format(arg.model, arg.train_pkl[12:-4], arg.idx), 'wb'))
#     pickle.dump(neg_X_text, open('results/model_{0}_{1}_{2}_neg.pkl'.format(arg.model, arg.train_pkl[12:-4], arg.idx), 'wb'))

#     arg.saved_model_path = arg.target_saved_model_folder + '{5}_model{8}_b{0}_el{1}_f{2}_h{3}_dp{9}_md{10}_{4}_idx{6}_kl{11}_uuid{7}.out'.format(arg.batch_size, arg.e_layers, arg.factor, arg.n_heads, arg.lr, arg.task, arg.idx, uid, arg.model, arg.dropout, arg.d_model, arg.kl)



    print('Finished Augmentation !')

    return new_aug_X_data, new_aug_y_data, new_aug_y_data_b, neg_X_data



if __name__ == "__main__":
    print(arg)
    evaluation = go(arg)
    print(arg)
