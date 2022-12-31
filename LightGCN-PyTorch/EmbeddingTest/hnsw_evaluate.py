import pickle
import numpy as np
from progressbar import *
import argparse


def load_test(test_file):
    test_data = {}
    with open(test_file) as f:
        for l in f.readlines():
            if len(l) > 0:
                l = l.strip('\n').split(' ')
                items = [int(i) for i in l[1:]]
                uid = int(l[0])
                test_data[uid] = items
    return test_data


def DCG(A, test_set):
    # ------ 计算 DCG ------ #
    dcg = 0
    for i in range(len(A)):
        # 给r_i赋值，若r_i在测试集中则为1，否则为0
        r_i = 0
        if A[i] in test_set:
            r_i = 1
        dcg += (2 ** r_i - 1) / np.log2((i + 1) + 1) # (i+1)是因为下标从0开始
    return dcg

def IDCG(A, test_set):
    # ------ 将在测试中的a排到前面去，然后再计算DCG ------ #
    A_temp_1 = [] # 临时A，用于存储r_i为1的a
    A_temp_0 = []  # 临时A，用于存储r_i为0的a
    for a in A:
        if a in test_set:
            # 若a在测试集中则追加到A_temp_1中
            A_temp_1.append(a)
        else:
            # 若a不在测试集中则追加到A_temp_0中
            A_temp_0.append(a)
    A_temp_1.extend(A_temp_0)
    idcg = DCG(A_temp_1, test_set)
    return idcg

def NDCG(A, test_set):
    dcg = DCG(A, test_set) # 计算DCG
    idcg = IDCG(A, test_set) # 计算IDCG
    if dcg == 0 or idcg == 0:
        ndcg = 0
    else:
        ndcg = dcg / idcg
    return ndcg

def MRR(predict, test):
    if predict:
        for node in test:
            if node in predict:
                return 1 / (test.index(node) + 1)
    else:
        return 0

def evaluate(pred, test, k):
    TruePositive = set(pred) & set(test)
    TP = len(TruePositive)
    recall = TP / len(test)
    precision = TP / k
    ndcg = NDCG(pred, test)
    mrr = MRR(TruePositive, pred)
    return recall, precision, ndcg, mrr


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate HNSW Graph")
    parser.add_argument('--dataset', type=str, default='gowalla',
                        help="available datasets: [gowalla, yelp2018]")
    parser.add_argument('--ef', type=int, default=128,
                        help="construction time/accuracy trade-off")
    return parser.parse_args()

if __name__ == '__main__':
    
    args = parse_args()
    dataset = args.dataset
    ef = args.ef
    epoch = 1000
    
    userEmbedding_path = '/home/yangzh/LightGCN-PyTorch/code/embeddings/%s/userEmbedding_epoch%d.npy'%(dataset, epoch)
    itemEmbedding_path = '/home/yangzh/LightGCN-PyTorch/code/embeddings/%s/itemEmbedding_epoch%d.npy'%(dataset, epoch)
    test_file = './data/%s/test.txt'%(dataset)
    index_file = './annGraph/%s_epoch%d_ef%d_item.ind'%(dataset, epoch, ef)
    
    print(dataset, 'loading...')
    embedding_user = np.load(userEmbedding_path) # load user embedding
    embedding_item = np.load(itemEmbedding_path) # load item embedding
    test_dict = load_test(test_file) # load test file
    with open(index_file, 'rb') as f: # load HNSW Graph
        hnsw_n = pickle.load(f)
    print('done!')
    
    n_user = len(embedding_user) # training n_user
    m_item = len(embedding_item) # training m_item
    print('测试集user:', len(test_dict), 'item:', m_item) # test n_user
    
    k_list = [10, 20]
    recall_dict, precision_dict, mrr_dict, ndcg_dict = {k:[] for k in k_list}, {k:[] for k in k_list}, {k:[] for k in k_list}, {k:[] for k in k_list}
    
    # show progressbar
    widgets = ['Evaluate: ',Percentage(), ' ', Bar('#'),' ', Timer(), ' ', ETA()]
    i = 0
    pbar = ProgressBar(widgets=widgets, maxval=len(test_dict)).start()
    
    for uid in test_dict:
        # if item
        idx = hnsw_n.search(embedding_user[uid], max(k_list))
        idx = [index for index, _ in idx] # [id1,id2,...]
        
        # if item+user   Graph中前m_item项为item，后面都是user
        # idx = hnsw_n.search(embedding_user[uid], 99)[1:] # search到的第一个点是自己,[(id1,dist),(id2,dist)...]
        # idx = [index for index, _ in idx] # [id1,id2,...]
        # idx = list(filter(lambda x:x < m_item, idx)) # 只保留前m_item项
        
        # if user+item   Graph中前n_user项为user，后面都是item
        # idx = hnsw_n.search(embedding_user[uid], 99)[1:] # search到的第一个点是自己,[(id1,dist),(id2,dist)...]
        # idx = [index - n_user for index, _ in idx] # [id1,id2,...]  令(-n_user, -1)为user, (0, m_item)为item
        # idx = list(filter(lambda x:x > -1, idx)) # Graph中前m_item项为item，后面都是user
        for k in k_list:
            pred = idx[:k]
            recall, precision, ndcg, mrr = evaluate(pred, test_dict[uid], k)
            recall_dict[k].append(recall)
            precision_dict[k].append(precision)
            ndcg_dict[k].append(ndcg)
            mrr_dict[k].append(mrr)
        
        pbar.update(i + 1)
        i += 1
    pbar.finish()
    
    for k in k_list:
        print("{:-^60s}".format("Split Line"))
        print('RECALL@%d:'%k, np.mean(recall_dict[k]))
        print('PRECISION@%d:'%k,np.mean(precision_dict[k]))
        print('NDCG@%d:'%k,np.mean(ndcg_dict[k]))
        print('MRR@%d:'%k, np.mean(mrr_dict[k]))
