import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from progressbar import *
import pickle
import argparse
import utils

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
    
    userEmbedding_path = '/home/yangzh/LightGCN-PyTorch/code/embeddings2/%s/userEmbedding_epoch%d.npy'%(dataset, epoch)
    itemEmbedding_path = '/home/yangzh/LightGCN-PyTorch/code/embeddings2/%s/itemEmbedding_epoch%d.npy'%(dataset, epoch)
    test_file = './data/%s/test.txt'%(dataset)
    index_file = './annGraph2/%s_epoch%d_ef%d_item.ind'%(dataset, epoch, ef)
    
    print(dataset, 'loading...')
    embedding_user = np.load(userEmbedding_path) # load user embedding
    embedding_item = np.load(itemEmbedding_path) # load item embedding
    test_dict = utils.load_test(test_file) # load test file
    with open(index_file, 'rb') as f: # load HNSW Graph
        hnsw_n = pickle.load(f)
    print('done!')
    
    n_user = len(embedding_user) # training n_user
    m_item = len(embedding_item) # training m_item
    print('测试集user:', len(test_dict), 'item:', m_item) # test n_user
    
    k_list = [10, 20]
    groundTrue_list = []
    pred_list = []
    
    # show progressbar
    widgets = ['Evaluate: ',Percentage(), ' ', Bar('#'),' ', Timer(), ' ', ETA()]
    i = 0
    pbar = ProgressBar(widgets=widgets, maxval=len(test_dict)).start()
    for uid in test_dict:
        groundTrue_list.append(test_dict[uid])
        
        # if item
        idx = hnsw_n.search(embedding_user[uid], max(k_list))
        idx = [index for index, _ in idx] # [id1,id2,...]
        pred_list.append(idx)
        
        pbar.update(i + 1)
        i += 1
    pbar.finish()
    
    result = utils.test_one_batch(pred_list, groundTrue_list, k_list)
    
    for i, k in enumerate(k_list):
        print("{:-^60s}".format("Split Line"))
        print(f"recall@{k}: {result['recall'][i]}")
        print(f"precision@{k}: {result['precision'][i]}")
        print(f"ndcg@{k}: {result['ndcg'][i]}")
