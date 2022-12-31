import numpy as np
import time
from progressbar import *
from hnsw_origin import HNSW
import pickle
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Build HNSW Graph")
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

    embedding_user = np.load(userEmbedding_path)
    embedding_item = np.load(itemEmbedding_path)
    n_user = len(embedding_user)
    m_item = len(embedding_item)

    # print(userEmbedding.shape, itemEmbedding.shape)
    print('user:%d, item:%d, total:%d'%(n_user, m_item, n_user+m_item))

    hnsw = HNSW('cosine', m0=16, ef=ef)

    widgets = ['BuildGraph: ',Percentage(), ' ', Bar('#'),' ', Timer(), ' ', ETA()]
    pbar = ProgressBar(widgets=widgets, maxval=m_item + n_user).start()
    start_time = time.time()
    # 只加item
    for i in range(m_item):
        hnsw.add(embedding_item[i])
        pbar.update(i + 1)
    end_time = time.time()
    pbar.finish()
    # pbar = ProgressBar(widgets=widgets, maxval=m_item + n_user).start()
    # start_time = time.time()
    # # 先加item，再加user
    # for i in range(m_item):
    #     hnsw.add(embedding_item[i])
    #     pbar.update(i + 1)
    # for i in range(n_user):
    #     hnsw.add(embedding_user[i])
    #     pbar.update(m_item + i + 1)
    # end_time = time.time()
    # pbar.finish()

    print(dataset, 'ef=%d, Index Time: %f' % (ef, end_time - start_time))
    # save index
    index_file = './annGraph2/%s_epoch%d_ef%d_item.ind'%(dataset, epoch, ef)
    with open(index_file, 'wb') as f:
        picklestring = pickle.dump(hnsw, f, pickle.HIGHEST_PROTOCOL)
    print('index saved in:', index_file)
