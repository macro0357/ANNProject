import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from progressbar import *
import argparse
import utils

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Embeddings")
    parser.add_argument('--dataset', type=str, default='gowalla',
                        help="available datasets: [gowalla, yelp2018]")
    parser.add_argument('--space', type=str, default='cosine',
                        help="distance type: [cosine, l2]")
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    dataset = args.dataset
    distance_type = args.space
    epoch = 1000
    
    userEmbedding_path = '/home/yangzh/LightGCN-PyTorch/code/embeddings2/%s/userEmbedding_epoch%d.npy'%(dataset, epoch)
    itemEmbedding_path = '/home/yangzh/LightGCN-PyTorch/code/embeddings2/%s/itemEmbedding_epoch%d.npy'%(dataset, epoch)
    test_file = './data/%s/test.txt'%(dataset)
    
    print(dataset, 'loading...')
    embedding_user = np.load(userEmbedding_path) # load user embedding
    embedding_item = np.load(itemEmbedding_path) # load item embedding
    test_dict = utils.load_test(test_file) # load test file
    print('done!')
    
    n_user = len(embedding_user) # training n_user
    m_item = len(embedding_item) # training m_item
    
    if distance_type == 'cosine':
        similarity_matrix = cosine_similarity(embedding_user, embedding_item)
    elif distance_type == 'l2':
        similarity_matrix = euclidean_distances(embedding_user, embedding_item)

    k_list = [10, 20]
    groundTrue_list = []
    pred_list = []
    
    # show progressbar
    widgets = ['Evaluate: ',Percentage(), ' ', Bar('#'),' ', Timer(), ' ', ETA()]
    i = 0
    pbar = ProgressBar(widgets=widgets, maxval=len(test_dict)).start()
    for uid in test_dict:
        groundTrue_list.append(test_dict[uid])
        
        if distance_type == 'cosine':
            pred = list(np.argsort(similarity_matrix[uid])[::-1][:max(k_list)])
        elif distance_type == 'l2':
            pred = list(np.argsort(similarity_matrix[uid])[:max(k_list)])
        pred_list.append(pred)
        
        pbar.update(i + 1)
        i += 1
    pbar.finish()
    
    result = utils.test_one_batch(pred_list, groundTrue_list, k_list)
    
    for i, k in enumerate(k_list):
        print("{:-^60s}".format("Split Line"))
        print(f"recall@{k}: {result['recall'][i]}")
        print(f"precision@{k}: {result['precision'][i]}")
        print(f"ndcg@{k}: {result['ndcg'][i]}")
