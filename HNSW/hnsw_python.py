from gensim.models import Word2Vec, KeyedVectors
import numpy as np
import time
from progressbar import *
import pickle
from hnsw_origin import HNSW

model = KeyedVectors.load_word2vec_format('../data/blogcatalog.embeddings', binary=False)
data = np.array(model.vectors)

dim = 128
num_elements = 10312


hnsw = HNSW('l2', m0=16, ef=128) # ef: construction time/accuracy trade-off 
# m0: defines tha maximum number of outgoing connections in the graph

widgets = ['Progress: ',Percentage(), ' ', Bar('#'),' ', Timer(), ' ', ETA()]

# show progressbar
pbar = ProgressBar(widgets=widgets, maxval=num_elements).start()
for i in range(len(data)):
    hnsw.add(data[i])
    pbar.update(i + 1)
pbar.finish()

# save index
with open('blog.ind', 'wb') as f:
    picklestring = pickle.dump(hnsw, f, pickle.HIGHEST_PROTOCOL)

# load index
fr = open('blog.ind','rb')
hnsw_n = pickle.load(fr)

add_point_time = time.time()
idx = hnsw_n.search(np.float32(np.random.random((1, dim))), 10)
search_time = time.time()
print("Searchtime: %f" % (search_time - add_point_time))
