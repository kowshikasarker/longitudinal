import os
from itertools import product
import pandas as pd

'''
emb_dim = [4, 8, 16, 32]
walk_length = [10, 50, 100, 250]
context_size = [2, 5, 10, 20]
walks_per_node = [5, 50, 250]
p = [10000, 1000, 100, 10]
q = [0.1, 0.25, 0.5, 0.9]
'''

'''
emb_dim = [4, 8]
walk_length = [10, 50]
context_size = [5, 10]
walks_per_node = [10, 50]
p = [10000, 10]
q = [0.1, 0.5]
'''

emb_dim = [16, 32]
walk_length = [10, 50]
context_size = [5, 10]
walks_per_node = [10, 50]
p = [10000, 10]
q = [0.1, 0.5]

param_count = len(emb_dim) * len(walk_length) * len(context_size) * len(walks_per_node) * len(p) * len(q)

param_name = ['param-'+str(i) for i in range(64, 64+param_count)]
param_combo = list(product(emb_dim, walk_length, context_size, walks_per_node, p, q))
#print('param_combo', param_combo)

param_df = pd.DataFrame(param_combo, index=param_name, columns=['emb_dim', 'walk_length', 'context_size', 'walks_per_node', 'p', 'q'])
param_df.index.name = 'param_name'
param_df.to_csv('/shared/nas/data/m1/ksarker2/Embedding/Results/homogenous/params.tsv', sep='\t', index=True, mode='a')

#print(param_df.head(5))

param_dict = param_df.to_dict(orient='index')
#print('param_dict', param_dict)

script = '/shared/nas/data/m1/ksarker2/Embedding/Code/node2vec-prior-embedding.py'


for react_set in range(1, 10):
    graph_path = '/shared/nas/data/m1/ksarker2/Embedding/Results/homogenous/reaction_set_' + str(react_set) + '/graphs/prior.pt'
    for param_name, params in param_dict.items():
        out_dir = '/shared/nas/data/m1/ksarker2/Embedding/Results/homogenous/reaction_set_' + str(react_set) + '/prior-embedding/' + param_name
        
        command = 'python3 ' + script + ' -g ' + graph_path + ' -d ' + str(params['emb_dim']) + ' -l ' + str(params['walk_length']) + ' -c ' + str(params['context_size']) + ' -w ' + str(params['walks_per_node']) + ' -p ' + str(params['p']) + ' -q ' + str(params['q']) + ' -o ' + out_dir
        print(command)
        exit_code = os.system(command)
        if(exit_code != 0):
            print('ERROR!')
    