import argparse, sys, os, pickle
import networkx as nx
from pathlib import Path
import pandas as pd
from itertools import repeat
from networkx.readwrite import json_graph
from torch_geometric.utils.convert import from_networkx
import torch

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--met_path", type=str,
                        help="path to the input metabolome in .tsv format",
                        required=True, default=None)
    parser.add_argument("-m2r", "--met_to_react_map", type=str,
                        help="metabolite to reaction mapping in .tsv format",
                        required=False, default=None)
    parser.add_argument("-r2s", "--react_to_sub_map", type=str,
                       help="reaction to subsystem mapping in .tsv format")
    parser.add_argument("-o", "--out_dir", type=str,
                        help="path to the output directory",
                        required=True, default=None)
    args = parser.parse_args()
    return args

def construct_homogenous_network(met_path, met_to_react_map, react_to_sub_map):
    met_df = pd.read_csv(met_path, sep='\t', index_col='Key')
    
    met_to_react = pd.read_csv(met_to_react_map, sep='\t') # reaction_set
    met_to_react = met_to_react[met_to_react.hmdb_id.isin(met_df.columns)]

    substrate_to_react = met_to_react[met_to_react['relation_type'] == 'substrate_of']
    product_to_react = met_to_react[met_to_react['relation_type'] == 'product_of']

    substrate_to_react['hmdb_id'] = substrate_to_react['hmdb_id'].astype(str) + '-'
    product_to_react['hmdb_id'] = product_to_react['hmdb_id'].astype(str) + '+'
    
    substrate_react_edges = list(zip(substrate_to_react.hmdb_id, substrate_to_react.reaction_id))
    product_react_edges = list(zip(product_to_react.hmdb_id, product_to_react.reaction_id))
    #print('substrate_react_edges', substrate_react_edges)
    #print('product_react_edges', product_react_edges)
    
    react_to_sub = pd.read_csv(react_to_sub_map, sep='\t', usecols=['reaction_id', 'subsystem'])
    react_to_sub = react_to_sub[react_to_sub.reaction_id.isin(met_to_react['reaction_id'])]
    react_sub_edges = list(zip(react_to_sub.reaction_id, react_to_sub.subsystem))
    #print('react_sub_edges', react_sub_edges)
    
    prior_edges = substrate_react_edges + product_react_edges + react_sub_edges
    print(len(met_df.columns), 'metabolites', len(set(met_to_react['reaction_id'])), 'reactions', len(set(react_to_sub['subsystem'])), 'subsystems', len(prior_edges), 'prior_edges')
    
    prior_graph = nx.Graph()
    prior_graph.add_edges_from(prior_edges)
    
    isolates = list(nx.isolates(prior_graph))
    print(len(isolates), 'isolates')
    prior_graph.remove_nodes_from(isolates)
    print(prior_graph.number_of_nodes(), 'nodes', prior_graph.number_of_edges(), 'edges')


    node_name = {}
    for node in (prior_graph.nodes):
        node_name[node] = node
    print('node_name', node_name)
    nx.set_node_attributes(prior_graph, node_name, 'node_name')
    
    nx.write_edgelist(prior_graph, "prior.edgelist")
    pickle.dump(prior_graph, open('prior.pkl', 'wb'))
    pyg_graph = from_networkx(prior_graph)
    torch.save(pyg_graph, 'prior.pt')
    
    non_met_nodes = [node for node in prior_graph.nodes if not node.startswith('HMDB')]
    #print('non_met_nodes', non_met_nodes)
    non_met_x = dict.fromkeys(non_met_nodes, 0) 
    print('non_met_x', non_met_x)
    
    for sample_id, row in met_df.iterrows(): # idx - sample id, row - hmdb change
        print('sample_id', sample_id)
        sample_graph = prior_graph.copy()
        
        x = {}
        
        for hmdb_id, change in row.items():
            if(change >= 0):
                x[hmdb_id+'+'] = abs(change)
                x[hmdb_id+'-'] = 0
                #sample_graph.add_edge(sample_id, hmdb_id+'+', change=change)
                
            else:
                x[hmdb_id+'-'] = abs(change)
                x[hmdb_id+'+'] = 0
        
        x.update(non_met_x)
        print('x', x)
        nx.set_node_attributes(sample_graph, x, 'x')
        pickle.dump(sample_graph, open(sample_id+'.pkl', 'wb'))

        
        pyg_graph = from_networkx(sample_graph)
        print('pyg_graph.node_name', pyg_graph.node_name)
        #print(pyg_graph.change.shape)
        #print(pyg_graph.change)
        #print(pyg_graph.to_dict())
        torch.save(pyg_graph, sample_id+'.pt')
        #print('edge_attr', pyg_graph.edge_attr)
        #print('change', pyg_graph.change)
        
        #print('edge_index', pyg_graph.edge_index)
        
        #print(len(change), 'sample edges')
                
def main(args):
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    os.chdir(args.out_dir)
    
    orig_stdout = sys.stdout
    orig_stderr = sys.stderr
    log_file = open('construct-network.log', 'w')
    sys.stdout = log_file
    sys.stderr = log_file
    
    graph_dir =  args.out_dir + '/graphs'
    Path(graph_dir).mkdir(parents=True, exist_ok=True)
    os.chdir(graph_dir)
    
    print(args)
    construct_homogenous_network(args.met_path, args.met_to_react_map, args.react_to_sub_map)

    sys.stdout = orig_stdout 
    sys.stderr = orig_stderr 
    log_file.close()

if __name__ == "__main__":
    main(parse_args())
