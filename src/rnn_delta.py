import argparse, os, sys
from pathlib import Path
from torch_geometric.data import Data
import json
import networkx as nx
from torch_geometric.utils.convert import from_networkx
import pandas as pd
import torch
from torch_geometric.loader import DataLoader
import torch.nn as nn
from torch.nn import PReLU, Linear
from itertools import product
from torch_geometric.nn import GCNConv
from torch.optim import Adam
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--met_path", type=str,
                        help="path to metabolite concentrations in .xlsx format",
                        required=True, default=None)
    parser.add_argument("-m2r", "--met_to_react_path", type=str,
                        help="path to metabolite to reaction mapping in .tsv format",
                        required=True, default=None)
    parser.add_argument("-r2s", "--react_to_sub_path", type=str,
                        help="path to reaction to subsystem mapping in .tsv format",
                        required=True, default=None)
    parser.add_argument("-o", "--out_dir", type=str,
                        help="path to output dir",
                        required=True, default=None)
    args = parser.parse_args()
    return args


class GCRNN(nn.Module):
    def __init__(self, input_dim, output_dim, add_self_loops):
        super(GCRNN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.conv = GCNConv(input_dim, output_dim, add_self_loops=add_self_loops)
        self.lin = Linear(output_dim, output_dim)
        self.act = nn.Tanh()
        
    def forward(self, x, edge_index):
        h_prev = torch.zeros(x.shape[0], self.output_dim)
        h_prev = h_prev.to('cuda:0')
        h = []
        for t in range(x.shape[1]):
            xt = x[:, t, :]
            print('xt.shape', xt.shape)
            i2h = self.conv(xt, edge_index)
            h2h = self.lin(h_prev)
            h_new = i2h + h2h
            h_new = self.act(h_new)
            print('h_new.shape', h_new.shape)
            h.append(h_new)
            h_prev = h_new
        h = torch.stack(h, dim=1)
        print('h.shape', h.shape)
        return h    

class Model4GCN(nn.Module):
    def __init__(self, hidden_dim, add_self_loops):
        super(Model4GCN, self).__init__()
        self.gcrnn1 = GCRNN(1, hidden_dim, add_self_loops)
        self.gcrnn2 = GCRNN(hidden_dim, hidden_dim, add_self_loops)
        self.gcrnn3 = GCRNN(hidden_dim, hidden_dim, add_self_loops)
        self.conv = GCNConv(hidden_dim, 1, add_self_loops=add_self_loops)

    def forward(self, x, edge_index):
        h = self.gcrnn1(x, edge_index)
        print('h', h.shape)
        h = self.gcrnn2(h, edge_index)
        print('h', h.shape)
        h = self.gcrnn3(h, edge_index)
        print('h', h.shape)
        out = []
        for i in range(h.shape[1]):
            ht = h[:, i, :]
            print('ht', ht.shape)
            outt = self.conv(ht, edge_index)
            print('outt', outt.shape)
            out.append(outt)
        out = torch.stack(out, dim=1)
        print('out', out.shape)
        return out

class Model2GCN(nn.Module):
    def __init__(self, hidden_dim, add_self_loops):
        super(Model2GCN, self).__init__()
        self.gcrnn = GCRNN(1, hidden_dim, add_self_loops)
        self.conv = GCNConv(hidden_dim, 1, add_self_loops=add_self_loops)

    def forward(self, x, edge_index):
        h = self.gcrnn(x, edge_index)
        print('h', h.shape)
        out = []
        for i in range(h.shape[1]):
            ht = h[:, i, :]
            print('ht', ht.shape)
            outt = self.conv(ht, edge_index)
            print('outt', outt.shape)
            out.append(outt)
        out = torch.stack(out, dim=1)
        print('out', out.shape)
        return out
    
def plot(x, y, legends, xlabel, ylabel, title, png_path):
    print('Plotting...')
    if(legends is None):
        for i in range(len(y)):
            plt.plot(x, y[i])
    else:
        for i in range(len(y)):
            plt.plot(x, y[i], label=legends[i])
            plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(png_path)
    print('Plot saved to', png_path)
    plt.close()
    
def plot_metrics(metric_path, metrics, png_dir):
    metric_df = pd.read_csv(metric_path, sep='\t')
    max_epoch = metric_df['epoch'].max()
    metric_df = metric_df.set_index('epoch')
        
    for metric in metrics:
        y_all = []
        for m in metric:
            y = []
            for epoch in range(1, max_epoch+1):
                y.append(metric_df.at[epoch, m])
            y_all.append(y)
            base_metric = metric[0].split('_')[-1]
            plot(list(range(1, max_epoch+1)), y_all, ['Train', 'Val'], 'Epoch', metric, base_metric + 'across epochs', png_dir+'/'+base_metric+'.png')
            
def init_model(hidden_dim, add_self_loops, model_name):
    if(model_name == 'Model4GCN'):
        return Model4GCN(hidden_dim, add_self_loops).to('cuda:0')
    elif(model_name == 'Model2GCN'):
        return Model2GCN(hidden_dim, add_self_loops).to('cuda:0')
    else:
        raise Exception("Unrecognized model name", model_name)
        
def train_sample(sample, hidden_dim, lr, add_self_loops, model_name, out_dir):
    print('sample', sample.sample_key, 'hidden_dim', hidden_dim, 'lr', lr, 'add_self_loops', add_self_loops, 'model_name', model_name, out_dir)
    
    model = init_model(hidden_dim, add_self_loops, model_name)
    print(model)
    loss_module = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=lr)
    
    metric_dict = {
        'epoch': [],
        'loss': []
    }
    max_epoch = 100
    #while(True):
    for epoch in range(1, max_epoch+1):
        print('Epoch', epoch)
        print(sample.edge_index.shape)
        out = model(sample.x, sample.edge_index)
        print('out.shape', out.shape)
        pred = torch.squeeze(out[sample.met_mask, :, :]) + torch.squeeze(sample.x[sample.met_mask, :, :])
        true = torch.squeeze(sample.y[sample.met_mask, :, :])
        optimizer.zero_grad()
        print('pred', pred.shape, 'true',  true.shape)
        loss = loss_module(pred, true)
        loss.backward()
        optimizer.step()       
        metric_dict['epoch'].append(epoch)
        metric_dict['loss'].append(loss.item())
        
        if(epoch > 1):
            if((metric_dict['loss'][-2] - metric_dict['loss'][-1])  < 1e-6):
                break
            else:
                torch.save(model, out_dir + '/best_model.pt')
        else:
            torch.save(model, out_dir + '/best_model.pt')
            
        #epoch += 1
                
    best_model = torch.load(out_dir + '/best_model.pt')
    
    loss_module = nn.MSELoss(reduction='sum')
    out = best_model(sample.x, sample.edge_index)
    pred = torch.squeeze(out[sample.met_mask, :, :]) + torch.squeeze(sample.x[sample.met_mask, :, :])
    true = torch.squeeze(sample.y[sample.met_mask, :, :])
    mse_loss = loss_module(pred, true).item()
    out = torch.squeeze(out) + torch.squeeze(sample.x)
    out_arr = out.cpu().detach().numpy()
    print('out_arr', out_arr.shape, type(out_arr))
    print(type(sample.nodes), type(sample.timestamps[1:]))
    outdf = pd.DataFrame(out_arr, index=sample.nodes, columns=sample.timestamps[1:])
    outdf.index.name = 'Key'
    outdf.to_csv(out_dir + '/' + sample.sample_key + '.tsv', sep='\t', index=True)
    
       
    metric_df = pd.DataFrame.from_dict(metric_dict)
    metric_df.to_csv(out_dir + '/metrics.tsv', sep='\t', index=False)
    plot_metrics(out_dir + '/metrics.tsv', [['loss']], out_dir)
    return max(metric_dict['epoch']), mse_loss
        
def train_model_for_each_sample(samples):
    hidden_dims = [1, 4, 8]
    lr_list = [0.01, 0.001, 0.0001, 0.00001, 0.000001]
    add_self_loops = [True, False]
    model_names = ['Model2GCN', 'Model4GCN']
    hparam_combo = list(product(hidden_dims, lr_list, add_self_loops, model_names))
    
    hparam_file = open('hyperparameters.tsv', 'w')
    
    for sample in samples:
        sample_key = sample.sample_key
        sample_dir = os.getcwd() + '/' + sample_key
        Path(sample_dir).mkdir(parents=True, exist_ok=True)
        
        hparam_file = open(sample_dir + '/hyperparameters.tsv', 'w')

        hparam_file.write('hparam_no' + '\t' + 'hidden_dim' + '\t' + 'lr' + '\t' + 'add_self_loops' + '\t' + 'model_name' + '\t' + 'last_epoch' + '\t' + 'mse_loss' + '\n')
 
        for hparam_no in range(len(hparam_combo)):
            hparam_label = 'hparams-'+str(hparam_no)
            hparam = hparam_combo[hparam_no]
            hparam_file.write(hparam_label + '\t' + str(hparam[0]) + '\t' + str(hparam[1]) + '\t' + str(hparam[2]) + '\t' + hparam[3])
            hparam_dir = sample_dir +'/' + hparam_label
            Path(hparam_dir).mkdir(parents=True, exist_ok=True)
            last_epoch, mse_loss = train_sample(sample, hparam[0], hparam[1], hparam[2], hparam[3], hparam_dir)
            hparam_file.write('\t' + str(last_epoch) + '\t' + str(mse_loss) + '\n')
            hparam_file.flush()
        hparam_file.close()
        
def construct_network(met_path, met_to_react_path, react_to_sub_path):
    met_dfs = pd.read_excel(met_path, sheet_name=None, index_col=0) 
    print(type(met_dfs))
    print(met_dfs.keys())
    
    mets = [set(met_df.index) for met_df in met_dfs.values()]
    mets = set.intersection(*mets)
    print(len(mets), 'mets')
    
    met_to_react = pd.read_csv(met_to_react_path, sep='\t')
    met_to_react = met_to_react[met_to_react.kegg_id.isin(mets)]
    
    react_to_sub = pd.read_csv(react_to_sub_path, sep='\t')
    react_to_sub = react_to_sub[react_to_sub.reaction_id.isin(met_to_react['reaction_id'])]
    
    met_nodes = list(mets.intersection(set(met_to_react['kegg_id'])))
    react_nodes = list(set(met_to_react['reaction_id']))
    sub_nodes = list(set(react_to_sub['subsystem']))
    nodes = met_nodes + react_nodes + sub_nodes
    
    node_id = dict(zip(nodes, list(range(len(nodes)))))
    with open("node_id.json", "w") as outfile: 
        json.dump(node_id, outfile)
    
    met_node_id = list(range(len(met_nodes)))
    excel_writer = excel_writer = pd.ExcelWriter('metabolome.xlsx')
    for sample_key in met_dfs.keys():
        met_df = met_dfs[sample_key]
        met_df = met_df.groupby(met_df.index).sum()
        met_df = met_df.loc[met_nodes, :]
        met_df = met_df.rename(index=node_id)
        met_df = met_df.loc[met_node_id, :]
        met_dfs[sample_key] = met_df
        print(sample_key, met_df.shape)
        met_df.to_excel(excel_writer, sheet_name=sample_key, index=True)
    excel_writer.close()
    
    met_to_react['met_node_id'] = met_to_react['kegg_id'].apply(lambda x: node_id[x])
    met_to_react['react_node_id'] = met_to_react['reaction_id'].apply(lambda x: node_id[x])
    met_to_react.to_csv('met_to_react.tsv', sep='\t', index=True)
    
    react_to_sub['react_node_id'] = react_to_sub['reaction_id'].apply(lambda x: node_id[x])
    react_to_sub['sub_node_id'] = react_to_sub['subsystem'].apply(lambda x: node_id[x])
    react_to_sub.to_csv('react_to_sub.tsv', sep='\t', index=True)
    
    met_to_sub = pd.merge(met_to_react, react_to_sub, on='reaction_id', how='inner')
    met_to_sub.to_csv('met_to_sub.tsv', sep='\t', index=True)
    
    met_sub_edges = list(zip(met_to_sub.met_node_id, met_to_sub.sub_node_id))
    sub_react_edges = list(zip(react_to_sub.sub_node_id, react_to_sub.react_node_id))
    react_met_edges = list(zip(met_to_react.react_node_id, met_to_react.met_node_id))
    edges = met_sub_edges + sub_react_edges + react_met_edges
    #print('edges', len(edges), 'met_sub_edges', len(met_sub_edges), 'sub_react_edges', len(sub_react_edges), 'react_met_edges', len(react_met_edges))
    
    met_mask = torch.ByteTensor([True] * len(met_nodes) + [False] * (len(react_nodes) + len(sub_nodes))).bool()
    #print('met_mask.dtype', met_mask.dtype)
    #print('met_mask', sum(met_mask))
    #print(met_mask)
    
    prior_graph = nx.DiGraph()
    prior_graph.add_edges_from(edges)
    prior_graph = from_networkx(prior_graph)
    prior_graph.nodes = nodes
    prior_graph.met_mask = met_mask
    #print(prior_graph.edge_index.shape[1], 'edges')
    #print('prior_graph.met_mask.dtype', prior_graph.met_mask.dtype)
    
    network_dir = "network"
    Path(network_dir).mkdir(parents=True, exist_ok=True)
    torch.save(prior_graph, network_dir+"/prior_graph.pt")
    
    sample_dir = network_dir + "/sample"
    Path(sample_dir).mkdir(parents=True, exist_ok=True)
    
    samples = []

    for sample_key in met_dfs.keys():
        sample_graph = prior_graph.clone()
        #print('sample_graph.met_mask.dtype', sample_graph.met_mask.dtype)
        #print(sample_key)
        met_df = met_dfs[sample_key]
        met_x = met_df.to_numpy()#Mets * #Time
        met_x = torch.Tensor(met_x) # expected -> 
        print('met_x', met_x.shape)
        non_met_x = torch.zeros(len(react_nodes)+len(sub_nodes), met_x.shape[1])
        print('non_met_x', non_met_x.shape)
        all_x = torch.cat([met_x, non_met_x], dim=0)
        print('all_x', all_x.shape)
        x = torch.unsqueeze(all_x[:, :-1], 2)
        print('x', x.shape)
        y = torch.unsqueeze(all_x[:, 1:], 2)
        print('y', y.shape)
        sample_graph.x = x
        sample_graph.y = y
        sample_graph.sample_key = sample_key
        sample_graph.timestamps = list(met_df.columns)
        sample_graph = sample_graph.to('cuda:0')
        torch.save(sample_graph, sample_dir+"/"+sample_key+".pt")
        samples.append(sample_graph)
    return samples
    
def main(args):
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    os.chdir(args.out_dir)
    
    orig_stdout = sys.stdout
    orig_stderr = sys.stderr
    log_file = open('rnn.log', 'w')
    sys.stdout = log_file
    sys.stderr = log_file
    
    samples = construct_network(args.met_path, args.met_to_react_path, args.react_to_sub_path)
    
    train_model_for_each_sample(samples)
    
    sys.stdout = orig_stdout 
    sys.stderr = orig_stderr 
    log_file.close()
    
if __name__ == "__main__":
    main(parse_args())