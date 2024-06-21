import os, argparse
from pathlib import Path
import torch
from torch_geometric.nn import Node2Vec
from tqdm.notebook import tqdm
import sys
import matplotlib.pyplot as plt
import time
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--graph_path", type=str,
                        help="path to the input graph in .pt format",
                        required=True, default=None)
    parser.add_argument("-d", "--emb_dim", type=int,
                        help="embedding dimension for node2vec",
                        required=True, default=None)
    parser.add_argument("-l", "--walk_length", type=int,
                        help="length of sampled random walks for node2vec",
                        required=True, default=None)
    parser.add_argument("-c", "--context_size", type=int,
                        help="context size for word2vec in for node2vec",
                        required=True, default=None)
    parser.add_argument("-w", "--walks_per_node", type=int,
                        help="walks per node for node2vec",
                        required=True, default=None)
    parser.add_argument("-p", "--p", type=float,
                        help="parameter p in for node2vec",
                        required=True, default=None)
    parser.add_argument("-q", "--q", type=float,
                        help="parameter q in for node2vec",
                        required=True, default=None)
    parser.add_argument("-o", "--out_dir", type=str,
                        help="path to the output directory",
                        required=True, default=None)
    args = parser.parse_args()
    return args

def plot(x, y, xlabel, ylabel, title, png_path):
    print('Plotting...')
    plt.plot(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(png_path)
    print('Plot saved to', png_path)
    plt.close()

def node2vec_embedding(graph_path, emb_dim, walk_length, context_size, walks_per_node, p, q):
    data = torch.load(graph_path)
    
    file = open('node_name.tsv', 'w')
    file.write('node_id' + '\t' + 'node_name' + '\n')
    for node, node_name in enumerate(data.node_name):
        file.write(str(node) + '\t' + node_name + '\n')
    file.flush()
    file.close()

    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('device', device, flush=True)
    
    model = Node2Vec(data.edge_index, embedding_dim=emb_dim, walk_length=walk_length,
                 context_size=context_size, walks_per_node=walks_per_node,
                 num_negative_samples=1, p=p, q=q, sparse=True).to(device)
    
    loader = model.loader(batch_size=256, shuffle=True, num_workers=0)
    optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=0.1)
    
    train_loss = []
    print('loader size', len(loader), flush=True)

    for epoch in range(1, 100):
        print('Epoch', epoch, 'started', flush=True)
        start_time = time.time()
        model.train()  # put model in train model
        total_loss = 0
        for pos_rw, neg_rw in tqdm(loader):
            print('pos_rw', pos_rw.shape, 'neg_rw', neg_rw.shape)
            optimizer.zero_grad()  # set the gradients to 0
            loss = model.loss(pos_rw.to(device), neg_rw.to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        duration = time.time() - start_time
        avg_loss = total_loss / len(loader)
        train_loss.append(avg_loss)
        print(f'Epoch: {epoch:02d}, Loss: {avg_loss:.4f}, Time: {duration:.2f}', flush=True)
        if((epoch > 1) and (avg_loss > train_loss[-2])):
            break
        
    plot(list(range(1, len(train_loss)+1)), train_loss, 'Epoch', 'Train Loss', 'Node2vec training loss', 'train_loss.png')

    embedding = "\t".join(["emb_"+str(i) for i in range(emb_dim)]) + "\n"

    print('arange', torch.arange(data.num_nodes, device=device))
    for tensor in model(torch.arange(data.num_nodes, device=device)):
        s = "\t".join([str(value) for value in tensor.detach().cpu().numpy()])
        embedding += s + "\n"
        
    # save the vectors
    with open("embedding.tsv", "w") as emb_file:
        emb_file.write(embedding)
        
    emb_df = pd.read_csv('embedding.tsv', sep='\t')
    emb_df = emb_df.reset_index(names='node_id')
    print('emb_df', emb_df.head(5))
    
    node_df = pd.read_csv('node_name.tsv', sep='\t')
    emb_df = pd.merge(node_df, emb_df, on='node_id')
    emb_df.to_csv('node_embedding.tsv', sep='\t', index=False)

def main(args):
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    os.chdir(args.out_dir)
    
    orig_stdout = sys.stdout
    orig_stderr = sys.stderr
    log_file = open('node2vec-prior-embedding.log', 'w')
    sys.stdout = log_file
    sys.stderr = log_file
    
    print(args, flush=True)
    
    node2vec_embedding(args.graph_path, args.emb_dim, args.walk_length, args.context_size, args.walks_per_node, args.p, args.q)

    sys.stdout = orig_stdout 
    sys.stderr = orig_stderr 
    log_file.close()

if __name__ == "__main__":
    main(parse_args())
