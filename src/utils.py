from sklearn.metrics import f1_score, roc_auc_score
from sklearn.metrics import precision_score, recall_score, confusion_matrix
import torch
import numpy as np
from torch_geometric.data import Data
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
from torch_scatter import scatter
from src.loader import load_graph
from src.classes import extract_all_features_single 


def to_dense_adj(edge_index, batch=None, edge_attr=None, max_num_nodes=None):
    r"""Converts batched sparse adjacency matrices given by edge indices and
    edge attributes to a single dense batched adjacency matrix.

    Args:
        edge_index (LongTensor): The edge indices.
        batch (LongTensor, optional): Batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns each
            node to a specific example. (default: :obj:`None`)
        edge_attr (Tensor, optional): Edge weights or multi-dimensional edge
            features. (default: :obj:`None`)
        max_num_nodes (int, optional): The size of the output node dimension.
            (default: :obj:`None`)

    :rtype: :class:`Tensor`
    """
    if batch is None:
        batch = edge_index.new_zeros(edge_index.max().item() + 1)

    batch_size = batch.max().item() + 1
    one = batch.new_ones(batch.size(0))
    num_nodes = scatter(one, batch, dim=0, dim_size=batch_size, reduce='add')
    cum_nodes = torch.cat([batch.new_zeros(1), num_nodes.cumsum(dim=0)])

    idx0 = batch[edge_index[0]]
    idx1 = edge_index[0] - cum_nodes[batch][edge_index[0]]
    idx2 = edge_index[1] - cum_nodes[batch][edge_index[1]]

    if max_num_nodes is None:
        max_num_nodes = num_nodes.max().item()

    elif idx1.max() >= max_num_nodes or idx2.max() >= max_num_nodes:
        mask = (idx1 < max_num_nodes) & (idx2 < max_num_nodes)
        idx0 = idx0[mask]
        idx1 = idx1[mask]
        idx2 = idx2[mask]
        edge_attr = None if edge_attr is None else edge_attr[mask]

    if edge_attr is None:
        edge_attr = torch.ones(idx0.numel(), device=edge_index.device)

    size = [batch_size, max_num_nodes, max_num_nodes]
    size += list(edge_attr.size())[1:]
    adj = torch.zeros(size, dtype=edge_attr.dtype, device=edge_index.device)

    flattened_size = batch_size * max_num_nodes * max_num_nodes
    adj = adj.view([flattened_size] + list(adj.size())[3:])
    idx = idx0 * max_num_nodes * max_num_nodes + idx1 * max_num_nodes + idx2
    scatter(edge_attr, idx, dim=0, out=adj, reduce='add')
    adj = adj.view(size)

    return adj

def score(pred, labels):
    tp, fp, tn, fn = confusion(labels, pred)
    print('tn, fp, fn, tp', tn, fp, fn, tp)
    
    labels = labels.cpu().numpy()
    pred = pred.cpu().numpy()

    accuracy = (pred == labels).sum() / len(pred)
    micro_f1 = f1_score(labels, pred, average='micro')
    macro_f1 = f1_score(labels, pred, average='macro')
    try:
        f1 = f1_score(labels, pred, average='weighted')
    except:
        print('Exception occurred while calculating F1 Score', labels, pred)
        f1 = 0
    auc = roc_auc_score(labels, pred)
    prec, recall = precision_score(labels, pred), recall_score(labels,pred)
    
    tn, fp, fn, tp = confusion_matrix(labels, pred).ravel()
    # print('tn, fp, fn, tp', tn, fp, fn, tp)
    
    fpr = fp/(fp+tn)
    # print('SCORE', accuracy, f1, auc)
    return {'acc':accuracy, 'f1':f1, 'auc':auc, 'prec':prec, 'recall':recall, 'fpr':fpr, 'mi_f1':micro_f1, 'ma_f1':macro_f1}


def confusion(truth, prediction):
    """ Returns the confusion matrix for the values in the `prediction` and `truth`
    tensors, i.e. the amount of positions where the values of `prediction`
    and `truth` are
    - 1 and 1 (True Positive)
    - 1 and 0 (False Positive)
    - 0 and 0 (True Negative)
    - 0 and 1 (False Negative)
    """

    confusion_vector = prediction / truth
    # Element-wise division of the 2 tensors returns a new tensor which holds a
    # unique value for each case:
    #   1     where prediction and truth are 1 (True Positive)
    #   inf   where prediction is 1 and truth is 0 (False Positive)
    #   nan   where prediction and truth are 0 (True Negative)
    #   0     where prediction is 0 and truth is 1 (False Negative)

    true_positives = torch.sum(confusion_vector == 1).item()
    false_positives = torch.sum(confusion_vector == float('inf')).item()
    true_negatives = torch.sum(torch.isnan(confusion_vector)).item()
    false_negatives = torch.sum(confusion_vector == 0).item()

    return true_positives, false_positives, true_negatives, false_negatives


def to_homogeneous(data, transform=None) -> Data:
        homo_data = data.clone()

        features_shape = sum([node_features.shape[1] for _, node_features in homo_data.x_dict.items()])
        mask_types = ['train_mask', 'val_mask']
        masks = {k: [] for k in mask_types}
        y = []
        
        node_map = {node_type: i for i, node_type in enumerate(homo_data.node_types)}
        edge_map = {i: (node_map[edge_type[0]], node_map[edge_type[2]]) for i, edge_type in enumerate(homo_data.edge_types)}
        
        l_padding = 0
        for node_type, node_features in homo_data.x_dict.items():
            node_features = node_features.numpy()
            r_padding = features_shape - node_features.shape[1] - l_padding
            features = []
            for node_feature in node_features:
                resized = np.pad(node_feature, (l_padding, r_padding), 'constant', constant_values=(0, 0))
                features.append(resized)

            if 'y' in homo_data[node_type]:
                y.append(homo_data[node_type].y)
                for mask_type in mask_types:
                    masks[mask_type].append(homo_data[node_type][mask_type])
            else:
                y.append(torch.full((node_features.shape[0],),2))
                for mask_type in mask_types:
                    masks[mask_type].append(torch.zeros(node_features.shape[0], dtype=torch.bool))

            l_padding += node_features.shape[1]
            homo_data[node_type].x =  torch.from_numpy(np.array(features)).float()

        homo_data = homo_data.to_homogeneous(add_edge_type=True, add_node_type=True)

        for mask_type, mask in masks.items():
            homo_data[mask_type] = torch.cat(mask)

        homo_data.y = torch.cat(y)
        homo_data.edge_map = edge_map
        num_nodes = homo_data.num_nodes
        
        if transform is not None:
            transform(homo_data)
        homo_data.num_nodes = num_nodes
            
        return homo_data

    
def plot_degree_dist_dom_ip_log(g, node_type, title=""):
    degrees = nx.degree(g)
    degree_df = pd.DataFrame(degrees, columns=['node_id', 'degree']).sort_values('node_id', ascending=True) 
    degree_df['node_type'] = node_type
    degree_df = degree_df.sort_values('degree', ascending=False)
    degree_df = degree_df[degree_df.degree > 0]
    
    domain_df = degree_df[degree_df.node_type == 0]
    ip_df = degree_df[degree_df.node_type == 1]
    
    dom_degrees = domain_df.degree.values
    ip_degrees = ip_df.degree.values
    max_deg = max(dom_degrees[0], ip_degrees[0])

    plt.figure(figsize=(4, 3))
    [[dom_counts, ip_counts],bins,_]=plt.hist([dom_degrees, ip_degrees],bins=max_deg)
    plt.xlabel('Degree')
    plt.ylabel('Frequency')
    plt.close()

    dom_countsnozero=dom_counts*1.
    dom_countsnozero[dom_counts==0]=-np.Inf
    ip_countsnozero=ip_counts*1.
    ip_countsnozero[ip_counts==0]=-np.Inf

    plt.figure(figsize=(3, 2.3))
    plt.scatter(bins[:-1],dom_countsnozero/float(sum(dom_counts)),s=10,marker='x',label='Domain')
    plt.scatter(bins[:-1],ip_countsnozero/float(sum(ip_counts)),s=10,marker='x',label="IP")
    plt.yscale('log'), plt.xscale('log')
    plt.xlabel('Degree (log)')
    plt.ylabel("Fraction of nodes (log)")
    plt.legend()
    plt.show()
    
def largest_indices(ary, n):
    """Returns the n largest indices from a numpy array."""
    flat = ary.flatten()
    indices = np.argpartition(flat, -n)[-n:]
    indices = indices[np.argsort(-flat[indices])]
    return np.unravel_index(indices, ary.shape)
    
def get_indices_of_k_smallest(arr, k):
    idx = np.argpartition(arr.ravel(), k)
    return tuple(np.array(np.unravel_index(idx, arr.shape))[:, range(min(k, 0), max(k, 0))])
    
def plot_degree_dist_labels_log(g, node_type, labels, title=""):
    degrees = nx.degree(g, nbunch=list(torch.nonzero(node_type == 0).t()[0].numpy()))
    degree_df = pd.DataFrame(degrees, columns=['node_id', 'degree']).sort_values('node_id', ascending=True) 
    degree_df['label'] = labels
    degree_df = degree_df.sort_values('degree', ascending=False)
    degree_df = degree_df[degree_df.degree > 0]

    ben_df = degree_df[degree_df.label == 0]
    mal_df = degree_df[degree_df.label == 1]
    
    ben_degrees = ben_df.degree.values
    mal_degrees = mal_df.degree.values
    max_deg = max(mal_degrees[0], ben_degrees[0])

    plt.figure(figsize=(4, 3))
    [[ben_counts, mal_counts],bins,_]=plt.hist([ben_degrees, mal_degrees],bins=max_deg)
    plt.xlabel('Degree')
    plt.ylabel('Frequency')
    plt.close()

    ben_countsnozero=ben_counts*1.
    ben_countsnozero[ben_counts==0]=-np.Inf
    mal_countsnozero=mal_counts*1.
    mal_countsnozero[mal_counts==0]=-np.Inf

    plt.figure(figsize=(3, 2.3))
    plt.scatter(bins[:-1],ben_countsnozero/float(sum(ben_counts)),s=10,marker='x',label='Benign',color='g')
    plt.scatter(bins[:-1],mal_countsnozero/float(sum(mal_counts)),s=10,marker='x',label="Malicious",color='r')
    plt.yscale('log'), plt.xscale('log')
    plt.xlabel('Degree (log)')
    plt.ylabel("Fraction of nodes (log)")
    plt.legend()
    plt.show()

    

    
    
    
    
    
def cal_n_add_facni(xx,data):
    graph_data = load_graph(xx)
    graph_nodes, edges, has_public, has_isolates, pruning, extras = graph_data
    K=data['domain_node'].x.shape[0]
    ee=np.array(extras)
    feats=np.zeros([K, 45])
    print("eeee", ee.shape)
    for i in range(K):          
        feats[i,:]=extract_all_features_single(ee[i,1])
    feats=torch.from_numpy(feats).float()
    return feats


# def extract_feat_adj(data):
#     feat_node=data.x_dict['domain_node']
#     feat_ip=data.x_dict['ip_node']   
#     adj_1=data.edge_index_dict[('domain_node', 'apex', 'domain_node')]
#     adj_2=data.edge_index_dict[('domain_node', 'resolves', 'ip_node')]
#     adj_3=data.edge_index_dict[('domain_node', 'similar', 'domain_node')]
#     adj_4=data.edge_index_dict[('ip_node', 'rev_resolves', 'domain_node')]
#     return feat_node, feat_ip, adj_1, adj_2, adj_3, adj_4


def extract_feat_adj2(data):
    feat_node=data.x_dict['domain_node']
    feat_ip=data.x_dict['ip_node']   
    adj_1=data.edge_index_dict[('domain_node', 'HAS_QUERY', 'host_node')]
    adj_2=data.edge_index_dict[('domain_node', 'apex', 'domain_node')]
    adj_3=data.edge_index_dict[('domain_node', 'resolves', 'ip_node')]
    adj_4=data.edge_index_dict[('domain_node', 'similar', 'domain_node')]
    adj_5=data.edge_index_dict[('host_node', 'rev_HAS_QUERY', 'domain_node')]
    adj_6=data.edge_index_dict[('ip_node', 'rev_resolves', 'domain_node')]

    return feat_node, feat_ip, adj_1, adj_2, adj_3, adj_4, adj_5, adj_6




def my_score(pred, labels):
    accuracy = (pred == labels).sum() / len(pred)
    return accuracy
# ASR performance
def calc_BD_metrics_feat(model2,data,ids_test):
    with torch.no_grad():
        model2.eval()
    mask = data['domain_node']['test_mask']
    x0=data['domain_node'].y[mask]
    x0=x0[ids_test]       
    y00 = model2(data.x_dict2, data.edge_index_dict).argmax(dim=-1) 
    y0=y00[mask]
    y0=y0[ids_test]
    locs=np.flatnonzero(data['domain_node'].y.cpu()) 
    num_of_1=0;
    num_of_1_forced_to_0=0;
    for jj in range(len(x0)):
        if x0[jj]==1:
            num_of_1=num_of_1+1
        if x0[jj]==1 and y0[jj]==0:
            num_of_1_forced_to_0=num_of_1_forced_to_0+1;
    if num_of_1==0:
        result=0.5555
    if num_of_1!=0:
        result= num_of_1_forced_to_0/num_of_1
    return result


# CAD performance
def calc_cad(model1,model2,data):
    model1.eval()
    with torch.no_grad():
        pred1 = model1(data.x_dict, data.edge_index_dict).argmax(dim=-1)    
    model2.eval()
    with torch.no_grad():
        pred2 = model2(data.x_dict, data.edge_index_dict).argmax(dim=-1)    
    mask = data['domain_node']['val_mask']
    acc1 = my_score(pred1[mask] ,data['domain_node'].y[mask])
    acc2 = my_score(pred2[mask] ,data['domain_node'].y[mask])
    return acc1-acc2

    
def get_matrices_from_data(zz, kk, data):
    G = nx.Graph()
    elist = []
    for x in range(len(zz)):
        innerlist = ((zz[x], kk[x]))
        elist.append(innerlist)
    mylist = zz + kk
    nodelist = list(dict.fromkeys(mylist))
    G.add_edges_from(elist)
    A=nx.to_numpy_matrix(G, nodelist)
    Atrain, Atest=get_2matrices_from_A( A, len(zz), data)
    return  Atrain, Atest, A


def get_2matrices_from_A(A,n,data):
    A=np.asmatrix(A)
    train_mask = data['domain_node'].train_mask
    xx_train=np.where(train_mask>0)   
    a=np.array(xx_train, dtype=bool)
    a=a * (a<n)
    xx_train=a
    Atrain=A.take(xx_train,axis=0).take(xx_train, axis=1) ### this is very useful
    test_mask = data['domain_node'].test_mask
    xx_test=np.where(test_mask.cpu()>0)    
    a=np.array(xx_test, dtype=bool)
    a=a * (a<n)
    xx_test=a
    print(A.shape, len(xx_test))
    Atest=A.take(xx_test,axis=0).take(xx_test, axis=1)
    return Atrain, Atest


    
