import argparse
import traceback
import time
import copy

import numpy as np
import dgl
import torch

from tgn import TGN
#from data_preprocess import TemporalWikipediaDataset, TemporalRedditDataset, TemporalDataset
from data_process import TemporalDataset
from dataloading import (FastTemporalEdgeCollator, FastTemporalSampler,
                         SimpleTemporalEdgeCollator, SimpleTemporalSampler,
                         TemporalEdgeDataLoader, TemporalSampler, TemporalEdgeCollator)

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import balanced_accuracy_score
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor

import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"

TRAIN_SPLIT = 0.7
VALID_SPLIT = 0.85

# set random Seed
np.random.seed(2021)
torch.manual_seed(2021)


def train(model, dataloader):
    model.eval()
    id = []
    embedding = []
    label = []
    for _, positive_pair_g, negative_pair_g, blocks in dataloader:
        positive_pair_g, negative_pair_g = positive_pair_g.to(args.device), negative_pair_g.to(args.device)
        feat_id, feat, lab = model.embed(positive_pair_g, negative_pair_g, blocks)
        for i in range(0, len(feat_id)):
            id.append(np.array(feat_id[i]))
        feat = feat.cpu().detach().numpy()
        lab = lab.cpu().detach().numpy()
        embedding = np.append(embedding,feat)
        label = np.append(label,lab)
    f_id = np.array(id)
    return f_id,embedding,label

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", type=int, default=50,
                        help='epochs for training on entire dataset')
    parser.add_argument("--batch_size", type=int,
                        default=200, help="Size of each batch")
    parser.add_argument("--embedding_dim", type=int, default=100,
                        help="Embedding dim for link prediction")
    parser.add_argument("--memory_dim", type=int, default=100,
                        help="dimension of memory")
    parser.add_argument("--temporal_dim", type=int, default=100,
                        help="Temporal dimension for time encoding")
    parser.add_argument("--memory_updater", type=str, default='gru',
                        help="Recurrent unit for memory update")
    parser.add_argument("--aggregator", type=str, default='last',
                        help="Aggregation method for memory update")
    parser.add_argument("--n_neighbors", type=int, default=10,
                        help="number of neighbors while doing embedding")
    parser.add_argument("--sampling_method", type=str, default='topk',
                        help="In embedding how node aggregate from its neighor")
    parser.add_argument("--num_heads", type=int, default=8,
                        help="Number of heads for multihead attention mechanism")
    parser.add_argument("--fast_mode", action="store_true", default=False,
                        help="Fast Mode uses batch temporal sampling, history within same batch cannot be obtained")
    parser.add_argument("--simple_mode", action="store_true", default=False,
                        help="Simple Mode directly delete the temporal edges from the original static graph")
    parser.add_argument("--num_negative_samples", type=int, default=1,
                        help="number of negative samplers per positive samples")
    parser.add_argument("--dataset", type=str, default="wikipedia",
                        help="dataset selection wikipedia/reddit")
    parser.add_argument("--k_hop", type=int, default=1,
                        help="sampling k-hop neighborhood")
    parser.add_argument("--not_use_memory", action="store_true", default=False,
                        help="Enable memory for TGN Model disable memory for TGN Model")
    parser.add_argument("--file",type=str, default="log.txt")
    parser.add_argument("--save_path", type=str, default="model")
    args = parser.parse_args()

    #args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.device = torch.device("cuda")
    data = TemporalDataset(args.dataset)


    # Pre-process data, mask new node in test set from original graph
    # 预处理数据，从原始图中屏蔽测试集中的新节点
    num_nodes = data.num_nodes()
    num_edges = data.num_edges()

    num_edges = data.num_edges()
    trainval_div = int(VALID_SPLIT*num_edges)

    # Select new node from test set and remove them from entire graph
    # 从测试集中选择新节点并将其从整个图中删除
    # unique()就是挑出tensor中的独立不重复元素
    test_split_ts = data.edata['timestamp'][trainval_div]
    test_nodes = torch.cat([data.edges()[0][trainval_div:], data.edges()[
                           1][trainval_div:]]).unique().cpu().numpy()
    test_new_nodes = np.random.choice(
        test_nodes, int(0.1*len(test_nodes)), replace=False)

    in_subg = dgl.in_subgraph(data, test_new_nodes) # 根据指定节点的入边生成边界子图
    out_subg = dgl.out_subgraph(data, test_new_nodes) # 根据给定节点的输出边创建新的图，除了提取子图之外，DGL还将提取的节点和边的特征复制到生成的图中
    #in_subg, out_subg = in_subg.to(args.device), out_subg.to(args.device)
    # Remove edge who happen before the test set to prevent from learning the connection info
    # 删除测试集之前发生的边缘，以防止学习连接信息
    new_node_in_eid_delete = in_subg.edata[dgl.EID][in_subg.edata['timestamp'] < test_split_ts]
    new_node_out_eid_delete = out_subg.edata[dgl.EID][out_subg.edata['timestamp'] < test_split_ts]
    new_node_eid_delete = torch.cat(
        [new_node_in_eid_delete, new_node_out_eid_delete]).unique()

    graph_new_node = copy.deepcopy(data) # 拷贝，拷贝对象及其子对象，哪怕以后对其有改动，也不会影响其第一次的拷贝，就是不随对象的后续改动而改变
    # relative order preseved 预设的相对顺序
    graph_new_node.remove_edges(new_node_eid_delete) # 删除指定的边并返回新图形，生成的图具有和输入图相同数量的节点，即使某些节点在边缘移除后变得孤立

    # Now for no new node graph, all edge id need to be removed
    # 现在，如果没有新的节点图，则需要删除所有边id
    in_eid_delete = in_subg.edata[dgl.EID]
    out_eid_delete = out_subg.edata[dgl.EID]
    eid_delete = torch.cat([in_eid_delete, out_eid_delete]).unique()

    graph_no_new_node = copy.deepcopy(data)
    #graph_no_new_node = graph_no_new_node.to(args.device)
    graph_no_new_node.remove_edges(eid_delete)

    # graph_no_new_node and graph_new_node should have same set of nid
    # graph_no_new_node和graph_new_node应具有相同的nid集

    # Sampler Initialization 采样器初始化
    if args.simple_mode:
        fan_out = [args.n_neighbors for _ in range(args.k_hop)]
        sampler = SimpleTemporalSampler(graph_no_new_node, fan_out)
        new_node_sampler = SimpleTemporalSampler(data, fan_out)
        edge_collator = SimpleTemporalEdgeCollator
    elif args.fast_mode:
        sampler = FastTemporalSampler(graph_no_new_node, k=args.n_neighbors)
        new_node_sampler = FastTemporalSampler(data, k=args.n_neighbors)
        edge_collator = FastTemporalEdgeCollator
    else:
        sampler = TemporalSampler(k=args.n_neighbors)
        edge_collator = TemporalEdgeCollator

    neg_sampler = dgl.dataloading.negative_sampler.Uniform(
        k=args.num_negative_samples) # dgl内置的负采样器
    # Set Train, validation, test and new node test id
    # 设置训练、验证、测试和新节点测试id
    # torch.arange()返回一个大小为(end-start)/step的一维张量，其值介于区间[start,end]，以step为步长等间隔取值，step默认值为1
    train_seed = torch.arange(int(TRAIN_SPLIT*graph_no_new_node.num_edges()))
    valid_seed = torch.arange(int(
        TRAIN_SPLIT*graph_no_new_node.num_edges()), trainval_div-new_node_eid_delete.size(0))
    test_seed = torch.arange(
        trainval_div-new_node_eid_delete.size(0), graph_no_new_node.num_edges())
    test_new_node_seed = torch.arange(
        trainval_div-new_node_eid_delete.size(0), graph_new_node.num_edges())

    # dgl.add_reverse_edges可为图中的每条边添加对应的反向边
    g_sampling = None if args.fast_mode else dgl.add_reverse_edges(
        graph_no_new_node, copy_edata=True)
    new_node_g_sampling = None if args.fast_mode else dgl.add_reverse_edges(
        graph_new_node, copy_edata=True)
    if not args.fast_mode:
        new_node_g_sampling.ndata[dgl.NID] = new_node_g_sampling.nodes()
        g_sampling.ndata[dgl.NID] = new_node_g_sampling.nodes()

    # we highly recommend that you always set the num_workers=0, otherwise the sampled subgraph may not be correct.
    # 我们强烈建议您始终将numworkers设置为0，否则采样的子图可能不正确。
    # 数据集采样
    train_dataloader = TemporalEdgeDataLoader(graph_no_new_node,
                                              train_seed,
                                              sampler,
                                              batch_size=args.batch_size,
                                              negative_sampler=neg_sampler,
                                              shuffle=False,
                                              drop_last=False,
                                              num_workers=0,
                                              collator=edge_collator,
                                              g_sampling=g_sampling,
                                              device=args.device)

    valid_dataloader = TemporalEdgeDataLoader(graph_no_new_node,
                                              valid_seed,
                                              sampler,
                                              batch_size=args.batch_size,
                                              negative_sampler=neg_sampler,
                                              shuffle=False,
                                              drop_last=False,
                                              num_workers=0,
                                              collator=edge_collator,
                                              g_sampling=g_sampling,
                                              device=args.device)

    test_dataloader = TemporalEdgeDataLoader(graph_no_new_node,
                                             test_seed,
                                             sampler,
                                             batch_size=args.batch_size,
                                             negative_sampler=neg_sampler,
                                             shuffle=False,
                                             drop_last=False,
                                             num_workers=0,
                                             collator=edge_collator,
                                             g_sampling=g_sampling,
                                             device=args.device)

    test_new_node_dataloader = TemporalEdgeDataLoader(graph_new_node,
                                                      test_new_node_seed,
                                                      new_node_sampler if args.fast_mode else sampler,
                                                      batch_size=args.batch_size,
                                                      negative_sampler=neg_sampler,
                                                      shuffle=False,
                                                      drop_last=False,
                                                      num_workers=0,
                                                      collator=edge_collator,
                                                      g_sampling=new_node_g_sampling,
                                                      device=args.device)

    edge_dim = data.edata['feats'].shape[1]
    num_node = data.num_nodes()

    #save_path = 'model/std_tgat/trained_model.pkl'
    model = torch.load(args.save_path)
    if isinstance(model,torch.nn.DataParallel):
        model = model.module
    f = open(args.file, 'w')
    train_id, train_emb, train_label = train(model, train_dataloader)
    number_id1 = len(train_id)
    number_emb1 = len(train_emb)
    train_emb = train_emb.reshape(number_id1, int(number_emb1/number_id1))
    train_label = train_label.reshape(-1,1)
    train_node = list(set(train_id))  #真正的节点序列
    train_emb_dict = dict.fromkeys(train_node, '2')
    train_label_dict = dict.fromkeys(train_node, '2')
    for i in range(0, number_id1):
        train_emb_dict[train_id[i]] = train_emb[i]
        train_label_dict[train_id[i]] = train_label[i]

    test_id, test_emb, test_label = train(model, test_dataloader)
    number_id2 = len(test_id)
    number_emb2 = len(test_emb)
    test_emb = test_emb.reshape(number_id2, int(number_emb2/number_id2))
    #test_label = test_label.reshape(-1, 1)
    test_node = list(set(test_id))  # 真正的节点序列
    test_emb_dict = dict.fromkeys(test_node, '2')
    test_label_dict = dict.fromkeys(test_node, '2')
    for i in range(0, number_id2):
        test_emb_dict[test_id[i]] = test_emb[i]
        test_label_dict[test_id[i]] = test_label[i]
    #print(test_emb_dict)
    #print(test_label_dict)
    test_new_id, test_new_emb, test_new_label = train(model, test_new_node_dataloader)
    number_id3 = len(test_new_id)
    number_emb3 = len(test_new_emb)
    test_new_emb = test_new_emb.reshape(number_id3, int(number_emb3 / number_id3))
    # test_new_label = test_new_label.reshape(-1, 1)
    test_new_node = list(set(test_new_id))  # 真正的节点序列
    test_new_emb_dict = dict.fromkeys(test_new_node, '2')
    test_new_label_dict = dict.fromkeys(test_new_node, '2')
    for i in range(0, number_id3):
        test_new_emb_dict[test_new_id[i]] = test_new_emb[i]
        test_new_label_dict[test_new_id[i]] = test_new_label[i]
    train_x_list = []
    train_y_list = []
    for j in range(0, len(train_node)):
        if train_label_dict[train_node[j]] != 2:
            train_x_list.append(train_emb_dict[train_node[j]])
            train_y_list.append(train_label_dict[train_node[j]])
    train_x = np.array(train_x_list)
    train_y = np.array(train_y_list)
    #train_y = train_y.reshape(-1,1)
    test_x_list = []
    test_y_list = []
    for k in range(0, len(test_node)):
        if test_label_dict[test_node[k]] != 2:
            test_x_list.append(test_emb_dict[test_node[k]])
            test_y_list.append(test_label_dict[test_node[k]])
    test_x = np.array(test_x_list)
    test_y = np.array(test_y_list)
    #test_y = test_y.reshape(-1, 1)
    test_new_x_list = []
    test_new_y_list = []
    for n in range(0, len(test_new_node)):
        if test_new_label_dict[test_new_node[n]] != 2:
            test_new_x_list.append(test_new_emb_dict[test_new_node[n]])
            test_new_y_list.append(test_new_label_dict[test_new_node[n]])
    test_new_x = np.array(test_new_x_list)
    test_new_y = np.array(test_new_y_list)
    # test_y = test_y.reshape(-1, 1)
    print(train_x.shape)
    print(train_y.shape)
    print(test_x.shape)
    print(test_y.shape)
    print(test_new_x.shape)
    print(test_new_y.shape)
    # 随机过采样
    ros = RandomOverSampler(random_state=0)
    train_x_ros, train_y_ros = ros.fit_resample(train_x, train_y)
    test_x_ros, test_y_ros = ros.fit_resample(test_x, test_y)
    test_new_x_ros, test_new_y_ros = ros.fit_resample(test_new_x, test_new_y)
    #train_y_ros = train_y_ros.reshape(-1,1)
    # SMOTE过采样
    model_smote = SMOTE(random_state=0)
    train_x_smote, train_y_smote = model_smote.fit_resample(train_x,train_y)
    test_x_smote, test_y_smote = model_smote.fit_resample(test_x, test_y)
    test_new_x_smote, test_new_y_smote = model_smote.fit_resample(test_new_x, test_new_y)
    #train_y_smote = train_y_smote.reshape(-1, 1)
    #train_x_smote, train_y_smote = train_x, train_y
    #test_x_smote, test_y_smote = test_x, test_y
    # 随机欠采样
    rus = RandomUnderSampler(random_state=0)
    train_x_rus, train_y_rus = rus.fit_resample(train_x, train_y)
    test_x_rus, test_y_rus = rus.fit_resample(test_x, test_y)
    test_new_x_rus, test_new_y_rus = rus.fit_resample(test_new_x, test_new_y)
    #train_y_rus = train_y_rus.reshape(-1, 1)
    # 综合采样
    kos = SMOTETomek(random_state=0)
    train_x_kos, train_y_kos = kos.fit_resample(train_x, train_y)
    test_x_kos, test_y_kos = kos.fit_resample(test_x, test_y)
    test_new_x_kos, test_new_y_kos = kos.fit_resample(test_new_x, test_new_y)
    #train_y_kos = train_y_kos.reshape(-1, 1)
    # 支持向量机模型
    #svc = SVC()
    #svc.fit(train_x_smote,train_y_smote.ravel())
    #predict_y = svc.predict(test_x_smote)
    # 决策树模型
    #dtc = DecisionTreeClassifier()
    #dtc.fit(train_x_smote,train_y_smote.ravel())
    #predict_y = dtc.predict(test_x_smote)
    # KNN模型
    #knn = KNeighborsClassifier()
    #knn.fit(train_x_smote,train_y_smote.ravel())
    #predict_y = knn.predict(test_x_smote)
    # AdaBoost模型
    #ada = AdaBoostClassifier()
    #ada.fit(train_x_smote,train_y_smote.ravel())
    #predict_y = ada.predict(test_x_smote)
    #cv = GridSearchCV(ada)
    # XGBoost
    #xgb = XGBRegressor()
    #xgb.fit(train_x_smote,train_y_smote.ravel())
    #predict_y = xgb.predict(test_x_smote)
    #gbdt = GradientBoostingClassifier()
    #gbdt.fit(train_x_smote,train_y_smote.ravel())
    #predict_y = gbdt.predict(test_x_smote)
    np.save('data/train_x.npy',train_x)
    np.save('data/train_y.npy', train_y)
    np.save('data/test_x.npy', test_x)
    np.save('data/test_y.npy', test_y)
    np.save('data/test_new_x.npy', test_new_x)
    np.save('data/test_new_y.npy', test_new_y)
    np.save('data/train_x_ros.npy', train_x_ros)
    np.save('data/train_y_ros.npy', train_y_ros)
    np.save('data/test_x_ros.npy', test_x_ros)
    np.save('data/test_y_ros.npy', test_y_ros)
    np.save('data/test_new_x_ros.npy', test_new_x_ros)
    np.save('data/test_new_y_ros.npy', test_new_y_ros)
    np.save('data/train_x_smote.npy', train_x_smote)
    np.save('data/train_y_smote.npy', train_y_smote)
    np.save('data/test_x_smote.npy', test_x_smote)
    np.save('data/test_y_smote.npy', test_y_smote)
    np.save('data/test_new_x_smote.npy', test_new_x_smote)
    np.save('data/test_new_y_smote.npy', test_new_y_smote)
    np.save('data/train_x_rus.npy', train_x_rus)
    np.save('data/train_y_rus.npy', train_y_rus)
    np.save('data/test_x_rus.npy', test_x_rus)
    np.save('data/test_y_rus.npy', test_y_rus)
    np.save('data/test_new_x_rus.npy', test_new_x_rus)
    np.save('data/test_new_y_rus.npy', test_new_y_rus)
    np.save('data/train_x_kos.npy', train_x_kos)
    np.save('data/train_y_kos.npy', train_y_kos)
    np.save('data/test_x_kos.npy', test_x_kos)
    np.save('data/test_y_kos.npy', test_y_kos)
    np.save('data/test_new_x_kos.npy', test_new_x_kos)
    np.save('data/test_new_y_kos.npy', test_new_y_kos)
    dataset = [[train_x, train_y, test_x, test_y, test_new_x, test_new_y],
               [train_x_ros, train_y_ros, test_x_ros, test_y_ros, test_new_x_ros, test_new_y_ros],
               [train_x_smote, train_y_smote, test_x_smote, test_y_smote, test_new_x_smote, test_new_y_smote],
               [train_x_rus, train_y_rus, test_x_rus, test_y_rus, test_new_x_rus, test_new_y_rus],
               [train_x_kos, train_y_kos, test_x_kos, test_y_kos, test_new_x_kos, test_new_y_kos]]
    log_content = []
    for datas, labels, test_datas, test_labels, test_new_datas, test_new_labels in dataset:
        #ada = AdaBoostClassifier()
        #ada.fit(datas, labels.ravel())
        #predict_y = ada.predict(test_x)
        knn = KNeighborsClassifier()
        knn.fit(datas, labels.ravel())
        predict_y = knn.predict(test_datas)
        predict_new_y = knn.predict(test_new_datas)
        log_content.append("test_y:{}\n".format(test_labels))
        log_content.append("predict_y:{}\n".format(predict_y))
        log_content.append(
            "accuracy_score:{:.3f}\n balanced_accuracy:{:.3f}\n average_precision:{:.3f}\n f1_score:{:.3f}\n auc:{:.3f}\n".format(
                accuracy_score(test_labels, predict_y), balanced_accuracy_score(test_labels, predict_y),
                average_precision_score(test_labels, predict_y), f1_score(test_labels, predict_y),roc_auc_score(test_labels, predict_y)
            ))
        log_content.append("test_new_y:{}\n".format(test_new_labels))
        log_content.append("predict_new_y:{}\n".format(predict_new_y))
        log_content.append(
            "accuracy_score:{:.3f}\n balanced_accuracy:{:.3f}\n average_precision:{:.3f}\n f1_score:{:.3f}\n auc:{:.3f}\n".format(
                accuracy_score(test_new_labels, predict_new_y), balanced_accuracy_score(test_new_labels, predict_new_y),
                average_precision_score(test_new_labels, predict_new_y), f1_score(test_new_labels, predict_new_y),
                roc_auc_score(test_new_labels, predict_new_y)
            ))
    f.writelines(log_content)
    print(log_content)