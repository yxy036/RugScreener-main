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

from sklearn.metrics import average_precision_score, roc_auc_score
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"

TRAIN_SPLIT = 0.7
VALID_SPLIT = 0.85

# set random Seed
np.random.seed(2021)
torch.manual_seed(2021)


def train(model, dataloader, sampler, criterion, optimizer, args):
    print("if on cuda:", next(model.parameters()).device)
    model.train()
    total_loss = 0
    batch_cnt = 0
    last_t = time.time()
    # （block是指采样得到的子图，可以把block看作一个数组，数组里的每一个元素是图上一层邻居的采样，Block内部节点是 从远到近的顺序排列内部的Block的，Block数组的下标从小到大对应着采样范围由外到内、覆盖范围由远及近，并且 blocks[i+1]的 source node 和 blocks[i]的target node是可以对应上的
    for _, positive_pair_g, negative_pair_g, blocks in dataloader:
        positive_pair_g, negative_pair_g = positive_pair_g.to(args.device), negative_pair_g.to(args.device)
        print("pos.device",positive_pair_g.device)
        print("neg_device",negative_pair_g.device)
        optimizer.zero_grad()
        if torch.cuda.device_count() > 1:
            #model = torch.nn.DataParallel(model)
            pred_pos, pred_neg = model.module.embed(
                            positive_pair_g, negative_pair_g, blocks)
        else:
            pred_pos, pred_neg = model.embed(
                positive_pair_g, negative_pair_g, blocks)
        print("pred.device",pred_pos.device)
        print("pred_device",pred_neg.device)
        pred_pos, pred_neg = pred_pos.to(args.device), pred_neg.to(args.device)
        #loss = criterion(pred_pos, torch.ones_like(pred_pos)).to(args.device)
        if torch.cuda.device_count() > 1:
            loss = torch.mean(criterion(pred_pos, torch.ones_like(pred_pos)).to(args.device))
            loss += torch.mean(criterion(pred_neg, torch.zeros_like(pred_neg)).to(args.device))
        else:
            loss = criterion(pred_pos, torch.ones_like(pred_pos)).to(args.device)
            loss += criterion(pred_neg, torch.zeros_like(pred_neg)).to(args.device)
        total_loss += float(loss)*args.batch_size
        retain_graph = True if batch_cnt == 0 and not args.fast_mode else False
        loss.backward(retain_graph=retain_graph)
        optimizer.step()
        model.module.detach_memory()
        if not args.not_use_memory:
            model.module.update_memory(positive_pair_g)
        if args.fast_mode:
            sampler.attach_last_update(model.module.memory.last_update_t)
        print("Batch: ", batch_cnt, "Time: ", time.time()-last_t)
        last_t = time.time()
        batch_cnt += 1
    return total_loss


def test_val(model, dataloader, sampler, criterion, args):
    model.eval()
    batch_size = args.batch_size
    total_loss = 0
    aps, aucs = [], []
    batch_cnt = 0
    with torch.no_grad():
        for _, postive_pair_g, negative_pair_g, blocks in dataloader:
            postive_pair_g, negative_pair_g = postive_pair_g.to(args.device), negative_pair_g.to(args.device)
            if torch.cuda.device_count() > 1:
                #model = torch.nn.DataParallel(model)
                pred_pos, pred_neg = model.module.embed(
                                        postive_pair_g, negative_pair_g, blocks)
            else:
                pred_pos, pred_neg = model.embed(
                    postive_pair_g, negative_pair_g, blocks)
            pred_pos, pred_neg = pred_pos.to(args.device), pred_neg.to(args.device)
            #loss = criterion(pred_pos, torch.ones_like(pred_pos)).to(args.device)
            if torch.cuda.device_count() > 1:
                loss = torch.mean(criterion(pred_pos, torch.ones_like(pred_pos)).to(args.device))
                loss += torch.mean(criterion(pred_neg, torch.zeros_like(pred_neg)).to(args.device))
            else:
                loss = criterion(pred_pos, torch.ones_like(pred_pos)).to(args.device)
                loss += criterion(pred_neg, torch.zeros_like(pred_neg)).to(args.device)
            total_loss += float(loss)*batch_size
            # 目标分数可以是肯定类别的概率估计值，置信度值或决策的非阈值度量
            y_pred = torch.cat([pred_pos, pred_neg], dim=0).sigmoid().cpu().numpy()
            # 真正的二进制标签
            y_true = torch.cat(
                [torch.ones(pred_pos.size(0)), torch.zeros(pred_neg.size(0))], dim=0)
            if not args.not_use_memory:
                model.module.update_memory(postive_pair_g)
            if args.fast_mode:
                sampler.attach_last_update(model.module.memory.last_update_t)

            aps.append(average_precision_score(y_true, y_pred)) # 预测值的平均准确率，计算AP
            aucs.append(roc_auc_score(y_true, y_pred)) # 计算AUC
            batch_cnt += 1
    return float(torch.tensor(aps).mean()), float(torch.tensor(aucs).mean())


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
    parser.add_argument("--save", type=str, default="model")
    args = parser.parse_args()

    assert not (
        args.fast_mode and args.simple_mode), "you can only choose one sampling mode"
    if args.k_hop != 1:
        assert args.simple_mode, "this k-hop parameter only support simple mode"
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

    model = TGN(edge_feat_dim=edge_dim,
                memory_dim=args.memory_dim,
                temporal_dim=args.temporal_dim,
                embedding_dim=args.embedding_dim,
                num_heads=args.num_heads,
                num_nodes=num_node,
                n_neighbors=args.n_neighbors,
                memory_updater_type=args.memory_updater,
                layers=args.k_hop)
    if torch.cuda.device_count() > 1:
        print("{} GPUs are available. Let's use them.".format(torch.cuda.device_count()))
        model = torch.nn.DataParallel(model)
    model = model.to(args.device)
    criterion = torch.nn.BCEWithLogitsLoss() # 损失函数
    criterion = criterion.to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001) # 优化器

    # Implement Logging mechanism 日志记录
    print("if on cuda:",next(model.parameters()).device)
    f = open(args.file, 'w')
    if args.fast_mode:
        sampler.reset()
    try:
        for i in range(args.epochs):
            train_loss = train(model, train_dataloader, sampler,
                               criterion, optimizer, args)

            val_ap, val_auc = test_val(
                model, valid_dataloader, sampler, criterion, args)
            memory_checkpoint = model.module.store_memory()
            if args.fast_mode:
                new_node_sampler.sync(sampler)
            test_ap, test_auc = test_val(
                model, test_dataloader, sampler, criterion, args)
            model.module.restore_memory(memory_checkpoint)
            if args.fast_mode:
                sample_nn = new_node_sampler
            else:
                sample_nn = sampler
            nn_test_ap, nn_test_auc = test_val(
                model, test_new_node_dataloader, sample_nn, criterion, args)
            log_content = []
            log_content.append("Epoch: {}; Training Loss: {} | Validation AP: {:.3f} AUC: {:.3f}\n".format(
                i, train_loss, val_ap, val_auc))
            log_content.append(
                "Epoch: {}; Test AP: {:.3f} AUC: {:.3f}\n".format(i, test_ap, test_auc))
            log_content.append("Epoch: {}; Test New Node AP: {:.3f} AUC: {:.3f}\n".format(
                i, nn_test_ap, nn_test_auc))

            f.writelines(log_content)
            model.module.reset_memory()
            if i < args.epochs-1 and args.fast_mode:
                sampler.reset()
            print(log_content[0], log_content[1], log_content[2])
            name = str(i)
            path = "model/" + args.save + "/trained_model_" + name + ".pkl"
            torch.save(model.module.state_dict(), path)
        save_path = "model/" + args.save + "/trained_model.pkl"
        torch.save(model, save_path)
    except KeyboardInterrupt:
        traceback.print_exc()
        error_content = "Training Interreputed!"
        f.writelines(error_content)
        f.close()
    print("========Training is Done========")
