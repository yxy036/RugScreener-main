import copy
import torch.nn as nn
import dgl
from modules import MemoryModule, MemoryOperation, MsgLinkPredictor, TemporalTransformerConv, TimeEncode
import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TGN(nn.Module):
    def __init__(self,
                 edge_feat_dim,
                 memory_dim,
                 temporal_dim,
                 embedding_dim,
                 num_heads,
                 num_nodes,
                 n_neighbors=10,
                 memory_updater_type='gru',
                 layers=1):
        super(TGN, self).__init__()
        self.memory_dim = memory_dim
        self.edge_feat_dim = edge_feat_dim
        self.temporal_dim = temporal_dim
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.n_neighbors = n_neighbors
        self.memory_updater_type = memory_updater_type
        self.num_nodes = num_nodes
        self.layers = layers

        self.temporal_encoder = TimeEncode(self.temporal_dim)

        self.memory = MemoryModule(self.num_nodes,
                                   self.memory_dim)

        self.memory_ops = MemoryOperation(self.memory_updater_type,
                                          self.memory,
                                          self.edge_feat_dim,
                                          self.temporal_encoder)

        self.embedding_attn = TemporalTransformerConv(self.edge_feat_dim,
                                                      self.memory_dim,
                                                      self.temporal_encoder,
                                                      self.embedding_dim,
                                                      self.num_heads,
                                                      layers=self.layers,
                                                      allow_zero_in_degree=True)

        self.msg_linkpredictor = MsgLinkPredictor(embedding_dim)

    def embed(self, postive_graph, negative_graph, blocks):
        emb_graph = blocks[0].to(device)
        emb_memory = self.memory.memory[emb_graph.ndata[dgl.NID], :].to(device)  # 在子图中通过dgl.NID获取父图中节点的映射
        emb_t = emb_graph.ndata['timestamp'].to(device)
        embedding = self.embedding_attn(emb_graph, emb_memory, emb_t)
        emb2pred = dict(
            zip(emb_graph.ndata[dgl.NID].tolist(), emb_graph.nodes().tolist())) # 将两个列表合并成一个字典
        # Since postive graph and negative graph has same is mapping由于正图和负图具有相同的映射
        feat_id = [emb2pred[int(n)] for n in postive_graph.ndata[dgl.NID]]
        feat = embedding[feat_id]
        in_degree = emb_graph.ndata['in_degree'][feat_id].unsqueeze(-1)
        out_degree = emb_graph.ndata['out_degree'][feat_id].unsqueeze(-1)
        c_degree = emb_graph.ndata['c_degree'][feat_id].unsqueeze(-1)
        pagerank = emb_graph.ndata['pagerank'][feat_id].unsqueeze(-1)
        c_closeness = emb_graph.ndata['c_closeness'][feat_id].unsqueeze(-1)
        c_betweenness = emb_graph.ndata['c_betweenness'][feat_id].unsqueeze(-1)
        c_eigenvector = emb_graph.ndata['c_eigenvector'][feat_id].unsqueeze(-1)
        feat_ret = torch.cat(
            [feat, in_degree, out_degree, c_degree, pagerank, c_closeness, c_betweenness, c_eigenvector], dim=1)
        label = emb_graph.ndata['label'][feat_id]
        #pred_pos, pred_neg = self.msg_linkpredictor(
        #    feat, postive_graph, negative_graph)
        #return pred_pos, pred_neg
        return feat_id, feat_ret, label

    def update_memory(self, subg):
        new_g = self.memory_ops(subg)
        self.memory.set_memory(new_g.ndata[dgl.NID], new_g.ndata['memory'])
        self.memory.set_last_update_t(
            new_g.ndata[dgl.NID], new_g.ndata['timestamp'])

    # Some memory operation wrappers 一些内存操作包装器
    def detach_memory(self):
        self.memory.detach_memory()

    def reset_memory(self):
        self.memory.reset_memory()

    def store_memory(self):
        memory_checkpoint = {}
        memory_checkpoint['memory'] = copy.deepcopy(self.memory.memory)
        memory_checkpoint['last_t'] = copy.deepcopy(self.memory.last_update_t)
        return memory_checkpoint

    def restore_memory(self, memory_checkpoint):
        self.memory.memory = memory_checkpoint['memory']
        self.memory.last_update_time = memory_checkpoint['last_t']
