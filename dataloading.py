import torch
import dgl

from dgl._dataloading.dataloader import EdgeCollator
from dgl._dataloading import BlockSampler
from dgl._dataloading.pytorch import _pop_subgraph_storage, _pop_storages, EdgeDataLoader
from dgl.base import DGLError

from functools import partial
import copy
import dgl.function as fn


def _prepare_tensor(g, data, name, is_distributed):
    return torch.tensor(data) if is_distributed else dgl.utils.prepare_tensor(g, data, name)


class TemporalSampler(BlockSampler):
    """ Temporal Sampler builds computational and temporal dependency of node representations via
    temporal neighbors selection and screening.
    The sampler expects input node to have same time stamps, in the case of TGN, it should be
    either positive [src,dst] pair or negative samples. It will first take in-subgraph of seed
    nodes and then screening out edges which happen after that timestamp. Finally it will sample
    a fixed number of neighbor edges using random or topk sampling.
    Parameters
    ----------
    sampler_type : str
        sampler indication string of the final sampler.
        If 'topk' then sample topk most recent nodes
        If 'uniform' then uniform randomly sample k nodes
    k : int
        maximum number of neighors to sampler
        default 10 neighbors as paper stated
    Examples
    ----------
    Please refers to examples/pytorch/tgn/train.py
    """

    def __init__(self, sampler_type='topk', k=10):
        super(TemporalSampler, self).__init__(1, False)
        if sampler_type == 'topk':
            self.sampler = partial(
                dgl.sampling.select_topk, k=k, weight='timestamp')
        elif sampler_type == 'uniform':
            self.sampler = partial(dgl.sampling.sample_neighbors, fanout=k)
        else:
            raise DGLError(
                "Sampler string invalid please use \'topk\' or \'uniform\'")

    def sampler_frontier(self,
                         block_id,
                         g,
                         seed_nodes,
                         timestamp):
        full_neighbor_subgraph = dgl.in_subgraph(g, seed_nodes)
        full_neighbor_subgraph = dgl.add_edges(full_neighbor_subgraph,
                                               seed_nodes, seed_nodes)

        temporal_edge_mask = (full_neighbor_subgraph.edata['timestamp'] < timestamp) + (
            full_neighbor_subgraph.edata['timestamp'] <= 0)
        temporal_subgraph = dgl.edge_subgraph(
            full_neighbor_subgraph, temporal_edge_mask)

        # Map preserve ID
        temp2origin = temporal_subgraph.ndata[dgl.NID]

        # The added new edgge will be preserved hence
        root2sub_dict = dict(
            zip(temp2origin.tolist(), temporal_subgraph.nodes().tolist()))
        temporal_subgraph.ndata[dgl.NID] = g.ndata[dgl.NID][temp2origin]
        seed_nodes = [root2sub_dict[int(n)] for n in seed_nodes]
        final_subgraph = self.sampler(g=temporal_subgraph, nodes=seed_nodes)
        final_subgraph.remove_self_loop()
        return final_subgraph

        # Temporal Subgraph
    def sample_blocks(self,
                      g,
                      seed_nodes,
                      timestamp):
        blocks = []
        frontier = self.sampler_frontier(0, g, seed_nodes, timestamp)
        #block = transform.to_block(frontier,seed_nodes)
        block = frontier
        if self.return_eids:
            self.assign_block_eids(block, frontier)
        blocks.append(block)
        return blocks


class TemporalEdgeCollator(EdgeCollator):
    """ Temporal Edge collator merge the edges specified by eid: items
    Since we cannot keep duplicated nodes on a graph we need to iterate though
    the incoming edges and expand the duplicated node and form a batched block
    graph capture the temporal and computational dependency.
    Parameters
    ----------
    g : DGLGraph
        The graph from which the edges are iterated in minibatches and the subgraphs
        are generated.
    eids : Tensor or dict[etype, Tensor]
        The edge set in graph :attr:`g` to compute outputs.
    graph_sampler : dgl.dataloading.BlockSampler
        The neighborhood sampler.
    g_sampling : DGLGraph, optional
        The graph where neighborhood sampling and message passing is performed.
        Note that this is not necessarily the same as :attr:`g`.
        If None, assume to be the same as :attr:`g`.
    exclude : str, optional
        Whether and how to exclude dependencies related to the sampled edges in the
        minibatch.  Possible values are
        * None, which excludes nothing.
        * ``'reverse_id'``, which excludes the reverse edges of the sampled edges.  The said
          reverse edges have the same edge type as the sampled edges.  Only works
          on edge types whose source node type is the same as its destination node type.
        * ``'reverse_types'``, which excludes the reverse edges of the sampled edges.  The
          said reverse edges have different edge types from the sampled edges.
        If ``g_sampling`` is given, ``exclude`` is ignored and will be always ``None``.
    reverse_eids : Tensor or dict[etype, Tensor], optional
        The mapping from original edge ID to its reverse edge ID.
        Required and only used when ``exclude`` is set to ``reverse_id``.
        For heterogeneous graph this will be a dict of edge type and edge IDs.  Note that
        only the edge types whose source node type is the same as destination node type
        are needed.
    reverse_etypes : dict[etype, etype], optional
        The mapping from the edge type to its reverse edge type.
        Required and only used when ``exclude`` is set to ``reverse_types``.
    negative_sampler : callable, optional
        The negative sampler.  Can be omitted if no negative sampling is needed.
        The negative sampler must be a callable that takes in the following arguments:
        * The original (heterogeneous) graph.
        * The ID array of sampled edges in the minibatch, or the dictionary of edge
          types and ID array of sampled edges in the minibatch if the graph is
          heterogeneous.
        It should return
        * A pair of source and destination node ID arrays as negative samples,
          or a dictionary of edge types and such pairs if the graph is heterogenenous.
        A set of builtin negative samplers are provided in
        :ref:`the negative sampling module <api-dataloading-negative-sampling>`.
    example
    ----------
    Please refers to examples/pytorch/tgn/train.py
    Temporal Edge collator合并eid:items指定的边
    由于我们不能在图上保留重复的节点，因此我们需要遍历传入的边并扩展重复的节点并形成一个批处理块图，以捕获时间和计算依赖性。
    g : DGLGraph,边在minibarcg中迭代并生成子图的图。
    eids : Tensor or dict[etype, Tensor]，图g中用于计算输出的边集。
        The edge set in graph :attr:`g` to compute outputs.
    graph_sampler : dgl.dataloading.BlockSampler，邻居采样器。
    g_sampling : DGLGraph, optional，执行邻域采样和消息传递的图形。注意，这不一定与g相同。如果无，则假设与g相同。
    exclude : str, optional
        是否以及如何排除与小批次中采样边相关的相关性。  可能的值为：
        * None, which excludes nothing.
        * ``'reverse_id'``, 这排除了采样边缘的反向边缘。所述反向边缘具有与采样边缘相同的边缘类型。仅适用于源节点类型与其目标节点类型相同的边类型。
        * ``'reverse_types'``, 这排除了采样边缘的反向边缘。所述反向边缘具有与采样边缘不同的边缘类型。
        如果给定“g_sampling”，则忽略“exclude”，并始终为“None”。
    reverse_eids : Tensor or dict[etype, Tensor], optional
        从原始边缘ID到其反向边缘ID的映射。
        必需，仅在“exclude”设置为“reverse_id”时使用。
        对于异构图，这将是边类型和边ID的字典。注意，只需要源节点类型与目标节点类型相同的边缘类型。
    reverse_etypes : dict[etype, etype], optional
        从边缘类型到其反向边缘类型的映射。
        必需，仅在“exclude”设置为“reverse_types”时使用。
    negative_sampler : callable, optional
        负采样器。如果不需要负采样，可以省略。
        负采样器必须是接受以下参数的可调用函数：
        * 原始（异构）图。
        * minibatch中采样边的ID数组，或者minibatch中的边类型字典和采样边ID数组（如果图形是异构的）。
        它应该会返回
        * 一对源和目的节点ID阵列作为负样本，或者如果图是异质的，则可以使用边缘类型和这样的对的字典。
        A set of builtin negative samplers are provided in
        :ref:`the negative sampling module <api-dataloading-negative-sampling>`.
    """

    def _collate_with_negative_sampling(self, items):
        items = _prepare_tensor(self.g_sampling, items, 'items', False)
        # Here node id will not change 此处节点id不会更改
        # 提取包含指定边的子图
        pair_graph = self.g.edge_subgraph(items, relabel_nodes=False)
        induced_edges = pair_graph.edata[dgl.EID] # 边的原始id
        # 使用dgl内置的负采样器
        neg_srcdst_raw = self.negative_sampler(self.g, items)
        # dgl.canonical_etypes以三元组的形式返回图中定义的边
        neg_srcdst = {self.g.canonical_etypes[0]: neg_srcdst_raw}
        dtype = list(neg_srcdst.values())[0][0].dtype
        neg_edges = {
            etype: neg_srcdst.get(etype, (torch.tensor(
                [], dtype=dtype), torch.tensor([], dtype=dtype)))
            for etype in self.g.canonical_etypes}
        # 根据边缘类型和边缘列表之间的字典创建异构图形
        neg_pair_graph = dgl.heterograph(
            neg_edges, {ntype: self.g.number_of_nodes(ntype) for ntype in self.g.ntypes})
        # 给定具有相同节点集的图列表，查找并消除所有图中的公共孤立节点
        pair_graph, neg_pair_graph = dgl.transforms.compact_graphs(
            [pair_graph, neg_pair_graph])
        # Need to remap id
        pair_graph.ndata[dgl.NID] = self.g.nodes()[pair_graph.ndata[dgl.NID]]
        neg_pair_graph.ndata[dgl.NID] = self.g.nodes()[
            neg_pair_graph.ndata[dgl.NID]]

        pair_graph.edata[dgl.EID] = induced_edges

        batch_graphs = []
        nodes_id = []
        timestamps = []

        for i, edge in enumerate(zip(self.g.edges()[0][items], self.g.edges()[1][items])):
            ts = pair_graph.edata['timestamp'][i]
            timestamps.append(ts)
            subg = self.graph_sampler.sample_blocks(self.g_sampling,
                                                    list(edge),
                                                    timestamp=ts)[0]
            subg.ndata['timestamp'] = ts.repeat(subg.num_nodes())
            nodes_id.append(subg.srcdata[dgl.NID])
            batch_graphs.append(subg)
        timestamps = torch.tensor(timestamps).repeat_interleave(
            self.negative_sampler.k)
        for i, neg_edge in enumerate(zip(neg_srcdst_raw[0].tolist(), neg_srcdst_raw[1].tolist())):
            ts = timestamps[i]
            subg = self.graph_sampler.sample_blocks(self.g_sampling,
                                                    [neg_edge[1]],
                                                    timestamp=ts)[0]
            subg.ndata['timestamp'] = ts.repeat(subg.num_nodes())
            batch_graphs.append(subg)
        blocks = [dgl.batch(batch_graphs)]
        input_nodes = torch.cat(nodes_id)
        return input_nodes, pair_graph, neg_pair_graph, blocks

    def collator(self, items):
        """
        The interface of collator, input items is edge id of the attached graph
        """
        result = super().collate(items)
        # Copy the feature from parent graph
        _pop_subgraph_storage(result[1], self.g)
        _pop_subgraph_storage(result[2], self.g)
        _pop_storages(result[-1], self.g_sampling)
        return result


class TemporalEdgeDataLoader(EdgeDataLoader):
    """ TemporalEdgeDataLoader is an iteratable object to generate blocks for temporal embedding
    as well as pos and neg pair graph for memory update.
    The batch generated will follow temporal order

    Parameters
    ----------
    g : dgl.Heterograph
        graph for batching the temporal edge id as well as generate negative subgraph
    eids : torch.tensor() or numpy array
        eids range which to be batched, it is useful to split training validation test dataset
    graph_sampler : dgl.dataloading.BlockSampler
        temporal neighbor sampler which sample temporal and computationally depend blocks for computation
    device : str
        'cpu' means load dataset on cpu
        'cuda' means load dataset on gpu
    collator : dgl.dataloading.EdgeCollator
        Merge input eid from pytorch dataloader to graph
    Example
    ----------
    Please refers to examples/pytorch/tgn/train.py

    TemporalEdgeDataLoader是一个可迭代的对象，用于生成用于时间嵌入的块以及用于内存更新的正负对图。
    生成的批次将遵循时间顺序
    参数
    ----------
    g:dgl.Heterograph 用于批处理时间边缘id以及生成负子图的图
    eid:torch.tensor（）或numpy数组，要批量处理的eids范围，分割训练验证测试数据集很有用
    graph_sampler:dgl.dataloading.BlockSampler，时间相邻采样器，对时间和计算相关的块进行采样以进行计算
    device：str，“cpu”表示在cpu上加载数据集，“cuda”表示在gpu上加载数据集
    collator：dgl.dataloading.EdgeCollator，将输入eid从pytorch数据加载器合并到图形
    """

    def __init__(self, g, eids, graph_sampler, device='cpu', collator=TemporalEdgeCollator, **kwargs):
        super().__init__(g, eids, graph_sampler, device, **kwargs) # 可变参数**kwargs有关键字参数，传入字典形式的参数
        collator_kwargs = {}
        dataloader_kwargs = {}
        for k, v in kwargs.items():
            if k in self.collator_arglist:
                collator_kwargs[k] = v
            else:
                dataloader_kwargs[k] = v
        # SimpleTemporalEdgeCollator返回的值为input_nodes, pair_graph, neg_pair_graph, [frontier]
        self.collator = collator(g, eids, graph_sampler, **collator_kwargs)

        assert not isinstance(g, dgl.distributed.DistGraph), \
            'EdgeDataLoader does not support DistGraph for now. ' \
            + 'Please use DistDataLoader directly.'
        # 对数据进行batch的划分，数据加载器
        self.dataloader = torch.utils.data.DataLoader(
            self.collator.dataset, collate_fn=self.collator.collate, **dataloader_kwargs)
        self.device = device

        # Precompute the CSR and CSC representations so each subprocess does not
        # duplicate. 预先计算CSR和CSC表示，以便每个子流程不会重复
        if dataloader_kwargs.get('num_workers', 0) > 0:
            g.create_formats_()

    def __iter__(self):
        return iter(self.dataloader)

# ====== Fast Mode ======

# Part of code in reservoir sampling comes from PyG library
# https://github.com/rusty1s/pytorch_geometric/nn/models/tgn.py


class FastTemporalSampler(BlockSampler):
    """Temporal Sampler which implemented with a fast query lookup table. Sample
    temporal and computationally depending subgraph.
    The sampler maintains a lookup table of most current k neighbors of each node
    each time, the sampler need to be updated with new edges from incoming batch to
    update the lookup table.
    Parameters
    ----------
    g : dgl.Heterograph
        graph to be sampled here it which only exist to provide feature and data reference
    k : int
        number of neighbors the lookup table is maintaining
    device : str
        indication str which represent where the data will be stored
        'cpu' store the intermediate data on cpu memory
        'cuda' store the intermediate data on gpu memory
    Example
    ----------
    Please refers to examples/pytorch/tgn/train.py
    """

    def __init__(self, g, k, device='cpu'):
        self.k = k
        self.g = g
        num_nodes = g.num_nodes()
        self.neighbors = torch.empty(
            (num_nodes, k), dtype=torch.long, device=device)
        self.e_id = torch.empty(
            (num_nodes, k), dtype=torch.long, device=device)
        self.__assoc__ = torch.empty(
            num_nodes, dtype=torch.long, device=device)
        self.last_update = torch.zeros(num_nodes, dtype=torch.double)
        self.reset()

    def sample_frontier(self,
                        block_id,
                        g,
                        seed_nodes):
        n_id = seed_nodes
        # Here Assume n_id is the bg nid
        neighbors = self.neighbors[n_id]
        nodes = n_id.view(-1, 1).repeat(1, self.k)
        e_id = self.e_id[n_id]
        mask = e_id >= 0

        neighbors[~mask] = nodes[~mask]
        # Screen out orphan node

        orphans = nodes[~mask].unique()
        nodes = nodes[mask]
        neighbors = neighbors[mask]

        e_id = e_id[mask]
        neighbors = neighbors.flatten()
        nodes = nodes.flatten()
        n_id = torch.cat([nodes, neighbors]).unique()
        self.__assoc__[n_id] = torch.arange(n_id.size(0), device=n_id.device)

        neighbors, nodes = self.__assoc__[neighbors], self.__assoc__[nodes]
        subg = dgl.graph((nodes, neighbors))

        # New node to complement orphans which haven't created
        subg.add_nodes(len(orphans))

        # Copy the seed node feature to subgraph
        subg.edata['timestamp'] = torch.zeros(subg.num_edges()).double()
        subg.edata['timestamp'] = self.g.edata['timestamp'][e_id]

        n_id = torch.cat([n_id, orphans])
        self.last_update = self.last_update.to('cpu')
        subg.ndata['timestamp'] = self.last_update[n_id]
        subg.edata['feats'] = torch.zeros(
            (subg.num_edges(), self.g.edata['feats'].shape[1])).float()
        subg.edata['feats'] = self.g.edata['feats'][e_id]
        subg = dgl.add_self_loop(subg)
        subg.ndata[dgl.NID] = n_id
        return subg

    def sample_blocks(self,
                      g,
                      seed_nodes):
        blocks = []
        frontier = self.sample_frontier(0, g, seed_nodes)
        block = frontier
        blocks.append(block)
        return blocks

    def add_edges(self, src, dst):
        """
        Add incoming batch edge info to the lookup table
        将传入的批次边缘信息添加到查找表
        Parameters
        ----------
        src : torch.Tensor
            src node of incoming batch of it should be consistent with self.g
        dst : torch.Tensor
            src node of incoming batch of it should be consistent with self.g
        """
        neighbors = torch.cat([src, dst], dim=0)
        nodes = torch.cat([dst, src], dim=0)
        e_id = torch.arange(self.cur_e_id, self.cur_e_id + src.size(0),
                            device=src.device).repeat(2)
        self.cur_e_id += src.numel()

        # Convert newly encountered interaction ids so that they point to
        # locations of a "dense" format of shape [num_nodes, size].
        nodes, perm = nodes.sort()
        neighbors, e_id = neighbors[perm], e_id[perm]

        n_id = nodes.unique()
        self.__assoc__[n_id] = torch.arange(n_id.numel(), device=n_id.device)

        dense_id = torch.arange(nodes.size(0), device=nodes.device) % self.k
        dense_id += self.__assoc__[nodes].mul_(self.k)

        dense_e_id = e_id.new_full((n_id.numel() * self.k, ), -1)
        dense_e_id[dense_id] = e_id
        dense_e_id = dense_e_id.view(-1, self.k)

        dense_neighbors = e_id.new_empty(n_id.numel() * self.k)
        dense_neighbors[dense_id] = neighbors
        dense_neighbors = dense_neighbors.view(-1, self.k)

        # Collect new and old interactions...
        e_id = torch.cat([self.e_id[n_id, :self.k], dense_e_id], dim=-1)
        neighbors = torch.cat(
            [self.neighbors[n_id, :self.k], dense_neighbors], dim=-1)

        # And sort them based on `e_id`.
        e_id, perm = e_id.topk(self.k, dim=-1)
        self.e_id[n_id] = e_id
        self.neighbors[n_id] = torch.gather(neighbors, 1, perm)

    def reset(self):
        """
        Clean up the lookup table
        """
        self.cur_e_id = 0
        self.e_id.fill_(-1)

    def attach_last_update(self, last_t):
        """
        Attach current last timestamp a node has been updated
        Parameters:
        ----------
        last_t : torch.Tensor
            last timestamp a node has been updated its size need to be consistent with self.g
        """
        self.last_update = last_t

    def sync(self, sampler):
        """
        Copy the lookup table information from another sampler
        This method is useful run the test dataset with new node,
        when test new node dataset the lookup table's state should
        be restored from the sampler just after validation
        Parameters
        ----------
        sampler : FastTemporalSampler
            The sampler from which current sampler get the lookup table info
        """
        self.cur_e_id = sampler.cur_e_id
        self.neighbors = copy.deepcopy(sampler.neighbors)
        self.e_id = copy.deepcopy(sampler.e_id)
        self.__assoc__ = copy.deepcopy(sampler.__assoc__)


class FastTemporalEdgeCollator(EdgeCollator):
    """ Temporal Edge collator merge the edges specified by eid: items
    Since we cannot keep duplicated nodes on a graph we need to iterate though
    the incoming edges and expand the duplicated node and form a batched block
    graph capture the temporal and computational dependency.
    Parameters
    ----------
    g : DGLGraph
        The graph from which the edges are iterated in minibatches and the subgraphs
        are generated.
    eids : Tensor or dict[etype, Tensor]
        The edge set in graph :attr:`g` to compute outputs.
    graph_sampler : dgl.dataloading.BlockSampler
        The neighborhood sampler.
    g_sampling : DGLGraph, optional
        The graph where neighborhood sampling and message passing is performed.
        Note that this is not necessarily the same as :attr:`g`.
        If None, assume to be the same as :attr:`g`.
    exclude : str, optional
        Whether and how to exclude dependencies related to the sampled edges in the
        minibatch.  Possible values are
        * None, which excludes nothing.
        * ``'reverse_id'``, which excludes the reverse edges of the sampled edges.  The said
          reverse edges have the same edge type as the sampled edges.  Only works
          on edge types whose source node type is the same as its destination node type.
        * ``'reverse_types'``, which excludes the reverse edges of the sampled edges.  The
          said reverse edges have different edge types from the sampled edges.
        If ``g_sampling`` is given, ``exclude`` is ignored and will be always ``None``.
    reverse_eids : Tensor or dict[etype, Tensor], optional
        The mapping from original edge ID to its reverse edge ID.
        Required and only used when ``exclude`` is set to ``reverse_id``.
        For heterogeneous graph this will be a dict of edge type and edge IDs.  Note that
        only the edge types whose source node type is the same as destination node type
        are needed.
    reverse_etypes : dict[etype, etype], optional
        The mapping from the edge type to its reverse edge type.
        Required and only used when ``exclude`` is set to ``reverse_types``.
    negative_sampler : callable, optional
        The negative sampler.  Can be omitted if no negative sampling is needed.
        The negative sampler must be a callable that takes in the following arguments:
        * The original (heterogeneous) graph.
        * The ID array of sampled edges in the minibatch, or the dictionary of edge
          types and ID array of sampled edges in the minibatch if the graph is
          heterogeneous.
        It should return
        * A pair of source and destination node ID arrays as negative samples,
          or a dictionary of edge types and such pairs if the graph is heterogenenous.
        A set of builtin negative samplers are provided in
        :ref:`the negative sampling module <api-dataloading-negative-sampling>`.
    example
    ----------
    Please refers to examples/pytorch/tgn/train.py
    """

    def _collate_with_negative_sampling(self, items):
        items = _prepare_tensor(self.g_sampling, items, 'items', False)
        # Here node id will not change
        pair_graph = self.g.edge_subgraph(items, relabel_nodes=False)
        induced_edges = pair_graph.edata[dgl.EID]

        neg_srcdst_raw = self.negative_sampler(self.g, items)
        neg_srcdst = {self.g.canonical_etypes[0]: neg_srcdst_raw}
        dtype = list(neg_srcdst.values())[0][0].dtype
        neg_edges = {
            etype: neg_srcdst.get(etype, (torch.tensor(
                [], dtype=dtype), torch.tensor([], dtype=dtype)))
            for etype in self.g.canonical_etypes}

        neg_pair_graph = dgl.heterograph(
            neg_edges, {ntype: self.g.number_of_nodes(ntype) for ntype in self.g.ntypes})
        pair_graph, neg_pair_graph = dgl.transforms.compact_graphs(
            [pair_graph, neg_pair_graph])
        # Need to remap id

        pair_graph.ndata[dgl.NID] = self.g.nodes()[pair_graph.ndata[dgl.NID]]
        neg_pair_graph.ndata[dgl.NID] = self.g.nodes()[
            neg_pair_graph.ndata[dgl.NID]]

        pair_graph.edata[dgl.EID] = induced_edges

        seed_nodes = pair_graph.ndata[dgl.NID]
        blocks = self.graph_sampler.sample_blocks(self.g_sampling, seed_nodes)
        blocks[0].ndata['timestamp'] = torch.zeros(
            blocks[0].num_nodes()).double()
        input_nodes = blocks[0].edges()[1]

        # update sampler
        _src = self.g.nodes()[self.g.edges()[0][items]]
        _dst = self.g.nodes()[self.g.edges()[1][items]]
        self.graph_sampler.add_edges(_src, _dst)
        return input_nodes, pair_graph, neg_pair_graph, blocks

    def collator(self, items):
        result = super().collate(items)
        # Copy the feature from parent graph
        _pop_subgraph_storage(result[1], self.g)
        _pop_subgraph_storage(result[2], self.g)
        _pop_storages(result[-1], self.g_sampling)
        return result


# ====== Simple Mode ======

# Part of code comes from paper
# "APAN: Asynchronous Propagation Attention Network for Real-time Temporal Graph Embedding"
# that will be appeared in SIGMOD 21, code repo https://github.com/WangXuhongCN/APAN

class SimpleTemporalSampler(BlockSampler):
    '''
    Simple Temporal Sampler just choose the edges that happen before the current timestamp, to build the subgraph of the corresponding nodes.
    And then the sampler uses the simplest static graph neighborhood sampling methods.
    简单时间采样器只需选择当前时间戳之前发生的边，以构建相应节点的子图。
    然后采样器使用最简单的静态图邻域采样方法。
    Parameters
    ----------
    fanouts : [int, ..., int] int list
        The neighbors sampling strategy
        邻居抽样策略
    '''

    def __init__(self, g, fanouts, return_eids=False):
        super().__init__(len(fanouts), return_eids)

        self.fanouts = fanouts
        self.ts = 0
        self.frontiers = [None for _ in range(len(fanouts))]

    def sample_frontier(self, block_id, g, seed_nodes):
        '''
        Deleting the the edges that happen after the current timestamp, then use a simple topk edge sampling by timestamp.
        删除在当前时间戳之后发生的边，然后使用简单的topk边按时间戳采样。
        '''
        fanout = self.fanouts[block_id]
        # List of neighbors to sample per edge type for each GNN layer, starting from the first layer.
        # 从第一层开始，每个GNN层的每个边缘类型要采样的邻居列表。
        g = dgl.in_subgraph(g, seed_nodes)
        g.remove_edges(torch.where(g.edata['timestamp'] > self.ts)[0])  # Deleting the the edges that happen after the current timestamp

        if fanout is None:  # full neighborhood sampling全邻域抽样
            frontier = g
        else:
            # 选择具有给定节点的k(这里是fanout)个最大（或k个最小）权重的相邻边，并返回诱导子图，即最近的时间戳边缘采样
            frontier = dgl.sampling.select_topk(g, fanout, 'timestamp', seed_nodes)
        self.frontiers[block_id] = frontier  # save frontier
        return frontier


class SimpleTemporalEdgeCollator(EdgeCollator):
    '''
    Temporal Edge collator merge the edges specified by eid: items
    Temporal Edge collator合并eid:items指定的边
    Parameters
    ----------
    g : DGLGraph
        The graph from which the edges are iterated in minibatches and the subgraphs
        are generated.
    eids : Tensor or dict[etype, Tensor]
        The edge set in graph :attr:`g` to compute outputs.
    graph_sampler : dgl.dataloading.BlockSampler
        The neighborhood sampler.
    g_sampling : DGLGraph, optional
        The graph where neighborhood sampling and message passing is performed.
        Note that this is not necessarily the same as :attr:`g`.
        If None, assume to be the same as :attr:`g`.
    exclude : str, optional
        Whether and how to exclude dependencies related to the sampled edges in the
        minibatch.  Possible values are
        * None, which excludes nothing.
        * ``'reverse_id'``, which excludes the reverse edges of the sampled edges.  The said
          reverse edges have the same edge type as the sampled edges.  Only works
          on edge types whose source node type is the same as its destination node type.
        * ``'reverse_types'``, which excludes the reverse edges of the sampled edges.  The
          said reverse edges have different edge types from the sampled edges.
        If ``g_sampling`` is given, ``exclude`` is ignored and will be always ``None``.
    reverse_eids : Tensor or dict[etype, Tensor], optional
        The mapping from original edge ID to its reverse edge ID.
        Required and only used when ``exclude`` is set to ``reverse_id``.
        For heterogeneous graph this will be a dict of edge type and edge IDs.  Note that
        only the edge types whose source node type is the same as destination node type
        are needed.
    reverse_etypes : dict[etype, etype], optional
        The mapping from the edge type to its reverse edge type.
        Required and only used when ``exclude`` is set to ``reverse_types``.
    negative_sampler : callable, optional
        The negative sampler.  Can be omitted if no negative sampling is needed.
        The negative sampler must be a callable that takes in the following arguments:
        * The original (heterogeneous) graph.
        * The ID array of sampled edges in the minibatch, or the dictionary of edge
          types and ID array of sampled edges in the minibatch if the graph is
          heterogeneous.
        It should return
        * A pair of source and destination node ID arrays as negative samples,
          or a dictionary of edge types and such pairs if the graph is heterogenenous.
        A set of builtin negative samplers are provided in
        :ref:`the negative sampling module <api-dataloading-negative-sampling>`.
    '''
    def __init__(self, g, eids, graph_sampler, g_sampling=None, exclude=None,
                reverse_eids=None, reverse_etypes=None, negative_sampler=None):
        super(SimpleTemporalEdgeCollator, self).__init__(g, eids, graph_sampler,
                                                         g_sampling, exclude, reverse_eids, reverse_etypes, negative_sampler)
        self.n_layer = len(self.graph_sampler.fanouts)

    def collate(self,items):
        '''
        items: edge id in graph g.
        We sample iteratively k-times and batch them into one single subgraph.
        我们迭代k次采样，并将它们批处理成一个子图
        '''
        # 仅在当前时间戳之前采样边，将当前时间戳恢复到图形采样器。
        current_ts = self.g.edata['timestamp'][items[0]]     #only sample edges before current timestamp
        self.graph_sampler.ts = current_ts    # restore the current timestamp to the graph sampler.

        # if link prefiction, we use a negative_sampler to generate neg-graph for loss computing.
        # 如果链路预作用，我们使用一个负采样器来生成用于损耗计算的负图。
        if self.negative_sampler is None:
            neg_pair_graph = None
            input_nodes, pair_graph, blocks = self._collate(items)
        else:
            input_nodes, pair_graph, neg_pair_graph, blocks = self._collate_with_negative_sampling(items)

        # we sampling k-hop subgraph and batch them into one graph
        # 我们对k跳子图进行采样，并将其批处理成一个图
        for i in range(self.n_layer-1):
            self.graph_sampler.frontiers[0].add_edges(*self.graph_sampler.frontiers[i+1].edges())
        frontier = self.graph_sampler.frontiers[0]
        # computing node last-update timestamp计算节点上次更新时间戳
        frontier.update_all(fn.copy_e('timestamp','ts'), fn.max('ts','timestamp'))

        return input_nodes, pair_graph, neg_pair_graph, [frontier]
