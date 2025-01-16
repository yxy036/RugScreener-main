import time
import csv
import pandas as pd
import os
import ssl
from six.moves import urllib
import numpy as np
import networkx as nx
import torch
import dgl

# 将日期时间转换为时间戳
def composetime(datatime):
    timearray = time.strptime(datatime,"%Y-%m-%d %H:%M:%S")
    time1 = time.mktime(timearray)
    timestamp = int(time1)
    return timestamp

# 将文件中的日期时间转换为时间戳表示
def transfer_time(filename):
    data = pd.read_csv(filename+'.csv')
    x = data['block_timestamp']
    # print(x)
    result = []
    for i in range(0,len(x)):
        datatime = x[i]
        timestamp = composetime(datatime)
        result.append(timestamp)
    # print(result)
    data['block_timestamp'] = result
    data.to_csv(filename+'_new.csv',index=False,encoding='utf-8')

# 将文件中的token_address转换为token_id
def address_to_id(filename):
    data = pd.read_csv(filename)
    token_address = data['token_address'] # 获取数据中的token_address
    print(token_address)
    token_id = []
    # 遍历数据中的所有token_address
    for i in range(0, len(token_address)):
        flag = 1
        data_token = pd.read_csv("address_to_id_token.csv")  # 读取存储token的address对应id的文件
        address_token = data_token['address']
        print(address_token)
        id_token = data_token['id']
        print(id_token)
        # 若存储address和id对应关系的文件中已有该token地址，则直接得到该地址对应的id
        for j in range(0, len(address_token)):
            if token_address[i] == address_token[j]:
                flag = 0
                token_id.append(id_token[j])
        # 若文件中没有此token地址，则添加该地址，并赋值对应的id
        if flag == 1:
            with open('address_to_id_token.csv', 'a+', encoding='utf-8', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([token_address[i],len(address_token)])
            token_id.append(len(address_token))
    data['token_address'] = token_id # 将对应的id序列赋值给文件中的token_address列
    data.to_csv(filename, index=False, encoding='utf-8')

# 将文件中的from_address和to_address转换为from_id和to_id，单独作为user_id
def add_to_id(filename):
    data = pd.read_csv(filename)
    from_address = data['from_address'] # 读取数据中的from_address
    to_address = data['to_address'] # 读取数据中的to_address
    from_id = []
    to_id = []
    # 遍历数据中的所有token_address
    for i in range(0, len(from_address)):
        flag_from = 1
        flag_to = 1
        data_user = pd.read_csv("address_to_id_user.csv")  # 读取存储投资者的address对应id的文件
        address_user = data_user['address']
        print(address_user)
        id_user = data_user['id']
        print(id_user)
        # 若存储address和id对应关系的文件中已有该地址，则直接得到该地址对应的id
        for j in range(0, len(address_user)):
            if from_address[i] == address_user[j]:
                flag_from = 0
                from_id.append(id_user[j])
            if to_address[i] == address_user[j]:
                flag_to = 0
                to_id.append(id_user[j])
        # 若文件中没有此token地址，则添加该地址，并赋值对应的id
        if flag_from == 1 and flag_to == 1:
            with open('address_to_id_user.csv', 'a+', encoding='utf-8', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([from_address[i], len(address_user)])
                writer.writerow([to_address[i], len(address_user) + 1])
            from_id.append(len(address_user))
            to_id.append(len(address_user) + 1)
        if flag_from == 1 and flag_to == 0:
            with open('address_to_id_user.csv', 'a+', encoding='utf-8', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([from_address[i], len(address_user)])
            from_id.append(len(address_user))
        if flag_from == 0 and flag_to == 1:
            with open('address_to_id_user.csv', 'a+', encoding='utf-8', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([to_address[i], len(address_user)])
            to_id.append(len(address_user))
    data['from_address'] = from_id  # 将对应的id序列赋值给文件中的from_address列
    data['to_address'] = to_id  # 将对应的id序列赋值给文件中的to_address列
    data.to_csv(filename, index=False, encoding='utf-8')

# 将文件中所有的address转换为id，综合排列
def a_to_i(filename):
    data = pd.read_csv(filename)
    token_address = data['token_address']  # 获取数据中的token_address
    from_address = data['from_address']  # 读取数据中的from_address
    to_address = data['to_address']  # 读取数据中的to_address
    token_id = []
    from_id = []
    to_id = []
    # 遍历数据中的所有token_address
    for i in range(0, len(token_address)):
        flag_token = 1
        flag_from = 1
        flag_to = 1
        data_ = pd.read_csv("address_to_id.csv")  # 读取存储投资者的address对应id的文件
        address = data_['address']
        print(address)
        id = data_['id']
        print(id)
        # 若存储address和id对应关系的文件中已有该地址，则直接得到该地址对应的id
        for j in range(0, len(address)):
            if token_address[i] == address[j]:
                flag_token = 0
                token_id.append(id[j])
            if from_address[i] == address[j]:
                flag_from = 0
                from_id.append(id[j])
            if to_address[i] == address[j]:
                flag_to = 0
                to_id.append(id[j])
        # 若文件中没有此token地址，则添加该地址，并赋值对应的id
        if flag_token == 1 and flag_from == 1 and flag_to == 1:
            with open('address_to_id.csv', 'a+', encoding='utf-8', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([token_address[i],len(address)])
                writer.writerow([from_address[i], len(address) + 1])
                writer.writerow([to_address[i], len(address) + 2])
            token_id.append(len(address))
            from_id.append(len(address) + 1)
            to_id.append(len(address) + 2)
        if flag_token == 1 and flag_from == 1 and flag_to == 0:
            with open('address_to_id.csv', 'a+', encoding='utf-8', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([token_address[i], len(address)])
                writer.writerow([from_address[i], len(address) + 1])
            token_id.append(len(address))
            from_id.append(len(address) + 1)
        if flag_token == 1 and flag_from == 0 and flag_to == 1:
            with open('address_to_id.csv', 'a+', encoding='utf-8', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([token_address[i], len(address)])
                writer.writerow([to_address[i], len(address) + 1])
            token_id.append(len(address))
            to_id.append(len(address) + 1)
        if flag_token == 1 and flag_from == 0 and flag_to == 0:
            with open('address_to_id.csv', 'a+', encoding='utf-8', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([token_address[i], len(address)])
            token_id.append(len(address))
        if flag_token == 0 and flag_from == 1 and flag_to == 1:
            with open('address_to_id.csv', 'a+', encoding='utf-8', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([from_address[i], len(address)])
                writer.writerow([to_address[i], len(address) + 1])
            from_id.append(len(address))
            to_id.append(len(address) + 1)
        if flag_token == 0 and flag_from == 1 and flag_to == 0:
            with open('address_to_id.csv', 'a+', encoding='utf-8', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([from_address[i], len(address)])
            from_id.append(len(address))
        if flag_token == 0 and flag_from == 0 and flag_to == 1:
            with open('address_to_id.csv', 'a+', encoding='utf-8', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([to_address[i], len(address)])
            to_id.append(len(address))
    data['token_address'] = token_id  # 将对应的id序列赋值给文件中的token_address列
    data['from_address'] = from_id  # 将对应的id序列赋值给文件中的from_address列
    data['to_address'] = to_id  # 将对应的id序列赋值给文件中的to_address列
    data.to_csv(filename, index=False, encoding='utf-8')

# === Below data preprocessing code are based on
# https://github.com/twitter-research/tgn

# Preprocess the raw data split each features
# 预处理原始数据分割每个特征

def preprocess(data_name):
    u_list, i_list, ts_list, label_list = [], [], [], []
    feat_l = []
    idx_list = []

    with open(data_name) as f:
        s = next(f)
        for idx, line in enumerate(f):
            e = line.strip().split(',')
            u = int(e[0])
            i = int(e[1])

            ts = float(e[2])
            label = float(e[3])  # int(e[3])

            feat = np.array([float(x) for x in e[4:]])

            u_list.append(u)
            i_list.append(i)
            ts_list.append(ts)
            label_list.append(label)
            idx_list.append(idx)

            feat_l.append(feat)
    return pd.DataFrame({'u': u_list,
                         'i': i_list,
                         'ts': ts_list,
                         'label': label_list,
                         'idx': idx_list}), np.array(feat_l)

# Re index nodes for DGL convience为DGL便利性重新索引节点
def reindex(df):
    new_df = df.copy()
    # assert (max(df.u.max(),df.i.max()) - min(df.u.min(),df.i.min()) + 1 == len(df.u) + len(df.i))
    # 缩小时间戳
    upper_ts = df.ts.min()
    new_df.ts = df.ts - upper_ts
    new_df.u += 1
    new_df.i += 1
    new_df.idx += 1
    return new_df

# Save edge list, features in different file for data easy process data
# 将边缘列表、特征保存在不同文件中，以便数据轻松处理数据
def run(data_name, bipartite=True):
    PATH = './data/{}.csv'.format(data_name)
    OUT_DF = './data/ml_{}.csv'.format(data_name)
    OUT_FEAT = './data/ml_{}.npy'.format(data_name)
    OUT_NODE_FEAT = './data/ml_{}_node.npy'.format(data_name)

    df, feat = preprocess(PATH)
    new_df = reindex(df)

    empty = np.zeros(feat.shape[1])[np.newaxis, :] # 在np.newaxis所在的位置增加一个一维，即大小变为（1，），在原基础外加一个[]
    feat = np.vstack([empty, feat]) # 返回竖直堆叠后的数组

    max_idx = max(new_df.u.max(), new_df.i.max())
    rand_feat = np.zeros((max_idx + 1, 1))  # 后面的1对应的是特征维度

    new_df.to_csv(OUT_DF)
    np.save(OUT_FEAT, feat)
    np.save(OUT_NODE_FEAT, rand_feat)

# === code from twitter-research-tgn end ===

def TemporalDataset(dataset):
    if not os.path.exists('./data/{}.bin'.format(dataset)):
        if not os.path.exists('./data/{}.csv'.format(dataset)):
            if not os.path.exists('./data'):
                os.mkdir('./data')

            url = 'https://snap.stanford.edu/jodie/{}.csv'.format(dataset)
            print("Start Downloading File....")
            context = ssl._create_unverified_context()
            data = urllib.request.urlopen(url, context=context)
            with open("./data/{}.csv".format(dataset), "wb") as handle:
                handle.write(data.read())

        print("Start Process Data ...")
        run(dataset)
        raw_connection = pd.read_csv('./data/ml_{}.csv'.format(dataset))
        raw_feature = np.load('./data/ml_{}.npy'.format(dataset))
        token_data = pd.read_csv('./data/transfer_std_label_new.csv')
        token = token_data['token_address'].to_numpy()
        lab = token_data['label']
        token_to_label = dict()
        for i in range(0,len(token)):
            token_to_label[token[i]] = lab[i]
        # -1 for re-index the node
        src = raw_connection['u'].to_numpy()-1
        dst = raw_connection['i'].to_numpy()-1
        node_list = list(set(np.append(src, dst)))
        token_list = list(set(token))
        node_to_label = dict.fromkeys(node_list, '2')
        for i in range(0, len(token_list)):
            node_to_label[token_list[i]] = token_to_label[token_list[i]]
        ns = list(node_to_label.keys())
        ls = list(node_to_label.values())
        nodes1 = []
        labels1 = []
        for i in range(0, len(ns)):
            nodes1.append(int(ns[i]))
            labels1.append(int(ls[i]))
        nodes = np.array(nodes1)
        labels = np.array(labels1)
        # Create directed graph
        g = dgl.graph((src, dst))
        in_degree_list = []
        out_degree_list = []
        for v in range(g.num_nodes()):
            in_degree_list.append(g.in_degrees(v))
            out_degree_list.append(g.out_degrees(v))
        in_degree = np.array(in_degree_list)
        out_degree = np.array(out_degree_list)
        nx_g = dgl.to_networkx(g)
        G = nx.DiGraph(nx_g)
        c_degree_dict = nx.degree_centrality(nx_g)
        pagerank_dict = nx.pagerank(G, max_iter=1000)
        c_closeness_dict = nx.closeness_centrality(nx_g)
        c_betweenness_dict = nx.betweenness_centrality(nx_g)
        c_eigenvector_dict = nx.eigenvector_centrality(G,max_iter=1000)
        c_degree_list = list(c_degree_dict.values())
        pagerank_list = list(pagerank_dict.values())
        c_closeness_list = list(c_closeness_dict.values())
        c_betweenness_list = list(c_betweenness_dict.values())
        c_eigenvector_list = list(c_eigenvector_dict.values())
        c_degree = np.array(c_degree_list)
        pagerank = np.array(pagerank_list)
        c_closeness = np.array(c_closeness_list)
        c_betweenness = np.array(c_betweenness_list)
        c_eigenvector = np.array(c_eigenvector_list)
        # print(c_degree_list)
        # print(c_closeness_list)
        # print(g)
        g.ndata['in_degree'] = torch.from_numpy(in_degree)
        g.ndata['out_degree'] = torch.from_numpy(out_degree)
        g.ndata['c_degree'] = torch.from_numpy(c_degree)
        g.ndata['pagerank'] = torch.from_numpy(pagerank)
        g.ndata['c_closeness'] = torch.from_numpy(c_closeness)
        g.ndata['c_betweenness'] = torch.from_numpy(c_betweenness)
        g.ndata['c_eigenvector'] = torch.from_numpy(c_eigenvector)
        g.edata['timestamp'] = torch.from_numpy(
            raw_connection['ts'].to_numpy())
        #g.edata['label'] = torch.from_numpy(raw_connection['label'].to_numpy())
        g.nodes[nodes].data['label'] = torch.from_numpy(labels)
        g.edata['feats'] = torch.from_numpy(raw_feature[1:, :]).float()
        dgl.save_graphs('./data/{}.bin'.format(dataset), [g])
    else:
        print("Data is exist directly loaded.")
        gs, _ = dgl.load_graphs('./data/{}.bin'.format(dataset))
        g = gs[0]
    return g

#if __name__=='__main__':
#    transfer_time("Token_Transfers")
#    with open('address_to_id.csv','w',encoding='utf-8',newline='') as f:
#        writer = csv.writer(f)
#        writer.writerow(['address','id'])
    #address_to_id("Token_Transfers_new.csv")
    #add_to_id("Token_Transfers_new.csv")
#    a_to_i("Token_Transfers_new.csv")