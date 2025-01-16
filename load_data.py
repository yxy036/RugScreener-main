import glob
import time

#csvx_list = glob.glob('data/subdata/*.csv')
#print('总共发现%s个CSV文件' % len(csvx_list))
#time.sleep(2)
#print('正在处理............')
#for i in csvx_list:
#    fr = open(i, 'r',encoding='utf-8').read()
#    with open('data/token_transfer.csv', 'a+', encoding='utf-8',newline='') as f:
#        f.write(fr)
#    print('写入成功')
#print('写入完毕！')
#print('10秒钟自动关闭程序！')
#time.sleep(10)

import pandas as pd
import numpy as np
import csv

# 将日期时间转换为时间戳
def composetime(datatime):
    timearray = time.strptime(datatime,"%Y-%m-%d %H:%M:%S")
    time1 = time.mktime(timearray)
    timestamp = int(time1)
    return timestamp

# 将文件中的address转换为id
def transfer(filename):
    data = pd.read_csv(filename+'.csv')
    token_address = data['token_address']  # 获取数据中的token_address
    from_address = data['from_address']  # 读取数据中的from_address
    to_address = data['to_address']  # 读取数据中的to_address
    #print(token_address)
    token_id = [] # 用于存储转换后的token_id
    from_id = []  # 用于存储转换后的from_id
    to_id = []  # 用于存储转换后的to_id
    # print(result)
    data_ = pd.read_csv("data/address_to_id_std_label.csv")  # 读取存储投资者的address对应id的文件
    address = data_['address']
    # print(address)
    id = data_['id']
    # print(id)
    dic = dict()
    for j in range(0, len(address)):
        dic[address[j]] = id[j]
    print(dic)
    # 遍历数据中的所有token_address
    for i in range(0, len(token_address)):
        print(i)
        token_id.append(dic.get(token_address[i]))
        from_id.append(dic.get(from_address[i]))
        to_id.append(dic.get(to_address[i]))
    data['token_address'] = token_id # 将对应的id序列赋值给文件中的token_address列
    data['from_address'] = from_id  # 将对应的id序列赋值给文件中的from_address列
    data['to_address'] = to_id  # 将对应的id序列赋值给文件中的to_address列
    data.to_csv(filename + '_new.csv', index=False, encoding='utf-8')

# 处理数据特征,对每个token的所有value进行归一化处理
def feature(filename):
    data = pd.read_csv(filename)
    token_address = data['token_address']
    from_address = data['from_address']
    to_address = data['to_address']
    value = data['value']
    block_timestamp = data['block_timestamp']
    label = data['label']
    address = token_address[0]
    address_token = []
    address_from = []
    address_to = []
    value_int = []
    timestamp = []
    label_result = []
    for i in range(0, len(token_address)):
        if token_address[i] == address:
            address_token.append(token_address[i])
            address_from.append(from_address[i])
            address_to.append(to_address[i])
            value_int.append(int(value[i]))
            timestamp.append(block_timestamp[i])
            label_result.append(label[i])
        else:
            address = token_address[i]
            mean = np.mean(value_int)
            value_max = np.max(value_int)
            value_min = np.min(value_int)
            mu = value_max - value_min
            std = np.std(value_int)
            value_to = []
            # 平均值归一法，（-1,1）
            # for j in range(0, len(value_int)):
            #    x = (value_int[j] - mean) / mu
            #    value_to.append(x)
            # 标准化归一法
            for j in range(0, len(value_int)):
                if std == 0:
                    x = (value_int[j] - mean + 1) / (std + 1)
                else:
                    x = (value_int[j] - mean) / std
                value_to.append(x)
            # max-min归一法
            # for j in range(0, len(value_int)):
            #    x = (value_int[j] - value_min) / mu
            #    value_to.append(x)
            with open('data/transfer_std_label.csv', 'a+', encoding='utf-8', newline='') as f:
                writer = csv.writer(f)
                for k in range(0, len(value_to)):
                    writer.writerow([address_token[k], address_from[k], address_to[k], value_to[k], timestamp[k],label_result[k]])
            address_token = []
            address_from = []
            address_to = []
            value_int = []
            timestamp = []
            label_result = []
            address_token.append(token_address[i])
            address_from.append(from_address[i])
            address_to.append(to_address[i])
            value_int.append(int(value[i]))
            timestamp.append(block_timestamp[i])
            label_result.append(label[i])
    mean = np.mean(value_int)
    value_max = np.max(value_int)
    value_min = np.min(value_int)
    mu = value_max - value_min
    std = np.std(value_int)
    value_to = []
    # 平均值归一法，（-1,1）
    # for j in range(0, len(value_int)):
    #    x = (value_int[j] - mean) / mu
    #    value_to.append(x)
    # 标准化归一法
    for j in range(0, len(value_int)):
        if std == 0:
            x = (value_int[j] - mean + 1) / (std + 1)
        else:
            x = (value_int[j] - mean) / std
        value_to.append(x)
    # max-min归一法
    # for j in range(0, len(value_int)):
    #    x = (value_int[j] - value_min) / mu
    #    value_to.append(x)
    with open('data/transfer_std_label.csv', 'a+', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        for k in range(0, len(value_to)):
            writer.writerow([address_token[k], address_from[k], address_to[k], value_to[k], timestamp[k],label_result[k]])

def load_edge(filename):
    data = pd.read_csv(filename)
    token_id = data['token_address']
    from_id = data['from_address']
    to_id = data['to_address']
    value = data['value']
    block_time = data['block_timestamp']
    label_ = data['label']
    src = []  # 有向边的起点
    dst = []  # 有向边的终点
    label = []  # 类别标签
    feature = []  # 特征
    timestamp = []  # 时间戳
    # 将每一条交易信息构造为两条有向边
    for i in range(0, len(token_id)):
        src.append(from_id[i])
        dst.append(token_id[i])
        src.append(token_id[i])
        dst.append(to_id[i])
        # 这里简单先用随机数随便赋值，获得真实数据后修改
        #r = random.randint(0, 1)  # 获取交易信息里的原始标签变为两条边的类别
        label.append(label_[i])
        label.append(label_[i])
        feature.append(value[i])
        feature.append(value[i])
        timestamp.append(block_time[i])
        timestamp.append(block_time[i])
    with open('data/data_std_label.csv','w',encoding='utf-8',newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['src','dst','timestamp','label','feature'])
        for j in range(0, len(src)):
            writer.writerow([src[j],dst[j],timestamp[j],label[j],feature[j]])

if __name__=='__main__':
    #data_yuan = pd.read_csv('data/token_transfer_label_yuan.csv')
    #x = data_yuan['block_timestamp']  # 获取数据中的时间
    ## print(token_address)
    #times = []  # 用于存储转换后的时间戳
    #for i in range(0, len(x)):
    #    datatime = x[i]
    #    timestamp = composetime(datatime)  # 将文件中的日期时间转换为时间戳表示
    #    times.append(timestamp)
    #data_yuan['block_timestamp'] = times
    #data_yuan.to_csv('data/token_transfer_label.csv', index=False, encoding='utf-8')
    with open('data/transfer_std_label.csv', 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['token_address', 'from_address','to_address','value', 'block_timestamp','label'])
    feature("data/token_transfer_label.csv")
    data = pd.read_csv('data/transfer_std_label.csv')
    token_address = data['token_address']
    from_address = data['from_address']
    to_address = data['to_address']
    address = []
    for i in range(0, len(to_address)):
        address.append(token_address[i])
        address.append(from_address[i])
        address.append(to_address[i])
    new_li = list(set(address))
    print(new_li)
    with open('data/address_to_id_std_label.csv', 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['address', 'id'])
        for i in range(0, len(new_li)):
            writer.writerow([new_li[i], i])
    transfer("data/transfer_std_label")
    load_edge("data/transfer_std_label_new.csv")
