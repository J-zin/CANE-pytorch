import torch
import random
import argparse
import numpy as np
import torch.optim as optim
from torch.autograd import Variable

import config
from model.CANE import CANE
from utils.DataSet import dataSet

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"

def option_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-d', type=str, default='cora')
    parser.add_argument('--rho', '-r', type=str, default='1.0,0.3,0.3')
    args = parser.parse_args()
    args.cuda = True if torch.cuda.is_available() else False

    return args

def load_data(args):
    dataset_name = args.dataset
    graph_path = os.path.join('temp/graph.txt')
    text_path = os.path.join(".", "datasets", dataset_name, 'data.txt')
    data = dataSet(text_path, graph_path)
    return data

def main(args):
    torch.manual_seed(1)
    LongTensor = torch.cuda.LongTensor if args.cuda else torch.LongTensor
    data = load_data(args)
    
    # model
    cane = CANE(data.num_vocab, data.num_nodes, args.rho)
    if args.cuda:
        cane = cane.cuda()

    # optimizer
    optimizer = optim.Adam(cane.parameters(), lr = config.lr)

    # training
    print('start training.......')
    for epoch in range(config.num_epoch):
        loss_epoch = 0
        batches = data.generate_batches()
        num_batch = len(batches)
        for i in range(num_batch):
            batch = batches[i]
            node1, node2, node3 = zip(*batch)
            node1, node2, node3 = np.array(node1), np.array(node2), np.array(node3)
            text1, text2, text3 = data.text[node1], data.text[node2], data.text[node3]

            node1, node2, node3 = Variable(LongTensor(node1)), Variable(LongTensor(node2)), Variable(LongTensor(node3))
            text1, text2, text3 = Variable(LongTensor(text1)), Variable(LongTensor(text2)), Variable(LongTensor(text3))
            
            loss_batch = cane(node1, node2, node3, text1, text2, text3)
            optimizer.zero_grad()
            loss_batch.backward()
            optimizer.step()

            loss_epoch += loss_batch.item()
        print('epoch: ', epoch + 1, ' loss: ', loss_epoch)
        
    # get embedding
    file = open('temp/embed.txt', 'wb')
    batches = data.generate_batches()
    num_batch = len(batches)
    embed = [[] for _ in range(data.num_nodes)]
    for i in range(num_batch):
        batch = batches[i]
        node1, node2, node3 = zip(*batch)
        node1, node2, node3 = np.array(node1), np.array(node2), np.array(node3)
        text1, text2, text3 = data.text[node1], data.text[node2], data.text[node3]

        node1, node2, node3 = Variable(LongTensor(node1)), Variable(LongTensor(node2)), Variable(LongTensor(node3))
        text1, text2, text3 = Variable(LongTensor(text1)), Variable(LongTensor(text2)), Variable(LongTensor(text3))

        convA, convB, TA, TB = cane.get_emb(node1, node2, node3, text1, text2, text3)
        for i in range(config.batch_size):
            em = list(TA[i].data.cpu())
            embed[node1[i].item()].append(em)
            em = list(TB[i].data.cpu())
            embed[node2[i].item()].append(em)
    for i in range(data.num_nodes):
        if embed[i]:
            # print embed[i]
            tmp = np.sum(embed[i], axis=0) / len(embed[i])
            file.write((' '.join(map(str, tmp)) + '\n').encode())
        else:
            file.write('\n'.encode())


if __name__ == "__main__":
    args = option_parse()
    main(args)