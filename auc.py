import os
import random
import numpy as np
from sklearn.linear_model import LogisticRegression

# link prediction
def link_prediction():
    node2vec = {}
    f = open('temp/embed.txt', 'rb')
    for i, j in enumerate(f):
        if j.decode() != '\n':
            node2vec[i] = list(map(float, j.strip().decode().split(' ')))
    f1 = open(os.path.join('temp/test_graph.txt'), 'rb')
    edges = [list(map(int, i.strip().decode().split('\t'))) for i in f1]
    nodes = list(set([i for j in edges for i in j]))
    a = 0
    b = 0
    for i, j in edges:
        if i in node2vec.keys() and j in node2vec.keys():
            dot1 = np.dot(node2vec[i], node2vec[j])
            random_node = random.sample(nodes, 1)[0]
            while random_node == j or random_node not in node2vec.keys():
                random_node = random.sample(nodes, 1)[0]
            dot2 = np.dot(node2vec[i], node2vec[random_node])
            if dot1 > dot2:
                a += 1
            elif dot1 == dot2:
                a += 0.5
            b += 1
    print("Link Prediction Auc value:", float(a) / b)

def node_classify():
    node2vec = {}
    # dataset_name = "cora"
    f = open('temp/embed.txt', 'rb')
    for i, j in enumerate(f):
        if j.decode() != '\n':
            node2vec[i] = list(map(float, j.strip().decode().split(' ')))

    node2label = {}
    f = open('datasets/cora/group.txt', 'rb')
    for i, j in enumerate(f):
        if j.decode() != '\n':
            node2label[i] = int(j.strip().decode())
    
    embs = []
    labels = []
    for key, val in node2vec.items():
        embs.append(val)
        labels.append(node2label[key])
    
    train_num = int(len(embs) * 0.9)
    train_X, train_Y = embs[:train_num], labels[:train_num]
    test_X, test_Y = embs[train_num:], labels[train_num:]
    clf = LogisticRegression(random_state=0, solver='lbfgs', multi_class='auto', penalty='l2').fit(train_X, train_Y)
    print("Node Classification Auc value:", clf.score(test_X, test_Y))

        

if __name__ == "__main__":
    link_prediction()

    # node_classify()