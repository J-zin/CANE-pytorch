from math import pow

import config

def InitNegTable(edges):
    a_list, b_list = zip(*edges)
    a_list = list(a_list)
    b_list = list(b_list)
    NEG_SAMPLE_POWER = config.NEG_SAMPLE_POWER
    node = a_list
    node.extend(b_list)
    
    node_degree = {}
    for i in node:
        if i in node_degree:
            node_degree[i] += 1
        else:
            node_degree[i] = 1
    sum_degree = 0
    for i in node_degree.values():
        sum_degree += pow(i, 0.75)

    por = 0
    cur_sum = 0
    vid = -1
    neg_table = []
    degree_list = list(node_degree.values())
    node_id = list(node_degree.keys())
    for i in range(config.neg_table_size):
        if ((i + 1) / float(config.neg_table_size)) > por:
            cur_sum += pow(degree_list[vid + 1], NEG_SAMPLE_POWER)
            por = cur_sum / sum_degree
            vid += 1
        neg_table.append(node_id[vid])
        
    return neg_table