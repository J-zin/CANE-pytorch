### CANE-pytorch

A Pytorch implementation of [ACL2017 paper: "CANE: Context-Aware Network Embedding for Relation Modeling"](https://www.aclweb.org/anthology/P17-1158/), The original implementation in tensorflow can be found at https://github.com/thunlp/CANE.

#### Run

Run the following command for training CANE:

`python3 run.py --dataset [cora,HepTh,zhihu] --gpu gpu_id --ratio [0.15,0.25,...] --rho rho_value`

For example, you can train like:

`python run.py --dataset zhihu --gpu 0 --ratio 0.55 --rho 1.0,0.3,0.3`

#### Experimental Results

The experimental results of link prediction

|       | 0.15 | 0.25 | 0.35 | 0.45 | 0.55 | 0.65 | 0.75 | 0.85 | 0.95 |
| ----- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| cora  | 73.8 | 78.2 | 82.0 | 83.4 | 88.1 | 89.8 | 91.7 | 92.4 | 98.7 |
| HepTh | 77.5 | 80.2 | 87.2 | 89.9 | 90.2 | 91.9 | 94.9 | 95.8 | 92.8 |
| zhihu |      |      |      |      |      |      |      |      |      |

The experimental results of node classification

|      | 0.15 | 0.25 | 0.35 | 0.45 | 0.55 | 0.65 | 0.75 | 0.85 | 0.95 |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| cora | 45.4 | 50.7 | 59.2 | 54.2 | 64.0 | 66.5 | 68.4 | 67.7 | 70.3 |

> Note: the parameters are not optimized on both two tasks.

#### Dependencies

- pytorch == 1.2.0

