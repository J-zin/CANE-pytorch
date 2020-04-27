import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', '-d', type=str, default='cora')
parser.add_argument('--gpu', '-g', type=str, default='0')
parser.add_argument('--ratio', '-r', type=float, default=0.95)
parser.add_argument('--rho', '-rh', type=str, default='1.0,0.3,0.3')
args = parser.parse_args()

if __name__ == "__main__":
    os.makedirs("temp", exist_ok=True)
    os.system("python3 prepareData.py --ratio %s --dataset %s" % (args.ratio, args.dataset))
    os.system("CUDA_VISIBLE_DEVICES=%s python3 main.py --dataset %s --rho %s" % (args.gpu, args.dataset, args.rho))
    os.system("python3 auc.py")