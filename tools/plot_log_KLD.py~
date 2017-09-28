#!/usr/bin/env python
import os, random, sys, argparse, linecache
import matplotlib
import numpy as np, matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Convert Morph database to LMDB')
parser.add_argument('--log', type=str, help='log', required=True)
parser.add_argument('--figure', type=str, help='where to save plot figure', required=False, default='log.eps')
args = parser.parse_args()
MAE = []
iteration = []
LR = []
LR_iter = []

if __name__ == '__main__':
  with open(args.log, 'r') as f:
    for idx,line in enumerate(f):
      line = line.strip()
      if ('Iteration' in line) and ('lr' in line):
        line1 = line.split()
        LR.append(float(line1[-1]))
        LR_iter.append(int(line1[-4][:-1]))
      if ('MAE' in line) and ('=' in line):
        line = line.split()
        next_line = linecache.getline(args.log,idx + 2).strip()
        if ('loss' in next_line) and ('Iteration' in next_line):
          next_line = next_line.split()
          iteration.append(int(next_line[-4][:-1]))
          MAE.append(float(line[-1]))
  plt.plot(np.array(iteration), np.array(MAE), label='MAE(best %f)'%np.min(np.array(MAE)))
  plt.hold(True)
  plt.grid(True)
  plt.plot(np.array(LR_iter), np.array(LR)*100, label='learning rate' + r'$*\frac{1}{100}$')
  plt.legend()
  plt.title("MAE vs Iter")
  plt.xlabel("Iter")
  plt.savefig(args.figure)
