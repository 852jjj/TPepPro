import os
import torch
from sklearn import metrics
import warnings
warnings.filterwarnings("ignore")
torch.set_default_tensor_type('torch.cuda.FloatTensor')
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
from my_train_and_validation import *

def main():
    
    if len(sys.argv) > 1:
        datasetname, rst_file, pkl_path,batchsize = sys.argv[1:]
        batchsize = int(batchsize)
    else: 
        datasetname = 'yeast'
        rst_file = './results/xxx.tsv'
        pkl_path = './model_pkl/GAT'
        batchsize = 32
    #losses,accs,testResults = train(trainArgs)

    train(trainArgs)

if __name__ == "__main__":
    main()