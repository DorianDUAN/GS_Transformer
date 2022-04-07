from torch.utils.data import DataLoader
import torch

import numpy as np
import pandas as pd

from models import Transformer
from create_data import create_prediction_input,StockDataset
from trans_train import train

def predict(data,input):
    # fixed hyperparams
    BATCH_SIZE = 10
    Y_OUT = 1 #length of output
    NUM_WORKERS = 0
    LR = 0.001 #learning rate
    TRAIN_SIZE = 450 
    N_EPOCHS = 5
    N_IN = 25 #length of input
    K = 2 # lags 
    SEED = 0

    # model parameters
    dim_input = 1 #number of features
    output_sequence_length = 1 #length of output
    dec_seq_len = 1 #length of output
    dim_val = 64
    dim_attn = 12#12
    n_heads = 8 
    n_encoder_layers = 4
    n_decoder_layers = 2

    MODEL_PATH = 'weights/{e}_{d}_{v}_{n}_{y}_{k}_seed{seed}'.format(e=n_encoder_layers, 
    d=n_decoder_layers, v=dim_val, n=N_IN, y=Y_OUT,k=K, seed=SEED)

    #init network
    net = Transformer(dim_val, dim_attn, dim_input, dec_seq_len, output_sequence_length, n_decoder_layers, n_encoder_layers, n_heads)

    #generat the training set
    X_train, y_train, min_, max_= create_prediction_input(data, N_IN, Y_OUT,TRAIN_SIZE,K)
    train_dataset = StockDataset(X_train, y_train)
    train_loader = DataLoader(dataset=train_dataset,batch_size=BATCH_SIZE)

    train(net, N_EPOCHS, train_loader, LR, MODEL_PATH)
    predict_input = []
    for i in input:
        list = [i]
        predict_input.append(list)

    predict_input = torch.FloatTensor([predict_input for i in range(2)])
    y_pred = net(predict_input)
    result = y_pred.detach().numpy().tolist()
    # undo the Min-Max

    return result[0][0] * (max_ - min_) + min_
