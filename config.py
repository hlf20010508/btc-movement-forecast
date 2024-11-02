import torch
from Informer.utils.tools import dotdict
import pandas as pd
import os

args = dotdict()

args.model = "informer"
args.root_path = "./data/"
args.data_path = "btcusdt_train.csv"
args.features = "MS"
args.target = "Close"
args.detail_freq = "h"
args.freq = args.detail_freq[-1:]
args.checkpoints = "./informer_checkpoints"
args.seq_len = 800
args.label_len = 400
args.pred_len = 200

columns = pd.read_csv(os.path.join(args.root_path, args.data_path), nrows=0).columns

args.enc_in = len(columns) - 1
args.dec_in = args.enc_in
args.c_out = 1
args.d_model = 512
args.n_heads = 8
args.e_layers = 2
args.d_layers = 1
args.s_layers = "3,2,1"
args.d_ff = 2048
args.factor = 5
args.padding = 0
args.distil = True
args.dropout = 0.05
args.attn = "prob"
args.embed = "timeF"
args.activation = "gelu"
args.output_attention = False
args.do_predict = False
args.mix = True
args.cols = None
args.num_workers = 0
args.itr = 2
args.train_epochs = 6
args.batch_size = 32
args.patience = 3
args.learning_rate = 0.0001
args.des = "exp"
args.loss = "mse"
args.lradj = "type1"
args.use_amp = False
args.inverse = False
args.use_gpu = (
    True if torch.cuda.is_available() or torch.backends.mps.is_available() else False
)
args.gpu = 0
args.use_multi_gpu = False
args.devices = "0,1,2,3"

# setting record of experiments
setting = "{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_at{}_fc{}_eb{}_dt{}_mx{}_{}".format(
    args.model,
    args.data,
    args.features,
    args.seq_len,
    args.label_len,
    args.pred_len,
    args.d_model,
    args.n_heads,
    args.e_layers,
    args.d_layers,
    args.d_ff,
    args.attn,
    args.factor,
    args.embed,
    args.distil,
    args.mix,
    args.des,
)
