import torch
from Informer.utils.tools import dotdict


args = dotdict()

args.model = "informer"  # model of experiment, options: [informer, informerstack, informerlight(TBD)]

args.data = "custom"  # data
args.root_path = "./data/"  # root path of data file
args.data_path = "btcusdt.csv"  # data file
args.features = "MS"  # forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate
args.target = "Close"  # target feature in S or MS task
args.freq = "4h"  # freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h
args.checkpoints = "./informer_checkpoints"  # location of model checkpoints

args.seq_len = 96  # input sequence length of Informer encoder
args.label_len = 48  # start token length of Informer decoder
args.pred_len = 24  # prediction sequence length
# Informer decoder input: concat[start token series(label_len), zero padding series(pred_len)]

args.enc_in = 5  # encoder input size
args.dec_in = 5  # decoder input size
args.c_out = 1  # output size
args.factor = 5  # probsparse attn factor
args.d_model = 512  # dimension of model
args.n_heads = 8  # num of heads
args.e_layers = 2  # num of encoder layers
args.d_layers = 1  # num of decoder layers
args.d_ff = 2048  # dimension of fcn in model
args.dropout = 0.05  # dropout
args.attn = "prob"  # attention used in encoder, options:[prob, full]
args.embed = "timeF"  # time features encoding, options:[timeF, fixed, learned]
args.activation = "gelu"  # activation
args.distil = True  # whether to use distilling in encoder
args.output_attention = False  # whether to output attention in ecoder
args.mix = True
args.padding = 0

args.batch_size = 32
args.learning_rate = 0.0001
args.loss = "mse"
args.lradj = "type2"
args.use_amp = False  # whether to use automatic mixed precision training

args.num_workers = 0
args.train_epochs = 6
args.patience = 3
args.des = "exp"

args.use_gpu = True if torch.cuda.is_available() or torch.mps.is_available() else False
args.gpu = 0

args.use_multi_gpu = False
args.devices = "0,1,2,3"

args.detail_freq = args.freq
args.freq = args.freq[-1:]

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
