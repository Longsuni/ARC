import torch
import numpy as np
from data_utils import utils
from data_utils.parse_args import args
from models.ARC import ARC
import itertools
import os
import warnings
import random
from torch.optim import lr_scheduler

features, mob_adj, poi_sim, land_sim = utils.load_data()


def main():
    warnings.filterwarnings("ignore")
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str('0')
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda:0' if use_cuda else 'cpu')

    seed = args.random_seed
    np.random.seed(seed)
    random.seed(seed + 1)
    torch.manual_seed(seed + 2)
    torch.cuda.manual_seed(seed + 3)
    torch.backends.cudnn.deterministic = True

    model = ARC(args)

    optimizer = torch.optim.Adam(
        itertools.chain(model.encoder_z1.parameters(),
                        model.encoder_z2.parameters(),
                        model.encoder_z3.parameters(),
                        model.autoencoder_a.parameters(),
                        model.autoencoder_b.parameters(),
                        model.autoencoder_c.parameters()
                        ), lr=args.learning_rate, weight_decay=args.weight_decay)

    scheduler = lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.3, verbose=False)

    model.to_device(device)

    data = features
    model.train_model(data, optimizer, task=args.task, city=args.city, sc=scheduler,epochs = args.epochs)

if __name__ == '__main__':
    main()
