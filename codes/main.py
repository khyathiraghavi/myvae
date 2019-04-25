import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch.optim as optim
import numpy as np
from torch.autograd import Variable
import argparse

from control_text_gen.dataset import *
from control_text_gen.model import VAE_base

from experiment_runner import ExperimentRunner 


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='my implementation of vae for controlling text generation')

    parser.add_argument('--use_gpu', default=True, action='store_true',
                        help='whether to run in the GPU')
    parser.add_argument('--save', default=False, action='store_true',
                        help='whether to save model or not')
    parser.add_argument('--batch_size', default=32, help='batch size')
    parser.add_argument('--fix_length', default=16, help='fix length for padding sequences')
    parser.add_argument('--emb_dim', default=50, help='z dimension')
    parser.add_argument('--z_dim', default=64, help='z dimension')
    parser.add_argument('--h_dim', default=64, help='hidden dimension')
    parser.add_argument('--dropout', default=0.3, help='dropout')
    parser.add_argument('--drop_words', default=0.3, help='word dropout')
    parser.add_argument('--lr', default=1e-3, help='learning rate')
    parser.add_argument('--lr_decay_steps', default=100000, help='learning rate')
    parser.add_argument('--decay_steps', default=1000000, help='learning rate decay steps')
    parser.add_argument('--epochs', default=20000, help='number of epochs')
    parser.add_argument('--log_interval', default=1, help='logging interval')
    parser.add_argument('--c_dim', default=2, help='c dimension')
    parser.add_argument('--model', default='base_vae', help='model definition')
    parser.add_argument('--freeze_embeddings', default=False, help='freezing embeddings')
    parser.add_argument('--pretrained_embeddings', default=None, help='freezing embeddings')
    
    parser.add_argument('--weight_kl', default=0.01, help='weight for kl loss term')
    parser.add_argument('--kloss_max', default=0.15, help='freezing embeddings')
    parser.add_argument('--kl_anneat_start', default=3000, help='freezing embeddings')

    args = parser.parse_args()

    text_dataset = TextDataset(emb_dim=args.emb_dim, batch_size=args.batch_size, fix_length=args.fix_length)

    if args.model == "base_vae":
        print ("in here")
        args.freeze_embeddings = False
        model = VAE_base(args, text_dataset)
        experiment_runner = ExperimentRunner(text_dataset, model, args)
        experiment_runner.train()


