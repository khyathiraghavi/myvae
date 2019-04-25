import dill
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
from itertools import chain

class VAE_base(nn.Module):
    def __init__(self, args, text_dataset):
        super(VAE_base, self).__init__()
       
        self.args = args
        self.unk_idx = 0
        self.pad_idx = 1
        self.start_idx = 2
        self.eos_idx = 3
        self.max_len = args.fix_length-1

        self.vocab_len = text_dataset.vocab_len
        self.h_dim = args.h_dim
        self.z_dim = args.z_dim
        self.c_dim = args.c_dim
        self.dropout = args.dropout
        self.use_gpu = args.use_gpu
        self.pretrained_emb = args.pretrained_embeddings


        # word embedding layer
        if self.pretrained_emb is None:
            self.emb_dim = self.h_dim
        else:
            self.emb_dim = self.pretrained_emb.size(1)
        self.word_emb = nn.Embedding(self.vocab_len, self.emb_dim, padding_idx = self.pad_idx)

        if args.freeze_embeddings:
           self.word_emb.weight.requires_grad = False
          
        # encoder is GRU with FC
        self.encoder = nn.GRU(self.emb_dim, self.h_dim)
        self.q_mu = nn.Linear(self.h_dim, self.z_dim) 
        self.q_logvar = nn.Linear(self.h_dim, self.z_dim)

        # decoder is GRU with z and c appended to its inputs
        zc_len = self.z_dim + self.c_dim
        self.decoder = nn.GRU(self.emb_dim + zc_len, zc_len, dropout = self.dropout)
        self.p_fc = nn.Linear(zc_len, self.vocab_len)

        # discriminator model
        self.conv3 = nn.Conv2d(1, 100, (3, self.emb_dim))
        self.conv4 = nn.Conv2d(1, 100, (4, self.emb_dim))
        self.conv5 = nn.Conv2d(1, 100, (5, self.emb_dim))

        self.discriminator_fc = nn.Sequential(
                                 nn.Dropout(0.5),
                                 nn.Linear(300, 2) )

        self.discriminator = nn.ModuleList([self.conv3, self.conv4, self.conv5, self.discriminator_fc])


        # grouping parameters
        self.encoder_params = chain(
                               self.encoder.parameters(),
                               self.q_mu.parameters(),
                               self.q_logvar.parameters() )
        self.decoder_params = chain(
                               self.decoder.parameters(),
                               self.p_fc.parameters() )

        self.vae_params = chain(
                              self.word_emb.parameters(), self.encoder_params, self.decoder_params )

        #self.discriminator_params = filter(lambda p: p.requires_grad, self.discriminator.parameters())


        if self.use_gpu:
            self.cuda()

    def forward(self, sent, use_cp=True): #sent is 16Xbs
        #batch_size = self.args.batch_size
        batch_size = sent.size(1)
        pad_words = Variable(torch.LongTensor([self.pad_idx]).repeat(1, batch_size)) 
        pad_words = pad_words.cuda() if self.use_gpu else pad_words
        
        inp_enc = sent
        inp_dec = sent
        tgt_dec = torch.cat([sent[1:], pad_words], dim=0) 
      









