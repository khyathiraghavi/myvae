import dill as pickle
import torch
import torch.optim as optim

from control_text_gen.dataset import *
from control_text_gen.model import VAE_base

class ExperimentRunner(object):
    def __init__(self, text_dataset, model, args):
        self.text_dataset = text_dataset
        self.model = model
        self.args = args
        
        if self.args.use_gpu:
            self.model.cuda()

        self.optimizer = optim.Adam(model.vae_params, lr=self.args.lr)

    def train(self):
        for epoch in range(10):
            self.model.train()
            inputs, labels = self.text_dataset.next_batch(self.args.use_gpu, split='train')
            
            loss_recon, loss_kl = self.model.forward(inputs)

        
                
            

        







