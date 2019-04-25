import torch
from torchtext import data, datasets
from torchtext.vocab import GloVe

class TextDataset(object):
    def __init__(self, emb_dim, batch_size, fix_length):

        self.emb_dim = emb_dim

        self.TEXT = data.Field(init_token='<start>', eos_token='<eos>', lower=True, tokenize='spacy', fix_length=fix_length)
        self.LABEL = data.Field(sequential=False, unk_token=None)
        
        f = lambda sent: len(sent.text) <= 15 and sent.label != 'neutral'

        train, val, test =  datasets.SST.splits(
                                self.TEXT, self.LABEL, fine_grained=False, train_subtrees=False, filter_pred=f) 

        #self.TEXT.build_vocab(train, vectors=Glove('6B', dim=self.emb_dim))
        self.TEXT.build_vocab(train, vectors="glove.6B.50d")
        self.LABEL.build_vocab(train)

        self.vocab_len = len(self.TEXT.vocab.itos)

        self.train_iter, self.val_iter, _ = data.BucketIterator.splits(
                                              (train, val, test), batch_size=batch_size, device='cuda', shuffle=True, repeat=True)


        self.train_iter = iter(self.train_iter)
        self.val_iter = iter(self.val_iter)


    def get_vocab_vectors(self):
        return self.TEXT.vocab.vectors

    def next_batch(self, use_gpu=True, split='train'):
        if split=='train':
            batch_data = next(self.train_iter)
        elif split=='val':
            batch_data = next(self.val_iter)
        if use_gpu:
            return batch_data.text.cuda(), batch_data.label.cuda()
        return batch_data.text, batch_data.label

    def idxs2sent(self, idxs):
        return ' '.join([self.TEXT.vocab.itos[i] for i in idxs])

    def idx2lab(self, idx):
        return self.LABEL.vocab.itos[idx]
        



