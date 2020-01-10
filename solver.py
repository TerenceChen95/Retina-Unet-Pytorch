import torch
from torch.autograd import Variable
import torch.optim as optim
from models.net2 import UNET
import torch.nn as nn
import os
from clr import CyclicLR

class Solver(object):
    def __init__(self, config):
        self.n_classes = config['n_classes']
        self.model = UNET(1, self.n_classes)
        if self.n_classes > 1:
            self.criterion = nn.CrossEntropyLoss()
        else:
            self.criterion = nn.BCEWithLogitsLoss()

        self.lr = config['lr']
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-3, betas=(0.8, 0.9))
        self.device = config['device']
        self.num_epochs = config['num_epochs']
        if config['N_subimgs'] % config['batch_size'] != 0:    
            self.train_step = config['N_subimgs'] // config['batch_size'] + 1
        else:
            self.train_step = config['N_subimgs'] / config['batch_size']
       
        self.model_save_dir = config['save_pth']
        self.best_loss = 10
        self.scheduler = CyclicLR(self.optimizer, step_size= 2*(config['N_subimgs'] % config['batch_size']), mode='triangular2')

        if self.device is not None:
            self.device = torch.device(self.device)
            self.model.to(self.device)

    def restore_best(self):
        model_pth = os.path.join(self.model_save_dir, 'BEST_checkpoint.pth.tar')
        checkpoint = torch.load(model_pth)
        state_dict = checkpoint['model']
        best_loss = checkpoint['loss']
        epoch = checkpoint['epoch']
        return epoch+1, best_loss
    
    def restore_model(self):
        model_pth = os.path.join(self.model_save_dir, 'checkpoint.pth.tar')
        checkpoint = torch.load(model_pth)
        state_dict = checkpoint['model']
        epoch = checkpoint['epoch']
        return epoch+1
    
    def save_checkpoint(self, state, path):
        torch.save(state, os.path.join(path, 'BEST_checkpoint.pth.tar'))

    def update_lr(self, lr):
        for param in self.optimizer.param_groups:
            param['lr'] = lr

    def train(self, prefetcher, resume=True, best=True):
        if best and resume:
            start_epoch, best_loss = self.restore_best()
            self.best_loss = best_loss.to(self.device)
            print('Start from %d, so far the best loss is %.6f' \
                    % (start_epoch, best_loss))
        elif resume:
            start_epoch = self.restore_model()
            print('Start from %d' % (start_epoch))
        else:
            start_epoch = 0
        #not really epoch, consider using step for naming
        for i in range(start_epoch, self.num_epochs):
            epoch_loss = 0
            self.model.train()
            self.scheduler.batch_step()
            for j in range(self.train_step):
                self.optimizer.zero_grad()
                img, label = prefetcher.next()
                img = Variable(img.to(self.device, dtype=torch.float32))
                label = Variable(label.to(self.device, dtype=torch.float32))
                output = self.model(img)
                loss = self.criterion(output, label)
                epoch_loss += loss
                loss.backward()
                self.optimizer.step()
                
                if loss < self.best_loss:
                    state = {}
                    state['loss'] = loss
                    state['model'] = self.model.state_dict()
                    state['epoch'] = i
                    print('loss decrease, saving model...........')
                    self.save_checkpoint(state, self.model_save_dir)
                    self.best_loss = loss
            
            aver_loss = epoch_loss / self.train_step
            print('training %d epoch, average loss is %.6f' % (i, aver_loss))
    






