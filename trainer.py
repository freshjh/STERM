import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.base_model import BaseModel, init_weights
import sys
from models import get_model

class Trainer(BaseModel):
    def name(self):
        return 'Trainer'

    def __init__(self, opt):
        super(Trainer, self).__init__(opt)
        self.opt = opt  
        self.model = get_model(opt.arch)
        torch.nn.init.normal_(self.model.fc.weight.data, 0.0, opt.init_gain)
        torch.nn.init.zeros_(self.model.kdporj.projector.fc2.weight.data)
        torch.nn.init.zeros_(self.model.logitporj.projector.fc2.weight.data)
 
        para_name = []
        # unfreeze the forgery encoder, detector and projector
        if opt.fix_backbone:
            params = []
            for name, p in self.model.named_parameters():
                para_name.append(name)
                # print(name)
                if  name=="fc.weight" or name=="fc.bias": 
                    params.append(p) 
                elif 'forgery_encoder' in name:
                     params.append(p) 
                elif 'forgery_proj' in name:
                     params.append(p) 
                elif 'authenticity_proj' in name:
                     params.append(p) 
                else:
                    p.requires_grad = False
        else:
            print("Your backbone is not fixed. Are you sure you want to proceed? If this is a mistake, enable the --fix_backbone command during training and rerun")
            import time 
            time.sleep(3)
            params = self.model.parameters()


        if opt.optim == 'adam':
            self.optimizer = torch.optim.AdamW(params, lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=opt.weight_decay)
        elif opt.optim == 'sgd':
            self.optimizer = torch.optim.SGD(params, lr=opt.lr, momentum=0.0, weight_decay=opt.weight_decay)
        else:
            raise ValueError("optim should be [adam, sgd]")

        self.loss_fn = nn.BCEWithLogitsLoss()

        # alignment loss
        self.alignment = nn.L1Loss()
        self.model.to(opt.gpu_ids[0])


       

    def set_input(self, input):
        self.input = input[0].to(self.device)
        self.label = input[1].to(self.device).float()


    def forward(self):
        self.new_feat, self.output_proj, self.old_feat, self.output, self.output_old = self.model(self.input)
        self.output_proj = self.output_proj.view(-1).unsqueeze(1)

        self.output = self.output.view(-1).unsqueeze(1)
        self.output_old = self.output_old.view(-1).unsqueeze(1)


    
    def get_loss(self):
        lce = self.loss_fn(self.output.squeeze(1), self.label)
        # semantic distribution alignment loss
        sda_loss = self.alignment(self.new_feat, self.old_feat)
        # authenticity discrepancy alignment loss
        ada_loss = self.alignment(self.output_proj, self.output_old.detach())
        return lce+self.opt.DisWeight*(sda_loss+ada_loss)

    def optimize_parameters(self):
        self.forward()
        self.loss = self.get_loss()
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()
        


