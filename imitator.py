import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

import utils
import loss
from networks import *

import torch
torch.cuda.current_device()
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn as nn

import renderer


# Decide which device we want to run on
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Imitator():

    def __init__(self, args, dataloaders):

        self.dataloaders = dataloaders

        self.rderr = renderer.Renderer(renderer=args.renderer)

        # define G
        self.net_G = define_G(rdrr=self.rderr, netG=args.net_G).to(device)

        # Learning rate
        self.lr = args.lr

        # define optimizers
        self.optimizer_G = optim.Adam(
            self.net_G.parameters(), lr=self.lr, betas=(0.9, 0.999))

        # define lr schedulers
        self.exp_lr_scheduler_G = lr_scheduler.StepLR(
            self.optimizer_G, step_size=100, gamma=0.1)

        # define some other vars to record the training states
        self.running_acc = []
        self.epoch_acc = 0
        self.best_val_acc = 0.0
        self.best_epoch_id = 0
        self.epoch_to_start = 0
        self.max_num_epochs = args.max_num_epochs
        self.G_pred_foreground = None
        self.G_pred_alpha = None
        self.batch = None
        self.G_loss = None
        self.is_training = False
        self.batch_id = 0
        self.epoch_id = 0
        self.checkpoint_dir = args.checkpoint_dir
        self.vis_dir = args.vis_dir

        # define the loss functions
        self._pxl_loss = loss.PixelLoss(p=2)

        self.VAL_ACC = np.array([], np.float32)
        if os.path.exists(os.path.join(self.checkpoint_dir, 'val_acc.npy')):
            self.VAL_ACC = np.load(os.path.join(self.checkpoint_dir, 'val_acc.npy'))

        # check and create model dir
        if os.path.exists(self.checkpoint_dir) is False:
            os.mkdir(self.checkpoint_dir)
        if os.path.exists(self.vis_dir) is False:
            os.mkdir(self.vis_dir)

        # visualize model
        if args.print_models:
            self._visualize_models()


    def _visualize_models(self):

        from torchviz import make_dot

        # visualize models with the package torchviz
        data = next(iter(self.dataloaders['train']))
        y = self.net_G(data['A'].to(device))
        mygraph = make_dot(y.mean(), params=dict(self.net_G.named_parameters()))
        mygraph.render('G')


    def _load_checkpoint(self):

        if os.path.exists(os.path.join(self.checkpoint_dir, 'last_ckpt.pt')):
            print('loading last checkpoint...')
            # load the entire checkpoint
            checkpoint = torch.load(os.path.join(self.checkpoint_dir, 'last_ckpt.pt'))

            # update net_G states
            self.net_G.load_state_dict(checkpoint['model_G_state_dict'])
            self.optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
            self.exp_lr_scheduler_G.load_state_dict(
                checkpoint['exp_lr_scheduler_G_state_dict'])
            self.net_G.to(device)

            # update some other states
            self.epoch_to_start = checkpoint['epoch_id'] + 1
            self.best_val_acc = checkpoint['best_val_acc']
            self.best_epoch_id = checkpoint['best_epoch_id']

            print('Epoch_to_start = %d, Historical_best_acc = %.4f (at epoch %d)' %
                  (self.epoch_to_start, self.best_val_acc, self.best_epoch_id))
            print()

        else:
            print('training from scratch...')


    def _save_checkpoint(self, ckpt_name):
        torch.save({
            'epoch_id': self.epoch_id,
            'best_val_acc': self.best_val_acc,
            'best_epoch_id': self.best_epoch_id,
            'model_G_state_dict': self.net_G.state_dict(),
            'optimizer_G_state_dict': self.optimizer_G.state_dict(),
            'exp_lr_scheduler_G_state_dict': self.exp_lr_scheduler_G.state_dict()
        }, os.path.join(self.checkpoint_dir, ckpt_name))


    def _update_lr_schedulers(self):
        self.exp_lr_scheduler_G.step()


    def _compute_acc(self):

        target_foreground = self.gt_foreground.to(device).detach()
        target_alpha_map = self.gt_alpha.to(device).detach()
        foreground = self.G_pred_foreground.detach()
        alpha_map = self.G_pred_alpha.detach()

        psnr1 = utils.cpt_batch_psnr(foreground, target_foreground, PIXEL_MAX=1.0)
        psnr2 = utils.cpt_batch_psnr(alpha_map, target_alpha_map, PIXEL_MAX=1.0)
        return (psnr1 + psnr2)/2.0


    def _collect_running_batch_states(self):
        self.running_acc.append(self._compute_acc().item())

        m = len(self.dataloaders['train'])
        if self.is_training is False:
            m = len(self.dataloaders['val'])

        if np.mod(self.batch_id, 100) == 1:
            print('Is_training: %s. [%d,%d][%d,%d], G_loss: %.5f, running_acc: %.5f'
                  % (self.is_training, self.epoch_id, self.max_num_epochs-1, self.batch_id, m,
                     self.G_loss.item(), np.mean(self.running_acc)))

        if np.mod(self.batch_id, 1000) == 1:
            vis_pred_foreground = utils.make_numpy_grid(self.G_pred_foreground)
            vis_gt_foreground = utils.make_numpy_grid(self.gt_foreground)
            vis_pred_alpha = utils.make_numpy_grid(self.G_pred_alpha)
            vis_gt_alpha = utils.make_numpy_grid(self.gt_alpha)

            vis = np.concatenate([vis_pred_foreground, vis_gt_foreground,
                                  vis_pred_alpha, vis_gt_alpha], axis=0)
            vis = np.clip(vis, a_min=0.0, a_max=1.0)
            file_name = os.path.join(
                self.vis_dir, 'istrain_'+str(self.is_training)+'_'+
                              str(self.epoch_id)+'_'+str(self.batch_id)+'.jpg')
            plt.imsave(file_name, vis)



    def _collect_epoch_states(self):

        self.epoch_acc = np.mean(self.running_acc)
        print('Is_training: %s. Epoch %d / %d, epoch_acc= %.5f' %
              (self.is_training, self.epoch_id, self.max_num_epochs-1, self.epoch_acc))
        print()


    def _update_checkpoints(self):

        # save current model
        self._save_checkpoint(ckpt_name='last_ckpt.pt')
        print('Lastest model updated. Epoch_acc=%.4f, Historical_best_acc=%.4f (at epoch %d)'
              % (self.epoch_acc, self.best_val_acc, self.best_epoch_id))
        print()

        self.VAL_ACC = np.append(self.VAL_ACC, [self.epoch_acc])
        np.save(os.path.join(self.checkpoint_dir, 'val_acc.npy'), self.VAL_ACC)

        # update the best model (based on eval acc)
        if self.epoch_acc > self.best_val_acc:
            self.best_val_acc = self.epoch_acc
            self.best_epoch_id = self.epoch_id
            self._save_checkpoint(ckpt_name='best_ckpt.pt')
            print('*' * 10 + 'Best model updated!')
            print()


    def _clear_cache(self):
        self.running_acc = []


    def _forward_pass(self, batch):
        self.batch = batch
        z_in = batch['A'].to(device)
        self.G_pred_foreground, self.G_pred_alpha = self.net_G(z_in)


    def _backward_G(self):

        self.gt_foreground = self.batch['B'].to(device)
        self.gt_alpha = self.batch['ALPHA'].to(device)

        _, _, h, w = self.G_pred_alpha.shape
        self.gt_foreground = torch.nn.functional.interpolate(self.gt_foreground, (h, w), mode='area')
        self.gt_alpha = torch.nn.functional.interpolate(self.gt_alpha, (h, w), mode='area')

        pixel_loss1 = self._pxl_loss(self.G_pred_foreground, self.gt_foreground)
        pixel_loss2 = self._pxl_loss(self.G_pred_alpha, self.gt_alpha)
        self.G_loss = 100 * (pixel_loss1 + pixel_loss2) / 2.0
        self.G_loss.backward()


    def train_models(self):

        self._load_checkpoint()

        # loop over the dataset multiple times
        for self.epoch_id in range(self.epoch_to_start, self.max_num_epochs):

            ################## train #################
            ##########################################
            self._clear_cache()
            self.is_training = True
            self.net_G.train()  # Set model to training mode
            # Iterate over data.
            for self.batch_id, batch in enumerate(self.dataloaders['train'], 0):
                self._forward_pass(batch)
                # update G
                self.optimizer_G.zero_grad()
                self._backward_G()
                self.optimizer_G.step()
                self._collect_running_batch_states()
            self._collect_epoch_states()
            self._update_lr_schedulers()

            ########### Update_Checkpoints ###########
            ##########################################
            self._update_checkpoints()

