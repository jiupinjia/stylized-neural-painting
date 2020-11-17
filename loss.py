import torch
torch.cuda.current_device()
import torch.nn as nn
import torchvision
import utils
import matplotlib.pyplot as plt
import numpy as np
import random
import pytorch_batch_sinkhorn as spc

# Decide which device we want to run on
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class PixelLoss(nn.Module):

    def __init__(self, p=1):
        super(PixelLoss, self).__init__()
        self.p = p

    def forward(self, canvas, gt, ignore_color=False):
        if ignore_color:
            canvas = torch.mean(canvas, dim=1)
            gt = torch.mean(gt, dim=1)
        loss = torch.mean(torch.abs(canvas-gt)**self.p)
        return loss


class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        vgg = torchvision.models.vgg16(pretrained=True).to(device)
        blocks = []
        blocks.append(vgg.features[:4].eval())
        blocks.append(vgg.features[4:9].eval())
        blocks.append(vgg.features[9:16].eval())
        blocks.append(vgg.features[16:23].eval())
        for bl in blocks:
            for p in bl:
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1)
        self.resize = resize

    def forward(self, input, target, ignore_color=False):
        self.mean = self.mean.type_as(input)
        self.std = self.std.type_as(input)
        if ignore_color:
            input = torch.mean(input, dim=1, keepdim=True)
            target = torch.mean(target, dim=1, keepdim=True)
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        input = (input-self.mean) / self.std
        target = (target-self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        loss = 0.0
        x = input
        y = target
        for block in self.blocks:
            x = block(x)
            y = block(y)
            loss += torch.nn.functional.l1_loss(x, y)
        return loss



class VGGStyleLoss(torch.nn.Module):
    def __init__(self, transfer_mode, resize=True):
        super(VGGStyleLoss, self).__init__()
        vgg = torchvision.models.vgg16(pretrained=True).to(device)
        for i, layer in enumerate(vgg.features):
            if isinstance(layer, torch.nn.MaxPool2d):
                vgg.features[i] = torch.nn.AvgPool2d(kernel_size=2, stride=2, padding=0)

        blocks = []
        if transfer_mode == 0:  # transfer color only
            blocks.append(vgg.features[:4].eval())
            blocks.append(vgg.features[4:9].eval())
        else: # transfer both color and texture
            blocks.append(vgg.features[:4].eval())
            blocks.append(vgg.features[4:9].eval())
            blocks.append(vgg.features[9:16].eval())
            blocks.append(vgg.features[16:23].eval())

        for bl in blocks:
            for p in bl:
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)

        self.transform = torch.nn.functional.interpolate
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
        self.resize = resize

    def gram_matrix(self, y):
        (b, ch, h, w) = y.size()
        features = y.view(b, ch, w * h)
        features_t = features.transpose(1, 2)
        gram = features.bmm(features_t) / (ch * w * h)
        return gram

    def forward(self, input, target):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        input = (input - self.mean) / self.std
        target = (target - self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        loss = 0.0
        x = input
        y = target
        for block in self.blocks:
            x = block(x)
            y = block(y)
            gm_x = self.gram_matrix(x)
            gm_y = self.gram_matrix(y)
            loss += torch.sum((gm_x-gm_y)**2)
        return loss



class SinkhornLoss(nn.Module):

    def __init__(self, epsilon=0.01, niter=5, normalize=False):
        super(SinkhornLoss, self).__init__()
        self.epsilon = epsilon
        self.niter = niter
        self.normalize = normalize

    def _mesh_grids(self, batch_size, h, w):

        a = torch.linspace(0.0, h - 1.0, h).to(device)
        b = torch.linspace(0.0, w - 1.0, w).to(device)
        y_grid = a.view(-1, 1).repeat(batch_size, 1, w) / h
        x_grid = b.view(1, -1).repeat(batch_size, h, 1) / w
        grids = torch.cat([y_grid.view(batch_size, -1, 1), x_grid.view(batch_size, -1, 1)], dim=-1)
        return grids

    def forward(self, canvas, gt):

        batch_size, c, h, w = gt.shape
        if h > 24:
            canvas = nn.functional.interpolate(canvas, [24, 24], mode='area')
            gt = nn.functional.interpolate(gt, [24, 24], mode='area')
            batch_size, c, h, w = gt.shape

        canvas_grids = self._mesh_grids(batch_size, h, w)
        gt_grids = torch.clone(canvas_grids)

        # randomly select a color channel, to speedup and consume memory
        i = random.randint(0, 2)

        img_1 = canvas[:, [i], :, :]
        img_2 = gt[:, [i], :, :]

        mass_x = img_1.reshape(batch_size, -1)
        mass_y = img_2.reshape(batch_size, -1)
        if self.normalize:
            loss = spc.sinkhorn_normalized(
                canvas_grids, gt_grids, epsilon=self.epsilon, niter=self.niter,
                mass_x=mass_x, mass_y=mass_y)
        else:
            loss = spc.sinkhorn_loss(
                canvas_grids, gt_grids, epsilon=self.epsilon, niter=self.niter,
                mass_x=mass_x, mass_y=mass_y)


        return loss




