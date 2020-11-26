import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage.measure import compare_ssim as sk_cpt_ssim


import os
import glob
import random

import torch
torch.cuda.current_device()
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms, utils
import renderer



M_RENDERING_SAMPLES_PER_EPOCH = 50000

class PairedDataAugmentation:

    def __init__(
            self,
            img_size,
            with_random_hflip=False,
            with_random_vflip=False,
            with_random_rot90=False,
            with_random_rot180=False,
            with_random_rot270=False,
            with_random_crop=False,
            with_random_patch=False
    ):
        self.img_size = img_size
        self.with_random_hflip = with_random_hflip
        self.with_random_vflip = with_random_vflip
        self.with_random_rot90 = with_random_rot90
        self.with_random_rot180 = with_random_rot180
        self.with_random_rot270 = with_random_rot270
        self.with_random_crop = with_random_crop
        self.with_random_patch = with_random_patch

    def transform(self, img1, img2):

        # resize image and covert to tensor
        img1 = TF.to_pil_image(img1)
        img1 = TF.resize(img1, [self.img_size, self.img_size], interpolation=3)
        img2 = TF.to_pil_image(img2)
        img2 = TF.resize(img2, [self.img_size, self.img_size], interpolation=3)

        if self.with_random_hflip and random.random() > 0.5:
            img1 = TF.hflip(img1)
            img2 = TF.hflip(img2)

        if self.with_random_vflip and random.random() > 0.5:
            img1 = TF.vflip(img1)
            img2 = TF.vflip(img2)

        if self.with_random_rot90 and random.random() > 0.5:
            img1 = TF.rotate(img1, 90)
            img2 = TF.rotate(img2, 90)

        if self.with_random_rot180 and random.random() > 0.5:
            img1 = TF.rotate(img1, 180)
            img2 = TF.rotate(img2, 180)

        if self.with_random_rot270 and random.random() > 0.5:
            img1 = TF.rotate(img1, 270)
            img2 = TF.rotate(img2, 270)

        if self.with_random_crop and random.random() > 0.5:
            i, j, h, w = transforms.RandomResizedCrop(size=self.img_size). \
                get_params(img=img1, scale=(0.5, 1.0), ratio=(0.9, 1.1))
            img1 = TF.resized_crop(
                img1, i, j, h, w, size=(self.img_size, self.img_size))
            img2 = TF.resized_crop(
                img2, i, j, h, w, size=(self.img_size, self.img_size))

        if self.with_random_patch:
            i, j, h, w = transforms.RandomResizedCrop(size=self.img_size). \
                get_params(img=img1, scale=(1/16.0, 1/9.0), ratio=(0.9, 1.1))
            img1 = TF.resized_crop(
                img1, i, j, h, w, size=(self.img_size, self.img_size))
            img2 = TF.resized_crop(
                img2, i, j, h, w, size=(self.img_size, self.img_size))

        # to tensor
        img1 = TF.to_tensor(img1)
        img2 = TF.to_tensor(img2)

        return img1, img2



class StrokeDataset(Dataset):

    def __init__(self, args, is_train=True):
        if '-light' in args.net_G:
            CANVAS_WIDTH = 32
        else:
            CANVAS_WIDTH = 128
        self.rderr = renderer.Renderer(
            renderer=args.renderer, CANVAS_WIDTH=CANVAS_WIDTH, train=True)
        self.is_train = is_train

    def __len__(self):
        if self.is_train:
            return M_RENDERING_SAMPLES_PER_EPOCH
        else:
            return int(M_RENDERING_SAMPLES_PER_EPOCH / 20)

    def __getitem__(self, idx):

        self.rderr.foreground = None
        self.rderr.stroke_alpha_map = None

        self.rderr.random_stroke_params()
        self.rderr.draw_stroke()

        # to tensor
        params = torch.tensor(np.array(self.rderr.stroke_params, dtype=np.float32))
        params = torch.reshape(params, [-1, 1, 1])
        foreground = TF.to_tensor(np.array(self.rderr.foreground, dtype=np.float32))
        stroke_alpha_map = TF.to_tensor(np.array(self.rderr.stroke_alpha_map, dtype=np.float32))

        data = {'A': params, 'B': foreground, 'ALPHA': stroke_alpha_map}

        return data




def get_renderer_loaders(args):

    training_set = StrokeDataset(args, is_train=True)
    val_set = StrokeDataset(args, is_train=False)

    datasets = {'train': training_set, 'val': val_set}
    dataloaders = {x: DataLoader(datasets[x], batch_size=args.batch_size,
                                 shuffle=True, num_workers=4)
                   for x in ['train', 'val']}

    return dataloaders



def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad



def make_numpy_grid(tensor_data):
    tensor_data = tensor_data.detach()
    vis = utils.make_grid(tensor_data)
    vis = np.array(vis.cpu()).transpose((1,2,0))
    if vis.shape[2] == 1:
        vis = np.stack([vis, vis, vis], axis=-1)
    return vis.clip(min=0, max=1)



def tensor2img(tensor_data):
    if tensor_data.shape[0] > 1:
        raise NotImplementedError('batch size > 1, please use make_numpy_grid')
    tensor_data = tensor_data.detach()[0, :]
    img = np.array(tensor_data.cpu()).transpose((1, 2, 0))
    if img.shape[2] == 1:
        img = np.stack([img, img, img], axis=-1)
    return img.clip(min=0, max=1)



def cpt_ssim(img, img_gt, normalize=False):

    if normalize:
        img = (img - img.min()) / (img.max() - img.min() + 1e-9)
        img_gt = (img_gt - img_gt.min()) / (img_gt.max() - img_gt.min() + 1e-9)

    SSIM = sk_cpt_ssim(img, img_gt, data_range=img_gt.max() - img_gt.min())

    return SSIM


def cpt_psnr(img, img_gt, PIXEL_MAX=1.0, normalize=False):

    if normalize:
        img = (img - img.min()) / (img.max() - img.min() + 1e-9)
        img_gt = (img_gt - img_gt.min()) / (img_gt.max() - img_gt.min() + 1e-9)

    mse = np.mean((img - img_gt) ** 2)
    psnr = 20 * np.log10(PIXEL_MAX / np.sqrt(mse))

    return psnr


def cpt_cos_similarity(img, img_gt, normalize=False):

    if normalize:
        img = (img - img.min()) / (img.max() - img.min() + 1e-9)
        img_gt = (img_gt - img_gt.min()) / (img_gt.max() - img_gt.min() + 1e-9)

    cos_dist = np.sum(img*img_gt) / np.sqrt(np.sum(img**2)*np.sum(img_gt**2) + 1e-9)

    return cos_dist


def cpt_batch_psnr(img, img_gt, PIXEL_MAX):
    mse = torch.mean((img - img_gt) ** 2)
    psnr = 20 * torch.log10(PIXEL_MAX / torch.sqrt(mse))
    return psnr


def rotate_pt(pt, rotate_center, theta, return_int=True):

    # theta in [0, pi]
    x, y = pt[0], pt[1]
    xc, yc = rotate_center[0], rotate_center[1]

    x_ = (x-xc) * np.cos(theta) + (y-yc) * np.sin(theta) + xc
    y_ = -1 * (x-xc) * np.sin(theta) + (y-yc) * np.cos(theta) + yc

    if return_int:
        x_, y_ = int(x_), int(y_)

    pt_ = (x_, y_)

    return pt_


def img2patches(img, m_grid, s, to_tensor=True):
    # input img: h, w, 3 (np.float32)
    # output patches: N, 3, s, s (tensor, float32)

    img = cv2.resize(img, (m_grid * s, m_grid * s))
    img_batch = np.zeros([m_grid ** 2, 3, s, s], np.float32)
    for y_id in range(m_grid):
        for x_id in range(m_grid):
            patch = img[y_id * s:y_id * s + s,
                    x_id * s:x_id * s + s, :].transpose([2, 0, 1])
            img_batch[y_id * m_grid + x_id, :, :, :] = patch

    if to_tensor:
        img_batch = torch.tensor(img_batch)

    return img_batch



def patches2img(img_batch, m_grid, to_numpy=True):
    # input img_batch: m_grid**2, 3, s, s (tensor)
    # output img: s*m_grid, s*m_grid, 3 (np.float32)

    _, _, s, _ = img_batch.shape
    img = torch.zeros([s*m_grid, s*m_grid, 3])

    for y_id in range(m_grid):
        for x_id in range(m_grid):
            patch = img_batch[y_id * m_grid + x_id, :, :, :]
            img[y_id * s:y_id * s + s, x_id * s:x_id * s + s, :] \
                = patch.permute([1, 2, 0])
    if to_numpy:
        img = img.detach().numpy()
    else:
        img = img.permute([2,0,1]).unsqueeze(0)

    return img




def create_transformed_brush(brush, canvas_w, canvas_h,
                      x0, y0, w, h, theta, R0, G0, B0, R2, G2, B2):

    brush_alpha = np.stack([brush, brush, brush], axis=-1)
    brush_alpha = (brush_alpha > 0).astype(np.float32)
    brush_alpha = (brush_alpha*255).astype(np.uint8)
    colormap = np.zeros([brush.shape[0], brush.shape[1], 3], np.float32)
    for ii in range(brush.shape[0]):
        t = ii / brush.shape[0]
        this_color = [(1 - t) * R0 + t * R2,
                      (1 - t) * G0 + t * G2,
                      (1 - t) * B0 + t * B2]
        colormap[ii, :, :] = np.expand_dims(this_color, axis=0)

    brush = np.expand_dims(brush, axis=-1).astype(np.float32) / 255.
    brush = (brush * colormap * 255).astype(np.uint8)
    # plt.imshow(brush), plt.show()

    M1 = build_transformation_matrix([-brush.shape[1]/2, -brush.shape[0]/2, 0])
    M2 = build_scale_matrix(sx=w/brush.shape[1], sy=h/brush.shape[0])
    M3 = build_transformation_matrix([0,0,theta])
    M4 = build_transformation_matrix([x0, y0, 0])

    M = update_transformation_matrix(M1, M2)
    M = update_transformation_matrix(M, M3)
    M = update_transformation_matrix(M, M4)

    brush = cv2.warpAffine(
        brush, M, (canvas_w, canvas_h),
        borderMode=cv2.BORDER_CONSTANT, flags=cv2.INTER_AREA)
    brush_alpha = cv2.warpAffine(
        brush_alpha, M, (canvas_w, canvas_h),
        borderMode=cv2.BORDER_CONSTANT, flags=cv2.INTER_AREA)

    return brush, brush_alpha


def build_scale_matrix(sx, sy):
    transform_matrix = np.zeros((2, 3))
    transform_matrix[0, 0] = sx
    transform_matrix[1, 1] = sy
    return transform_matrix


def update_transformation_matrix(M, m):

    # extend M and m to 3x3 by adding an [0,0,1] to their 3rd row
    M_ = np.concatenate([M, np.zeros([1,3])], axis=0)
    M_[-1, -1] = 1
    m_ = np.concatenate([m, np.zeros([1,3])], axis=0)
    m_[-1, -1] = 1

    M_new = np.matmul(m_, M_)
    return M_new[0:2, :]
#

def build_transformation_matrix(transform):
    """Convert transform list to transformation matrix

    :param transform: transform list as [dx, dy, da]
    :return: transform matrix as 2d (2, 3) numpy array
    """
    transform_matrix = np.zeros((2, 3))

    transform_matrix[0, 0] = np.cos(transform[2])
    transform_matrix[0, 1] = -np.sin(transform[2])
    transform_matrix[1, 0] = np.sin(transform[2])
    transform_matrix[1, 1] = np.cos(transform[2])
    transform_matrix[0, 2] = transform[0]
    transform_matrix[1, 2] = transform[1]

    return transform_matrix



