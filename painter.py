import os
import cv2
import random

import utils
import loss
from networks import *
import morphology

import renderer

import torch
torch.cuda.current_device()

# Decide which device we want to run on
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class PainterBase():
    def __init__(self, args):
        self.args = args

        self.rderr = renderer.Renderer(renderer=args.renderer,
                                       CANVAS_WIDTH=args.canvas_size, canvas_color=args.canvas_color)

        # define G
        self.net_G = define_G(rdrr=self.rderr, netG=args.net_G).to(device)

        # define some other vars to record the training states
        self.x_ctt = None
        self.x_color = None
        self.x_alpha = None

        self.G_pred_foreground = None
        self.G_pred_alpha = None
        self.G_final_pred_canvas = torch.zeros(
            [1, 3, self.net_G.out_size, self.net_G.out_size]).to(device)

        self.G_loss = torch.tensor(0.0)
        self.step_id = 0
        self.anchor_id = 0
        self.renderer_checkpoint_dir = args.renderer_checkpoint_dir
        self.output_dir = args.output_dir
        self.lr = args.lr

        # define the loss functions
        self._pxl_loss = loss.PixelLoss(p=1)
        self._sinkhorn_loss = loss.SinkhornLoss(epsilon=0.01, niter=5, normalize=False)

        # some other vars to be initialized in child classes
        self.input_aspect_ratio = None
        self.img_path = None
        self.img_batch = None
        self.img_ = None
        self.final_rendered_images = None
        self.m_grid = None
        self.m_strokes_per_block = None

        if os.path.exists(self.output_dir) is False:
            os.mkdir(self.output_dir)

    def _load_checkpoint(self):

        # load renderer G
        if os.path.exists((os.path.join(
                self.renderer_checkpoint_dir, 'last_ckpt.pt'))):
            print('loading renderer from pre-trained checkpoint...')
            # load the entire checkpoint
            checkpoint = torch.load(os.path.join(
                self.renderer_checkpoint_dir, 'last_ckpt.pt'))
            # update net_G states
            self.net_G.load_state_dict(checkpoint['model_G_state_dict'])
            self.net_G.to(device)
            self.net_G.eval()
        else:
            print('pre-trained renderer does not exist...')
            exit()


    def _compute_acc(self):

        target = self.img_batch.detach()
        canvas = self.G_pred_canvas.detach()
        psnr = utils.cpt_batch_psnr(canvas, target, PIXEL_MAX=1.0)

        return psnr

    def _save_stroke_params(self, v):

        d_shape = self.rderr.d_shape
        d_color = self.rderr.d_color
        d_alpha = self.rderr.d_alpha

        x_ctt = v[:, :, 0:d_shape]
        x_color = v[:, :, d_shape:d_shape+d_color]
        x_alpha = v[:, :, d_shape+d_color:d_shape+d_color+d_alpha]
        print('saving stroke parameters...')
        file_name = os.path.join(
            self.output_dir, self.img_path.split('/')[-1][:-4])
        np.savez(file_name + '_strokes.npz', x_ctt=x_ctt,
                 x_color=x_color, x_alpha=x_alpha)

    def _shuffle_strokes_and_reshape(self, v):

        grid_idx = list(range(self.m_grid ** 2))
        random.shuffle(grid_idx)
        v = v[grid_idx, :, :]
        v = np.reshape(np.transpose(v, [1,0,2]), [-1, self.rderr.d])
        v = np.expand_dims(v, axis=0)

        return v

    def _render(self, v, save_jpgs=True, save_video=True):

        v = v[0,:,:]

        if self.input_aspect_ratio < 1:
            out_h = int(self.args.canvas_size * self.input_aspect_ratio)
            out_w = self.args.canvas_size
        else:
            out_h = self.args.canvas_size
            out_w = int(self.args.canvas_size / self.input_aspect_ratio)

        file_name = os.path.join(
            self.output_dir, self.img_path.split('/')[-1][:-4])

        if save_video:
            video_writer = cv2.VideoWriter(
                file_name + '_animated.mp4', cv2.VideoWriter_fourcc(*'MP4V'), 40,
                (out_w, out_h))

        print('rendering canvas...')
        self.rderr.create_empty_canvas()
        for i in range(v.shape[0]):  # for each stroke
            self.rderr.stroke_params = v[i, :]
            if self.rderr.check_stroke():
                self.rderr.draw_stroke()
            this_frame = self.rderr.canvas
            this_frame = cv2.resize(this_frame, (out_w, out_h), cv2.INTER_AREA)
            if save_jpgs:
                plt.imsave(file_name + '_rendered_stroke_' + str((i+1)).zfill(4) +
                           '.png', this_frame)
            if save_video:
                video_writer.write((this_frame[:,:,::-1] * 255.).astype(np.uint8))

        if save_jpgs:
            print('saving input photo...')
            out_img = cv2.resize(self.img_, (out_w, out_h), cv2.INTER_AREA)
            plt.imsave(file_name + '_input.png', out_img)

        final_rendered_image = np.copy(this_frame)
        if save_jpgs:
            print('saving final rendered result...')
            plt.imsave(file_name + '_final.png', final_rendered_image)

        return final_rendered_image




    def _normalize_strokes(self, v):

        v = np.array(v.detach().cpu())

        if self.rderr.renderer in ['watercolor', 'markerpen']:
            # x0, y0, x1, y1, x2, y2, radius0, radius2, ...
            xs = np.array([0, 4])
            ys = np.array([1, 5])
            rs = np.array([6, 7])
        elif self.rderr.renderer in ['oilpaintbrush', 'rectangle']:
            # xc, yc, w, h, theta ...
            xs = np.array([0])
            ys = np.array([1])
            rs = np.array([2, 3])
        else:
            raise NotImplementedError('renderer [%s] is not implemented' % self.rderr.renderer)

        for y_id in range(self.m_grid):
            for x_id in range(self.m_grid):
                y_bias = y_id / self.m_grid
                x_bias = x_id / self.m_grid
                v[y_id * self.m_grid + x_id, :, ys] = \
                    y_bias + v[y_id * self.m_grid + x_id, :, ys] / self.m_grid
                v[y_id * self.m_grid + x_id, :, xs] = \
                    x_bias + v[y_id * self.m_grid + x_id, :, xs] / self.m_grid
                v[y_id * self.m_grid + x_id, :, rs] /= self.m_grid

        return v


    def initialize_params(self):

        self.x_ctt = np.random.rand(
            self.m_grid*self.m_grid, self.m_strokes_per_block,
            self.rderr.d_shape).astype(np.float32)
        self.x_ctt = torch.tensor(self.x_ctt).to(device)

        self.x_color = np.random.rand(
            self.m_grid*self.m_grid, self.m_strokes_per_block,
            self.rderr.d_color).astype(np.float32)
        self.x_color = torch.tensor(self.x_color).to(device)

        self.x_alpha = np.random.rand(
            self.m_grid*self.m_grid, self.m_strokes_per_block,
            self.rderr.d_alpha).astype(np.float32)
        self.x_alpha = torch.tensor(self.x_alpha).to(device)


    def stroke_sampler(self, anchor_id):

        if anchor_id == self.m_strokes_per_block:
            return

        err_maps = torch.sum(
            torch.abs(self.img_batch - self.G_final_pred_canvas),
            dim=1, keepdim=True).detach()

        for i in range(self.m_grid*self.m_grid):
            this_err_map = err_maps[i,0,:,:].cpu().numpy()
            ks = int(this_err_map.shape[0] / 8)
            this_err_map = cv2.blur(this_err_map, (ks, ks))
            this_err_map = this_err_map ** 4
            this_img = self.img_batch[i, :, :, :].detach().permute([1, 2, 0]).cpu().numpy()

            self.rderr.random_stroke_params_sampler(
                err_map=this_err_map, img=this_img)

            self.x_ctt.data[i, anchor_id, :] = torch.tensor(
                self.rderr.stroke_params[0:self.rderr.d_shape])
            self.x_color.data[i, anchor_id, :] = torch.tensor(
                self.rderr.stroke_params[self.rderr.d_shape:self.rderr.d_shape+self.rderr.d_color])
            self.x_alpha.data[i, anchor_id, :] = torch.tensor(self.rderr.stroke_params[-1])


    def _backward_x(self):

        self.G_loss = 0
        self.G_loss += self.args.beta_L1 * self._pxl_loss(
            canvas=self.G_final_pred_canvas, gt=self.img_batch)
        if self.args.with_ot_loss:
            self.G_loss += self.args.beta_ot * self._sinkhorn_loss(
                self.G_final_pred_canvas, self.img_batch)
        self.G_loss.backward()


    def _forward_pass(self):

        self.x = torch.cat([self.x_ctt, self.x_color, self.x_alpha], dim=-1)

        v = torch.reshape(self.x[:, 0:self.anchor_id+1, :],
                          [self.m_grid*self.m_grid*(self.anchor_id+1), -1, 1, 1])
        self.G_pred_foregrounds, self.G_pred_alphas = self.net_G(v)

        self.G_pred_foregrounds = morphology.Dilation2d(m=1)(self.G_pred_foregrounds)
        self.G_pred_alphas = morphology.Erosion2d(m=1)(self.G_pred_alphas)

        self.G_pred_foregrounds = torch.reshape(
            self.G_pred_foregrounds, [self.m_grid*self.m_grid, self.anchor_id+1, 3,
                                      self.net_G.out_size, self.net_G.out_size])
        self.G_pred_alphas = torch.reshape(
            self.G_pred_alphas, [self.m_grid*self.m_grid, self.anchor_id+1, 3,
                                 self.net_G.out_size, self.net_G.out_size])

        for i in range(self.anchor_id+1):
            G_pred_foreground = self.G_pred_foregrounds[:, i]
            G_pred_alpha = self.G_pred_alphas[:, i]
            self.G_pred_canvas = G_pred_foreground * G_pred_alpha \
                                 + self.G_pred_canvas * (1 - G_pred_alpha)

        self.G_final_pred_canvas = self.G_pred_canvas




class Painter(PainterBase):

    def __init__(self, args):
        super(Painter, self).__init__(args=args)

        self.m_grid = args.m_grid

        self.max_m_strokes = args.max_m_strokes

        self.img_path = args.img_path
        self.img_ = cv2.imread(args.img_path, cv2.IMREAD_COLOR)
        self.img_ = cv2.cvtColor(self.img_, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.
        self.input_aspect_ratio = self.img_.shape[0] / self.img_.shape[1]
        self.img_ = cv2.resize(self.img_, (self.net_G.out_size * args.m_grid,
                                           self.net_G.out_size * args.m_grid), cv2.INTER_AREA)

        self.m_strokes_per_block = int(args.max_m_strokes / (args.m_grid * args.m_grid))

        self.img_batch = utils.img2patches(self.img_, args.m_grid, self.net_G.out_size).to(device)

        self.final_rendered_images = None


    def _drawing_step_states(self):
        acc = self._compute_acc().item()
        print('iteration step %d, G_loss: %.5f, step_psnr: %.5f, strokes: %d / %d'
              % (self.step_id, self.G_loss.item(), acc,
                 (self.anchor_id+1)*self.m_grid*self.m_grid,
                 self.max_m_strokes))
        vis2 = utils.patches2img(self.G_final_pred_canvas, self.m_grid).clip(min=0, max=1)
        if self.args.disable_preview:
            pass
        else:
            cv2.imshow('G_pred', vis2[:,:,::-1])
            cv2.imshow('input', self.img_[:, :, ::-1])
            cv2.waitKey(1)





class ProgressivePainter(PainterBase):

    def __init__(self, args):
        super(ProgressivePainter, self).__init__(args=args)

        self.max_divide = args.max_divide

        self.max_m_strokes = args.max_m_strokes

        self.m_strokes_per_block = self.stroke_parser()

        self.m_grid = 1

        self.img_path = args.img_path
        self.img_ = cv2.imread(args.img_path, cv2.IMREAD_COLOR)
        self.img_ = cv2.cvtColor(self.img_, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.
        self.input_aspect_ratio = self.img_.shape[0] / self.img_.shape[1]
        self.img_ = cv2.resize(self.img_, (self.net_G.out_size * args.max_divide,
                                           self.net_G.out_size * args.max_divide), cv2.INTER_AREA)


    def stroke_parser(self):

        total_blocks = 0
        for i in range(0, self.max_divide + 1):
            total_blocks += i ** 2

        return int(self.max_m_strokes / total_blocks)


    def _drawing_step_states(self):
        acc = self._compute_acc().item()
        print('iteration step %d, G_loss: %.5f, step_acc: %.5f, grid_scale: %d / %d, strokes: %d / %d'
              % (self.step_id, self.G_loss.item(), acc,
                 self.m_grid, self.max_divide,
                 self.anchor_id + 1, self.m_strokes_per_block))
        vis2 = utils.patches2img(self.G_final_pred_canvas, self.m_grid).clip(min=0, max=1)
        if self.args.disable_preview:
            pass
        else:
            cv2.imshow('G_pred', vis2[:,:,::-1])
            cv2.imshow('input', self.img_[:, :, ::-1])
            cv2.waitKey(1)



class NeuralStyleTransfer(PainterBase):

    def __init__(self, args):
        super(NeuralStyleTransfer, self).__init__(args=args)

        self.args = args

        self._style_loss = loss.VGGStyleLoss(transfer_mode=args.transfer_mode, resize=True)

        print('loading pre-generated vector file...')
        if os.path.exists(args.vector_file) is False:
            exit('vector file does not exist, pls check --vector_file, or run demo.py fist')
        else:
            npzfile = np.load(args.vector_file)

        self.x_ctt = torch.tensor(npzfile['x_ctt']).to(device)
        self.x_color = torch.tensor(npzfile['x_color']).to(device)
        self.x_alpha = torch.tensor(npzfile['x_alpha']).to(device)
        self.m_grid = int(np.sqrt(self.x_ctt.shape[0]))

        self.anchor_id = self.x_ctt.shape[1] - 1

        img_ = cv2.imread(args.content_img_path, cv2.IMREAD_COLOR)
        img_ = cv2.cvtColor(img_, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.
        self.input_aspect_ratio = img_.shape[0] / img_.shape[1]
        self.img_ = cv2.resize(img_, (self.net_G.out_size*self.m_grid,
                                      self.net_G.out_size*self.m_grid), cv2.INTER_AREA)
        self.img_batch = utils.img2patches(self.img_, self.m_grid, self.net_G.out_size).to(device)

        style_img = cv2.imread(args.style_img_path, cv2.IMREAD_COLOR)
        self.style_img_ = cv2.cvtColor(style_img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.
        self.style_img = cv2.blur(cv2.resize(self.style_img_, (128, 128)), (2, 2))
        self.style_img = torch.tensor(self.style_img).permute([2, 0, 1]).unsqueeze(0).to(device)

        self.content_img_path = args.content_img_path
        self.img_path = args.content_img_path
        self.style_img_path = args.style_img_path


    def _style_transfer_step_states(self):
        acc = self._compute_acc().item()
        print('running style transfer... iteration step %d, G_loss: %.5f, step_psnr: %.5f'
              % (self.step_id, self.G_loss.item(), acc))
        vis2 = utils.patches2img(self.G_final_pred_canvas, self.m_grid).clip(min=0, max=1)
        if self.args.disable_preview:
            pass
        else:
            cv2.imshow('G_pred', vis2[:,:,::-1])
            cv2.imshow('input', self.img_[:, :, ::-1])
            cv2.imshow('style_img', self.style_img_[:, :, ::-1])
            cv2.waitKey(1)


    def _backward_x_sty(self):
        canvas = utils.patches2img(
            self.G_final_pred_canvas, self.m_grid, to_numpy=False).to(device)
        self.G_loss = self.args.beta_L1 * self._pxl_loss(
            canvas=self.G_final_pred_canvas, gt=self.img_batch, ignore_color=True)
        self.G_loss += self.args.beta_sty * self._style_loss(canvas, self.style_img)
        self.G_loss.backward()


    def _render_on_grids(self, v):

        rendered_imgs = []

        self.rderr.create_empty_canvas()

        grid_idx = list(range(self.m_grid ** 2))
        random.shuffle(grid_idx)
        for j in range(v.shape[1]):  # for each group of stroke
            for i in range(len(grid_idx)):  # for each random patch
                self.rderr.stroke_params = v[grid_idx[i], j, :]
                if self.rderr.check_stroke():
                    self.rderr.draw_stroke()
                rendered_imgs.append(self.rderr.canvas)

        return rendered_imgs


    def _save_style_transfer_images(self, final_rendered_image):

        if self.input_aspect_ratio < 1:
            out_h = int(self.args.canvas_size * self.input_aspect_ratio)
            out_w = self.args.canvas_size
        else:
            out_h = self.args.canvas_size
            out_w = int(self.args.canvas_size / self.input_aspect_ratio)

        print('saving style transfer results...')

        file_dir = os.path.join(
            self.output_dir, self.content_img_path.split('/')[-1][:-4])

        out_img = cv2.resize(self.style_img_, (out_w, out_h), cv2.INTER_AREA)
        plt.imsave(file_dir + '_style_img_' +
                   self.style_img_path.split('/')[-1][:-4] + '.png', out_img)

        out_img = cv2.resize(final_rendered_image, (out_w, out_h), cv2.INTER_AREA)
        plt.imsave(file_dir + '_style_transfer_' +
                   self.style_img_path.split('/')[-1][:-4] + '.png', out_img)
