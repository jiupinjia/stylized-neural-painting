import argparse

import torch
torch.cuda.current_device()
import torch.optim as optim

from painter import *

# settings
parser = argparse.ArgumentParser(description='STYLIZED NEURAL PAINTING')
parser.add_argument('--img_path', type=str, default='./test_images/sunflowers.jpg', metavar='str',
                    help='path to test image (default: ./test_images/sunflowers.jpg)')
parser.add_argument('--renderer', type=str, default='rectangle', metavar='str',
                    help='renderer: [watercolor, markerpen, oilpaintbrush, rectangle (default oilpaintbrush)')
parser.add_argument('--canvas_color', type=str, default='black', metavar='str',
                    help='canvas_color: [black, white] (default black)')
parser.add_argument('--canvas_size', type=int, default=512, metavar='str',
                    help='size of the canvas for stroke rendering')
parser.add_argument('--max_m_strokes', type=int, default=500, metavar='str',
                    help='max number of strokes (default 500)')
parser.add_argument('--max_divide', type=int, default=5, metavar='N',
                    help='divide an image up-to max_divide x max_divide patches (default 5)')
parser.add_argument('--beta_L1', type=float, default=1.0,
                    help='weight for L1 loss (default: 1.0)')
parser.add_argument('--with_ot_loss', action='store_true', default=False,
                    help='imporve the convergence by using optimal transportation loss')
parser.add_argument('--beta_ot', type=float, default=0.1,
                    help='weight for optimal transportation loss (default: 0.1)')
parser.add_argument('--net_G', type=str, default='zou-fusion-net', metavar='str',
                    help='net_G: plain-dcgan, plain-unet, huang-net, or zou-fusion-net (default: zou-fusion-net)')
parser.add_argument('--renderer_checkpoint_dir', type=str, default=r'./checkpoints_G_rectangle', metavar='str',
                    help='dir to load neu-renderer (default: ./checkpoints_G_rectangle)')
parser.add_argument('--lr', type=float, default=0.005,
                    help='learning rate for stroke searching (default: 0.005)')
parser.add_argument('--output_dir', type=str, default=r'./output', metavar='str',
                    help='dir to save painting results (default: ./output)')
args = parser.parse_args()


# Decide which device we want to run on
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def optimize_x(pt):

    pt._load_checkpoint()
    pt.net_G.eval()

    print('begin drawing...')

    PARAMS = np.zeros([1, 0, pt.rderr.d], np.float32)

    if pt.rderr.canvas_color == 'white':
        CANVAS_tmp = torch.ones([1, 3, 128, 128]).to(device)
    else:
        CANVAS_tmp = torch.zeros([1, 3, 128, 128]).to(device)

    for pt.m_grid in range(1, pt.max_divide + 1):

        pt.img_batch = utils.img2patches(pt.img_, pt.m_grid).to(device)
        pt.G_final_pred_canvas = CANVAS_tmp

        pt.initialize_params()
        pt.x_ctt.requires_grad = True
        pt.x_color.requires_grad = True
        pt.x_alpha.requires_grad = True
        utils.set_requires_grad(pt.net_G, False)

        pt.optimizer_x = optim.RMSprop([pt.x_ctt, pt.x_color, pt.x_alpha], lr=pt.lr, centered=True)

        pt.step_id = 0
        for pt.anchor_id in range(0, pt.m_strokes_per_block):
            pt.stroke_sampler(pt.anchor_id)
            iters_per_stroke = 20
            for i in range(iters_per_stroke):
                pt.G_pred_canvas = CANVAS_tmp

                # update x
                pt.optimizer_x.zero_grad()

                pt.x_ctt.data = torch.clamp(pt.x_ctt.data, 0, 1)
                pt.x_ctt.data[:, :, -1] = torch.clamp(pt.x_ctt.data[:, :, -1], 0, 0)
                pt.x_color.data = torch.clamp(pt.x_color.data, 0, 1)
                pt.x_alpha.data = torch.clamp(pt.x_alpha.data, 1, 1)

                pt._forward_pass()
                pt._backward_x()

                pt.x_ctt.data = torch.clamp(pt.x_ctt.data, 0, 1)
                pt.x_ctt.data[:, :, -1] = torch.clamp(pt.x_ctt.data[:, :, -1], 0, 0)
                pt.x_color.data = torch.clamp(pt.x_color.data, 0, 1)
                pt.x_alpha.data = torch.clamp(pt.x_alpha.data, 1, 1)

                pt._drawing_step_states()

                pt.optimizer_x.step()
                pt.step_id += 1

        v = pt._normalize_strokes(pt.x)
        PARAMS = np.concatenate([PARAMS, np.reshape(v, [1, -1, pt.rderr.d])], axis=1)
        CANVAS_tmp = pt._render(PARAMS)[-1]
        CANVAS_tmp = utils.img2patches(CANVAS_tmp, pt.m_grid + 1, to_tensor=True).to(device)

    pt._save_stroke_params(PARAMS)
    pt.final_rendered_images = pt._render(PARAMS)
    pt._save_rendered_images()



if __name__ == '__main__':

    pt = ProgressivePainter(args=args)
    optimize_x(pt)

