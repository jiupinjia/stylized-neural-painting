import argparse
import torch
import torch.optim as optim

from painter import *

# settings
parser = argparse.ArgumentParser(description='STYLIZED NEURAL PAINTING')
parser.add_argument('--renderer', type=str, default='oilpaintbrush', metavar='str',
                    help='renderer: [watercolor, markerpen, oilpaintbrush, rectangle (default oilpaintbrush)')
parser.add_argument('--vector_file', type=str, default='./output/sunflowers_strokes.npz', metavar='str',
                    help='path to pre-generated stroke vector file (default: ...)')
parser.add_argument('--style_img_path', type=str, default='./style_images/fire.jpg', metavar='str',
                    help='path to style image (default: ...)')
parser.add_argument('--content_img_path', type=str, default='./test_images/sunflowers.jpg', metavar='str',
                    help='path to content image (default: ...)')
parser.add_argument('--transfer_mode', type=int, default=1, metavar='N',
                    help='style transfer mode, 0: transfer color only, 1: transfer both color and texture, '
                         'defalt: 1')
parser.add_argument('--canvas_color', type=str, default='black', metavar='str',
                    help='canvas_color: [black, white] (default black)')
parser.add_argument('--canvas_size', type=int, default=512, metavar='str',
                    help='size of the canvas for stroke rendering')
parser.add_argument('--keep_aspect_ratio', action='store_true', default=False,
                    help='keep input aspect ratio when saving outputs')
parser.add_argument('--beta_L1', type=float, default=1.0,
                    help='weight for L1 loss (default: 1.0)')
parser.add_argument('--beta_sty', type=float, default=0.5,
                    help='weight for vgg style loss (default: 0.5)')
parser.add_argument('--net_G', type=str, default='zou-fusion-net-light', metavar='str',
                    help='net_G: plain-dcgan, plain-unet, huang-net, zou-fusion-net, '
                         'or zou-fusion-net-light (default: zou-fusion-net-light)')
parser.add_argument('--renderer_checkpoint_dir', type=str, default=r'./checkpoints_G_oilpaintbrush_light', metavar='str',
                    help='dir to load neu-renderer (default: ./checkpoints_G_oilpaintbrush_light)')
parser.add_argument('--lr', type=float, default=0.005,
                    help='learning rate for stroke searching (default: 0.005)')
parser.add_argument('--output_dir', type=str, default=r'./output', metavar='str',
                    help='dir to save style transfer results (default: ./output)')
parser.add_argument('--disable_preview', action='store_true', default=False,
                    help='disable cv2.imshow, for running remotely without x-display')
args = parser.parse_args()


# Decide which device we want to run on
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def optimize_x(pt):

    pt._load_checkpoint()
    pt.net_G.eval()

    if args.transfer_mode == 0: # transfer color only
        pt.x_ctt.requires_grad = False
        pt.x_color.requires_grad = True
        pt.x_alpha.requires_grad = False
    else: # transfer both color and texture
        pt.x_ctt.requires_grad = True
        pt.x_color.requires_grad = True
        pt.x_alpha.requires_grad = True

    pt.optimizer_x_sty = optim.RMSprop([pt.x_ctt, pt.x_color, pt.x_alpha], lr=pt.lr)

    iters_per_stroke = 100
    for i in range(iters_per_stroke):
        pt.optimizer_x_sty.zero_grad()

        pt.x_ctt.data = torch.clamp(pt.x_ctt.data, 0.1, 1 - 0.1)
        pt.x_color.data = torch.clamp(pt.x_color.data, 0, 1)
        pt.x_alpha.data = torch.clamp(pt.x_alpha.data, 0, 1)

        if args.canvas_color == 'white':
            pt.G_pred_canvas = torch.ones([pt.m_grid*pt.m_grid, 3, pt.net_G.out_size, pt.net_G.out_size]).to(device)
        else:
            pt.G_pred_canvas = torch.zeros(pt.m_grid*pt.m_grid, 3, pt.net_G.out_size, pt.net_G.out_size).to(device)

        pt._forward_pass()
        pt._style_transfer_step_states()
        pt._backward_x_sty()
        pt.optimizer_x_sty.step()

        pt.x_ctt.data = torch.clamp(pt.x_ctt.data, 0.1, 1 - 0.1)
        pt.x_color.data = torch.clamp(pt.x_color.data, 0, 1)
        pt.x_alpha.data = torch.clamp(pt.x_alpha.data, 0, 1)

        pt.step_id += 1

    print('saving style transfer result...')
    v_n = pt._normalize_strokes(pt.x)
    v_n = pt._shuffle_strokes_and_reshape(v_n)
    final_rendered_image = pt._render(v_n, save_jpgs=False, save_video=False)
    pt._save_style_transfer_images(final_rendered_image)


if __name__ == '__main__':

    pt = NeuralStyleTransfer(args=args)
    optimize_x(pt)

