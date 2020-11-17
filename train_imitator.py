import argparse
import numpy as np
import matplotlib.pyplot as plt
import utils
from imitator import*

# settings
parser = argparse.ArgumentParser(description='ZZX TRAIN IMITATOR')
parser.add_argument('--renderer', type=str, default='oilpaintbrush', metavar='str',
                    help='renderer: [watercolor, markerpen, oilpaintbrush, rectangle'
                         'bezier, circle, square, rectangle] (default ...)')
parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 4)')
parser.add_argument('--print_models', action='store_true', default=False,
                    help='visualize and print networks')
parser.add_argument('--net_G', type=str, default='zou-fusion-net', metavar='str',
                    help='net_G: plain-dcgan or plain-unet or huang-net or zou-fusion-net')
parser.add_argument('--checkpoint_dir', type=str, default=r'./checkpoints_G', metavar='str',
                    help='dir to save checkpoints (default: ...)')
parser.add_argument('--vis_dir', type=str, default=r'./val_out_G', metavar='str',
                    help='dir to save results during training (default: ./val_out_G)')
parser.add_argument('--lr', type=float, default=2e-4,
                    help='learning rate (default: 0.0002)')
parser.add_argument('--max_num_epochs', type=int, default=400, metavar='N',
                    help='max number of training epochs (default 400)')
args = parser.parse_args()


if __name__ == '__main__':

    dataloaders = utils.get_renderer_loaders(args)
    imt = Imitator(args=args, dataloaders=dataloaders)
    imt.train_models()

    # # How to check if the data is loading correctly?
    # dataloaders = utils.get_renderer_loaders(args)
    # for i in range(100):
    #     data = next(iter(dataloaders['train']))
    #     vis_A = data['A']
    #     vis_B = utils.make_numpy_grid(data['B'])
    #     print(data['A'].cpu().numpy().shape[1])
    #     print(data['B'].shape)
    #     plt.imshow(vis_B)
    #     plt.show()


