import argparse

import torch
torch.cuda.current_device()
import torch.optim as optim
import runway
from painter import *
# Decide which device we want to run on
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from PIL import Image

# settings
parser = argparse.ArgumentParser(description='STYLIZED NEURAL PAINTING')
args = parser.parse_args(args=[])
args.renderer = 'oilpaintbrush' # [watercolor, markerpen, oilpaintbrush, rectangle]
args.canvas_color = 'black' # [black, white]
args.canvas_size = 512 # size of the canvas for stroke rendering'
args.keep_aspect_ratio = False # whether to keep input aspect ratio when saving outputs
args.max_divide = 5 # divide an image up-to max_divide x max_divide patches
args.beta_L1 = 1.0 # weight for L1 loss
args.with_ot_loss = False # set True for imporving the convergence by using optimal transportation loss, but will slow-down the speed
args.beta_ot = 0.1 # weight for optimal transportation loss
args.net_G = 'zou-fusion-net' # renderer architecture
args.renderer_checkpoint_dir = './checkpoints_G_oilpaintbrush' # dir to load the pretrained neu-renderer
args.lr = 0.005 # learning rate for stroke searching
args.output_dir = './output' # dir to save painting results
args.disable_preview = True # disable cv2.imshow, for running remotely without x-display

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

        pt.img_batch = utils.img2patches(pt.img_, pt.m_grid, pt.net_G.out_size).to(device)
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
            iters_per_stroke = int(500 / pt.m_strokes_per_block)
            for i in range(iters_per_stroke):
                pt.G_pred_canvas = CANVAS_tmp

                # update x
                pt.optimizer_x.zero_grad()

                pt.x_ctt.data = torch.clamp(pt.x_ctt.data, 0.1, 1 - 0.1)
                pt.x_color.data = torch.clamp(pt.x_color.data, 0, 1)
                pt.x_alpha.data = torch.clamp(pt.x_alpha.data, 0, 1)

                pt._forward_pass()
                pt._drawing_step_states()
                pt._backward_x()

                pt.x_ctt.data = torch.clamp(pt.x_ctt.data, 0.1, 1 - 0.1)
                pt.x_color.data = torch.clamp(pt.x_color.data, 0, 1)
                pt.x_alpha.data = torch.clamp(pt.x_alpha.data, 0, 1)

                pt.optimizer_x.step()
                pt.step_id += 1

        v = pt._normalize_strokes(pt.x)
        v = pt._shuffle_strokes_and_reshape(v)
        PARAMS = np.concatenate([PARAMS, v], axis=1)
        CANVAS_tmp = pt._render(PARAMS, save_jpgs=False, save_video=False)
        CANVAS_tmp = utils.img2patches(CANVAS_tmp, pt.m_grid + 1, pt.net_G.out_size).to(device)

    pt._save_stroke_params(PARAMS)
    final_rendered_image = pt._render(PARAMS, save_jpgs=False, save_video=True)

    return final_rendered_image

@runway.command('translate', inputs={'source_imgs': runway.image(description='input image to be translated'), 'Strokes': runway.number(min=100, max=700, default=100,description='number of strokes')
}, outputs={'image': runway.image(description='output image containing the translated result')})
def translate(learn, inputs):
    os.makedirs('images', exist_ok=True)
    inputs['source_imgs'].save('images/temp.jpg')
    paths = os.path.join('images','temp.jpg')
    args.img_path = paths
    args.max_m_strokes = inputs['Strokes']
    pt = ProgressivePainter(args=args)
    final_rendered_image = optimize_x(pt)
    formatted = (final_rendered_image * 255 / np.max(final_rendered_image)).astype('uint8')
    img = Image.fromarray(formatted)
    return img


if __name__ == '__main__':
    runway.run(port=8889)
