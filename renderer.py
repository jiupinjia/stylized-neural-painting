import numpy as np
import cv2
import random
import utils

import matplotlib.pyplot as plt


def _random_floats(low, high, size):
    return [random.uniform(low, high) for _ in range(size)]


def _normalize(x, width):
    return (int)(x * (width - 1) + 0.5)


class Renderer():

    def __init__(self, renderer='oilpaintbrush', CANVAS_WIDTH=128, train=False, canvas_color='black'):

        self.CANVAS_WIDTH = CANVAS_WIDTH
        self.renderer = renderer
        self.stroke_params = None
        self.canvas_color = canvas_color

        self.canvas = None
        self.create_empty_canvas()

        self.train = train

        if self.renderer in ['markerpen']:
            self.d = 12 # x0, y0, x1, y1, x2, y2, radius0, radius2, R, G, B, A
            self.d_shape = 8
            self.d_color = 3
            self.d_alpha = 1
        elif self.renderer in ['watercolor']:
            self.d = 15 # x0, y0, x1, y1, x2, y2, radius0, radius2, R0, G0, B0, R2, G2, B2, A
            self.d_shape = 8
            self.d_color = 6
            self.d_alpha = 1
        elif self.renderer in ['oilpaintbrush']:
            self.d = 12 # xc, yc, w, h, theta, R0, G0, B0, R2, G2, B2, A
            self.d_shape = 5
            self.d_color = 6
            self.d_alpha = 1
            self.brush_small_vertical = cv2.imread(
                r'./brushes/brush_fromweb2_small_vertical.png', cv2.IMREAD_GRAYSCALE)
            self.brush_small_horizontal = cv2.imread(
                r'./brushes/brush_fromweb2_small_horizontal.png', cv2.IMREAD_GRAYSCALE)
            self.brush_large_vertical = cv2.imread(
                r'./brushes/brush_fromweb2_large_vertical.png', cv2.IMREAD_GRAYSCALE)
            self.brush_large_horizontal = cv2.imread(
                r'./brushes/brush_fromweb2_large_horizontal.png', cv2.IMREAD_GRAYSCALE)
        elif self.renderer in ['rectangle']:
            self.d = 9 # xc, yc, w, h, theta, R, G, B, A
            self.d_shape = 5
            self.d_color = 3
            self.d_alpha = 1
        else:
            raise NotImplementedError(
                'Wrong renderer name %s (choose one from [watercolor, markerpen, oilpaintbrush, rectangle] ...)'
                % self.renderer)

    def create_empty_canvas(self):
        if self.canvas_color == 'white':
            self.canvas = np.ones(
                [self.CANVAS_WIDTH, self.CANVAS_WIDTH, 3]).astype('float32')
        else:
            self.canvas = np.zeros(
                [self.CANVAS_WIDTH, self.CANVAS_WIDTH, 3]).astype('float32')


    def random_stroke_params(self):
        self.stroke_params = np.array(_random_floats(0, 1, self.d), dtype=np.float32)

    def random_stroke_params_sampler(self, err_map, img):

        map_h, map_w, c = img.shape

        err_map = cv2.resize(err_map, (self.CANVAS_WIDTH, self.CANVAS_WIDTH))
        err_map[err_map < 0] = 0
        if np.all((err_map == 0)):
            err_map = np.ones_like(err_map)
        err_map = err_map / (np.sum(err_map) + 1e-99)

        index = np.random.choice(range(err_map.size), size=1, p=err_map.ravel())[0]

        cy = (index // self.CANVAS_WIDTH) / self.CANVAS_WIDTH
        cx = (index % self.CANVAS_WIDTH) / self.CANVAS_WIDTH

        if self.renderer in ['markerpen']:
            # x0, y0, x1, y1, x2, y2, radius0, radius2, R, G, B, A
            x0, y0, x1, y1, x2, y2 = cx, cy, cx, cy, cx, cy
            x = [x0, y0, x1, y1, x2, y2]
            r = _random_floats(0.1, 0.5, 2)
            color = img[int(cy*map_h), int(cx*map_w), :].tolist()
            alpha = _random_floats(0.8, 0.98, 1)
            self.stroke_params = np.array(x + r + color + alpha, dtype=np.float32)
        elif self.renderer in ['watercolor']:
            # x0, y0, x1, y1, x2, y2, radius0, radius2, R0, G0, B0, R2, G2, B2, A
            x0, y0, x1, y1, x2, y2 = cx, cy, cx, cy, cx, cy
            x = [x0, y0, x1, y1, x2, y2]
            r = _random_floats(0.1, 0.5, 2)
            color = img[int(cy*map_h), int(cx*map_w), :].tolist()
            color = color + color
            alpha = _random_floats(0.98, 1.0, 1)
            self.stroke_params = np.array(x + r + color + alpha, dtype=np.float32)
        elif self.renderer in ['oilpaintbrush']:
            # xc, yc, w, h, theta, R0, G0, B0, R2, G2, B2, A
            x = [cx, cy]
            wh = _random_floats(0.1, 0.5, 2)
            theta = _random_floats(0, 1, 1)
            color = img[int(cy*map_h), int(cx*map_w), :].tolist()
            color = color + color
            alpha = _random_floats(0.98, 1.0, 1)
            self.stroke_params = np.array(x + wh + theta + color + alpha, dtype=np.float32)
        elif self.renderer in ['rectangle']:
            # xc, yc, w, h, theta, R, G, B, A
            x = [cx, cy]
            wh = _random_floats(0.1, 0.5, 2)
            theta = [0]
            color = img[int(cy*map_h), int(cx*map_w), :].tolist()
            alpha = _random_floats(0.8, 0.98, 1)
            self.stroke_params = np.array(x + wh + theta + color + alpha, dtype=np.float32)


    def check_stroke(self):
        r_ = 1.0
        if self.renderer in ['markerpen', 'watercolor']:
            r_ = max(self.stroke_params[6], self.stroke_params[7])
        elif self.renderer in ['oilpaintbrush']:
            r_ = max(self.stroke_params[2], self.stroke_params[3])
        elif self.renderer in ['rectangle']:
            r_ = max(self.stroke_params[2], self.stroke_params[3])
        if r_ > 0.025:
            return True
        else:
            return False


    def draw_stroke(self):

        if self.renderer == 'watercolor':
            return self._draw_watercolor()
        elif self.renderer == 'markerpen':
            return self._draw_markerpen()
        elif self.renderer == 'oilpaintbrush':
            return self._draw_oilpaintbrush()
        elif self.renderer == 'rectangle':
            return self._draw_rectangle()


    def _draw_watercolor(self):

        # x0, y0, x1, y1, x2, y2, radius0, radius2, R0, G0, B0, R2, G2, B2, A
        x0, y0, x1, y1, x2, y2, radius0, radius2 = self.stroke_params[0:8]
        R0, G0, B0, R2, G2, B2, ALPHA = self.stroke_params[8:]
        x1 = x0 + (x2 - x0) * x1
        y1 = y0 + (y2 - y0) * y1
        x0 = _normalize(x0, self.CANVAS_WIDTH)
        x1 = _normalize(x1, self.CANVAS_WIDTH)
        x2 = _normalize(x2, self.CANVAS_WIDTH)
        y0 = _normalize(y0, self.CANVAS_WIDTH)
        y1 = _normalize(y1, self.CANVAS_WIDTH)
        y2 = _normalize(y2, self.CANVAS_WIDTH)
        radius0 = (int)(1 + radius0 * self.CANVAS_WIDTH // 4)
        radius2 = (int)(1 + radius2 * self.CANVAS_WIDTH // 4)

        stroke_alpha_value = self.stroke_params[-1]

        self.foreground = np.zeros_like(
            self.canvas, dtype=np.uint8) # uint8 for antialiasing
        self.stroke_alpha_map = np.zeros_like(
            self.canvas, dtype=np.uint8) # uint8 for antialiasing

        alpha = (stroke_alpha_value * 255,
                 stroke_alpha_value * 255,
                 stroke_alpha_value * 255)
        tmp = 1. / 100
        for i in range(100):
            t = i * tmp
            x = (int)((1 - t) * (1 - t) * x0 + 2 * t * (1 - t) * x1 + t * t * x2)
            y = (int)((1 - t) * (1 - t) * y0 + 2 * t * (1 - t) * y1 + t * t * y2)
            radius = (int)((1 - t) * radius0 + t * radius2)
            color = ((1-t)*R0*255 + t*R2*255,
                     (1-t)*G0*255 + t*G2*255,
                     (1-t)*B0*255 + t*B2*255)
            cv2.circle(self.foreground, (x, y), radius, color, -1, lineType=cv2.LINE_AA)
            cv2.circle(self.stroke_alpha_map, (x, y), radius, alpha, -1, lineType=cv2.LINE_AA)

        if not self.train:
            self.foreground = cv2.dilate(self.foreground, np.ones([2, 2]))
            self.stroke_alpha_map = cv2.erode(self.stroke_alpha_map, np.ones([2, 2]))

        self.foreground = np.array(self.foreground, dtype=np.float32)/255.
        self.stroke_alpha_map = np.array(self.stroke_alpha_map, dtype=np.float32)/255.
        self.canvas = self._update_canvas()


    def _draw_rectangle(self):

        # xc, yc, w, h, theta, R, G, B, A
        x0, y0, w, h, theta = self.stroke_params[0:5]
        R0, G0, B0, ALPHA = self.stroke_params[5:]
        x0 = _normalize(x0, self.CANVAS_WIDTH)
        y0 = _normalize(y0, self.CANVAS_WIDTH)
        w = (int)(1 + w * self.CANVAS_WIDTH // 4)
        h = (int)(1 + h * self.CANVAS_WIDTH // 4)
        theta = np.pi*theta
        stroke_alpha_value = self.stroke_params[-1]

        self.foreground = np.zeros_like(
            self.canvas, dtype=np.uint8) # uint8 for antialiasing
        self.stroke_alpha_map = np.zeros_like(
            self.canvas, dtype=np.uint8) # uint8 for antialiasing

        color = (R0 * 255, G0 * 255, B0 * 255)
        alpha = (stroke_alpha_value * 255,
                 stroke_alpha_value * 255,
                 stroke_alpha_value * 255)
        ptc = (x0, y0)
        pt0 = utils.rotate_pt(pt=(x0 - w, y0 - h), rotate_center=ptc, theta=theta)
        pt1 = utils.rotate_pt(pt=(x0 + w, y0 - h), rotate_center=ptc, theta=theta)
        pt2 = utils.rotate_pt(pt=(x0 + w, y0 + h), rotate_center=ptc, theta=theta)
        pt3 = utils.rotate_pt(pt=(x0 - w, y0 + h), rotate_center=ptc, theta=theta)

        ppt = np.array([pt0, pt1, pt2, pt3], np.int32)
        ppt = ppt.reshape((-1, 1, 2))
        cv2.fillPoly(self.foreground, [ppt], color, lineType=cv2.LINE_AA)
        cv2.fillPoly(self.stroke_alpha_map, [ppt], alpha, lineType=cv2.LINE_AA)

        if not self.train:
            self.foreground = cv2.dilate(self.foreground, np.ones([2, 2]))
            self.stroke_alpha_map = cv2.erode(self.stroke_alpha_map, np.ones([2, 2]))

        self.foreground = np.array(self.foreground, dtype=np.float32)/255.
        self.stroke_alpha_map = np.array(self.stroke_alpha_map, dtype=np.float32)/255.
        self.canvas = self._update_canvas()


    def _draw_markerpen(self):

        # x0, y0, x1, y1, x2, y2, radius0, radius2, R, G, B, A
        x0, y0, x1, y1, x2, y2, radius, _ = self.stroke_params[0:8]
        R0, G0, B0, ALPHA = self.stroke_params[8:]
        x1 = x0 + (x2 - x0) * x1
        y1 = y0 + (y2 - y0) * y1
        x0 = _normalize(x0, self.CANVAS_WIDTH)
        x1 = _normalize(x1, self.CANVAS_WIDTH)
        x2 = _normalize(x2, self.CANVAS_WIDTH)
        y0 = _normalize(y0, self.CANVAS_WIDTH)
        y1 = _normalize(y1, self.CANVAS_WIDTH)
        y2 = _normalize(y2, self.CANVAS_WIDTH)
        radius = (int)(1 + radius * self.CANVAS_WIDTH // 4)

        stroke_alpha_value = self.stroke_params[-1]

        self.foreground = np.zeros_like(
            self.canvas, dtype=np.uint8) # uint8 for antialiasing
        self.stroke_alpha_map = np.zeros_like(
            self.canvas, dtype=np.uint8) # uint8 for antialiasing

        if abs(x0-x2) + abs(y0-y2) < 4: # too small, dont draw
            self.foreground = np.array(self.foreground, dtype=np.float32) / 255.
            self.stroke_alpha_map = np.array(self.stroke_alpha_map, dtype=np.float32) / 255.
            self.canvas = self._update_canvas()
            return

        color = (R0 * 255, G0 * 255, B0 * 255)
        alpha = (stroke_alpha_value * 255,
                 stroke_alpha_value * 255,
                 stroke_alpha_value * 255)
        tmp = 1. / 100
        for i in range(100):
            t = i * tmp
            x = (1 - t) * (1 - t) * x0 + 2 * t * (1 - t) * x1 + t * t * x2
            y = (1 - t) * (1 - t) * y0 + 2 * t * (1 - t) * y1 + t * t * y2

            ptc = (x, y)
            dx = 2 * (t - 1) * x0 + 2 * (1 - 2 * t) * x1 + 2 * t * x2
            dy = 2 * (t - 1) * y0 + 2 * (1 - 2 * t) * y1 + 2 * t * y2

            theta = np.arctan2(dx, dy) - np.pi/2
            pt0 = utils.rotate_pt(pt=(x - radius, y - radius), rotate_center=ptc, theta=theta)
            pt1 = utils.rotate_pt(pt=(x + radius, y - radius), rotate_center=ptc, theta=theta)
            pt2 = utils.rotate_pt(pt=(x + radius, y + radius), rotate_center=ptc, theta=theta)
            pt3 = utils.rotate_pt(pt=(x - radius, y + radius), rotate_center=ptc, theta=theta)
            ppt = np.array([pt0, pt1, pt2, pt3], np.int32)
            ppt = ppt.reshape((-1, 1, 2))
            cv2.fillPoly(self.foreground, [ppt], color, lineType=cv2.LINE_AA)
            cv2.fillPoly(self.stroke_alpha_map, [ppt], alpha, lineType=cv2.LINE_AA)

        if not self.train:
            self.foreground = cv2.dilate(self.foreground, np.ones([2, 2]))
            self.stroke_alpha_map = cv2.erode(self.stroke_alpha_map, np.ones([2, 2]))

        self.foreground = np.array(self.foreground, dtype=np.float32)/255.
        self.stroke_alpha_map = np.array(self.stroke_alpha_map, dtype=np.float32)/255.
        self.canvas = self._update_canvas()



    def _draw_oilpaintbrush(self):

        # xc, yc, w, h, theta, R0, G0, B0, R2, G2, B2, A
        x0, y0, w, h, theta = self.stroke_params[0:5]
        R0, G0, B0, R2, G2, B2, ALPHA = self.stroke_params[5:]
        x0 = _normalize(x0, self.CANVAS_WIDTH)
        y0 = _normalize(y0, self.CANVAS_WIDTH)
        w = (int)(1 + w * self.CANVAS_WIDTH)
        h = (int)(1 + h * self.CANVAS_WIDTH)
        theta = np.pi*theta

        if w * h / (self.CANVAS_WIDTH**2) > 0.1:
            if h > w:
                brush = self.brush_large_vertical
            else:
                brush = self.brush_large_horizontal
        else:
            if h > w:
                brush = self.brush_small_vertical
            else:
                brush = self.brush_small_horizontal
        self.foreground, self.stroke_alpha_map = utils.create_transformed_brush(
            brush, self.CANVAS_WIDTH, self.CANVAS_WIDTH,
            x0, y0, w, h, theta, R0, G0, B0, R2, G2, B2)

        if not self.train:
            self.foreground = cv2.dilate(self.foreground, np.ones([2, 2]))
            self.stroke_alpha_map = cv2.erode(self.stroke_alpha_map, np.ones([2, 2]))

        self.foreground = np.array(self.foreground, dtype=np.float32)/255.
        self.stroke_alpha_map = np.array(self.stroke_alpha_map, dtype=np.float32)/255.
        self.canvas = self._update_canvas()


    def _update_canvas(self):
        return self.foreground * self.stroke_alpha_map + \
               self.canvas * (1 - self.stroke_alpha_map)
