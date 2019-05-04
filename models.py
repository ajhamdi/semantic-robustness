import sys
import numpy as np
from numpy.linalg import inv
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from scipy import stats
from torch.optim import lr_scheduler
import neural_renderer as nr
from interval import interval
from utils import int2binarray
# aLL torch models are put here


class renderer_model_2(nn.Module):
    def __init__(self, network_model, vertices, faces, camera_distance, elevation, azimuth, image_size, device=None):
        super(renderer_model_2, self).__init__()
        self.register_buffer('vertices', vertices)
        self.register_buffer('faces', faces)

        # create textures
        texture_size = 2
        textures = torch.ones(self.faces.shape[0], self.faces.shape[1],
                              texture_size, texture_size, texture_size, 3, dtype=torch.float32)
        self.register_buffer('textures', textures)
        self.device = device

        # define the DNN model as part of the model of the renderer
        self.network_model = network_model

        self.register_buffer('camera_distance', torch.from_numpy(
            np.array(camera_distance)).float().unsqueeze_(0))

        # camera parameters
#         self.camera_position = nn.Parameter(torch.from_numpy(np.array([6, 10, -14], dtype=np.float32)))
        self.azimuth = nn.Parameter(torch.from_numpy(
            np.array(azimuth)).float().unsqueeze_(0))  # if bach remove unsqueeze
        self.elevation = nn.Parameter(torch.from_numpy(
            np.array(elevation)).float().unsqueeze_(0))  # if anthc remove unsqueeze

        # setup renderer
        renderer = nr.Renderer(camera_mode='look_at', image_size=image_size)
#         renderer.eye = self.camera_position
        self.renderer = renderer

    def forward(self, eval_point):
        self.azimuth.data.set_(torch.from_numpy(
            np.array(eval_point[0])).float().to(self.device))
        self.elevation.data.set_(torch.from_numpy(
            np.array(eval_point[1])).float().to(self.device))
        self.renderer.eye = nr.get_points_from_angles(
            self.camera_distance, self.elevation, self.azimuth)
        images = self.renderer(self.vertices, self.faces, self.textures)
#         image = images.detach().cpu().numpy()[0].transpose((1, 2, 0))  # [image_size, image_size, RGB]
#         imsave("/tmp/aa.png",(255*image).astype(np.uint8))
        prop = torch.functional.F.softmax(self.network_model(images), dim=1)
        return prop


class renderer_model(nn.Module):
    def __init__(self, network_model, vertices, faces, camera_distance, elevation, azimuth, image_size, device=None):
        super(renderer_model, self).__init__()
        self.register_buffer('vertices', vertices)
        self.register_buffer('faces', faces)

        # create textures
        texture_size = 2
        textures = torch.ones(self.faces.shape[0], self.faces.shape[1],
                              texture_size, texture_size, texture_size, 3, dtype=torch.float32)
        self.register_buffer('textures', textures)
        self.device = device

        # define the DNN model as part of the model of the renderer
        self.network_model = network_model

        self.register_buffer('camera_distance', torch.from_numpy(
            np.array(camera_distance)).float().unsqueeze_(0))
        self.register_buffer('elevation', torch.from_numpy(
            np.array(elevation)).float().unsqueeze_(0))

        # camera parameters
#         self.camera_position = nn.Parameter(torch.from_numpy(np.array([6, 10, -14], dtype=np.float32)))
        self.azimuth = nn.Parameter(torch.from_numpy(
            np.array(azimuth)).float().unsqueeze_(0))  # if anthc remove unsqueeze

        # setup renderer
        renderer = nr.Renderer(camera_mode='look_at', image_size=image_size)
#         renderer.eye = self.camera_position
        self.renderer = renderer

    def forward(self, azimuth):
        self.azimuth.data.set_(torch.from_numpy(
            np.array(azimuth)).float().to(self.device))
        self.renderer.eye = nr.get_points_from_angles(
            self.camera_distance, self.elevation, self.azimuth)
        images = self.renderer(self.vertices, self.faces, self.textures)
#         image = images.detach().cpu().numpy()[0].transpose((1, 2, 0))  # [image_size, image_size, RGB]
#         imsave("/tmp/aa.png",(255*image).astype(np.uint8))
        prop = torch.functional.F.softmax(self.network_model(images), dim=1)

        return prop

    from interval import interval


class ndinterval():
    def __init__(self, a, b):
        if len(a) != len(b):
            print("not valid n-dim interval")
        elif "interval" not in sys.modules:
            print("pip install pyinterval module first !!")
        else:
            self.n = len(a)
            self.two_to_n = 2**self.n
            self.mask = np.array([int2binarray(x, n_bits=self.n)
                                  for x in range(self.two_to_n)]).T
            self.mask_c = np.logical_not(self.mask).astype(np.int)
            self.a = np.array(a)
            self.b = np.array(b)
            self.update()
            self.old_a = self.a
            self.old_b = self.b

    def step_size(self):
        return np.sum(self.a - self.old_a) + np.sum(self.b - self.old_b)

    def size(self):
        return np.prod(self.r)

    def update(self):
        self.region = [interval([self.a[ii], self.b[ii]])
                       for ii in range(self.n)]
        self.r = np.array([x[0][1] - x[0][0] for x in self.region])
        self.R = inv(np.diag(self.r))
        self.corners_matrix = np.matmul(np.ones([self.two_to_n, 1]), np.expand_dims(
            self.a, axis=0)) + self.mask.T * np.matmul(np.ones([self.two_to_n, 1]), np.expand_dims(self.r, axis=0))
        self.corners_set = [self.corners_matrix[ii, ::]
                            for ii in range(self.two_to_n)]

    def size_normalized(self):
        return self.size() / self.two_to_n

    def __str__(self):
        return str(self.region)

    def __call__(self, a, b):
        self.old_a = self.a.copy()
        self.old_b = self.b.copy()
        self.a = a.copy()
        self.b = b.copy()
        self.update()

    def __and__(self, interval2):
        return [x & y for x, y in zip(self.region, interval2.region)]

    def __or__(self, interval2):
        return [x | y for x, y in zip(self.region, interval2.region)]


