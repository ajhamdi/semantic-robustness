import sys
import os
import numpy as np
from numpy.linalg import inv
import torch
import glob
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from scipy import stats
from torch.optim import lr_scheduler
import neural_renderer as nr
from interval import interval  # pip install pyinterval module first
import torch.utils.data as data
import torch.nn.functional as F
from utils import int2binarray, listdir_nohidden
# aLL torch models are put here


type_to_index_map = {
    'aeroplane': 0, "bathtub": 1, 'bench': 2, 'bottle': 3, 'chair': 4,
    "cup": 5, "piano": 6, 'rifle': 7, 'vase': 8, "toilet": 9}

index_to_type_map = dict([[v, k] for k, v in type_to_index_map.items()])


def list_features_shapenet_classes(class_dir, epoch=160):
    objec_nb_list = []
    shapes_list = list(
        glob.glob(class_dir+"/*/models/features_{}.npz".format(epoch)))
    mesh_file_list = list(glob.glob(class_dir+"/*"))
    mesh_file_list = [os.path.join(
        x, "models", "features_{}.npz".format(epoch)) for x in mesh_file_list]
    for ii, sh in enumerate(mesh_file_list):
        if sh in shapes_list:
            objec_nb_list.append(ii)
    return zip(objec_nb_list, shapes_list)


class renderer_model_n(nn.Module):
    def __init__(self, network_model, vertices, faces, camera_distance, elevation, azimuth, image_size, device=None, light_direction=[0, 1, 0], light_intensity_directional=0.5,texture_color=[1,1,1], n=10):
        super(renderer_model_n, self).__init__()
        self.register_buffer('vertices', vertices.to(device))
        self.register_buffer('faces', faces.to(device))
        # self.vertices = nn.Parameter(vertices.float()).to(device)
        # self.faces = nn.Parameter(faces.float()).to(device)
        # self.register_buffer('light_direction', light_direction)
        # self.register_buffer('light_intensity_directional',
        #                      light_intensity_directional)
        # self.register_buffer('texture_color', texture_color)
        self.network_model = network_model
        # create textures
        texture_size = 2
        self.device = device
        self.n = n

        if n == 1:
            textures = torch.ones(self.faces.shape[0], self.faces.shape[1],texture_size, texture_size, texture_size, 3, dtype=torch.float32)
            self.register_buffer('textures', textures)
            self.register_buffer('camera_distance', torch.from_numpy(np.array(camera_distance)).float().unsqueeze_(0))
            self.azimuth = nn.Parameter(torch.from_numpy(np.array(azimuth)).float().unsqueeze_(0))  # if bach remove unsqueeze
            self.register_buffer('elevation', torch.from_numpy(np.array(elevation)).float().unsqueeze_(0))
            self.renderer = nr.Renderer(camera_mode='look_at', image_size=image_size)


        elif n == 2:
            textures = torch.ones(self.faces.shape[0], self.faces.shape[1],texture_size, texture_size, texture_size, 3, dtype=torch.float32)
            self.register_buffer('textures', textures)
            self.register_buffer('camera_distance', torch.from_numpy(np.array(camera_distance)).float().unsqueeze_(0))
            self.azimuth = nn.Parameter(torch.from_numpy(np.array(azimuth)).float().unsqueeze_(0))  # if bach remove unsqueeze
            self.elevation = nn.Parameter(torch.from_numpy(np.array(elevation)).float().unsqueeze_(0))  # if anthc remove unsqueeze
            self.renderer = nr.Renderer(camera_mode='look_at', image_size=image_size)


        elif n == 3:
            textures = torch.ones(self.faces.shape[0], self.faces.shape[1],
                                  texture_size, texture_size, texture_size, 3, dtype=torch.float32)
            self.register_buffer('textures', textures)
            self.camera_distance = nn.Parameter(torch.from_numpy(
                np.array(camera_distance)).float().unsqueeze_(0))
            self.azimuth = nn.Parameter(torch.from_numpy(np.array(azimuth)).float().unsqueeze_(0))  # if bach remove unsqueeze
            self.elevation = nn.Parameter(torch.from_numpy(np.array(elevation)).float().unsqueeze_(0))  # if anthc remove unsqueeze
            self.renderer = nr.Renderer(camera_mode='look_at', image_size=image_size)


        elif n == 6 :
            self.light_direction = nn.Parameter(nn.functional.normalize(
                torch.FloatTensor(light_direction), dim=0, eps=1e-16).to(self.device))
            textures = torch.ones(self.faces.shape[0], self.faces.shape[1],
                                  texture_size, texture_size, texture_size, 3, dtype=torch.float32)
            self.register_buffer('textures', textures)
            self.camera_distance = nn.Parameter(torch.from_numpy(
                np.array(camera_distance)).float().unsqueeze_(0))
            self.azimuth = nn.Parameter(torch.from_numpy(
                np.array(azimuth)).float().unsqueeze_(0))  # if bach remove unsqueeze
            self.elevation = nn.Parameter(torch.from_numpy(np.array(elevation)).float().unsqueeze_(0))  # if anthc remove unsqueeze
            self.renderer = nr.Renderer(camera_mode='look_at', image_size=image_size, light_direction=self.light_direction)

        elif n == 10:
            self.light_intensity_directional = nn.Parameter(torch.from_numpy(
                np.array(light_intensity_directional)).float().unsqueeze_(0).to(self.device))
            self.light_direction = nn.Parameter(nn.functional.normalize(
                torch.FloatTensor(light_direction), dim=0, eps=1e-16).to(self.device))
            self.texture_color = nn.Parameter(torch.torch.from_numpy(
                np.array(texture_color)).float().to(self.device))
            textures = torch.ones(self.faces.shape[0], self.faces.shape[1],
                                  texture_size, texture_size, texture_size, 3, dtype=torch.float32).to(self.device)
            self.textures = self.texture_color * nn.Parameter(textures)
            self.camera_distance = nn.Parameter(torch.from_numpy(
                np.array(camera_distance)).float().unsqueeze_(0))
            self.azimuth = nn.Parameter(torch.from_numpy(
                np.array(azimuth)).float().unsqueeze_(0))  # if bach remove unsqueeze
            self.elevation = nn.Parameter(torch.from_numpy(
                np.array(elevation)).float().unsqueeze_(0))  # if anthc remove unsqueeze
            self.renderer = nr.Renderer(camera_mode='look_at', image_size=image_size,
                                        light_direction=self.light_direction, light_intensity_directional=self.light_intensity_directional)

        elif n == 0:
            self.texture_color = torch.torch.from_numpy(np.array(texture_color)).float().to(self.device)
            textures =  torch.ones(self.faces.shape[0], self.faces.shape[1],
                                  texture_size, texture_size, texture_size, 3, dtype=torch.float32).to(self.device)
            self.textures = nn.Parameter(self.texture_color * textures)
            self.register_buffer('light_intensity_directional', torch.from_numpy(
                np.array(light_intensity_directional)).float().to(self.device))
            self.register_buffer('light_direction', torch.from_numpy(
                np.array(light_direction)).float().to(self.device))
            self.register_buffer('camera_distance', torch.from_numpy(np.array(camera_distance)).float().unsqueeze_(0))
            self.register_buffer('azimuth', torch.from_numpy(
                np.array(azimuth)).float().unsqueeze_(0))  # if bach remove unsqueeze
            self.register_buffer('elevation', torch.from_numpy(np.array(elevation)).float().unsqueeze_(0))
            self.renderer = nr.Renderer(camera_mode='look_at',
             image_size=image_size,light_direction=self.light_direction,
              light_intensity_directional=self.light_intensity_directional)
            
    def forward(self):
        # if self.n == 0:
        #     self.textures.data.set_(torch.ones(self.faces.shape[0], self.faces.shape[1],
        #                                        texture_size, texture_size, texture_size, 3, dtype=torch.float32).float().unsqueeze_(0).to(self.device))
        # if self.n >= 1:
        #     self.azimuth.data.set_(torch.from_numpy(np.array(eval_point[0])).float().to(self.device))
        # if self.n >= 2:
        #     self.elevation.data.set_(torch.from_numpy(np.array(eval_point[1])).float().to(self.device))
        # if self.n >= 3:
        #     self.camera_distance.data.set_(torch.from_numpy(np.array(eval_point[2])).float().to(self.device))
        # if self.n >= 6:
        #     self.light_direction.data.set_(torch.from_numpy(
        #         np.array(eval_point[3:6])).float().to(self.device))
        # if self.n >= 7:
        #     self.camera_distance.data.set_(torch.from_numpy(np.array(eval_point[2])).float().to(self.device))

        self.renderer.eye = nr.get_points_from_angles(
            self.camera_distance, self.elevation, self.azimuth)
        images = self.renderer(self.vertices, self.faces, self.textures)[0]
#         image = images.detach().cpu().numpy()[0].transpose((1, 2, 0))  # [image_size, image_size, RGB]
#         imsave("/tmp/aa.png",(255*image).astype(np.uint8))
        prop = torch.functional.F.softmax(self.network_model(images), dim=1)
        return prop

    def render(self):
        self.renderer.eye = nr.get_points_from_angles(
            self.camera_distance, self.elevation, self.azimuth)
        images = self.renderer(self.vertices, self.faces, self.textures)[0]
        return images.detach().cpu().numpy()[0].transpose((1, 2, 0))


    def forward_(self,obj_class):
        prop = self.forward()
        return prop[0, obj_class].detach().cpu().numpy()

    def backward_(self, obj_class):
        prop = self.forward()
        labels = torch.tensor([obj_class]).to(self.device)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(prop, labels)
        self.zero_grad()
        loss.backward(retain_graph=False)
        if self.n ==0:
            return self.textures.grad.cpu().numpy()
            # return self.texture_color.grad.cpu().numpy()
        else :
            grad_dict = {}
            grad_list = []
            if self.n >=1 :
                grad_list.append(self.azimuth.grad.cpu().numpy().item())
                grad_dict["azimuth"] = self.azimuth.grad.cpu().numpy().item()
            if self.n >= 2:
                grad_list.append(self.elevation.grad.cpu().numpy().item())
                grad_dict["elevation"] = self.elevation.grad.cpu().numpy().item()

            if self.n >= 3:
                grad_list.append(
                    self.camera_distance.grad.cpu().numpy().item())
                grad_dict["camera_distance"] = self.camera_distance.grad.cpu().numpy().item()

            if self.n >= 6:
                light_direction_grad = self.light_direction.grad.cpu().numpy()
                grad_list.extend([*light_direction_grad])
                grad_dict["light_direction"] = np.array([*light_direction_grad])
                grad_dict["light_direction_x"] = [*light_direction_grad][0]
                grad_dict["light_direction_y"] = [*light_direction_grad][1]
                grad_dict["light_direction_z"] = [*light_direction_grad][2]
            if self.n >= 7:
                light_intensity_directional_grad = self.light_intensity_directional.grad.cpu().numpy().item()
                grad_list.append(light_intensity_directional_grad)
                grad_dict["light_intensity_directional"] = self.light_intensity_directional.grad.cpu(
                ).numpy().item()

                # intensity_ambient_grad = self.intensity_ambient.grad.cpu().numpy()
                # grad_list.append(intensity_ambient_grad)
            if self.n >=10:
                color_texture_grad = self.texture_color.grad.cpu().numpy()
                grad_list.extend([*color_texture_grad])
                grad_dict["texture_color"] = np.array([*color_texture_grad])
                grad_dict["texture_color_R"] = [*color_texture_grad][0]
                grad_dict["texture_color_G"] = [*color_texture_grad][1]
                grad_dict["texture_color_B"] = [*color_texture_grad][2]

            return grad_dict #,grad_list[0:self.n]

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
        images = self.renderer(self.vertices, self.faces, self.textures)[0]
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
        images = self.renderer(self.vertices, self.faces, self.textures)[0]
#         image = images.detach().cpu().numpy()[0].transpose((1, 2, 0))  # [image_size, image_size, RGB]
#         imsave("/tmp/aa.png",(255*image).astype(np.uint8))
        prop = torch.functional.F.softmax(self.network_model(images), dim=1)

        return prop


class renderer_model_1(nn.Module):
    def __init__(self, network_model, vertices, faces, camera_distance, elevation, azimuth, image_size, device=None):
        super(renderer_model_1, self).__init__()
        self.vertices = vertices.to(device)
        self.faces = faces.to(device)

        # create textures
        texture_size = 2
        self.textures = torch.ones(self.faces.shape[0], self.faces.shape[1],texture_size, texture_size, texture_size, 3, dtype=torch.float32).requires_grad_(requires_grad=False).to(device)
        # self.register_buffer('textures', textures)
        self.device = device

        # define the DNN model as part of the model of the renderer
        self.network_model = network_model

        self.camera_distance = torch.from_numpy(np.array(camera_distance)).float(
        ).unsqueeze_(0).requires_grad_(requires_grad=False).to(self.device)
        self.elevation = torch.from_numpy(np.array(elevation)).float(
        ).unsqueeze_(0).requires_grad_(requires_grad=False).to(self.device)


        # camera parameters
#         self.camera_position = nn.Parameter(torch.from_numpy(np.array([6, 10, -14], dtype=np.float32)))
        self.azimuth = nn.Parameter(torch.from_numpy(np.array(azimuth)).float().unsqueeze_(0))
        renderer = nr.Renderer(camera_mode='look_at',
                               image_size=image_size)
        self.renderer = renderer

        # setup renderer
        
#         renderer.eye = self.camera_position

    def forward(self):
        # self.azimuth.data.set_(torch.from_numpy(
        #     np.array(azimuth)).float().to(self.device))


        self.renderer.eye = nr.get_points_from_angles(self.camera_distance, self.elevation, self.azimuth)
        images = self.renderer(self.vertices, self.faces, self.textures)[0]
#         image = images.detach().cpu().numpy()[0].transpose((1, 2, 0))  # [image_size, image_size, RGB]
#         imsave("/tmp/aa.png",(255*image).astype(np.uint8))
        prop = torch.functional.F.softmax(self.network_model(images), dim=1)

        return prop

    def backward_(self, obj_class):
        with torch.autograd.set_detect_anomaly(True):
            prop = self.forward()
            # torch.from_numpy(np.tile(np.eye(1000)[obj_class],(1,prop.size()[0]))).float().to(device)
            labels = torch.tensor([obj_class]).to(self.device)
            criterion = nn.CrossEntropyLoss()
            loss = criterion(prop, labels)
            self.zero_grad()
            loss.backward(retain_graph=False)
        return self.azimuth.grad.cpu().numpy()

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


class ShapeFeatures(data.Dataset):
    def __init__(self, model=None, network_name=None, classification=False, root_dir=None, part="train"):
        self.data_dir = root_dir
        self.root =  os.path.join(self.data_dir, "scale")
        self.part = part
        self.pt_train_max = 6
        self.pt_test_max = 4
        self.model = model
        self.classification = classification
        self.CALSS_THRESHOLD = 0.05
        self.network_name = network_name
        print("##########", self.root)
        self.data = []
        for class_name in listdir_nohidden(self.root):
            type_index = type_to_index_map[class_name]
            type_root = os.path.join(os.path.join(self.root, class_name))
            # for filename in os.listdir(type_root):
            #     if filename.endswith('.npz'):
            #         self.data.append((os.path.join(type_root, filename), type_index))
            for object_nb, filename in list_features_shapenet_classes(type_root, epoch=410):
                if filename.endswith('.npz'):
                    if self.part == "train":
                        self.data.extend(self.pt_train_max *
                                         [(filename, type_index, object_nb)])
                    else:
                        self.data.extend(self.pt_test_max *
                                         [(filename, type_index, object_nb)])
#                     self.data.append((filename, type_index,object_nb))

    def __getitem__(self, i):
        path, class_nbr, object_nb = self.data[i]
#         mesh = trimesh.load_mesh(path, file_type='obj', resolver=None)
#         while len(mesh.faces)> self.FACE_THRESHOLD :
#             path, class_nbr = self.data[np.random.randint(0,len(self.data))]
#             mesh = trimesh.load_mesh(path, file_type='obj', resolver=None)

        obj_file = os.path.join(os.path.split(path)[0], "model_normalized.obj")
        shape_feature = np.load(path)['features']

        if self.part == "train":
            optim_dict = torch.load(os.path.join(self.data_dir, "checkpoint", self.network_name, str(
                class_nbr), str(object_nb), "optim_t.pt"))
            pt_idx = i % self.pt_train_max
        else:
            optim_dict = torch.load(os.path.join(
                self.data_dir, "checkpoint", self.network_name, str(class_nbr), str(object_nb), "optim.pt"))
            pt_idx = i % self.pt_test_max
#         self.pt_idx = np.random.choice(range(len(optim_dict['initial_point'])))
        initial_point = optim_dict['initial_point'][pt_idx]
        srvr = optim_dict["OIR_B"]["regions"][pt_idx].size()/9000
        if self.classification:
            srvr = 3*srvr
            srvr = np.int((srvr > self.CALSS_THRESHOLD))
#         print(srvr)


#         # data augmentation
#         if self.augment_data and self.part == 'train':
#             sigma, clip = 0.01, 0.05
#             jittered_data = np.clip(sigma * np.random.randn(*face[:, :12].shape), -1 * clip, clip)
#             face = np.concatenate((face[:, :12] + jittered_data, face[:, 12:]), 1)

        # to tensor
        shape_feature = torch.from_numpy(shape_feature).float().squeeze()
        initial_point = torch.from_numpy(initial_point).float().squeeze()
#         if self.classification:
#             srvr = torch.from_numpy(np.array(srvr)).long()
#         else :
        srvr = torch.from_numpy(np.array(srvr)).float()

        return shape_feature, srvr, initial_point, obj_file, class_nbr

    def __len__(self):
        return len(self.data)


def fix_regions(ndregion, smalles_a, largest_b, eps=0.00001):
    np.clip(ndregion.b, smalles_a, largest_b, out=ndregion.b)
    np.clip(ndregion.a, smalles_a, largest_b, out=ndregion.a)
    problems = np.where(np.abs(ndregion.b - ndregion.a) < eps)[0]
    for problem in problems:
        ndregion.b[problem] = ndregion.a[problem] + eps
    ndregion.update()
class SRVR_Classifier(torch.nn.Module):
    def __init__(self, n_feature, n_output, depth):
        super(SRVR_Classifier, self).__init__()
        self.depth = depth-4
        self.hidden1 = torch.nn.Linear(n_feature, 1000)   # hidden layer
        self.hidden = torch.nn.Linear(500, 500)   # hidden layer
        self.hidden2 = torch.nn.Linear(1000, 500)
        self.hidden3 = torch.nn.Linear(500, 50)
        self.predict = torch.nn.Linear(50, n_output)   # output layer

    def forward(self, x):
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))  # activation function for hidden layer
        for ii in range(self.depth):
            x = F.relu(self.hidden(x))
        x = F.relu(self.hidden3(x))
        x = self.predict(x)             # linear output
        return F.sigmoid(x)
