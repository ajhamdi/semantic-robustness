from __future__ import division, print_function, absolute_import

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import logging
import pandas as pd
import json
from ops import *
import torch
from torchvision import datasets, models, transforms
from utils import *
from models import *
from scale.true_dict import TRUE_DICT
import imageio

# import tensorflow as tf
import glob
import copy
import pandas as pd
# from tqdm import tqdm
import numpy as np


object_list = ['aeroplane', "bathtub", 'bench', 'bottle', 'chair', "cup",
               "piano", 'rifle', 'vase', "toilet"]  # ["teapot","truck","boat","tv"]
# this is the list tht represent the 10 classes in ImageNEt class labels
obj_class_list = [404, 435, 703, 898, 559, 968, 579, 413, 883, 861]
camera_distance = 2.732
image_size = 224
left_limit = np.array([0, -10])
right_limit = np.array([360, 90])
data_dir = os.getcwd()
sys.path.append(data_dir)



def main(network, gpu, class_nb,object_nb,override):

    device = torch.device("cuda:"+str(gpu) if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(gpu)

    network_name, network_model = get_trained_network(network, device)
    print("network: {}  gpu: {}  class: {}  object: {} , override: {}".format(
        network_name, gpu, class_nb, object_nb, override))

    models_dicts = {network_name: network_model}
    all_initial_points = [np.array([130,  30]), np.array(
        [50, 20]), np.array([200,  15]), np.array([310,  50])]

    optim_SRVR(network_name, class_nb, object_nb,
               all_initial_points,override, models_dicts, device)
    

def optim_SRVR(network_name, class_nb, object_nb, all_initial_points,override, models_dicts, device):
    shapes_dir = os.path.join(data_dir, "scale", object_list[class_nb])
    mesh_file = os.path.join(shapes_dir, TRUE_DICT[str(
        class_nb)][str(object_nb)], "models", "model_normalized.obj")
    test_optimization(models_dicts[network_name], network_name, class_nb, object_nb, all_initial_points,
                      obj_class_list, mesh_file=mesh_file, setup=None, data_dir=data_dir, override=override, device=device)

if __name__ == '__main__':
    parser = ArgumentParser(description='parse args',
                            formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument('-n', '--network', required=True, type=str, choices=['incept', 'alexnet', 'vgg',"resnet"],
                        help='network type of the experiments')
    parser.add_argument('-g', '--gpu', required=True, type=int,
                        help='the GPU number in which the exp perfoprmed ')
    parser.add_argument('-s', '--class_nb', default=0, type=int,
                        help='number of the class used for the experiment [0-9]')
    parser.add_argument('-j', '--object_nb', default=0, type=int,
                        help='number of the 3D object used for the experiment [0-9]')
    parser.add_argument('-o', '--override', dest='override',action='store_true', help='override exisisting results')


    args = parser.parse_args()

    main(**vars(args))
