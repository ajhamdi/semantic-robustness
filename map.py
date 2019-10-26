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
from true_dict import TRUE_DICT
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
models_dir = os.path.join(data_dir, "models")
dataset_root = os.path.join(data_dir, "scale")
scores_dir = os.path.join(data_dir, "results", "scores")
sys.path.append(data_dir)

def main(all_shapes, class_nb, object_nb, override, precisions, custom_shapes):
    print(all_shapes, class_nb, object_nb, override, precisions, custom_shapes)
    # with open(tags_filename, 'r') as fobj:
    #     tags = json.load(fobj)
    # print(bool(all_points),override,reduced)
    all_shapes = bool(all_shapes)
    override = bool(override)
    custom_shapes = bool(custom_shapes)
    data_dir = os.getcwd()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("Warninign ....... no Cuda !!")
    object_list = ['aeroplane', "bathtub", 'bench', 'bottle', 'chair', "cup",
                   "piano", 'rifle', 'vase', "toilet"]  # ["teapot","truck","boat","tv"]
    obj_class_list = [404, 435, 703, 898, 559, 968,
                      579, 413, 883, 861]  # [849,586,606,549]
    # define the deep model
    resnet = models.resnet50(pretrained=True).eval().to(device)
    # resnet = nn.DataParallel(resnet)
    alexnet = models.alexnet(pretrained=True).eval().to(device)
    # alexnet = nn.DataParallel(alexnet)
    vgg = models.vgg11_bn(pretrained=True).eval().to(device)
    # vgg = nn.DataParallel(vgg)
    incept = models.inception_v3(pretrained=True).eval().to(device)
    # incept = nn.DataParallel(incept)
    models_dicts = {"ResNet50": resnet, "AlexNet": alexnet,
                    "VGG": vgg, "Inceptionv3": incept}
    setup = {}
    setup["a"] = [0, -10]
    setup["b"] = [360, 90]
    setup["precisions"] = [precisions, precisions]

    # if network == "resnet":
    #     network_model =  models.resnet50(pretrained=True).eval().to(device)  ; network_name = "ResNet50"
    # elif network == "incept":
    #     network_model =  models.inception_v3(pretrained=True).eval().to(device)  ; network_name = "Inceptionv3"
    # elif network == "vgg":
    #     network_model =  models.vgg11_bn(pretrained=True).eval().to(device)  ; network_name = "VGG"
    # elif network == "alexnet":
    #     network_model =  models.alexnet(pretrained=True).eval().to(device)  ; network_name = "AlexNet"
    # else:
    #     print("NO available network with this name ... Sorry !")
    #     raise Exception("NO NETWORK")
    special_list = [1, 2, 3, 7, 8, 9]
    if all_shapes:
        shapes_dir = os.path.join(data_dir, "scale", object_list[class_nb])
        shapes_list = list(glob.glob(shapes_dir+"/*"))
        for k, v in models_dicts.items():
            print("doing ...", k)
            for myobject_nb in range(len(shapes_list)):
                print("doing shape ...", myobject_nb)
                map_network(v, k, class_nb, myobject_nb, obj_class_list=obj_class_list,
                            object_list=object_list, setup=setup, data_dir=data_dir, override=override, device=device)
    elif custom_shapes:
        for k, v in models_dicts.items():
            print("doing ...", k)
            for myobject_nb in special_list:
                print("doing shape ...", myobject_nb)
                map_network(v, k, class_nb, myobject_nb, obj_class_list=obj_class_list,
                            object_list=object_list, setup=setup, data_dir=data_dir, override=override, device=device)
    else:
        for k, v in models_dicts.items():
            print("doing ...", k)
            print("doing shape ...", object_nb)
            map_network(v, k, class_nb, object_nb, obj_class_list=obj_class_list,
                        object_list=object_list, setup=setup, data_dir=data_dir, override=override, device=device)


if __name__ == '__main__':
    parser = ArgumentParser(description='extract values of summary tags from tensorboard events files and dump it in a csv file',
                            formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument('-a', '--all_shapes', required=True, type=int,
                        help='perform all the points of all experiments')
    parser.add_argument('-c', '--class_nb', default=1, type=int,
                        help='the claass number of the experoemtn being performed')
    parser.add_argument('-o', '--object_nb', default=1, type=int,
                        help='shape number being processes ')
    parser.add_argument('-r', '--override', default=0, type=int,
                        help='overriding existing dictionary files and writing the new ones')
    parser.add_argument('-p', '--precisions', default=800, type=int,
                        help='number of precisions for the experiment')
    parser.add_argument('-s', '--custom_shapes', default=0, type=int,
                        help='is it custom shape list ')

    args = parser.parse_args()
    # numeric_level = getattr(logging, args.loglevel.upper(), None)
    # if not isinstance(numeric_level, int):
    #     raise ValueError('Invalid log level: %s' % args.loglevel)
    # logging.basicConfig(format='%(asctime)s:%(levelname)s: %(message)s',
    #                 level=numeric_level)
    # delattr(args, 'loglevel')

    main(**vars(args))
