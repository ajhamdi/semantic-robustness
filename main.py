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
import glob
# from tqdm import tqdm
import numpy as np


def main(network, all_points, class_nb, object_nb, override, reduced, iterations, custom_points, custom_list):
    print(network, all_points, class_nb,
          object_nb, override, reduced, iterations)

    all_points = bool(all_points)
    override = bool(override)
    reduced = bool(reduced)
    custom_points = bool(custom_points)
    data_dir = os.getcwd()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("Warninign ....... no Cuda !!")
    object_list = ['aeroplane', "bathtub", 'bench', 'bottle', 'chair', "cup",
                   "piano", 'rifle', 'vase', "toilet"]  # ["teapot","truck","boat","tv"]
    # this is the list tht represent the 10 classes in ImageNEt class labels
    obj_class_list = [404, 435, 703, 898, 559, 968, 579, 413, 883, 861]
    # models_dicts = {"ResNet50":resnet , "AlexNet": alexnet , "VGG":vgg , "Inceptionv3":incept}
    setup = {}
    setup["learning_rate"] = 0.1
    setup["alpha"] = 0.05
    setup["beta"] = 0.0009
    setup["reg"] = 0.1
    setup["n_iterations"] = iterations
    # all_initial_points = [np.array([130,30]),np.array([200,15]),np.array([310,50])]
    special_list = [3, 5, 6, 7, 9]
    if network == "resnet":
        network_model = models.resnet50(pretrained=True).eval().to(device)
        network_name = "ResNet50"
    elif network == "incept":
        network_model = models.inception_v3(pretrained=True).eval().to(device)
        network_name = "Inceptionv3"
    elif network == "vgg":
        network_model = models.vgg11_bn(pretrained=True).eval().to(device)
        network_name = "VGG"
    elif network == "alexnet":
        network_model = models.alexnet(pretrained=True).eval().to(device)
        network_name = "AlexNet"
    else:
        print("NO available network with this name ... Sorry !")
        raise Exception("NO NETWORK")

    if all_points:
        print("you asked for all points ... ")
        all_initial_points = [np.array([130, 30]), np.array(
            [50, 20]), np.array([200, 15]), np.array([310, 50])]
        shapes_dir = os.path.join(data_dir, "scale", object_list[class_nb])
        shapes_list = list(glob.glob(shapes_dir+"/*"))
        for myobject_nb in range(len(shapes_list)):
            test_optimization_2(network_model, network_name, class_nb, myobject_nb, all_initial_points, obj_class_list,
                                object_list, setup, data_dir=data_dir, override=override, reduced=reduced, device=device)
    elif custom_points:
        all_initial_points = [np.array([310, 50])]
        test_optimization_2(network_model, network_name, class_nb, object_nb, all_initial_points, obj_class_list,
                            object_list, setup, data_dir=data_dir, override=override, reduced=reduced, device=device)

    # else :
    #     print("you asked for custom points ... ")
    #     all_initial_points = [np.array([130,30]),np.array([50,20]),np.array([200,15]),np.array([310,50])]
    #     shapes_dir = os.path.join(data_dir,"scale",object_list[class_nb])
    #     shapes_list = list(glob.glob(shapes_dir+"/*"))
    #     # if class_nb == 0 or class_nb == 3  :
    #     #     custom_list_full = [5,6,7,8,9]
    #     # elif class_nb == 6 or class_nb == 7 :
    #     #     custom_list_full = [8,9]
    #     # for myobject_nb in range(len(shapes_list)):
    #     myobject_nb = 4
    #     if network == "incept" and class_nb==9:
    #         myobject_nb = 9
    #     test_optimization(network_model,network_name,class_nb,myobject_nb,all_initial_points,obj_class_list,object_list,setup,data_dir=data_dir,override=override,reduced=reduced ,device=device)


if __name__ == '__main__':
    parser = ArgumentParser(description='extract values of summary tags from tensorboard events files and dump it in a csv file',
                            formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument('-n', '--network', required=True, type=str,
                        help='network type of the experiments')
    parser.add_argument('-a', '--all_points', required=True, type=int,
                        help='perform all the points of all experiments')
    parser.add_argument('-c', '--class_nb', default=1, type=int,
                        help='the claass number of the experoemtn being performed')
    parser.add_argument('-o', '--object_nb', default=1, type=int,
                        help='shape number being processes ')
    parser.add_argument('-r', '--override', default=0, type=int,
                        help='overriding existing dictionary files and writing the new ones')
    parser.add_argument('-d', '--reduced', default=0, type=int,
                        help='do not do all the experiments .. because it might fail at some.')
    parser.add_argument('-i', '--iterations', default=800, type=int,
                        help='number of iterations for the experiment')
    parser.add_argument('-y', '--custom_points', default=0, type=int,
                        help='is it custom point')
    parser.add_argument('-k', '--custom_list', default=0, type=int,
                        help='number part of cutom list or index custom list')

    args = parser.parse_args()
    # numeric_level = getattr(logging, args.loglevel.upper(), None)
    # if not isinstance(numeric_level, int):
    #     raise ValueError('Invalid log level: %s' % args.loglevel)
    # logging.basicConfig(format='%(asctime)s:%(levelname)s: %(message)s',
    #                 level=numeric_level)
    # delattr(args, 'loglevel')

    main(**vars(args))
