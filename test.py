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
import tensorflow as tf
import glob
# from tqdm import tqdm
import numpy as np


def main(network, is_weight_learning, is_shape_features, is_initial_point, network_depth,  iterations, cluster_exp):
    print(network, is_weight_learning, is_shape_features,
          is_initial_point, network_depth, iterations, cluster_exp)

    is_weight_learning = bool(is_weight_learning)
    is_shape_features = bool(is_shape_features)
    is_initial_point = bool(is_initial_point)

    exp_nb = np.random.randint(100000, 1000000)
    # exp_nb = 60339
    BATCH_SIZE = 32
    # network_depth = 4
    is_classificaion = True
    # is_weight_learning = False
    is_early_features = False
    # is_shape_features = True
    # is_initial_point = True
    PCA_size = 40
    lambda1 = 0.001
    lr = 0.003
    epochs = iterations
    cfg = {"BATCH_SIZE": BATCH_SIZE, "is_weight_learning": is_weight_learning, "is_shape_features": is_shape_features, "is_initial_point": is_initial_point, "network_depth": network_depth,
           "PCA_size": PCA_size, "lambda1": lambda1, 'learning_rate': lr, "epochs": epochs, "is_classificaion": is_classificaion, "exp_nb": exp_nb, "is_early_features": is_early_features}
    data_dir = os.getcwd()
    models_dir = os.path.join(data_dir, "models")


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("Warninign ....... no Cuda !!")
    object_list = ['aeroplane', "bathtub", 'bench', 'bottle', 'chair', "cup",
                   "piano", 'rifle', 'vase', "toilet"]  # ["teapot","truck","boat","tv"]
    # this is the list tht represent the 10 classes in ImageNEt class labels
    obj_class_list = [404, 435, 703, 898, 559, 968, 579, 413, 883, 861]


    # all_initial_points = [np.array([130,30]),np.array([200,15]),np.array([310,50])]

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

    # models_dicts = {"ResNet50":resnet , "AlexNet": alexnet , "VGG":vgg , "Inceptionv3":incept}
    log_dir = os.path.join(
        data_dir, "logs/{}_{}".format(network_name, cfg["exp_nb"]))
    check_folder(log_dir)
    if not hasattr(log_dir, 'add_summary'):
        log_dir = tf.summary.FileWriter(log_dir)
    load = os.path.join(models_dir, "{}_{}_best_model.pt".format(
        network_name, cfg["exp_nb"]))
    save = os.path.join(models_dir, "{}_{}_best_model.pt".format(
        network_name, cfg["exp_nb"]))

    # writer.file_writer?
    if is_weight_learning:
        input_size = PCA_size*1000
    else:
        input_size = 1000
    if is_shape_features:
        input_size += 256
    if is_initial_point:
        input_size += 2
    model = SRVR_Classifier(n_feature=input_size,  n_output=1,
                            depth=cfg["network_depth"]).to(device)     # define the network
    #  model = Regressor(n_feature=PCA_size*2048+256+2,  n_output=1).to(device)     # define the network

    # print(net)  # net architecture

    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    if is_classificaion:
        loss_func = torch.nn.BCELoss()
    else:
        loss_func = torch.nn.MSELoss()  # this is for regression mean squared loss

    # data_set = ShapeFeatures(models_dicts[network_name],network_name,classification=cfg["is_classificaion"],part)
    data_set = {x: ShapeFeatures(network_model, network_name,
                                 classification=cfg["is_classificaion"], root_dir=data_dir, part=x) for x in ['train', 'test']}
    print("the train dtaset size :", len(
        data_set["train"]), "  the test dtaset size :", len(data_set["test"]))
    # raise Exception("@@@@")
    # data_loader = data.DataLoader(data_set, batch_size=BATCH_SIZE, num_workers=4, shuffle=True, pin_memory=False)
    data_loader = {x: data.DataLoader(data_set[x], batch_size=cfg['BATCH_SIZE'],
                                      num_workers=4, shuffle=True, pin_memory=False) for x in ['train', 'test']}

    if load is not None:
        try:
            best_state = torch.load(load)
            model.load_state_dict(best_state['model'])
            optimizer.load_state_dict(best_state['optimizer'])
    #         scheduler.load_state_dict(best_state['scheduler'])
        except FileNotFoundError:
            msg = 'Couldn\'t find checkpoint file! {} (training with random initialization)'
            print(msg.format(load))
            load = None

    # otherwise, start from the current initialization
    if load is None:
        best_state = {
            'epoch': -1,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            #         'scheduler': scheduler.state_dict(),
            'accuracy': float(0.0),
        }

    summaries = {}
    [summaries.update({"cfg/"+K: V}) for K, V in cfg.items()]
    values = [tf.Summary.Value(tag=k, simple_value=v)
              for k, v in summaries.items()]
    log_dir.add_summary(tf.Summary(value=values), 0)
    log_dir.flush()

    for epoch in range(epochs):
        print('-' * 60)
        print('Epoch: {} / {}'.format(epoch, epochs))
        print('-' * 60)

        running_loss = {}
        running_corrects = {}
        for x in ['train', 'test']:
            running_loss[x], running_corrects[x] = full_epoch(
                model, optimizer, loss_func, data_loader[x], data_set[x], device, cfg)

        epoch_loss = {x: running_loss[x] /
                      len(data_set[x]) for x in ['train', 'test']}
        epoch_accuracy = {
            x: running_corrects[x]/len(data_set[x]) for x in ['train', 'test']}
        [print('{} Loss: {:.4f} , accuracy: {:.4f}  '.format(
            x, epoch_loss[x], epoch_accuracy[x])) for x in ['train', 'test']]
        values = [tf.Summary.Value(tag='metrics/epoch_{}_loss'.format(x),
                                   simple_value=epoch_loss[x]) for x in ['train', 'test']]
        values += [tf.Summary.Value(tag='metrics/epoch_{}_accuracy'.format(
            x), simple_value=epoch_accuracy[x]) for x in ['train', 'test']]

        log_dir.add_summary(tf.Summary(value=values), epoch)
        log_dir.flush()

        if best_state['accuracy'] < epoch_accuracy["test"]:
            best_state = {
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                #             'scheduler': scheduler.state_dict(),
                'accuracy': epoch_accuracy["test"],
            }
            values = [tf.Summary.Value(
                tag='cfg/best_accuracy', simple_value=epoch_accuracy["test"])]
            log_dir.add_summary(tf.Summary(value=values), 0)
            log_dir.flush()

            if save is not None:
                torch.save(best_state, save)


if __name__ == '__main__':
    parser = ArgumentParser(description='extract values of summary tags from tensorboard events files and dump it in a csv file',
                            formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument('-n', '--network', required=True, type=str,
                        help='network type of the experiments')
    parser.add_argument('-c', '--is_weight_learning', required=True, type=int,
                        help='the claass number of the experoemtn being performed')
    parser.add_argument('-o', '--is_shape_features', required=True, type=int,
                        help='shape number being processes ')
    parser.add_argument('-r', '--is_initial_point', required=True, type=int,
                        help='overriding existing dictionary files and writing the new ones')
    parser.add_argument('-d', '--network_depth', default=4, type=int,
                        help='do not do all the experiments .. because it might fail at some.')
    parser.add_argument('-i', '--iterations', default=120, type=int,
                        help='number of iterations for the experiment')
    parser.add_argument('-y', '--cluster_exp', default=0, type=int,
                        help='is it custom point')

    args = parser.parse_args()
    # numeric_level = getattr(logging, args.loglevel.upper(), None)
    # if not isinstance(numeric_level, int):
    #     raise ValueError('Invalid log level: %s' % args.loglevel)
    # logging.basicConfig(format='%(asctime)s:%(levelname)s: %(message)s',
    #                 level=numeric_level)
    # delattr(args, 'loglevel')

    main(**vars(args))
