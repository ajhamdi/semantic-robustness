from __future__ import division, print_function, absolute_import

from hyperopt import STATUS_OK
from hyperopt import fmin
from hyperopt import Trials
from hyperopt import rand, tpe
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



def main(network,exp, cluster,gpu,override,samples_nb):
    print("network", network,"exp",exp, "cluster", cluster, "gpu", gpu,
          "override", override, "samples_nb", samples_nb)


    # cfg = {"BATCH_SIZE": BATCH_SIZE, "is_weight_learning": is_weight_learning, "is_shape_features": is_shape_features, "is_initial_point": is_initial_point, "network_depth": network_depth,
    #        "PCA_size": PCA_size, "lambda1": lambda1, 'learning_rate': lr, "epochs": epochs, "is_classificaion": is_classificaion, "exp_nb": exp_nb, "is_early_features": is_early_features}

    # all_initial_points = [np.array([130,30]),np.array([200,15]),np.array([310,50])]
    device = torch.device("cuda:"+str(gpu) if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(gpu)

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
    models_dicts = {network_name: network_model}
    save_file = os.path.join(scores_dir, "SRVR_scores_{}_{}.csv".format(exp,str(list(network_name)[0])))


    final_results = ListDict(["network", "point_nb", "class_nb",
                          "azimuth", "method", "object_nb", "elevation", "pred"])


    # models_dicts = {"ResNet50":resnet , "AlexNet": alexnet , "VGG":vgg , "Inceptionv3":incept}
    if exp == "random":
        for class_nb in range(10):
            for object_nb in range(10):
                print("class: {}  object: {}".format(class_nb, object_nb))
                random_SRVR(network_name, class_nb, object_nb, samples_nb,
                            final_results, save_file, override, models_dicts, device)
    elif exp == "bay":
        for class_nb in range(10):
            for object_nb in range(10):
                print("class: {}  object: {}".format(class_nb, object_nb))
                bayesian_SRVR(network_name, class_nb, object_nb, samples_nb,
                              final_results, save_file, override, models_dicts, device)
    elif exp == "test":
        class_nb = 6
        object_nb = 0
        elevation = 35
        azimuth = 0
        shapes_dir = os.path.join(data_dir, "scale", object_list[class_nb])
        mesh_file = os.path.join(shapes_dir, TRUE_DICT[str(
            class_nb)][str(object_nb)], "models", "model_normalized.obj")
        vertices, faces = load_mymesh(mesh_file)
        renderer = renderer_model_2(models_dicts[network_name], vertices, faces,
                                    camera_distance, elevation, azimuth, image_size, device).to(device)

        def f(x): return query_robustness(renderer, obj_class_list[class_nb], x)
        def f_grad(x): return query_gradient_2(renderer, obj_class_list[class_nb], x)
        print("the current funtion value at the current azimuth {} and elevation {}  is {}".format(
            azimuth, elevation, f([azimuth, elevation])))
        print("the current gradient value at the current azimuth {} and elevation {}  is [{},{}]".format(
            azimuth, elevation, f_grad([azimuth, elevation])[0], f_grad([azimuth, elevation])[1]))


def random_SRVR(network_name, class_nb, object_nb, samples_nb, final_results, save_file, override, models_dicts,device):
    samples = np.array([np.random.uniform(
        low=left_limit[ii], high=right_limit[ii], size=samples_nb) for ii in range(len(left_limit))]).T
    shapes_dir = os.path.join(
        data_dir, "scale", object_list[class_nb])
#             shapes_list = list(glob.glob(shapes_dir+"/*"))
    mesh_file = os.path.join(shapes_dir, TRUE_DICT[str(
        class_nb)][str(object_nb)], "models", "model_normalized.obj")
#             mesh_file_list = [os.path.join(x,"models","model_normalized.obj") for x in shapes_list]
    for point_nb, sample in enumerate(list(samples)):
        pred = render_evaluate(mesh_file, camera_distance=camera_distance, elevation=sample[1], azimuth=sample[0], light_direction=[
            0, 1, 0], image_size=image_size, data_dir=data_dir, model=models_dicts[network_name], class_label=obj_class_list[class_nb], save_image=False, device=device)
        result = {"network": network_name, "class_nb": class_nb, "point_nb": point_nb, "method": "random",
                  "object_nb": object_nb, "azimuth": sample[0], "elevation": sample[1], "pred": pred}
        final_results.append(result)
    if not os.path.isfile(save_file) or bool(override):
        save_results(save_file=save_file, results=final_results)


def bayesian_SRVR(network_name, class_nb, object_nb, samples_nb, final_results, save_file, override, models_dicts,device):
    from hyperopt import hp
    from hyperopt.pyll.stochastic import sample
    tpe_trials = Trials()
    tpe_algo = tpe.suggest
    vars_list = ["x"+str(ii) for ii in range(len(left_limit))]
    space = {}
    for keys in vars_list:
        space[keys] = hp.uniform(keys, 0, 1)
    global ITERATION
    ITERATION = 0
    shapes_dir = os.path.join(data_dir, "scale", object_list[class_nb])
    mesh_file = os.path.join(shapes_dir, TRUE_DICT[str(class_nb)][str(object_nb)], "models", "model_normalized.obj")
    def objective(xs):
        ########### @@@@@@@@@@@@@@@@@ play with the uinput to make it vector !!
        global ITERATION
        ITERATION += 1
        def broadcast_azimuth(xx): return xx * 360.0
        def broadcast_elevation(xx): return xx * 100.0 - 10

        keylist = xs.keys()
        list(keylist).sort()
        x = [xs[ii] for ii in keylist]
        x[0] = broadcast_azimuth(x[0])
        x[1] = broadcast_elevation(x[1])

    #     print("@@@@@@@@@@@@@","ITERATION : ", ITERATION)
        # raise Exception
        try:
            pred = render_evaluate(mesh_file, camera_distance=camera_distance, elevation=x[1], azimuth=x[0], light_direction=[
                0, 1, 0], image_size=image_size, data_dir=data_dir, model=models_dicts[network_name], class_label=obj_class_list[class_nb], save_image=False, device=device)
        except:
            try:
                pred = render_evaluate(mesh_file, camera_distance=camera_distance, elevation=x[1], azimuth=x[0], light_direction=[
                    0, 1, 0], image_size=image_size, data_dir=data_dir, model=models_dicts[network_name], class_label=obj_class_list[class_nb], save_image=False , device=device)
            except:
                pred = result["pred"][-1]
        result = {"network": network_name, "class_nb": class_nb, "point_nb": ITERATION-1, "method": "bayesian",
                "object_nb": object_nb, "azimuth": x[0], "elevation": x[1], "pred": float(pred)}
        final_results.append(result)
        return {'loss': 1-float(pred), 'xs': xs, 'iteration': ITERATION, 'status': STATUS_OK}


    tpe_best = fmin(fn=objective, space=space, algo=tpe_algo, trials=tpe_trials,
                    max_evals=samples_nb, rstate=np.random.RandomState(50))
    # print('Minimum loss attained with TPE:    {:.4f}'.format(tpe_trials.best_trial['result']['loss']))
    # self.all_Xs = [[vars_dics[keys] for keys in vars_list]for vars_dics in [x['xs'] for x in tpe_trials.results]]
    # tpe_results = pd.DataFrame({'loss': [x['loss'] for x in tpe_trials.results], 'iteration': [x['iteration'] for x in tpe_trials.results],'x':self.all_Xs })
    # tpe_results.to_csv(os.path.join(self.generated_frames_train_dir,'baysian.csv'),sep=',',index=False)
    save_results(results=final_results, save_file=save_file)

if __name__ == '__main__':
    parser = ArgumentParser(description='extract values of summary tags from tensorboard events files and dump it in a csv file',
                            formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument('-n', '--network', required=True, type=str, choices=['incept', 'alexnet', 'vgg',"resnet"],
                        help='network type of the experiments')
    parser.add_argument('-e', '--exp', required=True, type=str, choices=['random', 'bay',"test"],
                        help='the exp name random or Baysian SRVR exp')
    parser.add_argument('-c', '--cluster', required=True, type=str, choices=['pc', 'semantic'],
                        help='the cluster name')
    parser.add_argument('-g', '--gpu', required=True, type=int,
                        help='the GPU number in which the exp perfoprmed ')
    parser.add_argument('-o', '--override', required=True, type=int,
                        help='override exisisting results')
    parser.add_argument('-s', '--samples_nb', default=1000, type=int,
                        help='number of samples for the experiment')


    args = parser.parse_args()
    # numeric_level = getattr(logging, args.loglevel.upper(), None)
    # if not isinstance(numeric_level, int):
    #     raise ValueError('Invalid log level: %s' % args.loglevel)
    # logging.basicConfig(format='%(asctime)s:%(levelname)s: %(message)s',
    #                 level=numeric_level)
    # delattr(args, 'loglevel')

    main(**vars(args))
