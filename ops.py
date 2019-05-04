import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
import glob
import imageio
from scipy import stats
from torch.optim import lr_scheduler
import neural_renderer as nr
from models import *


def load_mymesh(filename_obj):
    vertices, faces = nr.load_obj(filename_obj)
    # [num_vertices, XYZ] -> [batch_size=1, num_vertices, XYZ]
    vertices = vertices[None, :, :]
    faces = faces[None, :, :]  # [num_faces, 3] -> [batch_size=1, num_faces, 3]
    return vertices, faces


def optimize_n_boundary(f, f_grad, initial_point, learning_rate=0.1, alpha=0.1, beta=0.01, reg=0.001, n_iterations=500, exp_type="inner"):
    optimization_trace = []
    loss_trace = []
    a_grad = 0
    b_grad = 0
#     f = lambda x:  query_robustness(renderer,obj_class,x)
#     f_grad = lambda x: query_gradient(renderer,obj_class,x)
    a = initial_point - 0.00001
    b = initial_point + 0.00001
    loss = - f(initial_point)
    my_region = ndinterval(a, b)
    if exp_type == "OIR_B" or exp_type == "OIR_W":
        M_C_c = (1 + alpha) ** (my_region.n - 1) * ((1 + 0.5 * alpha)
                                                    * my_region.mask_c + 0.5 * alpha * my_region.mask)
        M_C = (1 + alpha) ** (my_region.n - 1) * ((1 + 0.5 * alpha)
                                                  * my_region.mask + 0.5 * alpha * my_region.mask_c)
        M_D_c = (2 - (2 * my_region.n - 1) * beta) * \
            my_region.mask_c - beta * my_region.mask
        M_D = (2 - (2 * my_region.n - 1) * beta) * \
            my_region.mask - beta * my_region.mask_c
        A = a - 0.5 * alpha * my_region.r
        B = b + 0.5 * alpha * my_region.r
        outer_region = ndinterval(A, B)

    if exp_type == "naive":
        for t in range(n_iterations):
            #     #     evaluationg the functions and a rough estimate of the loss complex
            f_D = np.expand_dims(
                np.array([f(x) for x in my_region.corners_set]), axis=0).T
            region_size = my_region.size_normalized()

        #     recording hte curent state before ubdate
            optimization_trace.append((a, b))
    #         loss = np.mean(np.array([loss,fa,fb]))
            loss = - my_region.size_normalized() * np.sum(np.squeeze(f_D)) + \
                reg * np.linalg.norm(my_region.r)
            print("iteration: %3d,   current loss = %1.4f , boundaries: " %
                  (t, loss), a, b)
            loss_trace.append(loss)
            a_grad = 2 * my_region.size_normalized() * my_region.R @(my_region.mask_c @ f_D) - reg * np.expand_dims(my_region.r, axis=0).T
            b_grad = 2 * my_region.size_normalized() * my_region.R @(-my_region.mask @ f_D) + reg * np.expand_dims(my_region.r, axis=0).T

            a = a - learning_rate * (np.squeeze(a_grad))
            b = b - learning_rate * (np.squeeze(b_grad))
            my_region(a, b)

    elif exp_type == "OIR_B":
        #     Evaluating the outer boundary
        for t in range(n_iterations):
            #     #     evaluationg the functions and a rough estimate of the loss complex
            f_D = np.expand_dims(
                np.array([f(x) for x in my_region.corners_set]), axis=0).T
            region_size = my_region.size_normalized()

            A = a - 0.5 * alpha * my_region.r
            B = b + 0.5 * alpha * my_region.r
            outer_region(A, B)
            f_C = np.expand_dims(
                np.array([f(x) for x in outer_region.corners_set]), axis=0).T
            optimization_trace.append((a, b))
            loss = my_region.size_normalized() * ((1 + alpha)**my_region.n *
                                                  np.sum(np.squeeze(f_C)) - 2 * np.sum(np.squeeze(f_D)))
            print("iteration: %3d,   current loss = %1.4f , boundaries: " %
                  (t, loss), a, b)
            loss_trace.append(loss)
#             print(A,a,b,B)
#             fA =  query_robustness(network_model,obj_class,vertices,faces,camera_distance,elevation,A,image_size)
#             fB =  query_robustness(network_model,obj_class,vertices,faces,camera_distance,elevation,B,image_size)
            a_grad = 2 * my_region.size_normalized() * my_region.R @(2 * my_region.mask_c @ f_D - M_C_c @ f_C)
            b_grad = 2 * my_region.size_normalized() * my_region.R @(-2 * my_region.mask @ f_D + M_C @ f_C)

            a = a - learning_rate * (np.squeeze(a_grad))
            b = b - learning_rate * (np.squeeze(b_grad))
            my_region(a, b)

    elif exp_type == "OIR_W":
        for t in range(n_iterations):
            #     #     evaluationg the functions and a rough estimate of the loss complex
            f_D = np.expand_dims(
                np.array([f(x) for x in my_region.corners_set]), axis=0).T
            region_size = my_region.size_normalized()
            A = a - 0.5 * alpha * my_region.r
            B = b + 0.5 * alpha * my_region.r
            outer_region(A, B)
            f_C = np.expand_dims(
                np.array([f(x) for x in outer_region.corners_set]), axis=0).T
            optimization_trace.append((a, b))
            loss = (1 + alpha)**my_region.n * \
                np.sum(np.squeeze(f_C)) / (np.sum(np.squeeze(f_D))) - 1
            print("iteration: %3d,   current loss = %1.4f , boundaries: " %
                  (t, loss), a, b)
            loss_trace.append(loss)
            G_D = np.array([f_grad(x) for x in my_region.corners_set])
            ss_c = []
            ss = []
            for kk in range(my_region.n):
                temp = [my_region.r[ii] * ((my_region.mask_c[ii, ::] - my_region.mask[ii, ::]) * my_region.mask_c[kk, ::]).reshape(1, -1) @ G_D[::, ii].reshape(-1, 1) for ii in range(my_region.n) if ii != kk]
                ss_c.append(np.sum(temp))
                temp = [my_region.r[ii] * ((my_region.mask[ii, ::] - my_region.mask_c[ii, ::]) * my_region.mask[kk, ::]).reshape(1, -1) @ G_D[::, ii].reshape(-1, 1) for ii in range(my_region.n) if ii != kk]
                ss.append(np.sum(temp))
            s_c = my_region.R @ np.array([ss_c]).reshape(-1, 1)
            s = my_region.R @ np.array([ss]).reshape(-1, 1)
            a_grad = my_region.size_normalized() * (my_region.R @ M_D_c @ f_D + beta * np.diag(my_region.mask_c @ G_D).reshape(-1, 1) + beta * s_c)
            b_grad = my_region.size_normalized() * (-my_region.R @ M_D @ f_D + beta * np.diag(my_region.mask @ G_D).reshape(-1, 1) + beta * s)

            a = a - learning_rate * (np.squeeze(a_grad))
            b = b - learning_rate * (np.squeeze(b_grad))
            my_region(a, b)

    elif exp_type == "trap":
        for t in range(n_iterations):
            #     #     evaluationg the functions and a rough estimate of the loss complex
            f_D = np.expand_dims(
                np.array([f(x) for x in my_region.corners_set]), axis=0).T
            region_size = my_region.size_normalized()
            loss = - my_region.size_normalized() * np.sum(np.squeeze(f_D)) + \
                reg * np.linalg.norm(my_region.r)
            print("iteration: %3d,   current loss = %1.4f , boundaries: " %
                  (t, loss), a, b)
            loss_trace.append(loss)
            G_D = np.array([f_grad(x) for x in my_region.corners_set])
            a_grad = my_region.size_normalized() * (-np.diag(my_region.mask_c @ G_D).reshape(-1, 1) + np.sum(np.squeeze(f_D)) * my_region.R @(np.ones([my_region.n, 1]))) - reg * my_region.r.reshape(-1, 1)
            b_grad = my_region.size_normalized() * (-np.diag(my_region.mask @ G_D).reshape(-1, 1) - np.sum(np.squeeze(f_D)) * my_region.R @(np.ones([my_region.n, 1]))) + reg * my_region.r.reshape(-1, 1)

            a = a - learning_rate * (np.squeeze(a_grad))
            b = b - learning_rate * (np.squeeze(b_grad))
            my_region(a, b)

    #     Update rule according to gradient descent and record the new loss
#         print(np.squeeze(a_grad))
    return optimization_trace, loss_trace, my_region

def test_optimization_2(network_model,network_name,class_nb,object_nb,all_initial_points,obj_class_list,object_list,setup=None,data_dir=None,override=False,reduced=False ,device="cuda:0"):
#         class_nb = 1
    camera_distance = 2.732
    azimuth = 50
    domain_begin = 0 ; domain_end = 360 ; domain_precision = 5
    analysis_domain = range(domain_begin, domain_end, domain_precision)
    elevation = 35
    image_size = 224
    if setup:
        learning_rate=setup["learning_rate"] ; alpha=setup["alpha"] ; beta=setup["beta"]; reg=setup["reg"]  ; n_iterations=setup["n_iterations"]
    else : 
        learning_rate=0.1 ;alpha=0.05 ; beta=0.0009 ; reg=0.1 ; n_iterations=800
    shapes_dir = os.path.join(data_dir,"scale",object_list[class_nb])
    shapes_list = list(glob.glob(shapes_dir+"/*"))

#     object_nb = 1
    mesh_file = os.path.join(shapes_list[object_nb],"models","model_normalized.obj")
    mesh_file_list = [os.path.join(x,"models","model_normalized.obj") for x in shapes_list]
    _,shape_id = os.path.split(shapes_list[object_nb])
    vertices, faces =  load_mymesh(mesh_file)
    renderer =  renderer_model_2(network_model,vertices,faces,camera_distance,elevation,azimuth,image_size,device).to(device)
    f = lambda x:  query_robustness(renderer,obj_class_list[class_nb],x)
    f_grad = lambda x: query_gradient_2(renderer,obj_class_list[class_nb],x)
    if not reduced :
        exp_type_list = ["naive","OIR_B","OIR_W"]
    else :
        exp_type_list = ["naive","OIR_B",]
    file = os.path.join(data_dir,"optim_%s_%d_%d.pt" %(network_name,class_nb,object_nb))
    if not os.path.exists(file) or override:
        optim_dict = {}
        for exp in exp_type_list:
            optim_dict[exp] = {}
            optim_dict[exp]["optim_trace"] =[] ; optim_dict[exp]["loss_trace"] = [] ; optim_dict[exp]["regions"] = [] 
        optim_dict["initial_point"] = all_initial_points
        optim_dict["class_nb"] = class_nb ; optim_dict["shape_id"] = shape_id
    #     exp_type_list = ["inner","inner_outer_naive","inner_outer_grad","trap"]
        # network_prop_dicts["Inceptionv3"][class_n][int(initial_point/2)]
        for initial_point in all_initial_points:
            for exp in exp_type_list:
                optimization_trace, loss_trace, result_region = optimize_n_boundary(f,f_grad,initial_point,learning_rate=learning_rate,alpha=alpha,beta=beta,reg=reg,n_iterations=n_iterations,exp_type=exp)
                optim_dict[exp]["optim_trace"].append(optimization_trace) ; optim_dict[exp]["loss_trace"].append(loss_trace) ; optim_dict[exp]["regions"].append(result_region)
        #         optim_dict[exp]["optim_trace"] = optimization_trace ; optim_dict[exp]["loss_trace"] = loss_trace ; optim_dict[exp]["regions"] = result_region 
        torch.save(optim_dict, file)
    optim_dict = torch.load(file)
    return optim_dict

def test_optimization_1(network_model,network_name,class_nb,object_nb,all_initial_points,obj_class_list,object_list,setup=None,data_dir=None,override=False,reduced=False ,device="cuda:0"):
#         class_nb = 1
    camera_distance = 2.732
    azimuth = 50
    domain_begin = 0 ; domain_end = 360 ; domain_precision = 5
    analysis_domain = range(domain_begin, domain_end, domain_precision)
    elevation = 35
    image_size = 224
    if setup:
        learning_rate=setup["learning_rate"] ; alpha=setup["alpha"] ; beta=setup["beta"]; reg=setup["reg"]  ; n_iterations=setup["n_iterations"]
    else : 
        learning_rate=0.1 ;alpha=0.05 ; beta=0.0009 ; reg=0.1 ; n_iterations=600
    shapes_dir = os.path.join(data_dir,"scale",object_list[class_nb])
    shapes_list = list(glob.glob(shapes_dir+"/*"))

#     object_nb = 1
    mesh_file = os.path.join(shapes_list[object_nb],"models","model_normalized.obj")
    mesh_file_list = [os.path.join(x,"models","model_normalized.obj") for x in shapes_list]
    _,shape_id = os.path.split(shapes_list[object_nb])
    vertices, faces =  load_mymesh(mesh_file)
    renderer =  renderer_model(network_model,vertices,faces,camera_distance,elevation,azimuth,image_size,device).to(device)
    f = lambda x:  query_robustness(renderer,obj_class_list[class_nb],x)
    f_grad = lambda x: query_gradient(renderer,obj_class_list[class_nb],x)
    if not reduced :
        exp_type_list = ["naive","OIR_B","OIR_W"]
    else :
        exp_type_list = ["naive","OIR_B",]
    file = os.path.join(data_dir,"optim_%s_%d_%d.pt" %(network_name,class_nb,object_nb))
    if not os.path.exists(file) or override:
        optim_dict = {}
        for exp in exp_type_list:
            optim_dict[exp] = {}
            optim_dict[exp]["optim_trace"] =[] ; optim_dict[exp]["loss_trace"] = [] ; optim_dict[exp]["regions"] = [] 
        optim_dict["initial_point"] = all_initial_points
        optim_dict["class_nb"] = class_nb ; optim_dict["shape_id"] = shape_id
    #     exp_type_list = ["inner","inner_outer_naive","inner_outer_grad","trap"]
        # network_prop_dicts["Inceptionv3"][class_n][int(initial_point/2)]
        for initial_point in all_initial_points:
            for exp in exp_type_list:
                optimization_trace, loss_trace, result_region = optimize_n_boundary(f,f_grad,initial_point,learning_rate=learning_rate,alpha=alpha,beta=beta,reg=reg,n_iterations=n_iterations,exp_type=exp)
                optim_dict[exp]["optim_trace"].append(optimization_trace) ; optim_dict[exp]["loss_trace"].append(loss_trace) ; optim_dict[exp]["regions"].append(result_region)
        #         optim_dict[exp]["optim_trace"] = optimization_trace ; optim_dict[exp]["loss_trace"] = loss_trace ; optim_dict[exp]["regions"] = result_region 
        torch.save(optim_dict, file)
    optim_dict = torch.load(file)
    return optim_dict

def evaluate_robustness(model, shapes_list, class_label, camera_distance, elevation, analysis_domain, image_size,class_nb, data_dir=None, device=None,save_gif=True):
    """
    evluate the robustness of the DNN model over the fulll range of domain analysis ias azimujth angles and record a gif of teh rotated object 
    """
    texture_size = 2
    image_collection = []
    all_prop_profile = []
    all_class_profile = []

    # load .obj
    for exp in range(len(shapes_list)):
        prop_profile = []
        class_profile = []
        vertices, faces = nr.load_obj(shapes_list[exp])
        # [num_vertices, XYZ] -> [batch_size=1, num_vertices, XYZ]
        vertices = vertices[None, :, :]
        # [num_faces, 3] -> [batch_size=1, num_faces, 3]
        faces = faces[None, :, :]

        # create texture [batch_size=1, num_faces, texture_size, texture_size, texture_size, RGB]
        textures = torch.ones(
            1, faces.shape[1], texture_size, texture_size, texture_size, 3, dtype=torch.float32).to(device)

        # create renderer
        renderer = nr.Renderer(camera_mode='look_at', image_size=image_size)

        print("processing..\n", shapes_list[exp])
        model.eval()
        # draw object
        for num, azimuth in enumerate(analysis_domain):
            #     loop.set_description('Drawing')
            renderer.eye = nr.get_points_from_angles(
                camera_distance, elevation, azimuth)
            # [batch_size, RGB, image_size, image_size]
            images = renderer(vertices, faces, textures,)
            with torch.no_grad():
                prop = torch.functional.F.softmax(model(images), dim=1)
                class_profile.append(torch.max(prop, 1)[
                                     1].detach().cpu().numpy())
                prop_profile.append(
                    prop[0, class_label].detach().cpu().numpy())
            image = images.detach().cpu().numpy()[0].transpose(
                (1, 2, 0))  # [image_size, image_size, RGB]
            image_collection.append((255 * image).astype(np.uint8))
        all_prop_profile.append(
            prop_profile), all_class_profile.append(class_profile)
    if save_gif:
        imageio.mimsave(os.path.join(data_dir, "results",
                                     "class_%d_.gif" % (class_nb)), image_collection)
    return all_prop_profile, all_class_profile


def evaluate_robustness_2(renderer_2, a, b, precesions, class_nb, obj_class_list):
    def f(x): return query_robustness(renderer_2, obj_class_list[class_nb], x)
    x = np.arange(a[0], b[0], precesions[0])
    y = np.arange(a[1], b[1], precesions[1])
    xx, yy = np.meshgrid(x, y)
    z = [[f([xx[ii, jj], yy[ii, jj]]) for jj in range(xx.shape[1])]
         for ii in range(xx.shape[0])]
    z = np.array(z)
    return z, xx, yy


def render_from_point(obj_file, camera_distance, elevation, azimuth, image_size, data_dir=None, name=None):
    texture_size = 2
    # load .obj
    vertices, faces = nr.load_obj(obj_file)
    # [num_vertices, XYZ] -> [batch_size=1, num_vertices, XYZ]
    vertices = vertices[None, :, :]
    faces = faces[None, :, :]  # [num_faces, 3] -> [batch_size=1, num_faces, 3]

    # create texture [batch_size=1, num_faces, texture_size, texture_size, texture_size, RGB]
    textures = torch.ones(
        1, faces.shape[1], texture_size, texture_size, texture_size, 3, dtype=torch.float32).to(device)
    renderer = nr.Renderer(camera_mode='look_at', image_size=image_size)
    renderer.eye = nr.get_points_from_angles(
        camera_distance, elevation, azimuth)
    # [batch_size, RGB, image_size, image_size]
    images = renderer(vertices, faces, textures,)
    filename = os.path.split(obj_file)[1]
    if not data_dir:
        data_dir, filename = os.path.split(obj_file)
    if not name:
        filename = os.path.splitext(filename)[0]
    else:
        filename = name
    file = os.path.join(data_dir, "examples", filename + "_%d.jpg" % (azimuth))
    path, _ = os.path.split(file)
    if not os.path.exists(path):
        os.makedirs(path)
    image = images.detach().cpu().numpy()[0].transpose(
        (1, 2, 0))  # [image_size, image_size, RGB]
    imsave(file, (255 * image).astype(np.uint8))


def render_region(obj_file, camera_distance, elevations, azimuths, image_size, data_dir=None, name=None):
    filename = os.path.split(obj_file)[1]
    if not data_dir:
        data_dir, filename = os.path.split(obj_file)
    if not name:
        filename = os.path.splitext(filename)[0]
    else:
        filename = name
    texture_size = 2
    # load .obj
    vertices, faces = nr.load_obj(obj_file)
    # [num_vertices, XYZ] -> [batch_size=1, num_vertices, XYZ]
    vertices = vertices[None, :, :]
    faces = faces[None, :, :]  # [num_faces, 3] -> [batch_size=1, num_faces, 3]

    # create texture [batch_size=1, num_faces, texture_size, texture_size, texture_size, RGB]
    textures = torch.ones(
        1, faces.shape[1], texture_size, texture_size, texture_size, 3, dtype=torch.float32).to(device)
    renderer = nr.Renderer(camera_mode='look_at', image_size=image_size)
    image_collection = []
    for elevation in elevations:
        for azimuth in azimuths:
            renderer.eye = nr.get_points_from_angles(
                camera_distance, elevation, azimuth)

            # [batch_size, RGB, image_size, image_size]
            images = renderer(vertices, faces, textures,)
            file = os.path.join(data_dir, "examples", filename,
                                "%d_%d.jpg" % (elevation, azimuth))
            path, _ = os.path.split(file)
            if not os.path.exists(path):
                os.makedirs(path)
            image = images.detach().cpu().numpy()[0].transpose(
                (1, 2, 0))  # [image_size, image_size, RGB]
            imsave(file, (255 * image).astype(np.uint8))
            image_collection.append((255 * image).astype(np.uint8))
    imageio.mimsave(os.path.join(data_dir, "examples", filename,
                                 "%s_vidoe.gif" % (filename)), image_collection)


def render_evaluate(obj_file, camera_distance, elevation, azimuth, light_direction=[0, 1, 0], image_size=224, data_dir=None, model=None, class_label=0):
    texture_size = 2
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # load .obj
    vertices, faces = nr.load_obj(obj_file)
    # [num_vertices, XYZ] -> [batch_size=1, num_vertices, XYZ]
    vertices = vertices[None, :, :]
    faces = faces[None, :, :]  # [num_faces, 3] -> [batch_size=1, num_faces, 3]

    # create texture [batch_size=1, num_faces, texture_size, texture_size, texture_size, RGB]
    textures = torch.ones(
        1, faces.shape[1], texture_size, texture_size, texture_size, 3, dtype=torch.float32).to(device)
    light_direction = nn.functional.normalize(torch.FloatTensor(
        light_direction), dim=0, eps=1e-16).numpy().tolist()
    renderer = nr.Renderer(
        camera_mode='look_at', image_size=image_size, light_direction=light_direction)
    renderer.eye = nr.get_points_from_angles(
        camera_distance, elevation, azimuth)
    # [batch_size, RGB, image_size, image_size]
    images = renderer(vertices, faces, textures,)
    print(typ(images))
    print(len(images))
    print(images.shape)
    if not data_dir:
        data_dir, filename = os.path.split(obj_file)
        filename = os.path.splitext(filename)[0]
    else:
        filename = "class"
    image = images.detach().cpu().numpy()[0].transpose(
        (1, 2, 0))  # [image_size, image_size, RGB]
    imsave(os.path.join(data_dir, "examples", filename + "_%d_%d_%d.jpg" %
                        (azimuth, elevation, camera_distance)), (255 * image).astype(np.uint8))
    if model:
        with torch.no_grad():
            prop = torch.functional.F.softmax(model(images), dim=1)
        return prop[0, class_label].detach().cpu().numpy()


def query_robustness(renderer, obj_class, querry_point):
    with torch.no_grad():
        prop = renderer(querry_point)

    return prop[0, obj_class].detach().cpu().numpy()


#     texture_size = 2
#     # create texture [batch_size=1, num_faces, texture_size, texture_size, texture_size, RGB]
#     textures = torch.ones(1, faces.shape[1], texture_size, texture_size, texture_size, 3, dtype=torch.float32).to(device)

#     # create renderer
#     renderer = nr.Renderer(camera_mode='look_at',image_size=image_size)
#     model.eval()
#     # draw object
# #     loop.set_description('Drawing')
#     renderer.eye = nr.get_points_from_angles(camera_distance, elevation, azimuth)
#     images = renderer(vertices, faces, textures,)  # [batch_size, RGB, image_size, image_size]
#     with torch.no_grad():
#         prop = torch.functional.F.softmax(model(images),dim=1)
#     return prop[0,obj_class].detach().cpu().numpy()

def query_gradient(renderer, obj_class, querry_point):
    prop = renderer(querry_point)
    # torch.from_numpy(np.tile(np.eye(1000)[obj_class],(1,prop.size()[0]))).float().to(device)
    labels = torch.tensor([obj_class]).to(renderer.device)
    criterion = nn.CrossEntropyLoss()
    loss = criterion(prop, labels)
    renderer.zero_grad()
    loss.backward(retain_graph=True)
    return renderer.azimuth.grad.cpu().numpy()


def query_gradient_2(renderer, obj_class, querry_point):
    prop = renderer(querry_point)
    # torch.from_numpy(np.tile(np.eye(1000)[obj_class],(1,prop.size()[0]))).float().to(device)
    labels = torch.tensor([obj_class]).to(renderer.device)
    criterion = nn.CrossEntropyLoss()
    loss = criterion(prop, labels)
    renderer.zero_grad()
    loss.backward(retain_graph=True)
    return renderer.azimuth.grad.cpu().numpy(), renderer.elevation.grad.cpu().numpy()
