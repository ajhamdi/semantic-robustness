
# coding: utf-8

# ### libraries used in teh experiment

# In[271]:


# from __future__ import division
import os
import sys
import argparse
import glob

import torch
import logging
import torch.nn as nn
import numpy as np
from skimage.io import imread, imsave
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from scipy import stats
from torch.optim import lr_scheduler
import numpy as np
from numpy.linalg import inv
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import math
import time
import copy
import seaborn as sns
import shutil
import matplotlib.patches as patches
from interval import interval


import imageio
from IPython.display import display, HTML 
import trimesh
# trimesh.util.attach_to_log(logging.DEBUG,me)

try:
    import neural_renderer as nr
except:
    sys.path.append('../neural_renderer')
    import neural_renderer as nr

# data_dir = os.path.join('./', 'data/3d_renderer')


# ### supporting functions

# In[272]:


def make_gif(filename):
    with imageio.get_writer(filename, mode='I') as writer:
        for filename in sorted(glob.glob('/tmp/_tmp_*.png')):
            writer.append_data(imread(filename))
            os.remove(filename)
    writer.close()
def load_mymesh(filename_obj):
    vertices, faces = nr.load_obj(filename_obj)
    vertices = vertices[None, :, :]  # [num_vertices, XYZ] -> [batch_size=1, num_vertices, XYZ]
    faces = faces[None, :, :]  # [num_faces, 3] -> [batch_size=1, num_faces, 3]
    return vertices ,faces


# ### defining the models 

# In[465]:


# class renderer_model_2(nn.Module):
#     def __init__(self, network_model,vertices,faces,camera_distance,elevation,azimuth,image_size):
#         super(renderer_model_2, self).__init__()
#         self.register_buffer('vertices', vertices)
#         self.register_buffer('faces', faces)

#         # create textures
#         texture_size = 2
#         textures = torch.ones(self.faces.shape[0], self.faces.shape[1], texture_size, texture_size, texture_size, 3, dtype=torch.float32)
#         self.register_buffer('textures', textures)

#         # define the DNN model as part of the model of the renderer 
#         self.network_model =  network_model
        
#         self.register_buffer('camera_distance', torch.from_numpy(np.array(camera_distance)).float().unsqueeze_(0))
        
        
#         # camera parameters
# #         self.camera_position = nn.Parameter(torch.from_numpy(np.array([6, 10, -14], dtype=np.float32)))
#         self.azimuth = nn.Parameter(torch.from_numpy(np.array(azimuth)).float().unsqueeze_(0))  ### if bach remove unsqueeze
#         self.elevation = nn.Parameter(torch.from_numpy(np.array(elevation)).float().unsqueeze_(0))  ### if anthc remove unsqueeze        


#         # setup renderer
#         renderer = nr.Renderer(camera_mode='look_at',image_size=image_size)
# #         renderer.eye = self.camera_position
#         self.renderer = renderer

#     def forward(self,eval_point):
#         self.azimuth.data.set_(torch.from_numpy(np.array(eval_point[0])).float().to(device))
#         self.elevation.data.set_(torch.from_numpy(np.array(eval_point[1])).float().to(device))
#         self.renderer.eye =nr.get_points_from_angles(self.camera_distance, self.elevation, self.azimuth)
#         images = self.renderer(self.vertices, self.faces,self.textures)
# #         image = images.detach().cpu().numpy()[0].transpose((1, 2, 0))  # [image_size, image_size, RGB]
# #         imsave("/tmp/aa.png",(255*image).astype(np.uint8))
#         prop = torch.functional.F.softmax(self.network_model(images),dim=1)
#         return prop
    
class renderer_model_2(nn.Module):
    def __init__(self, network_model,vertices,faces,camera_distance,elevation,azimuth,image_size,device=None):
        super(renderer_model_2, self).__init__()
        self.register_buffer('vertices', vertices)
        self.register_buffer('faces', faces)

        # create textures
        texture_size = 2
        textures = torch.ones(self.faces.shape[0], self.faces.shape[1], texture_size, texture_size, texture_size, 3, dtype=torch.float32)
        self.register_buffer('textures', textures)
        self.device = device

        # define the DNN model as part of the model of the renderer 
        self.network_model =  network_model
        
        self.register_buffer('camera_distance', torch.from_numpy(np.array(camera_distance)).float().unsqueeze_(0))
        
        
        # camera parameters
#         self.camera_position = nn.Parameter(torch.from_numpy(np.array([6, 10, -14], dtype=np.float32)))
        self.azimuth = nn.Parameter(torch.from_numpy(np.array(azimuth)).float().unsqueeze_(0))  ### if bach remove unsqueeze
        self.elevation = nn.Parameter(torch.from_numpy(np.array(elevation)).float().unsqueeze_(0))  ### if anthc remove unsqueeze        


        # setup renderer
        renderer = nr.Renderer(camera_mode='look_at',image_size=image_size)
#         renderer.eye = self.camera_position
        self.renderer = renderer

    def forward(self,eval_point):
        self.azimuth.data.set_(torch.from_numpy(np.array(eval_point[0])).float().to(self.device))
        self.elevation.data.set_(torch.from_numpy(np.array(eval_point[1])).float().to(self.device))
        self.renderer.eye = nr.get_points_from_angles(self.camera_distance, self.elevation, self.azimuth)
        images = self.renderer(self.vertices, self.faces,self.textures)
#         image = images.detach().cpu().numpy()[0].transpose((1, 2, 0))  # [image_size, image_size, RGB]
#         imsave("/tmp/aa.png",(255*image).astype(np.uint8))
        prop = torch.functional.F.softmax(self.network_model(images),dim=1)
        return prop
class renderer_model(nn.Module):
    def __init__(self, network_model,vertices,faces,camera_distance,elevation,azimuth,image_size,device=None):
        super(renderer_model, self).__init__()
        self.register_buffer('vertices', vertices)
        self.register_buffer('faces', faces)

        # create textures
        texture_size = 2
        textures = torch.ones(self.faces.shape[0], self.faces.shape[1], texture_size, texture_size, texture_size, 3, dtype=torch.float32)
        self.register_buffer('textures', textures)

        # define the DNN model as part of the model of the renderer 
        self.network_model =  network_model
        
        self.register_buffer('camera_distance', torch.from_numpy(np.array(camera_distance)).float().unsqueeze_(0))
        self.register_buffer('elevation', torch.from_numpy(np.array(elevation)).float().unsqueeze_(0))
        
        
        # camera parameters
#         self.camera_position = nn.Parameter(torch.from_numpy(np.array([6, 10, -14], dtype=np.float32)))
        self.azimuth = nn.Parameter(torch.from_numpy(np.array(azimuth)).float().unsqueeze_(0))  ### if anthc remove unsqueeze


        # setup renderer
        renderer = nr.Renderer(camera_mode='look_at',image_size=image_size)
#         renderer.eye = self.camera_position
        self.renderer = renderer

    def forward(self,azimuth):
        self.azimuth.data.set_(torch.from_numpy(np.array(azimuth)).float().to(device))
        self.renderer.eye =nr.get_points_from_angles(self.camera_distance, self.elevation, self.azimuth)
        images = self.renderer(self.vertices, self.faces,self.textures)
#         image = images.detach().cpu().numpy()[0].transpose((1, 2, 0))  # [image_size, image_size, RGB]
#         imsave("/tmp/aa.png",(255*image).astype(np.uint8))
        prop = torch.functional.F.softmax(self.network_model(images),dim=1)
    
        return prop


# ### defining the evaluation functions 

# In[466]:


# to gpu
def evaluate_robustness(model,shapes_list,class_label,camera_distance,elevation,analysis_domain,image_size,data_dir=None,save_gif=True):
    """
    evluate the robustness of the DNN model over the fulll range of domain analysis ias azimujth angles and record a gif of teh rotated object 
    """ 
    texture_size = 2
    image_collection =[]
    all_prop_profile = []
    all_class_profile =[]
    
    # load .obj
    for exp in range(len(shapes_list)):
        prop_profile = []
        class_profile = []
        vertices, faces = nr.load_obj(shapes_list[exp])
        vertices = vertices[None, :, :]  # [num_vertices, XYZ] -> [batch_size=1, num_vertices, XYZ]
        faces = faces[None, :, :]  # [num_faces, 3] -> [batch_size=1, num_faces, 3]

        # create texture [batch_size=1, num_faces, texture_size, texture_size, texture_size, RGB]
        textures = torch.ones(1, faces.shape[1], texture_size, texture_size, texture_size, 3, dtype=torch.float32).to(device)

        # create renderer
        renderer = nr.Renderer(camera_mode='look_at',image_size=image_size)

        print("processing..\n", shapes_list[exp])
        model.eval()
        # draw object
        for num, azimuth in enumerate(analysis_domain):
        #     loop.set_description('Drawing')
            renderer.eye = nr.get_points_from_angles(camera_distance, elevation, azimuth)
            images = renderer(vertices, faces, textures,)  # [batch_size, RGB, image_size, image_size]
            with torch.no_grad():
                prop = torch.functional.F.softmax(model(images),dim=1)
                class_profile.append(torch.max(prop,1)[1].detach().cpu().numpy())
                prop_profile.append(prop[0,class_label].detach().cpu().numpy())
            image = images.detach().cpu().numpy()[0].transpose((1, 2, 0))  # [image_size, image_size, RGB]
            image_collection.append((255*image).astype(np.uint8))
        all_prop_profile.append(prop_profile) , all_class_profile.append(class_profile)
    if save_gif :
        imageio.mimsave(os.path.join(data_dir,"results","class_%d_.gif" %(class_nb)),image_collection)
    return all_prop_profile , all_class_profile

def evaluate_robustness_2(renderer_2,a,b,precesions,class_nb,obj_class_list):
    f = lambda x:  query_robustness(renderer_2,obj_class_list[class_nb],x)
    x = np.arange(a[0], b[0], precesions[0])
    y = np.arange(a[1], b[1], precesions[1])
    xx, yy = np.meshgrid(x, y)
    z = [ [f([xx[ii,jj],yy[ii,jj]]) for jj in range(xx.shape[1]) ] for ii in range(xx.shape[0]) ]
    z = np.array(z)
    return z,xx,yy

def render_from_point(obj_file,camera_distance,elevation,azimuth,image_size,data_dir=None):
    texture_size = 2
    # load .obj
    vertices, faces = nr.load_obj(obj_file)
    vertices = vertices[None, :, :]  # [num_vertices, XYZ] -> [batch_size=1, num_vertices, XYZ]
    faces = faces[None, :, :]  # [num_faces, 3] -> [batch_size=1, num_faces, 3]

    # create texture [batch_size=1, num_faces, texture_size, texture_size, texture_size, RGB]
    textures = torch.ones(1, faces.shape[1], texture_size, texture_size, texture_size, 3, dtype=torch.float32).to(device)
    renderer = nr.Renderer(camera_mode='look_at',image_size=image_size)
    renderer.eye = nr.get_points_from_angles(camera_distance, elevation, azimuth)
    images = renderer(vertices, faces, textures,)  # [batch_size, RGB, image_size, image_size]
    if not data_dir:
        data_dir , filename = os.path.split(obj_file)
        filename = os.path.splitext(filename)[0]
    else:
        filename = "class"
    image = images.detach().cpu().numpy()[0].transpose((1, 2, 0))  # [image_size, image_size, RGB]
    imsave(os.path.join(data_dir,"examples",filename+"_%d.jpg" %(azimuth)),(255*image).astype(np.uint8))
    

def render_evaluate(obj_file,camera_distance,elevation,azimuth,light_direction=[0,1,0],image_size=224,data_dir=None,model=None,class_label=0):
    texture_size = 2
    # load .obj
    vertices, faces = nr.load_obj(obj_file)
    vertices = vertices[None, :, :]  # [num_vertices, XYZ] -> [batch_size=1, num_vertices, XYZ]
    faces = faces[None, :, :]  # [num_faces, 3] -> [batch_size=1, num_faces, 3]

    # create texture [batch_size=1, num_faces, texture_size, texture_size, texture_size, RGB]
    textures = torch.ones(1, faces.shape[1], texture_size, texture_size, texture_size, 3, dtype=torch.float32).to(device)
    light_direction = nn.functional.normalize(torch.FloatTensor(light_direction), dim=0, eps=1e-16).numpy().tolist()
    renderer = nr.Renderer(camera_mode='look_at',image_size=image_size,light_direction=light_direction)
    renderer.eye = nr.get_points_from_angles(camera_distance, elevation, azimuth)
    images = renderer(vertices, faces, textures,)  # [batch_size, RGB, image_size, image_size]
    if not data_dir:
        data_dir , filename = os.path.split(obj_file)
        filename = os.path.splitext(filename)[0]
    else:
        filename = "class"
    image = images.detach().cpu().numpy()[0].transpose((1, 2, 0))  # [image_size, image_size, RGB]
    imsave(os.path.join(data_dir,"examples",filename+"_%d_%d_%d.jpg" %(azimuth,elevation,camera_distance)),(255*image).astype(np.uint8))
    if model:
        with torch.no_grad():
            prop = torch.functional.F.softmax(model(images),dim=1)
        return prop[0,class_label].detach().cpu().numpy()
    
    
def query_robustness(renderer,obj_class,querry_point):
    with torch.no_grad():
        prop = renderer(querry_point)
        
    return prop[0,obj_class].detach().cpu().numpy()
    
    
    
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

def query_gradient(renderer,obj_class,querry_point):
    prop = renderer(querry_point)
    labels = torch.tensor([obj_class]).to(device)  #torch.from_numpy(np.tile(np.eye(1000)[obj_class],(1,prop.size()[0]))).float().to(device)
    criterion = nn.CrossEntropyLoss()
    loss = criterion(prop,labels)
    renderer.zero_grad()
    loss.backward(retain_graph=True)
    return renderer.azimuth.grad.cpu().numpy()

def query_gradient_2(renderer,obj_class,querry_point):
    prop = renderer(querry_point)
    labels = torch.tensor([obj_class]).to(device)  #torch.from_numpy(np.tile(np.eye(1000)[obj_class],(1,prop.size()[0]))).float().to(device)
    criterion = nn.CrossEntropyLoss()
    loss = criterion(prop,labels)
    renderer.zero_grad()
    loss.backward(retain_graph=True)
    return renderer.azimuth.grad.cpu().numpy(),renderer.elevation.grad.cpu().numpy()


# ### visulaizing the probability distribution of softmax and the mesh being analyzed

# In[456]:



def visulize_average_softmax(class_nb,prop_profiles,class_profiles,analysis_domain,object_list,shape_list,remove_bad=False):
   network = "ResNet50"
   bad_list = []
   for exp in range(len(prop_profiles)):
       _,shape_name = os.path.split(shape_list[exp])
       shape_avg = np.mean(np.array(prop_profiles[exp]))
       shape_std = np.std(np.array(prop_profiles[exp]))
       print("class: %s, shape %s ,  mean: %1.3f , std: %1.3f" %(object_list[class_nb],shape_name,shape_avg,shape_std))
       print("most common class label is %d with frequency %2.1f percent " %(int(stats.mode(np.array(class_profiles[exp]))[0]),
                                                                             100*float(stats.mode(np.array(class_profiles[exp]))[1])/len(class_profiles[exp])))
       if (shape_avg < 0.1) :
           bad_list.append(exp)
   
   if remove_bad :
       for ii in bad_list:
           shutil.rmtree(shapes_list[ii], ignore_errors=True)
       prop_profiles = [i for j, i in enumerate(prop_profiles) if j not in bad_list]
       class_profiles = [i for j, i in enumerate(class_profiles) if j not in bad_list]
       


   plt.figure(figsize = (8, 6))
   plt.title("Average %s score on %d shapes of %s class vs azimuth rotations" %(network,len(prop_profiles),object_list[class_nb].upper()),fontsize=15)
   plt.xlabel("the azimuth rotation around teh object (degrees)",fontsize=13)
   plt.ylabel("%s softmax prob of that class" %(network),fontsize=13)
   for exp in range(1):
       # # for ii in range(self.nb_parameters):
       x = np.array(analysis_domain)
       y = np.poly1d(np.mean(np.array(prop_profiles),axis=0))
       z = np.polyfit(x, y, 20)
#        plt.scatter(x,y,alpha=0.5,s=10,label=object_list[exp])
       plt.plot(x,y,linewidth=1,alpha=1,label="%s" %(network))
           
       # sns.kdeplot(np.array(prop_profile).tolist(), linewidth = 2, shade = False, label=self.paramters_list[ii],clip=(-1,1))
   plt.legend(fontsize=12)
   plt.xlim(min(analysis_domain),max(analysis_domain))    
   plt.ylim(0,1)    
   plt.savefig(os.path.join(data_dir,"results","azimuth_performance_%s.pdf"%(object_list[class_nb].upper())))
   # plt.close()
   


def visulize_average_network(class_nb,network_dicts,analysis_domain,object_list,plot_global=False):
   """
   the network_dicts has dixtionary that contain keys of network names and values of prop profiles for these networks on the 
   target analysss domain for 10 differet nshapes in each 
   
   """
   plt.figure(figsize = (8, 6))
   plt.title("Average class scores of %s class vs azimuth rotations" %(object_list[class_nb]).upper(),fontsize=15)
   plt.xlabel("the azimuth rotation around teh object (degrees)",fontsize=13)
   plt.ylabel("softmax prob of the %s class" %(object_list[class_nb]).upper(),fontsize=13)
   for k ,v  in network_dicts.items():
       # # for ii in range(self.nb_parameters):
       x = np.array(analysis_domain)
       y = np.poly1d(np.mean(np.array(v[class_nb]),axis=0))
       plt.plot(x,y,linewidth=1,alpha=1,label=k)
   plt.legend(fontsize=12)
   plt.xlim(min(analysis_domain),max(analysis_domain))
   plt.ylim(0,1)    
   plt.savefig(os.path.join(data_dir,"results","average_azimuth_performance_%s.pdf"%(object_list[class_nb].upper())))
   
   if plot_global:
       plt.figure(figsize = (8, 6))
       plt.title("Average scores of all classes vs azimuth rotations" ,fontsize=15)
       plt.xlabel("the azimuth rotation around the object (degrees)",fontsize=13)
       plt.ylabel("average softmax prob of all classes",fontsize=13)
       for k ,v  in network_dicts.items():
           # # for ii in range(self.nb_parameters):
           x = np.array(analysis_domain)
           y = np.poly1d(np.mean(np.mean(np.array(v),axis=0),axis=0))
           plt.plot(x,y,linewidth=1,alpha=1,label=k)
       plt.legend(fontsize=12)
       plt.xlim(min(analysis_domain),max(analysis_domain))
       plt.ylim(0,1)    
       plt.savefig(os.path.join(data_dir,"results","average_azimuth_performance_%s.pdf"%(object_list[class_nb].upper())))

       
# def plot_2d_contour(f,a,b,precesion=0.1,levels=[0.5],rectangles=None,data_dir=None):
#     x = np.arange(a[0], b[0], precesion)
#     y = np.arange(a[1], b[1], precesion)
#     xx, yy = np.meshgrid(x, y)
#     z = [ [f([xx[ii,jj],yy[ii,jj]]) for jj in range(xx.shape[1]) ] for ii in range(xx.shape[0]) ]
#     z = np.array(z)
#     print("@@@@@",xx.shape,yy.shape,z.shape)
#     plt.figure(figsize = (8, 6))
#     plt.rc('text', usetex=True)
#     plt.rc('font', family='serif')
#     fig,ax = plt.subplots(1)
#     plt.title("function contours" ,fontsize=15)
#     plt.xlabel(r'$x_{1}$',fontsize=13)
#     plt.ylabel(r'$x_{2}$',fontsize=13)
#         # # for ii in range(self.nb_parameters):
# #     plt.scatter(xx.reshape(-1),yy.reshape(-1),alpha=0.5,s=0.01,label="low")
#     CS  = plt.contour(xx,yy,z,levels,linewidths=2)
#     ax.clabel(CS, CS.levels, inline=True, fontsize=8)
# #         sns.kdeplot(np.array(prop_profile).tolist(), linewidth = 2, shade = False, label=self.paramters_list[ii],clip=(-1,1))
#     plt.xlim(a[0],b[0])
#     plt.ylim(a[1],b[1])    
#     colors_list = [(1,0,0),(0.,0.8588,0.921),(0.94117,0.7098,0.0666),(0.796,0.215,0.941)]
#     if rectangles:
#         for idx,rectangle in enumerate(rectangles):
#             rect = patches.Rectangle((rectangle[0],rectangle[1]),rectangle[2],rectangle[3],linewidth=1.2,edgecolor=colors_list[idx],facecolor='none',label="R%d" %(idx+1))
#             ax.add_patch(rect)
#         plt.legend(fontsize=8)

#     if data_dir:
#         plt.savefig(os.path.join(data_dir,"results","2D.pdf"))
           
def region2rectangle(region):
   rectangle = (region.a[0],region.a[1],region.r[0],region.r[1])
   return rectangle

def visualize_network_2(map_dict,object_list,optim_dict=None,data_dir=None,heatmap=False,levels=5):
   network = "ResNet50"
   class_nb = map_dict["class_nb"] ; yy = map_dict["yy"] ; xx= map_dict["xx"] ; z = map_dict["z"]

   plt.figure(figsize = (8, 6))
#     plt.rc('text', usetex=True)
#     plt.rc('font', family='serif')
   fig,ax = plt.subplots(1)
   plt.title("%s softmax scores of %s class on 2D semantics" %(network,object_list[class_nb].upper()),fontsize=15)
   plt.xlabel("the azimuth rotation around teh object (degrees)",fontsize=13)
   plt.ylabel("the elevation angle of view-point (degrees)",fontsize=13)
       # # for ii in range(self.nb_parameters):
   if not heatmap:
       CS  = plt.contour(xx,yy,z,levels,linewidths=0.5)
   else :
       c = ax.pcolormesh(xx, yy, z, cmap='RdBu', vmin=0, vmax=np.max(z.reshape(-1)))
       fig.colorbar(c, ax=ax)
#     ax.clabel(CS, CS.levels, inline=True, fontsize=8)
#         sns.kdeplot(np.array(prop_profile).tolist(), linewidth = 2, shade = False, label=self.paramters_list[ii],clip=(-1,1))
   plt.xlim(np.min(xx.reshape(-1)),np.max(xx.reshape(-1)))
   plt.ylim(np.min(yy.reshape(-1)),np.max(yy.reshape(-1)))    
   colors_dict = {"naive":(0.796,0.215,0.981),"OIR_B":(0.,0.8588,0.821),"OIR_W":(0.94117,0.9598,0.0666),"trap":(1,0,0)}
   if optim_dict:
       a = ax.scatter(np.array(optim_dict["initial_point"])[::,0],np.array(optim_dict["initial_point"])[::,1],c='r',marker='x',alpha=1,linewidths=12,label="initial points")
       artist_list = [] ; label_list = [] 
       label_list.append("initial points") ; artist_list.append(a)
       for exp,proprts in optim_dict.items():
           if exp != "naive" and exp != "OIR_B" and exp != "OIR_W":
               continue
           rectangles = [region2rectangle(optim_dict[exp]["regions"][ii]) for ii in range(len(optim_dict["initial_point"]))]
           rect = patches.Rectangle((rectangles[0][0],rectangles[0][1]),rectangles[0][2],rectangles[0][3],linewidth=1.8,edgecolor=colors_dict[exp],facecolor='none',label=exp.replace("_"," "))
           a = ax.add_patch(rect)
           label_list.append(exp.replace("_"," "))
           artist_list.append(a)
           for idx,rectangle in enumerate(rectangles):
               rect = patches.Rectangle((rectangle[0],rectangle[1]),rectangle[2],rectangle[3],linewidth=1.8,edgecolor=colors_dict[exp],facecolor='none',label=exp.replace("_"," "))
               ax.add_patch(rect)

       ax.legend(artist_list ,label_list,fontsize=8)



   if data_dir:
       if optim_dict:
           plt.savefig(os.path.join(data_dir,"results","2d","2D_%s_regions.pdf" %(object_list[class_nb])))
       else:
           plt.savefig(os.path.join(data_dir,"results","2d","2D_%s.pdf" %(object_list[class_nb])))
           


def visulaize_mesh(anchor,object_list,data_dir,view_3d=False):
   filename_obj =  os.path.join(data_dir,"{}.obj".format(object_list[anchor]))
   filename_out =  os.path.join(data_dir,"{}_out.gif".format(object_list[anchor]))
   display(HTML("<img src='{}'></img>".format(filename_out)))
   if view_3d:
       logging.disable(sys.maxsize)
       mesh = trimesh.load(filename_obj)
       return mesh

def visulize_trace(optimization_trace,loss_trace,object_list,object_number=0,exp_type=None):
   """
   visualize the trace taken by the optimization on the two variables (a,b) contained in the list optimization_trace
   
   """
   t = np.array(range(len(optimization_trace)))
   
   fig, ax1 = plt.subplots()
   plt.title("azimuth rotations trace during optimization of %s class in the %s exp" %(object_list[object_number].upper(),str(exp_type).upper()))
   ax1.set_xlabel("iterations (steps)")
   ax1.set_ylabel("zimuth rotations change " )
   aa = np.array([x[0] for x in optimization_trace])
   bb = np.array([x[1] for x in optimization_trace])
   ax1.plot(t,aa,linewidth=1,alpha=1,label="the left bound a")
   ax1.plot(t,bb,linewidth=1,alpha=1,label="the right bound b")
   ax1.tick_params(axis='y')
   plt.legend()
   ax1.set_xlim(0,len(optimization_trace))   
   ax1.set_ylim(np.min(aa),np.max(bb))
   
   ax2 = ax1.twinx()
   color = 'tab:red'
   ax2.set_ylabel('loss', color=color)  # we already handled the x-label with ax1
   ax2.plot(t, np.array(loss_trace), color=color)
   ax2.tick_params(axis='y', labelcolor=color)

#     fig.tight_layout()  # otherwise the right y-label is slightly clipped

# color = 'tab:red'
# ax1.set_xlabel('time (s)')
# ax1.set_ylabel('exp', color=color)
# ax1.plot(t, data1, color=color)
# ax1.tick_params(axis='y', labelcolor=color)



# ## visualizing the networks on azimuth range

# ### experimtn setup

# In[276]:


# object_list =  ['aeroplane',"bathtub",'bench', 'bottle','chair',"cup","piano",'rifle','vase',"toilet"] #["teapot","truck","boat","tv"]
# obj_class_list = [404,435,703,898,559,968,579,413,883,861] # [849,586,606,549] 


# class_nb = 1
# shapes_dir = os.path.join(data_dir,"scale",object_list[class_nb])
# shapes_list = list(glob.glob(shapes_dir+"/*"))

# object_nb = 0
# mesh_file = os.path.join(shapes_list[object_nb],"models","model_normalized.obj")
# mesh_file_list = [os.path.join(x,"models","model_normalized.obj") for x in shapes_list]

# camera_distance = 2.732
# domain_begin = 0 ; domain_end = 360 ; domain_precision = 15
# analysis_domain = range(domain_begin, domain_end, domain_precision)
# elevation = 35
# image_size = 224
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# #### define the deep model 
# resnet = models.resnet50(pretrained=True).eval().to(device)
# resnet = nn.DataParallel(resnet)
# alexnet = models.alexnet(pretrained=True).eval().to(device)
# alexnet = nn.DataParallel(alexnet)
# vgg = models.vgg11_bn(pretrained=True).eval().to(device)
# vgg = nn.DataParallel(vgg)
# incept = models.inception_v3(pretrained=True).eval().to(device)
# incept = nn.DataParallel(incept)
# models_dicts = {"ResNet50":resnet , "AlexNet": alexnet , "VGG":vgg , "Inceptionv3":incept}


# # In[69]:


# # try:
# #     mesh = trimesh.load(mesh_file)
# # except:
# #     print("Mesh Not Loaded ")
# # print(mesh_file)
# # mesh.show()


# # In[70]:


# file = os.path.join(data_dir,'all_networks_profiles.pt')
# if not os.path.exists(file):
#     networks_profiles_dict = {}
#     for network_name, network_model in models_dicts.items():
#         all_classes_prop_profiles = [] ;  all_classes_label_profile = []
#         for class_nb in range(len(obj_class_list)):
#             shapes_dir = os.path.join(data_dir,"scale",object_list[class_nb])
#             shapes_list = list(glob.glob(shapes_dir+"/*"))
#             mesh_file_list = [os.path.join(x,"models","model_normalized.obj") for x in shapes_list]
#             prop_profiles , _ =  evaluate_robustness(network_model,mesh_file_list,
#                         class_label=obj_class_list[class_nb],camera_distance=camera_distance,elevation=elevation,analysis_domain=analysis_domain,image_size=image_size,data_dir=data_dir,save_gif=False)
#             all_classes_prop_profiles.append(prop_profiles) 
#         networks_profiles_dict[network_name] = all_classes_prop_profiles.copy()
#     torch.save(networks_profiles_dict, file)
# networks_profiles_dict = torch.load(file)


# # In[23]:


# file = os.path.join(data_dir,'all_classes_profiles.pt')
# if not os.path.exists(file):
#     all_classes_prop_profiles = [] ;  all_classes_label_profile = []
#     for class_nb in range(1):
#         shapes_dir = os.path.join(data_dir,"scale",object_list[class_nb])
#         shapes_list = list(glob.glob(shapes_dir+"/*"))
#         mesh_file_list = [os.path.join(x,"models","model_normalized.obj") for x in shapes_list]
#         prop_profiles , label_profile =  evaluate_robustness(incept,mesh_file_list,
#                     class_label=obj_class_list[class_nb],camera_distance=camera_distance,elevation=elevation,analysis_domain=analysis_domain,image_size=image_size,data_dir=data_dir)
#         all_classes_prop_profiles.append(prop_profiles) ; all_classes_label_profile.append(label_profile)
#     scores_dict = {}
#     scores_dict["prop_profile"] = all_classes_prop_profiles  ;   scores_dict["label_profile"] = all_classes_label_profile
#     torch.save(scores_dict, file)
# scores_dict = torch.load(file)


# # In[24]:


# scores_dict = torch.load(os.path.join(data_dir,'all_classes_profiles.pt'))
# all_classes_prop_profiles = scores_dict["prop_profile"] ; all_classes_label_profile = scores_dict["label_profile"]
# for class_nb in range(1):#range(len(obj_class_list)):
#     shapes_dir = os.path.join(data_dir,"scale",object_list[class_nb])
#     shapes_list = list(glob.glob(shapes_dir+"/*"))
#     mesh_file_list = [os.path.join(x,"models","model_normalized.obj") for x in shapes_list]
#     visulize_average_softmax(class_nb,all_classes_prop_profiles[class_nb],all_classes_label_profile[class_nb],analysis_domain,object_list,shapes_list,remove_bad=False)
#     print("original class label %d \n-------------------\n" %(obj_class_list[class_nb]))


# # In[73]:


# networks_profiles_dict = torch.load(os.path.join(data_dir,'all_networks_profiles.pt'))
# for class_nb in range(len(obj_class_list)):
#     shapes_dir = os.path.join(data_dir,"scale",object_list[class_nb])
#     shapes_list = list(glob.glob(shapes_dir+"/*"))
#     mesh_file_list = [os.path.join(x,"models","model_normalized.obj") for x in shapes_list]
#     visulize_average_network(class_nb,networks_profiles_dict,analysis_domain=analysis_domain,object_list=object_list,plot_global=False)
#     print("finsihed processing class %s \n-------------------\n" %(object_list[class_nb]))


# # ### visualizing all the networks together 

# # In[10]:


# models_dicts = {"ResNet50":resnet , "AlexNet": alexnet , "VGG":vgg , "Inceptionv3":incept}


# # In[ ]:


# render_from_point(mesh_file,camera_distance=camera_distance,elevation=elevation,azimuth=68,image_size=image_size,data_dir=data_dir)


# # In[53]:





# In[480]:


def int2binarray(number, n_bits ): 
    if 2**n_bits <= number :
        print("not enouhg bits")
    else :
        a = format(number, "#0%db" %(n_bits+2))[2::]
        binary = np.array([int(x) for x in list(a)])
        return binary 
class n_interval():
    def __init__(self,a,b):
        if len(a) != len(b):
            print("not valid n-dim interval")
        elif "interval" not in sys.modules:
            print("pip install pyinterval module first !!")
        else:
            self.n = len(a)
            self.two_to_n = 2**self.n
            self.mask = np.array([int2binarray(x,n_bits=self.n) for x in range(self.two_to_n) ]).T
            self.mask_c = np.logical_not(self.mask).astype(np.int)
            self.a = np.array(a) ; self.b = np.array(b)
            self.update()
            self.old_a = self.a ; self.old_b = self.b
    def step_size(self):
        return np.sum(self.a - self.old_a) + np.sum(self.b - self.old_b)
    def size(self):
        return np.prod(self.r)
    def update(self):
        self.region = [interval([self.a[ii],self.b[ii]]) for ii in range(self.n)]
        self.r  = np.array([x[0][1] - x[0][0] for x in self.region])
        self.R = inv(np.diag(self.r))
        self.corners_matrix = np.ones([self.two_to_n,1]) @ np.expand_dims(self.a,axis=0) + self.mask.T * (np.ones([self.two_to_n,1]) @ np.expand_dims(self.r,axis=0))
        self.corners_set = [self.corners_matrix[ii,::] for ii in range(self.two_to_n)]
    def size_normalized(self):
        return self.size() / self.two_to_n
    def __str__(self):
        return str(self.region)
    def __call__(self,a,b):
        self.old_a = self.a.copy() ; self.old_b = self.b.copy()
        self.a = a.copy() ; self.b = b.copy()
        self.update()
    def __and__(self,interval2):
        return [x & y for x,y in zip(self.region,interval2.region)]
    def __or__(self,interval2):
        return [x | y for x,y in zip(self.region,interval2.region)]
    
# my_region = n_interval(np.array([0,3]),[3,8])
# my_region3 = n_interval(np.array([11,20]),[30,50])
# print(my_region)
# my_region2 = n_interval(np.array([-1,7]),[1,11])
# print(my_region2.mask_c)
# new_r = (my_region |  my_region2)
# print(new_r)
# [new_r[ii] | my_region3.region[ii] for ii in range( my_region3.n)]
# # my_region([1,3],[3,7])
# print(my_region.corners_set)


# ## n-interval operator

# In[352]:


def optimize_n_boundary(f,f_grad,initial_point,learning_rate=0.1,alpha=0.1,beta=0.01,reg=0.001,n_iterations=500,exp_type="inner"):
    optimization_trace = [] ; loss_trace = [] ; a_grad = 0 ; b_grad = 0
#     f = lambda x:  query_robustness(renderer,obj_class,x)
#     f_grad = lambda x: query_gradient(renderer,obj_class,x)
    a = initial_point - 0.00001 ; b =  initial_point + 0.00001
    loss = - f(initial_point) 
    my_region = n_interval(a,b)
    if exp_type == "OIR_B" or exp_type == "OIR_W":
        M_C_c = (1+alpha)** (my_region.n -1 ) *( (1+0.5 * alpha)*my_region.mask_c + 0.5*alpha*my_region.mask ) 
        M_C = (1+alpha)** (my_region.n -1 ) *( (1+0.5 * alpha)*my_region.mask + 0.5*alpha*my_region.mask_c )     
        M_D_c =  (2-my_region.n* beta)*my_region.mask_c - beta * my_region.mask
        M_D = (2-my_region.n* beta)*my_region.mask - beta * my_region.mask_c
        A = a - 0.5 * alpha * my_region.r ; B = b+ 0.5*alpha * my_region.r
        outer_region = n_interval(A,B)
    

    if exp_type == "naive":
        for t in range(n_iterations):
#     #     evaluationg the functions and a rough estimate of the loss complex
            f_D =  np.expand_dims(np.array([f(x) for x in my_region.corners_set]),axis=0).T
            region_size  = my_region.size_normalized()

        #     recording hte curent state before ubdate 
            optimization_trace.append((a,b))
    #         loss = np.mean(np.array([loss,fa,fb]))
            loss = - my_region.size_normalized()* np.sum(np.squeeze(f_D)) + reg *  np.linalg.norm(my_region.r)
            print("iteration: %3d,   current loss = %1.4f , boundaries: " %(t,loss), a ,b)
            loss_trace.append(loss)
            a_grad = 2*my_region.size_normalized()*my_region.R @(my_region.mask_c @ f_D) - reg* np.expand_dims(my_region.r,axis=0).T
            b_grad = 2*my_region.size_normalized()*my_region.R @(-my_region.mask @ f_D ) + reg* np.expand_dims(my_region.r,axis=0).T

            a = a - learning_rate * (np.squeeze(a_grad))
            b = b - learning_rate * (np.squeeze(b_grad))
            my_region(a,b)


    elif exp_type == "OIR_B":
        #     Evaluating the outer boundary
        for t in range(n_iterations):
#     #     evaluationg the functions and a rough estimate of the loss complex
            f_D =  np.expand_dims(np.array([f(x) for x in my_region.corners_set]),axis=0).T
            region_size  = my_region.size_normalized()

            A = a - 0.5 * alpha * my_region.r ; B = b+ 0.5*alpha * my_region.r
            outer_region(A,B)
            f_C =  np.expand_dims(np.array([f(x) for x in outer_region.corners_set]),axis=0).T
            loss = my_region.size_normalized()*((1+alpha)**my_region.n * np.sum(np.squeeze(f_C)) -2 * np.sum(np.squeeze(f_D))  )
            print("iteration: %3d,   current loss = %1.4f , boundaries: " %(t,loss), a ,b)
            loss_trace.append(loss)
#             print(A,a,b,B)
#             fA =  query_robustness(network_model,obj_class,vertices,faces,camera_distance,elevation,A,image_size)
#             fB =  query_robustness(network_model,obj_class,vertices,faces,camera_distance,elevation,B,image_size)
            a_grad = 2*my_region.size_normalized()*my_region.R @(2*my_region.mask_c @ f_D - M_C_c @ f_C)
            b_grad = 2*my_region.size_normalized()*my_region.R @(-2*my_region.mask @ f_D + M_C @ f_C)

            a = a - learning_rate * (np.squeeze(a_grad))
            b = b - learning_rate * (np.squeeze(b_grad))
            my_region(a,b)



    elif exp_type == "OIR_W":
        for t in range(n_iterations):
#     #     evaluationg the functions and a rough estimate of the loss complex
            f_D =  np.expand_dims(np.array([f(x) for x in my_region.corners_set]),axis=0).T
            region_size  = my_region.size_normalized()
            A = a - 0.5 * alpha * my_region.r ; B = b+ 0.5*alpha * my_region.r
            outer_region(A,B)
            f_C =  np.expand_dims(np.array([f(x) for x in outer_region.corners_set]),axis=0).T
            loss = (1+alpha)**my_region.n * np.sum(np.squeeze(f_C))/(np.sum(np.squeeze(f_D)))  -1 
            print("iteration: %3d,   current loss = %1.4f , boundaries: " %(t,loss), a ,b)
            loss_trace.append(loss)
            G_D = np.array([f_grad(x) for x in my_region.corners_set])
            ss_c =[] ; ss =[]
            for kk in range(my_region.n):
                temp = [my_region.r[ii] * ((my_region.mask_c[ii,::] - my_region.mask[ii,::])* my_region.mask_c[kk,::]).reshape(1,-1) @ G_D[::,ii].reshape(-1,1) for ii in range(my_region.n) if ii != kk]
                ss_c.append(np.sum(temp))
                temp = [my_region.r[ii] * ((my_region.mask[ii,::] - my_region.mask_c[ii,::])* my_region.mask[kk,::]).reshape(1,-1) @ G_D[::,ii].reshape(-1,1) for ii in range(my_region.n) if ii != kk]
                ss.append(np.sum(temp))
            s_c = my_region.R @ np.array([ss_c]).reshape(-1,1)
            s = my_region.R @ np.array([ss]).reshape(-1,1)
            a_grad = my_region.size_normalized()*(my_region.R @ M_D_c @ f_D + beta * np.diag(my_region.mask_c @ G_D).reshape(-1,1) + beta*s_c)
            b_grad =  my_region.size_normalized()*(-my_region.R @ M_D @ f_D + beta * np.diag(my_region.mask @ G_D).reshape(-1,1) + beta*s)

            a = a - learning_rate * (np.squeeze(a_grad))
            b = b - learning_rate * (np.squeeze(b_grad))
            my_region(a,b)



    elif exp_type == "trap":
        for t in range(n_iterations):
#     #     evaluationg the functions and a rough estimate of the loss complex
            f_D =  np.expand_dims(np.array([f(x) for x in my_region.corners_set]),axis=0).T
            region_size  = my_region.size_normalized()
            loss = - my_region.size_normalized()* np.sum(np.squeeze(f_D)) + reg *  np.linalg.norm(my_region.r)
            print("iteration: %3d,   current loss = %1.4f , boundaries: " %(t,loss), a ,b)
            loss_trace.append(loss)
            G_D = np.array([f_grad(x) for x in my_region.corners_set])
            a_grad = my_region.size_normalized()*(-np.diag(my_region.mask_c @ G_D).reshape(-1,1) + np.sum(np.squeeze(f_D)) * my_region.R @(np.ones([my_region.n,1]))) - reg* my_region.r.reshape(-1,1)
            b_grad = my_region.size_normalized()*(-np.diag(my_region.mask @ G_D).reshape(-1,1) - np.sum(np.squeeze(f_D)) * my_region.R @(np.ones([my_region.n,1]))) + reg* my_region.r.reshape(-1,1)

            a = a - learning_rate * (np.squeeze(a_grad))
            b = b - learning_rate * (np.squeeze(b_grad))
            my_region(a,b)

    #     Update rule according to gradient descent and record the new loss 
#         print(np.squeeze(a_grad))
    return optimization_trace, loss_trace , my_region


# ### running the optimization after initiailizaing the hyperparameters

# In[462]:


# object_list =  ['aeroplane',"bathtub",'bench', 'bottle','chair',"cup","piano",'rifle','vase',"toilet"] #["teapot","truck","boat","tv"]
# obj_class_list = [404,435,703,898,559,968,579,413,883,861] # [849,586,606,549] 
# #### define the deep model 
# resnet = models.resnet50(pretrained=True).eval().to(device)
# # resnet = nn.DataParallel(resnet)
# alexnet = models.alexnet(pretrained=True).eval().to(device)
# # alexnet = nn.DataParallel(alexnet)
# vgg = models.vgg11_bn(pretrained=True).eval().to(device)
# # vgg = nn.DataParallel(vgg)
# incept = models.inception_v3(pretrained=True).eval().to(device)
# # incept = nn.DataParallel(incept)
# models_dicts = {"ResNet50":resnet , "AlexNet": alexnet , "VGG":vgg , "Inceptionv3":incept}

# class_nb = 1
# shapes_dir = os.path.join(data_dir,"scale",object_list[class_nb])
# shapes_list = list(glob.glob(shapes_dir+"/*"))

# object_nb = 1
# mesh_file = os.path.join(shapes_list[object_nb],"models","model_normalized.obj")
# mesh_file_list = [os.path.join(x,"models","model_normalized.obj") for x in shapes_list]
# _,shape_id = os.path.split(shapes_list[object_nb])
    
# camera_distance = 2.732
# azimuth = 50
# domain_begin = 0 ; domain_end = 360 ; domain_precision = 5
# analysis_domain = range(domain_begin, domain_end, domain_precision)
# elevation = 35
# image_size = 224
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# vertices, faces =  load_mymesh(mesh_file)
# renderer =  renderer_model_2(network_model,vertices,faces,camera_distance,elevation,azimuth,image_size).to(device)
# f = lambda x:  query_robustness(renderer,obj_class_list[class_nb],x)
# f_grad = lambda x: query_gradient_2(renderer,obj_class_list[class_nb],x)


# f = lambda x: np.prod(np.array([np.sinc(y)**2 for y in list(x)]))
# def cosinc(x):
#     if x == 0:
#         return 0
#     else :
#         return np.cos(x)/(np.pi* x) - np.sin(x)/(np.pi**2 * x**2)
# f_grad = lambda x : np.array([2*np.sinc(y)*cosinc(y)*np.prod(np.array([np.sinc(z)**2 for z in list(x) if z!=y])) for y in list(x)])

# print("The function: ",f([azimuth,elevation]),f_grad([azimuth,elevation]))
# print("the record : ",networks_profiles_dict["ResNet50"][class_nb][object_nb][int(azimuth/2)])


# In[487]:


def test_optimization(network_model,network_name,class_nb,object_nb,all_initial_points,obj_class_list,object_list,setup=None,data_dir=None,override=False,reduced=False ,device="cuda:0"):
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
    file = os.path.join(data_dir,"checkpoint","optim_%s_%d_%d.pt" %(network_name,class_nb,object_nb))
    if not os.path.exists(file) or override:
        print("starting the expereimtn")
        optim_dict = {}
        for exp in exp_type_list:
            optim_dict[exp] = {}
            optim_dict[exp]["optim_trace"] =[] ; optim_dict[exp]["loss_trace"] = [] ; optim_dict[exp]["regions"] = [] 
        optim_dict["initial_point"] = all_initial_points
        optim_dict["class_nb"] = class_nb ; optim_dict["shape_id"] = shape_id ; optim_dict["network_name"] = network_name
    #     exp_type_list = ["inner","inner_outer_naive","inner_outer_grad","trap"]
        # network_prop_dicts["Inceptionv3"][class_n][int(initial_point/2)]
        for initial_point in all_initial_points:
            for exp in exp_type_list:
                optimization_trace, loss_trace, result_region = optimize_n_boundary(f,f_grad,initial_point,learning_rate=0.1,alpha=0.05,beta=0.0009,reg=0.1,n_iterations=800,exp_type=exp)
                optim_dict[exp]["optim_trace"].append(optimization_trace) ; optim_dict[exp]["loss_trace"].append(loss_trace) ; optim_dict[exp]["regions"].append(result_region)
        #         optim_dict[exp]["optim_trace"] = optimization_trace ; optim_dict[exp]["loss_trace"] = loss_trace ; optim_dict[exp]["regions"] = result_region 
        torch.save(optim_dict, file)
    optim_dict = torch.load(file)
    return optim_dict


# In[488]:




# In[191]:


# def map_network(network_model,network_name,class_nb,object_nb,obj_class_list,data_dir=None,override=False,device="cuda:0")
#     file = os.path.join(data_dir,'all_maps.pt')
    
#     z,xx,yy = evaluate_robustness_2(renderer,[0,-10],[360,90],[2,2],class_nb,obj_class_list)
#     map_dict = {"xx":xx , "yy":yy, "z":z ,"class_nb":class_nb,"shape_id":shape_id}


# In[483]:


# optim_dict["naive"]["optim_trace"][1]


# # In[457]:


# visualize_network_2(map_dict,object_list,optim_dict=optim_dict,data_dir=data_dir,heatmap=True)
# # plot_2d_contour(f,[0,0],[360,90],domain_precision,levels=20,rectangles=rectangles)


# # In[451]:


# len(optim_dict)


# # In[354]:


# shapes_list


# # In[281]:


# mesh_file = os.path.join(data_dir,"teapot.obj")


# # In[321]:


# render_evaluate(mesh_file,camera_distance=2,elevation=35,azimuth=0,light_direction=[0,1,0],image_size=224,data_dir=data_dir,model=resnet,class_label=849)

