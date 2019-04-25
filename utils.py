# supporting functions
import imageio
import numpy as np
import logging
import seaborn as sns
import shutil
import matplotlib.patches as patches
from IPython.display import display, HTML 
import trimesh
import matplotlib.pyplot as plt
def make_gif(filename):
    with imageio.get_writer(filename, mode='I') as writer:
        for filename in sorted(glob.glob('/tmp/_tmp_*.png')):
            writer.append_data(imread(filename))
            os.remove(filename)
    writer.close()


def gif_folder(data_dir, extension="jpg"):
    image_collection = []
    for img_name in glob.glob(data_dir + "/*." + extension):
        image_collection.append(imageio.imread(img_name))
    imageio.mimsave(os.path.join(data_dir, "video.gif"), image_collection)

def int2binarray(number, n_bits ): 
    if 2**n_bits <= number :
        print("not enouhg bits")
    else :
        a = format(number, "#0%db" %(n_bits+2))[2::]
        binary = np.array([int(x) for x in list(a)])
        return binary 
        
def visulize_function_profiles(function_profiles,analysis_domain,label_profiles,title,xlabel,ylabel,fit_poly=False,limit_y=True):
    plt.figure(figsize = (9, 6))
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    for exp in range(len(function_profiles)):
        x = np.array(analysis_domain)
        y= np.array(function_profiles[exp])
        if fit_poly:
            y = np.poly1d(np.array(function_profiles[exp]))
            z = np.polyfit(x, y, 20)
            f = np.poly1d(z)
            x_new = np.linspace(x[0], x[-1], 150)
            y_new = f(x_new)
            plt.plot(x_new,y_new,linewidth=2,alpha=1,label=label_profiles[exp])
        else:
            plt.plot(x,y,linewidth=1,alpha=1,label=label_profiles[exp])
            
    plt.legend()
    plt.xlim(min(analysis_domain),max(analysis_domain)) 
    if limit_y:
        plt.ylim(min(function_profiles[0]),max(function_profiles[0])) 
    
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
        


    plt.figure(figsize = (9, 6))
    plt.title("Average %s score on %d shapes of %s class vs azimuth rotations" %(network,len(prop_profiles),object_list[class_nb].upper()),fontsize=15)
    plt.xlabel("the azimuth rotation around the object (degrees)",fontsize=13)
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
    plt.figure(figsize = (9, 6))
    plt.title("Average class scores of %s class vs azimuth rotations" %(object_list[class_nb]).upper(),fontsize=15)
    plt.xlabel("the azimuth rotation around the object (degrees)",fontsize=13)
    plt.ylabel("softmax prob of the %s class" %(object_list[class_nb]).upper(),fontsize=13)
    for k ,v  in network_dicts.items():
        # # for ii in range(self.nb_parameters):
        x = np.array(analysis_domain)
        y = np.mean(np.array(v[class_nb]),axis=0)
        plt.plot(x,y,linewidth=1,alpha=1,label=k)
    plt.legend(fontsize=12)
    plt.xlim(min(analysis_domain),max(analysis_domain))
    plt.ylim(0,1)    
    plt.savefig(os.path.join(data_dir,"results","average_azimuth_performance_%s.pdf"%(object_list[class_nb].upper())))
    
    if plot_global:
        plt.figure(figsize = (9, 6))
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
        plt.savefig(os.path.join(data_dir,"results","%s_%s_1D.pdf"%(object_list[class_nb].upper(),"Average")))

        
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
def region2interval(region):
    rectangle = (region.a[0],0,region.r[0],1)
    return rectangle


def visualize_network_2(map_dict,object_list,optim_dict=None,data_dir=None,heatmap=True,object_nb=0):
    """
    create a figure of 2D case of regions areound some points if optim_dict != None , if optim_dict =None , then it only save the 2D map of the entwork in map_dict as pdf
    """
    network = map_dict["network_name"]
    class_nb = map_dict["class_nb"] ; yy = map_dict["yy"] ; xx= map_dict["xx"] ; z = map_dict["z"]

    plt.figure(figsize = (9, 6))
#     plt.rc('text', usetex=True)
#     plt.rc('font', family='serif')
    fig,ax = plt.subplots(1)
    plt.title("%s softmax scores of %s class on 2D semantics" %(network,object_list[class_nb].upper()),fontsize=15)
    plt.xlabel("Azimuth rotation around the object(degrees)",fontsize=13)
    plt.ylabel("Elevation angle of view-point",fontsize=13)
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
    colors_dict = {"naive":(0.796,0.215,0.981),"OIR_B":(1,0,0),"OIR_W":(0.94117,0.9598,0.0666),"trap":(0.,0.8588,0.821)}
    if optim_dict:
        a = ax.scatter(np.array(optim_dict["initial_point"])[::,0],np.array(optim_dict["initial_point"])[::,1],c='g',marker='x',alpha=1,linewidths=16,s=35,label="initial points")
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
            plt.savefig(os.path.join(data_dir,"results","2d","optim","%s_%s_%d_regions.pdf" %(network, object_list[class_nb],object_nb )),bbox_inches='tight')
        else:
            plt.savefig(os.path.join(data_dir,"results","2d","%s_%s_%d.pdf" %(network, object_list[class_nb],object_nb )),bbox_inches='tight')
            
def visualize_network_2_avg(map_dict_list,object_list,optim_dict=None,data_dir=None,heatmap=True):
    """
    create a an average 2d semantic map of all the maps in map_dict_list and save as pdf
    """
    network = map_dict["network_name"]
    class_nb = map_dict_list[0]["class_nb"] ; yy = map_dict_list[0]["yy"] ; xx= map_dict_list[0]["xx"] ; z = np.mean(np.array([map_dict_list[ii]["z"] for ii in range(len(map_dict_list))]),axis=0)

    plt.figure(figsize = (9, 6))
#     plt.rc('text', usetex=True)
#     plt.rc('font', family='serif')
    fig,ax = plt.subplots(1)
    plt.title("Average %s softmax scores of %s class" %(network,object_list[class_nb].upper()),fontsize=15)
    plt.xlabel("Azimuth rotation around the object(degrees)",fontsize=13)
    plt.ylabel("Elevation angle of view-point",fontsize=13)
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
    if data_dir:
        plt.savefig(os.path.join(data_dir,"results","2d","%s_%s_%s.pdf" %(network, object_list[class_nb],"Average" )),bbox_inches='tight')

def visualize_network_2_avg_avg(all_networks_list,object_list,optim_dict=None,data_dir=None,heatmap=True):
    """
    create a 
    """
#     network = map_dict["network_name"]
    class_nb = all_networks_list[0][0]["class_nb"] ; yy = all_networks_list[0][0]["yy"] ; xx= all_networks_list[0][0]["xx"] 
    
    z = np.mean(np.array([np.mean(np.array([x[ii]["z"] for ii in range(len(map_dict_list))]),axis=0) for x in all_networks_list]),axis=0)

    plt.figure(figsize = (9, 6))
#     plt.rc('text', usetex=True)
#     plt.rc('font', family='serif')
    fig,ax = plt.subplots(1)
    plt.title("Average confidence score of %s class in ImageNet" %(object_list[class_nb].upper()),fontsize=15)
    plt.xlabel("Azimuth rotation around the object(degrees)",fontsize=13)
    plt.ylabel("Elevation angle of view-point",fontsize=13)
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
    if data_dir:
        plt.savefig(os.path.join(data_dir,"results","2d","%s_%s_2D.pdf" %(object_list[class_nb],"Average" )),bbox_inches='tight')


def visulaize_mesh(anchor,object_list,data_dir,view_3d=False):
    filename_obj =  os.path.join(data_dir,"{}.obj".format(object_list[anchor]))
    filename_out =  os.path.join(data_dir,"{}_out.gif".format(object_list[anchor]))
    display(HTML("<img src='{}'></img>".format(filename_out)))
    if view_3d:
        logging.disable(sys.maxsize)
        mesh = trimesh.load(filename_obj)
        return mesh

def visualize_network_1(map_dict,object_list,optim_dict=None,data_dir=None,heatmap=True):
    """
    create a figure of 1D case of regions areound some points if optim_dict != None , if optim_dict =None , then it only save the 1D map of the entwork in map_dict as pdf
    """
    network = "ResNet50"
    class_nb = optim_dict["class_nb"] ; y = map_dict["y"] ; x= map_dict["x"] 

    plt.figure(figsize = (9, 6))
#     plt.rc('text', usetex=True)
#     plt.rc('font', family='serif')
    fig,ax = plt.subplots(1)
    plt.title("%s softmax scores of %s class" %(network,object_list[class_nb].upper()),fontsize=15)
    plt.xlabel("Azimuth rotation around the object(degrees)",fontsize=13)
    plt.ylabel("%s softmax scores" %(network),fontsize=13)
    
#     y = np.poly1d(np.array(y))
    plt.plot(x,y,linewidth=1.5,alpha=1)
    plt.xlim(0,360)
    plt.ylim(0,1)    
    colors_dict = {"naive":(0.796,0.215,0.981),"OIR_B":(0.,0.0588,0.221),"OIR_W":(0.24117,0.9598,0.0666),"trap":(1,0,0)}
    if optim_dict:
        a = ax.scatter(np.array(optim_dict["initial_point"]),np.array(0.2+np.zeros_like(optim_dict["initial_point"])),c='r',marker='x',alpha=1,linewidths=12,s=30,label="initial points")
        artist_list = [] ; label_list = [] 
        label_list.append("initial points") ; artist_list.append(a)
        for exp,proprts in optim_dict.items():
            if exp != "naive" and exp != "OIR_B" and exp != "OIR_W":
                continue
            rectangles = [region2interval(optim_dict[exp]["regions"][ii]) for ii in range(len(optim_dict["initial_point"]))]
            rect = patches.Rectangle((rectangles[0][0],rectangles[0][1]),rectangles[0][2],rectangles[0][3],linewidth=1,edgecolor=colors_dict[exp],facecolor='none',label=exp.replace("_"," "))
            a = ax.add_patch(rect)
            label_list.append(exp.replace("_"," "))
            artist_list.append(a)
            for idx,rectangle in enumerate(rectangles):
                rect = patches.Rectangle((rectangle[0],rectangle[1]),rectangle[2],rectangle[3],linewidth=1,edgecolor=colors_dict[exp],facecolor='none',label=exp.replace("_"," "))
                ax.add_patch(rect)

        ax.legend(artist_list ,label_list,fontsize=8)



    if data_dir:
        if optim_dict:
            plt.savefig(os.path.join(data_dir,"results","1d","1D_%s_regions.pdf" %(object_list[class_nb])),bbox_inches='tight')
        else:
            plt.savefig(os.path.join(data_dir,"results","1d","%s_%s_%d.pdf" %(network, object_list[class_nb],object_nb )),bbox_inches='tight')

def visulize_trace(optimization_traces,loss_traces,object_list,point_nb=0,exp_type=None,show_loss=True):
    """
    visualize the trace taken by the optimization on the two variables (a,b) contained in the list optimization_trace
    
    """
    optimization_trace = optimization_traces[point_nb] ; loss_trace = loss_traces[point_nb]
    t = np.array(range(len(loss_trace)))
    fig, ax1 = plt.subplots()
    plt.title("optimization trace of %s class in the %s exp" %("bathtub".upper(),str(exp_type).upper()))
    ax1.set_xlabel("iterations (steps)")
    ax1.set_ylabel("azimuth rotations change " )
    aa = np.array([x[0] for x in optimization_trace])
    bb = np.array([x[1] for x in optimization_trace])
    ax1.plot(t,aa,linewidth=1,alpha=1,label="the left bound a")
    ax1.plot(t,bb,linewidth=1,alpha=1,label="the right bound b")
    ax1.tick_params(axis='y')
    plt.legend()
    ax1.set_xlim(0,len(optimization_trace))   
    ax1.set_ylim(np.min(aa),np.max(bb))
    
    if show_loss:
        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('loss', color=color)  # we already handled the x-label with ax1
        ax2.plot(t, np.array(loss_trace), color=color)
        ax2.tick_params(axis='y', labelcolor=color)
    plt.savefig(os.path.join(data_dir,"results","1d","run%d_%d_%d_%s.pdf" %(point_nb,class_nb,object_nb,exp_type )))

def visulize_region_growing_1(optimization_traces,map_dict,object_list,initial_points=[],point_nb=0,data_dir=None,exp_type=None):
    """
    save jpgs of the 1D regions while growing   
    
    """
    point_nb = 0
    initial_point = initial_points[point_nb]
    optimization_trace = optimization_traces[point_nb]
    t = np.array(range(len(optimization_trace)))
    for ii,bound in enumerate(optimization_trace) :
        a,b = bound ; r = b-a
        plt.figure(figsize = (9, 6))
        fig,ax = plt.subplots(1)
        plt.title("%s softmax scores of %s class" %(network,object_list[class_nb].upper()),fontsize=15)
        plt.xlabel("Azimuth rotation around the object(degrees)",fontsize=13)
        plt.ylabel("%s softmax scores" %(network),fontsize=13)
        class_nb = optim_dict["class_nb"] ; y = map_dict["y"] ; x= map_dict["x"] 
        plt.plot(x,y,linewidth=1.5,alpha=1)
        plt.xlim(0,360)
        plt.ylim(0,1)    
        colors_dict = {"naive":(0.796,0.215,0.981),"OIR_B":(0.,0.0588,0.221),"OIR_W":(0.24117,0.9598,0.0666),"trap":(1,0,0)}
        ar = ax.scatter(np.array(initial_point),np.array(0.2+np.zeros_like(initial_point)),c='r',marker='x',alpha=1,linewidths=12,s=30,label="initial points")
        artist_list = [] ; label_list = [] 
        label_list.append("initial points") ; artist_list.append(ar)
        for exp,proprts in optim_dict.items():
            if exp != "naive" and exp != "OIR_B" and exp != "OIR_W":
                continue
            rectangles = [(a,0,r,1)]
            rect = patches.Rectangle((rectangles[0][0],rectangles[0][1]),rectangles[0][2],rectangles[0][3],linewidth=1,edgecolor=colors_dict[exp],facecolor='none',label=exp.replace("_"," "))
            ar = ax.add_patch(rect)
            label_list.append(exp.replace("_"," "))
            artist_list.append(ar)
            for idx,rectangle in enumerate(rectangles):
                rect = patches.Rectangle((rectangle[0],rectangle[1]),rectangle[2],rectangle[3],linewidth=1,edgecolor=colors_dict[exp],facecolor='none',label=exp.replace("_"," "))
                ax.add_patch(rect)
        ax.legend(artist_list ,label_list,fontsize=8)
        plt.savefig(os.path.join(data_dir,"examples","1d","%d.jpg" %(ii)))
   