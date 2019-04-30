<img src='.//results/teapot.jpg' align="right" width=500>
<img src='./some_examples/robustness_out.gif' align="center" width=250>  
<br><br><br><br>

# Semantic Robustness of Deep Neural Networks
### [Paper](https://arxiv.org/pdf/1904.04621.pdf) | [Tutorial](https://colab.research.google.com/drive/1cZzTPu1uwftnRLqtIIjjqw-YZSKh4QYn)<br>
Pytorch implementation of the paper on Arxive. The paper tries to address the robustness of Deep Neeural Networks, but not from pixel-level perturbation lense, rather from semantic lense in which the perturbation happens in the latent parameters that generate the image. This typer of robustness for safety-critical applications like self-driving cars in which tolerance of error is very low and risk of failure is high. <br><br>
[Towards Analyzing Semantic Robustness of Deep Neural Networks](https://arxiv.org/pdf/1904.04621.pdf)  
 [Abdullah Hamdi](http://www.fihm.ai), [Bernard Ghanem](http://www.bernardghanem.com/)

## Examples of Results
- ### visualizing the Deep networks average semantic profiles for 10 objects.
<img src='.//results/toilet.jpg' align="center" width=500>

- ### visualizing the Deep networks semantic profiles for average  objects.
<img src='.//results/toilet.jpg' align="center" width=500>

- ### visualizing the Deep networks semantic profiles for average  objects.
<img src='.//results/toilet.jpg' align="center" width=500>



## Prerequisites
- Linux 
- Python 2 or 3
- NVIDIA GPU (11G memory or larger) + CUDA cuDNN

## Getting Started
### Installation
- install [anaconda](https://docs.anaconda.com/anaconda/install/) and then run the following commans 
```bash
conda env create -f environment.yaml
source activate semantic
conda install -c anaconda cudatoolkit==9.0
pip install git+https://github.com/daniilidis-group/neural_renderer
```
- Clone this repo:
```bash
git clone https://github.com/ajhamdi/semantic-robustness
cd semantic-robustness
```


### Simple Colab Tutorial with a toy example:
We provide a simple tutorial on colab [here](https://colab.research.google.com/drive/1cZzTPu1uwftnRLqtIIjjqw-YZSKh4QYn) to test a toy example on some 3D objects and apply the bound optimzation algorithms for sample points in the 1D case directly on the cloud. The complete results obtained in the `results` directory are obtained as of the following sections  
The test results will be saved as ` to a html file here: `./results/__network name__/*.html`.

More example scripts can be found in the `scripts` directory.


### Dataset
- We collect 100 3D shapes from 10 classes from [ShapeNet](https://www.shapenet.org/) that are also exist in [ImagNet](http://www.image-net.org/) and made sure that networks trained on ImageNEt identifies these shapes of ShapeNet nefore proceeding. All the obj files are availabe in the `sacale` directory which contain the dataset. The following are visuailzations of the 3D dataset used in our paper.

<img src='./results/class_0_.gif' width=150>  <img src='./results/class_1_.gif' width=150>  <img src='./results/class_2_.gif' width=150>  <img src='./results/class_3_.gif' width=150>  <img src='./results/class_4_.gif' width=150>  <img src='./results/class_5_.gif' width=150> <img src='./results/class_6_.gif' width=150>  <img src='./results/class_7_.gif' width=150> <img src='./results/class_8_.gif' width=150>  <img src='./results/class_9_.gif' width=150> 

### Testing with your own 3D dataset and networks
- `TODO NEXT`

## Citation

If you find this useful for your research, please use the following.

```
@article{hamdi2019towards,
  title={Towards Analyzing Semantic Robustness of Deep Neural Networks},
  author={Hamdi, Abdullah and Ghanem, Bernard},
  journal={arXiv preprint arXiv:1904.04621},
  year={2019}
}
```

## Acknowledgments
This work is suuported by King Abdullah University of Science and Technology (KAUST).The code borrows heavily from [neural mesh renderer](https://github.com/daniilidis-group/neural_renderer).
