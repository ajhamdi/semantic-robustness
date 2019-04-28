<img src='./some_examples/robustness_out.gif' align="right" width=180> <br>
<img src='.//results/teapot.jpg' align="right" width=500><br>
<br><br><br><br>

# semantic-robustness
### [Paper](https://arxiv.org/pdf/1904.04621.pdf) | [Tutorial](https://colab.research.google.com/drive/1cZzTPu1uwftnRLqtIIjjqw-YZSKh4QYn)<br>
Pytorch implementation of our method for ...... <br><br>
[Towards Analyzing Semantic Robustness of Deep Neural Networks](https://arxiv.org/pdf/1904.04621.pdf)  
 [Abdullah Hamdi](http://www.fihm.ai), [Bernard Ghanem](http://www.bernardghanem.com/)
 
## examples
- Our label-to-streetview results
<p align='center'>  
  <img src='imgs/teaser_label.png' width='400'/>
  <img src='imgs/teaser_ours.jpg' width='400'/>
</p>
- Interactive editing results
<p align='center'>  
  <img src='imgs/teaser_style.gif' width='400'/>
  <img src='imgs/teaser_label.gif' width='400'/>
</p>
- Additional streetview results
<p align='center'>
  <img src='imgs/cityscapes_1.jpg' width='400'/>
  <img src='imgs/cityscapes_2.jpg' width='400'/>
</p>
<p align='center'>
  <img src='imgs/cityscapes_3.jpg' width='400'/>
  <img src='imgs/cityscapes_4.jpg' width='400'/>
</p>

- Label-to-face and interactive editing results
<p align='center'>
  <img src='imgs/face1_1.jpg' width='250'/>
  <img src='imgs/face1_2.jpg' width='250'/>
  <img src='imgs/face1_3.jpg' width='250'/>
</p>
<p align='center'>
  <img src='imgs/face2_1.jpg' width='250'/>
  <img src='imgs/face2_2.jpg' width='250'/>
  <img src='imgs/face2_3.jpg' width='250'/>
</p>

- Our editing interface
<p align='center'>
  <img src='imgs/city_short.gif' width='330'/>
  <img src='imgs/face_short.gif' width='450'/>
</p>

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


### Testing
- A few example Cityscapes test images are included in the `datasets` folder.
- Please download the pre-trained Cityscapes model from [here](https://drive.google.com/file/d/1h9SykUnuZul7J3Nbms2QGH1wa85nbN2-/view?usp=sharing) (google drive link), and put it under `./checkpoints/label2city_1024p/`
- Test the model (`bash ./scripts/test_1024p.sh`):
```bash
#!./scripts/test_1024p.sh
python test.py --name label2city_1024p --netG local --ngf 32 --resize_or_crop none
```
The test results will be saved to a html file here: `./results/label2city_1024p/test_latest/index.html`.

More example scripts can be found in the `scripts` directory.


### Dataset
- We use the Cityscapes dataset. To train a model on the full dataset, please download it from the [official website](https://www.cityscapes-dataset.com/) (registration required).
After downloading, please put it under the `datasets` folder in the same way the example images are provided.


### Training
- Train a model at 1024 x 512 resolution (`bash ./scripts/train_512p.sh`):
```bash
#!./scripts/train_512p.sh
python train.py --name label2city_512p
```
- To view training results, please checkout intermediate results in `./checkpoints/label2city_512p/web/index.html`.
If you have tensorflow installed, you can see tensorboard logs in `./checkpoints/label2city_512p/logs` by adding `--tf_log` to the training scripts.

### Cluster testing 
- Test the models on the cluster of GPUs ( if you have access to one ) 
```bash
bash gpu_start.sh
```
Note: this is not tested and we trained our model using single GPU only. Please use at your own discretion.

### Training with Automatic Mixed Precision (AMP) for faster speed
- To train with mixed precision support, please first install apex from: https://github.com/NVIDIA/apex
- You can then train the model by adding `--fp16`. For example,
```bash
#!./scripts/train_512p_fp16.sh
python -m torch.distributed.launch train.py --name label2city_512p --fp16
```
In our test case, it trains about 80% faster with AMP on a Volta machine.

### Training at full resolution
- To train the images at full resolution (2048 x 1024) requires a GPU with 24G memory (`bash ./scripts/train_1024p_24G.sh`), or 16G memory if using mixed precision (AMP).
- If only GPUs with 12G memory are available, please use the 12G script (`bash ./scripts/train_1024p_12G.sh`), which will crop the images during training. Performance is not guaranteed using this script.

### Training with your own dataset
- If you want to train with your own dataset, please generate label maps which are one-channel whose pixel values correspond to the object labels (i.e. 0,1,...,N-1, where N is the number of labels). This is because we need to generate one-hot vectors from the label maps. Please also specity `--label_nc N` during both training and testing.
- If your input is not a label map, please just specify `--label_nc 0` which will directly use the RGB colors as input. The folders should then be named `train_A`, `train_B` instead of `train_label`, `train_img`, where the goal is to translate images from A to B.
- If you don't have instance maps or don't want to use them, please specify `--no_instance`.
- The default setting for preprocessing is `scale_width`, which will scale the width of all training images to `opt.loadSize` (1024) while keeping the aspect ratio. If you want a different setting, please change it by using the `--resize_or_crop` option. For example, `scale_width_and_crop` first resizes the image to have width `opt.loadSize` and then does random cropping of size `(opt.fineSize, opt.fineSize)`. `crop` skips the resizing step and only performs random cropping. If you don't want any preprocessing, please specify `none`, which will do nothing other than making sure the image is divisible by 32.

## More Training/Test Details
- Flags: see `options/train_options.py` and `options/base_options.py` for all the training flags; see `options/test_options.py` and `options/base_options.py` for all the test flags.
- Instance map: we take in both label maps and instance maps as input. If you don't want to use instance maps, please specify the flag `--no_instance`.


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
This code borrows heavily from [neural mesh renderer](https://github.com/daniilidis-group/neural_renderer).
