# CADEL: Long-tailed Classification via CAscaded Deep Ensemble Learning

Zhi Chen, Jiang Duan, Li Kang and Guoping Qiu

This repository is the official PyTorch implementation of the paper [CADEL](https://arxiv.org/abs/).

## Environments

```shell
pytorch >= 1.8.0
timm == 0.3.2
```

1. If your PyTorch is 1.8.0+, a [fix](https://github.com/huggingface/pytorch-image-models/issues/420) is needed to work with timm.

## Data preparation

You can download the original datasets as follows:

- ImageNet_LT and Places_LT
  
  Download the [ImageNet_2014](http://image-net.org/index) and [Places_365](http://places2.csail.mit.edu/download.html).

- iNaturalist 2018
  
  - Download the dataset following [here](https://github.com/visipedia/inat_comp/tree/master/2018).

Change the `data_root` in `main_CNN.py and main_ViT.py` accordingly.

After preparation, the file structures are as follows:

```shell
/path/to/ImageNet-LT/
    train/
        class1/
            img1.jpeg
        class2/
            img2.jpeg
    val/
        class1/
            img3.jpeg
        class2/
            img4.jpeg
    train.txt
    val.txt
    test.txt
    num_shots.txt
```

train.txt, val.txt and test.txt list the file names, and  num_shots.txt gives the number of training images in each class. All these data files have been uploaded to this repo.

## Usage

1. You can see all our settings in ./config/

2. Typically, 2 GPUs and >=24 GB per GPU Memory are available. But when training ViT-B-16 with a training resolution of 384, bigger GPU Memory is required.

For the stage one training, you can train the model with DataParallel or DistributedDataParallel. Specially, for stage one training, the commands are:

```python
# Stage one
python main_CNN.py (or python main_ViT.py if you want to train ViT)
or
torch.distributed.launch --nproc_per_node=n main_CNN.py

where n is the number of gpus in your server. And you should divide the defaulting batch_size in our configs with n.

# Stage Two
python main_CNN_PC.py
```

## Results of CNNs

| Datasets    | Many | Medium | Few  | All  | Model                                                                                             |
| ----------- |:----:|:------:|:----:|:----:|:-------------------------------------------------------------------------------------------------:|
| ImageNet-LT | 67.5 | 55.6   | 43.2 | 58.5 | [ResNet50](https://drive.google.com/drive/folders/1oabSU5re8428jwtMj_iXX6fSc8stdzer?usp=sharing)  |
| ImageNet-LT | 68.8 | 55.8   | 44.0 | 59.2 | [ResNeXt50](https://drive.google.com/drive/folders/1X7EsmjGQpBVeHY1gfC7pfB1F2gJ-lukL?usp=sharing) |
| iNat18      | ---  | ---    | ---  | 73.5 | [ResNet50](https://drive.google.com/drive/folders/1NT4pnGHUQUcz7LR4DsQAzcgMz0TxsXiV?usp=sharing)  |
| Places-LT   | ---  | ---    | ---  | 41.4 | [ResNet152](https://drive.google.com/drive/folders/18osTC-Hx1iFuMV9A2tRSdjVxICMIHCkn?usp=sharing) |

## Results of ViTs

<table class="tg">
<thead>
  <tr>
    <th class="tg-nrix">Dataset</th>
    <th class="tg-nrix">Resolution</th>
    <th class="tg-nrix">Many</th>
    <th class="tg-nrix">Med.</th>
    <th class="tg-nrix">Few</th>
    <th class="tg-nrix">Acc</th>
    <th class="tg-nrix">Pretrain ckpt</th>
  </tr>
</thead>
<tbody> 
  <tr>
    <td class="tg-57iy">ImageNet-LT</td>
    <td class="tg-57iy">224*224</td>
    <td class="tg-57iy">70.3</td>
    <td class="tg-57iy">59.8</td>
    <td class="tg-57iy">47.5</td>
    <td class="tg-57iy"><a href="https://drive.google.com/drive/folders/1sXUf-g4quPMDmBuYyPZeh94VhmHH4zW9?usp=sharing">61.7</a></td>
    <td class="tg-57iy" rowspan="2"><a href="https://drive.google.com/file/d/1xMYcz01wutzwhPqNmMHUQE1CT76-Q0LU/view?usp=sharing">Res_224</a></td>
  </tr>
  <tr>
    <td class="tg-nrix">ImageNet-LT</td>
    <td class="tg-nrix">384*384</td>
    <td class="tg-nrix">73.0</td>
    <td class="tg-nrix">62.2</td>
    <td class="tg-nrix">50.3</td>
    <td class="tg-57iy"><a href="https://drive.google.com/drive/folders/18XCKaKe6xaUrqOvJzAz0zSTVc4tgQc3E?usp=sharing">64.7</a></td>
  </tr>
  <tr>
    <td class="tg-57iy">iNat18</td>
    <td class="tg-57iy">224*224</td>
    <td class="tg-57iy">77.7</td>
    <td class="tg-57iy">76.3</td>
    <td class="tg-57iy">75.1</td>
    <td class="tg-57iy"><a href="https://drive.google.com/drive/folders/1_iPWKjAs5toPKz9FUy6arH_FzYUK-WMg?usp=sharing">76.2</a></td>
    <td class="tg-57iy" rowspan="2"><a href="https://drive.google.com/file/d/1RW0sQayJ6vXBCC9N1bF-skfbvo9moZhw/view?usp=sharing">Res_128</a></td>
  </tr>
  <tr>
    <td class="tg-nrix">iNat18</td>
    <td class="tg-nrix">384*384</td>
    <td class="tg-nrix">75.0</td>
    <td class="tg-nrix">81.8</td>
    <td class="tg-nrix">85.4</td>
    <td class="tg-nrix"><span style="color:#000"><a href="https://drive.google.com/drive/folders/1gaWtrsznfw3_aqQiG4OSvgcewr3QmC-l?usp=sharing">82.7</a></span></td>
  </tr>
    <tr>
    <td class="tg-57iy">Places-LT</td>
    <td class="tg-57iy">224*224</td>
    <td class="tg-57iy">46.6</td>
    <td class="tg-57iy">46.7</td>
    <td class="tg-57iy">46.5</td>
    <td class="tg-57iy"><a href="https://drive.google.com/drive/folders/1j4Th5Mrtp8AtyRoL3AhS4AKC0jmHC8uU?usp=sharing">46.6</a></td>
    <td class="tg-57iy" rowspan="1"><a href="https://drive.google.com/file/d/1U4EQuUdshxISkDrJlnBSGUGjriDtwieW/view?usp=sharing">Image-1K-224</a></td>
  </tr>
  <tr>
    <td class="tg-nrix">Places-LT</td>
    <td class="tg-nrix">384*384</td>
    <td class="tg-nrix">47.9</td>
    <td class="tg-nrix">50.2</td>
    <td class="tg-nrix">38.5</td>
    <td class="tg-nrix"><span style="color:#000"><a href="https://drive.google.com/drive/folders/1PnSv2YpqSLkkeM2-SAdvrSjY55_Vv6dm?usp=sharing">47.1</a></span></td>
    <td class="tg-57iy" rowspan="1"><a href="https://drive.google.com/file/d/1EaIQKLBDec1a2wGt1IzO_RJvNlP9qEu6/view?usp=sharing">Image-1K-384</a></td>

</tr>
</tbody>
</table>

## Citation

If you find our idea or code inspiring, please cite our paper:

```bibtex
@article{CADEL,
  title={CADEL: Long-tailed Classification via CAscaded Deep Ensemble Learning},
  author={Zhi Chen, Jiang Duan, Li Kang, Xin Li and Guoping Qiu},
  year={2023},
  archivePrefix={arXiv},
  primaryClass={cs.AI}
}
```

This code is partially based on [cRT](https://github.com/facebookresearch/classifier-balancing) and  [LiVT](https://github.com/XuZhengzhuo/LiVT), if you use our code, please also cite：

```bibtex
@inproceedings{kang2019decoupling,
  title={Decoupling representation and classifier for long-tailed recognition},
  author={Kang, Bingyi and Xie, Saining and Rohrbach, Marcus and Yan, Zhicheng
          and Gordo, Albert and Feng, Jiashi and Kalantidis, Yannis},
  booktitle={Eighth International Conference on Learning Representations (ICLR)},
  year={2020}
}
```

```bibtex
@inproceedings{LiVT,
  title={Learning Imbalanced Data with Vision Transformers},
  author={Xu, Zhengzhuo and Liu, Ruikang and Yang, Shuo and Chai, Zenghao and Yuan, Chun},
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2023}
}
```

## Acknowledgements

This project is based on [cRT](https://github.com/facebookresearch/classifier-balancing) and [LiVT](https://github.com/XuZhengzhuo/LiVT).
