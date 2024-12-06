# FBKT

## Requirements
- [PyTorch >= version 1.1 and torchvision](https://pytorch.org)
- tqdm

## Datasets
We provide the source code on three benchmark datasets, i.e., CIFAR100, CUB200 and miniImageNet. Please follow the guidelines in [CEC](https://github.com/icoz69/CEC-CVPR2021) to prepare them.
As for StanfordCar in [Link](https://github.com/cyizhuo/Stanford-Cars-dataset?tab=readme-ov-file).

## Code Structures
There are five parts in the code.
 - `models`: It contains the backbone network and training protocols for the experiment.
 - `data`: Images and splits for the data sets.
- `dataloader`: Dataloader of different datasets.
 - `augmentations`: The augmentations methods.
 
## Training scripts

- CIFAR100

  ```
  python train.py -project fbkt -dataset cifar100 -lr_base 0.1 -lr_new 0.001 -decay 0.0005 -epochs_base 600 -schedule Cosine -gpu 0  -size_crops 32 18 -min_scale_crops 0.9 0.2 -max_scale_crops 1.0 0.7 -alpha 0.1 -beta 0.2
  ```
  
- CUB200
    ```
  python train.py -project fbkt -dataset cub200 -lr_base 0.002 -lr_new 0.000005 -decay 0.0005 -epochs_base 140 -schedule Milestone -milestones 75 95 120  -gpu '0'  -size_crops 224 96 -min_scale_crops 0.2 0.05 -max_scale_crops 1.0 0.14 -alpha 0.1 -beta 0.2
    ```

- miniImageNet
    ```
    python train.py -project fbkt -dataset mini_imagenet  -lr_base 0.001 -lr_new 0.1 -decay 0.0005 -epochs_base 140 -schedule Milestone -milestones 75 95 120  -gpu '0' -size_crops 84 50 -min_scale_crops 0.2 0.05 -max_scale_crops 1.0 0.14 -alpha 0.1 -beta 0.2
  ```
  
- StanfordCar
    ```
  python train.py -project fbkt -dataset stanfordcar -lr_base 0.002 -lr_new 0.000005 -decay 0.0005 -epochs_base 140 -schedule Milestone -milestones 75 95 120  -gpu '0'  -size_crops 224 96 -min_scale_crops 0.2 0.05 -max_scale_crops 1.0 0.14 -alpha 0.1 -beta 0.2
    ```

Remember to change `YOURDATAROOT` into your own data root. If you want to use incremental finetuning, set `-incft`. 

