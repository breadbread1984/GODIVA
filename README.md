# GODIVA
this project implements text2video algorithm introduced in paper [GODIVA: Generating open-doain videos from natural descriptions](https://www.arxiv-vanity.com/papers/2104.14806/)

## pretrain

generate imagenet dataset with [this script](https://github.com/tensorflow/models/blob/r1.13.0/research/slim/datasets/build_imagenet_data.py).

pretrain VQ-VAE on imagenet with command

```shell
python3 pretrain.py --mode train --type (original|ema_update) --train_dir <path/to/trainset> --test_dir <path/to/testset>
```

save checkpoint to pretrain model file with command

```shell
python3 pretrain.py --mode save --type (original|ema_update) --save_model
```

test pretrained model with command

```shell
python3 pretrain.py --mode test --type (original|ema_update) --img <path/to/image>
```

a pair of imagenet-pretrained ema update encoder and decoder are provided in this repo.

here are some reconstruction examples.

<p align="center">
 <table>
   <tr><td><img src="pics/car.png" /></td><td><img src="pics/cat.png" /></td><td><img src="pics/house.png" /></td><td><img src="pics/people.png"></td></tr>
 </table>
</p>

## generate moving mnist dataset

create moving single digit dataset with command

```shell
python3 dataset/mnist_caption_single.py
```

after executing successfully, a file named mnist_single_git.h5 is generated.

create moving double digits dataset with command

```shell
python3 dataset/mnist_caption_two_digit.py
```

after executing successfully, a file named mnist_two_gif.h5 is generated.
the dataset creation code is borrowed from [Sync-Draw](https://github.com/syncdraw/Sync-DRAW/tree/master/dataset) and slightly modified.

## train GODIVA on moving mnist dataset

train GODIVA with command

```shell
python3 train.py (single|double)
```
