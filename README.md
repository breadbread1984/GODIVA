# GODIVA
this project implements text2video algorithm introduced in paper [GODIVA: Generating open-doain videos from natural descriptions](https://www.arxiv-vanity.com/papers/2104.14806/)

## pretrain

generate imagenet dataset with [this script](https://github.com/tensorflow/models/blob/r1.13.0/research/slim/datasets/build_imagenet_data.py).

pretrain VQ-VAE on imagenet with command

```shell
python3 pretrain.py <path/to/trainset> <path/to/testset>
```

