# GODIVA
this project implements text2video algorithm introduced in paper [GODIVA: Generating open-doain videos from natural descriptions](https://www.arxiv-vanity.com/papers/2104.14806/)

## pretrain

generate imagenet dataset with [this script](https://github.com/tensorflow/models/blob/r1.13.0/research/slim/datasets/build_imagenet_data.py).

pretrain VQ-VAE on imagenet with command

```shell
python3 pretrain.py (original|ema_update) <path/to/trainset> <path/to/testset>
```

save checkpoint to pretrain model file with command

```shell
python3 save_pretrain_model.py (original|ema_update)
```

a pair of imagenet-pretrained ema update encoder and decoder are provided in this repo.
