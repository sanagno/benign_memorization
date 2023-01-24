## Code for the ICLR2023 paper `The Curious Case of Benign Memorization'.

Experiments when run using `Python 3.8.5`, `CUDA Version 11.3` and `Pytorch 1.10.0`.

No changes are expected for different versions.

To install dependencies simply run 

``pip install -r requirements.txt``.

To check the available flags run 

``python main.py --help``

Most importantly the flag `label_noise` controls the percentage of noise in the labels of the dataset and the flag `augmentation` control the type of augmentations to use. Valid options include `none` for no augmentations `full` for the set of augmentations used (except `mixup` that has its own flag) and `fullfixed_n` that specifies a fixed number of augmentations per sample. The  number `n` should be replaced by the specified number of augmentations per sample.

To reproduce our random label experiments with augmentations, simply run

``
CUDA_VISIBLE_DEVICES=0 python main.py --dataset CIFAR10 --model resnet18_small --epochs 10000 --batch-size 256 --comment random_labels --lr 3e-4 --image-size 32 --augmentation full --label-noise 1.0 --bottleneck-dim 65536
``

and wait for a few days.

If you like our paper please cite as:
```
@article{anagnostidis2022curious,
  title={The Curious Case of Benign Memorization},
  author={Anagnostidis, Sotiris and Bachmann, Gregor and Noci, Lorenzo and Hofmann, Thomas},
  journal={arXiv preprint arXiv:2210.14019},
  year={2022}
}
```
