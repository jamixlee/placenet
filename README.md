# PyTorch Implementation of PlaceNet


## Dependencies

- Python >= 3.6
- torch==1.6.0
- torchvision==0.7.0
- Pillow==7.2.0
- numpy==1.19.5

**(optional)**
- tensorboard
- libnccl2==2.7.8-1+cuda11.0
- libjemalloc-dev


## How to run

- Train: `run.sh`

  - This script basically uses all GPUs installed in the computer.
  - To use single GPU,
    - remove `ddp`, and
    - add `gpu=<the number of gpu you want to use>`


## How to monitor the training progress
- `tensorboard --logdir=<log path>`


## How to download the House dataset
- http://gofile.me/3TjAz/WmZgNAcyz
- To access the dataset, please send an email to `cylee@bi.snu.ac.kr`.


## House-Traveler
- You can also use your own 3D scenes to train PlaceNet.
- It will be helpful to use the House-Traveler, a trajectory generator that create natural paths in 3D environments.
- https://github.com/Yoo-Youngjae/house_traveler
