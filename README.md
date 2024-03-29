# PyTorch Implementation of PlaceNet

![placenet_teaser](https://user-images.githubusercontent.com/6002018/165482719-37003726-b5de-43f3-b597-e4f39867ea82.png)


## Dependencies

- python>=3.6
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
- `TBD`


## House-Traveler
- You can also use your own 3D scenes to train PlaceNet.
- It will be helpful to use the House-Traveler, a trajectory generator that create natural paths in 3D environments.
- https://github.com/Yoo-Youngjae/house_traveler


## Note

- To implement a 3D scene representation and rendering approach, we referenced the Generative Query Networks ([GQN](https://www.science.org/doi/10.1126/science.aar6170)).
  
- In the course of developing this project, we referenced various GQN implementations:

  - https://github.com/ogroth/tf-gqn
  - https://github.com/wohlert/generative-query-network-pytorch
  - https://github.com/masa-su/pixyzoo/tree/master/GQN
  - https://github.com/iShohei220/torch-gqn
  
  - Among them, I mostly refer to [iShohei220](https://github.com/iShohei220/torch-gqn)'s work, which is composed of the most intuitive codes.


## To be shared

- Pretrained models
- Visualization codes
- Input data template
