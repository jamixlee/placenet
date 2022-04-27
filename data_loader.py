import collections
import os
import io
import random
import numpy as np
from PIL import Image, ImageFile
import torch
from torch.utils.data import Dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True  # To prevent problem with corrupted images

# Class type of the pickled data
Scene = collections.namedtuple('Scene', ['image_r', 'image_d', 'image_o', 'poses'])

class HouseData(Dataset):
    def __init__(self, root_dir, dataset, image_size=64, attention=None, transform=None, target_transform=None):
        self.root_dir = root_dir
        self.dataset = dataset
        self.image_size = image_size
        self.transform = transform
        self.target_transform = target_transform
        self.attention = attention
        self.load_data = load_data

    def __len__(self):
        return len(os.listdir(self.root_dir))

    def __getitem__(self, idx):
        scene_path = os.path.join(self.root_dir, '{}.pt'.format(idx))
        poses, image = self.load_data(scene_path, self.attention)

        if self.target_transform:
            poses = self.target_transform(poses, self.dataset)

        poses = torch.FloatTensor(poses)
        image = torch.FloatTensor(image).permute(0,1,4,2,3) / 255

        return poses, image


def open_image(frame):
    image = Image.open(io.BytesIO(frame))
    image_np = np.array(image, dtype='u1')
    image.close()
    del image
    return image_np


def load_data(scene_path, attention):
    # read a file
    fp = open(scene_path, 'rb')
    data = torch.load(fp)
    fp.close()

    # get poses
    poses = data.poses[0]  # numpy.ndarray (not torch.tensor)

    # get scene images
    image_r = np.stack([open_image(frame) for frame in data.image_r])
    if attention is not None:
        if 'd' in attention:
            image_d = np.stack([open_image(frame) for frame in data.image_d])
        if 'o' in attention:
            image_o = np.stack([open_image(frame) for frame in data.image_o])

        if 'd' in attention and 'o' in attention:
            image = np.stack([image_r, image_d, image_o], axis=0)
        elif 'd' in attention:
            image = np.stack([image_r, image_d], axis=0)
        elif 'o' in attention:
            image = np.stack([image_r, image_o], axis=0)
        else:
            image = image_r[None, :, :, :, :]
    else:
        image = image_r[None, :, :, :, :]

    return poses, image


def norm_vec(v, range_in=None, range_out=None):
    if range_out is None:
        range_out = [-1, 1]
    if range_in is None:
        range_in = [np.min(v, 0), np.max(v, 0)]  #range_in = [torch.min(v,0), torch.max(v,0)]
    r_out = range_out[1] - range_out[0]
    r_in = range_in[1] - range_in[0]
    v = (r_out * (v-range_in[0]) / r_in) + range_out[0]
    return v


def transform_poses(poses, dataset):
    position, orientation = np.split(poses, [3], axis=-1)
    yaw, pitch = np.split(orientation, [1], axis=-1)

    if dataset == 'Mines':
        position = position.view(-1, 3)[:,[0, 2, 1]]  # reorder y and z axis for Minecraft dataset
        position = norm_vec(position, [0, 40])        # Normalize position vector to be scaled between -1 to 1
    elif dataset == 'House':
        position = norm_vec(position, [0, 100])
    elif dataset == 'Mazes':
        position = norm_vec(position, [-0.4, 0.4])
    else:
        pass

    pose_vector = [position, np.cos(yaw), np.sin(yaw), np.cos(pitch), np.sin(pitch)]
    poses = np.concatenate(pose_vector, axis=-1)

    return poses


def sample_batch(v_data, x_data, dataset, seed=None, obs_range=None, obs_count=None):
    random.seed(seed)

    # Maximum number of contexts
    if dataset == 'Mazes':
        num_context = 20
    elif dataset == 'Mines':
        num_context = 15
    elif dataset == 'House':
        num_context = 20
    else:
        num_context = 15

    # Random number of contexts
    if obs_count is None:
        obs_count = random.randint(3, num_context-3)

    # Get a random range with the size of 'obs_range'
    if obs_range is not None:
        l = random.randint(0, x_data.size(2) - obs_range)
        r = l + obs_range
    else:
        l = 0
        r = x_data.size(2)

    obs_list = list(range(l, r))

    # Sample a random query frame
    idx_query = random.randint(l, r-1)
    obs_list.remove(idx_query)

    # Sample M frames in random positions
    idx_context = random.sample(obs_list, obs_count)

    # Contexts
    v = v_data[:, idx_context]
    x = x_data[:, :, idx_context]

    # Query
    v_q = v_data[:, idx_query]
    x_q = x_data[:, 0, idx_query]

    return v, v_q, x, x_q
