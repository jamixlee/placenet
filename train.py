"""
Training script of PlaceNet
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import argparse
import random
import re
import time
import shutil
import importlib
import gc

import torch
from torchvision.utils import make_grid
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.backends.cudnn as cudnn
# from apex.parallel import DistributedDataParallel as DDP

from data_loader import HouseData, Scene, transform_poses, sample_batch
from model import PlaceNet
from scheduler import AnnealingStepLR
# from skimage.metrics import structural_similarity as SSIM
from ssim import SSIM
if importlib.util.find_spec('torch.utils.tensorboard'):
    from torch.utils.tensorboard import SummaryWriter
else:
    from tensorboardX import SummaryWriter


def main():
    # Check CUDA and set the GPU as the device for CUDA
    if not torch.cuda.is_available():
        raise RuntimeError("[ERR] CUDA is not available")

    # Random Seeding (setting for reproducibility)
    if args.seed is not None:
        # This will turn on the CUDNN deterministic setting,
        # which can slow down your training considerably.
        # You may see unexpected behavior when restarting from checkpoints.
        print("[LOG] Seed training mode")
        cudnn.benchmark = False
        cudnn.deterministic = True
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.set_printoptions(precision=10)  # Number of digits of precision for floating point output

    # Path to the log
    args.log_title = "{}-{}".format(args.dataset, args.log_dir)
    args.log_dir = os.path.join(args.root_log_dir, args.log_title)

    # Make the log directory
    make_log_dir(args.log_dir, args.resume)

    # Count the number of GPUs
    num_gpu_per_node = torch.cuda.device_count()
    print("[LOG] {} GPUs found".format(num_gpu_per_node))

    # Specific single GPU has been selected to be used
    if args.gpu is not None:
        print("[LOG] Single GPU per nodes: GPU-{} is used".format(args.gpu))

    # When num_nodes is more than 1, the distributed option will be activated
    args.distributed = args.num_nodes > 1 or args.ddp

    if args.ddp:
        print("[LOG] Data parallelism mode")
        # Since we have num_gpu_per_node processes per node, the total num_nodes needs to be adjusted accordingly
        args.num_nodes = num_gpu_per_node * args.num_nodes
        # Use torch.multiprocessing.spawn to launch distributed processes: the main_worker process function
        mp.spawn(main_worker, nprocs=num_gpu_per_node, args=(num_gpu_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, num_gpu_per_node, args)


def main_worker(gpu, num_gpu_per_node, args):
    # get data path
    train_data_dir = os.path.join(args.data_dir, args.data_dir_train)
    valid_data_dir = os.path.join(args.data_dir, args.data_dir_test)

    args.gpu = gpu
    if args.gpu is not None:
        print("[GPU-{}] Ready".format(args.gpu))

    # TensorBoard writer and the ELBO criterion
    if args.rank == 0:
        writer = SummaryWriter(log_dir=os.path.join(args.log_dir, 'runs'))
        elbo_min = 20000

    if args.distributed:
        if args.host_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.ddp:
            # For multiprocessing distributed training,
            # rank needs to be the global rank among all the processes
            args.rank = args.rank * num_gpu_per_node + gpu

        dist.init_process_group(backend=args.dist_backend, init_method=args.host_url,
                                # timeout=datetime.timedelta(seconds=1800),
                                world_size=args.num_nodes, rank=args.rank)

    ########################################################
    # Create model
    ########################################################
    print("[GPU-{}] Construct the PlaceNet model".format(args.gpu))
    model = PlaceNet(args.x_ch, args.z_ch, args.v_ch, args.r_ch, args.h_ch, args.image_size, args.num_layer,
                args.attention, args.att_weight, args.att_weight_grad, args.att_weight_delay)

    # Distribute the model into multiple GPUs
    if args.distributed:
        # For multiprocessing distributed,
        # DDP constructor should always set the single device scope,
        # otherwise, DDP will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per DDP,
            # we need to divide the batch size ourselves based on the total number of GPUs we have
            args.num_batch = int(args.num_batch / num_gpu_per_node)
            args.workers = int((args.workers + num_gpu_per_node - 1) / num_gpu_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model,
                                                              device_ids=[args.gpu],
                                                              output_device=args.gpu,
                                                              find_unused_parameters=True)
                                                              # delay_allreduce=True
        else:
            model.cuda()
            # DDP will divide and allocate batch_size to all available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model,
                                                              device_ids=[args.rank],
                                                              output_device=args.rank,
                                                              find_unused_parameters=True)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        model.cuda()

    ########################################################
    # Create optimizer
    ########################################################
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_alpha, betas=args.lr_beta, eps=1e-08,
                                 weight_decay=0, amsgrad=False)
    # # Using RAdam
    # optimizer = torch.optim.RAdam(model.parameters(), lr=5e-4, betas=(0.9, 0.999), eps=1e-8,
    #                             weight_decay=0, degenerated_to_sgd=False)

    ########################################################
    # Define the scheduler: from 5e-4 to 5e-5 until 1.6e6
    ########################################################
    scheduler = AnnealingStepLR(optimizer, mu_i=5e-4, mu_f=5e-5, n=1.6e6)

    ########################################################
    # Update model if the checkpoint is given
    ########################################################
    if args.resume:
        args.resume = os.path.join(args.log_dir, 'models', args.resume)
        if os.path.isfile(args.resume):
            if args.distributed:
                dist.barrier()  # Use a barrier() to make sure that process 1 loads the model after process 0 saves it
            print("[GPU-{}] Loading checkpoint from '{}'".format(args.gpu, args.resume))

            model, optimizer, start_epoch = load_checkpoint(model, optimizer, args)

            args.start_epoch = start_epoch
            print("[GPU-{}] Checkpoint loading complete (epoch: {})".format(args.gpu, args.start_epoch))
            if args.distributed:
                dist.barrier()  # wait until all processes have finished reading the checkpoint
        else:
            print("[ERR] No checkpoint found at '{}'".format(args.resume))
            sys.exit(1)

    ########################################################
    # Dataset Loading
    ########################################################
    print("[GPU-{}] Loading train data from '{}'".format(args.gpu, train_data_dir))
    train_dataset = HouseData(root_dir=train_data_dir, dataset=args.dataset, image_size=args.image_size,
                              attention=args.attention, target_transform=transform_poses)
    print("[GPU-{}] Loading valid data from '{}'".format(args.gpu, valid_data_dir))
    valid_dataset = HouseData(root_dir=valid_data_dir, dataset=args.dataset, image_size=args.image_size,
                              attention=args.attention, target_transform=transform_poses)

    if args.distributed:  # train_loader will be distributed using DistributedSampler
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.num_batch, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=args.num_batch, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    # Get valid data randomly as much as the batch size (for logging and visualization)
    v_data_valid, x_data_valid = next(iter(valid_loader))

    print("[GPU-{}] Start training".format(args.gpu))
    ########################################################
    # Epoch Iterations
    ########################################################
    for epoch in range(args.start_epoch, args.num_epoch):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        ########################################################
        # Define a progress meter
        ########################################################
        batch_train = AverageMeter('TIME', ':6.3f')  # Elapse time for mini-batch processing
        elbo_train = AverageMeter('ELBO', ': .4f')  # Estimated ELBO of the evidence
        kld_train = AverageMeter('KLD', ': .4f')  # KL divergence from q to pi

        # Return output measures
        progress = ProgressMeter(len(train_loader), [batch_train, elbo_train, kld_train],
                                 prefix="[GPU-{}] [{}]".format(args.gpu, epoch))

        # switch to train mode
        model.train()
        begin_train = time.time()
        train_iter = iter(train_loader)
        ################################################################################################
        # Training iteration
        ################################################################################################
        for i in range(len(train_loader)):
            # Load training data, and skip errors while loading
            try:
                v_data, x_data = next(train_iter)
            except Exception as e:
                print("[ERROR] ", e.__class__)
                train_iter = iter(train_loader)
                continue

            # Global training step
            t = len(train_loader) * epoch + i

            # Store data into GPUs
            if args.gpu is not None:
                v_data = v_data.cuda(args.gpu, non_blocking=True)
                x_data = x_data.cuda(args.gpu, non_blocking=True)
            else:
                v_data = v_data.cuda()
                x_data = x_data.cuda()

            # Sampling train data: {contexts, a query}
            if args.obs_train:
                v, v_q, x, x_q = sample_batch(v_data, x_data, args.dataset, obs_range=args.obs_range)
            else:
                v, v_q, x, x_q = sample_batch(v_data, x_data, args.dataset)

            # Pixel-variance annealing
            sigma = max(args.pixel_var[1] + (args.pixel_var[0] - args.pixel_var[1]) *
                        (1 - t / args.pixel_var_step), args.pixel_var[1])

            ## Forward and Get Loss (estimate ELBO) ################
            #
            #
            elbo, kld, bpd = model(v, v_q, x, x_q, sigma)
            #
            #
            ########################################################

            # Update the progress info: ELBO and KLD
            elbo_train.update(elbo)
            kld_train.update(kld)

            # Compute gradient and do SGD step
            optimizer.zero_grad()  # Initialize gradients
            elbo.backward()  # Back-propagation (to compute empirical ELBO gradients)
            optimizer.step()  # Update weights
            scheduler.step()  # Update optimizer state

            # Update the elapsed batch time, and initialize the stopwatch
            batch_train.update(time.time() - begin_train)
            begin_train = time.time()

            # Garbage collection
            if i % args.log_interval == 0:
                junk = gc.collect()

            # Print the progress information
            if i % args.print_interval == 0:
                progress.display(i)

            # Only 1st-ranked worker performs processes in below
            #
            # DON'T WORRY!
            # All processes should see same parameters as they all start from same random parameters
            # and gradients are synchronized in backward passes.
            # Therefore, monitoring or saving it in one process is sufficient.
            #
            if args.rank == 0:
                ########################################################
                # Writing logs on TensorBoard
                ########################################################
                writer.add_scalar('Train/elbo', elbo, t)
                writer.add_scalar('Train/kld', kld, t)
                if t >= args.pixel_var_step:
                    writer.add_scalar('Train/bpd', bpd, t)

                ########################################################
                # Test
                ########################################################
                if t % args.log_interval == 0:
                    valid(v_data_valid, x_data_valid, model, t, args, writer)

        ########################################################
        # Save the current networks
        ########################################################
        if args.rank == 0 and elbo < elbo_min:
            if epoch > 0 and epoch % args.model_save_interval == 0:
                filename = args.log_dir + "/models/model-{}.pth".format(t)
                save_checkpoint(model.state_dict(), optimizer.state_dict(), epoch, filename)

                # remove the old file
                if t > args.model_save_interval * args.num_saved_model:
                    manage_num_model(os.path.join(args.log_dir, 'models'), args.num_saved_model)

                # update elbo criterion
                elbo_min = elbo.clone().detach()

    ########################################################
    # Finale
    ########################################################
    if args.rank == 0:
        filename = args.log_dir + "/models/model-final.pth"
        save_checkpoint(model.state_dict(), optimizer.state_dict(), args.num_epoch, filename)

    # Clean up the DDP
    print("[GPU-{}] Done".format(args.gpu))
    dist.destroy_process_group()


########################################################
# Test
########################################################
def valid(v_data_valid, x_data_valid, model, t, args, writer):
    with torch.no_grad():
        model.eval()

        batch_test = AverageMeter('TIME', ':6.3f')
        elbo_test = AverageMeter('ELBO', ':.4f')
        kld_test = AverageMeter('KLD', ' :.4f')
        begin_test = time.time()

        if args.gpu is not None:
            v_data_valid = v_data_valid.cuda(args.gpu, non_blocking=True)
            x_data_valid = x_data_valid.cuda(args.gpu, non_blocking=True)
        else:
            v_data_valid = v_data_valid.cuda()
            x_data_valid = x_data_valid.cuda()

        # sampling valid data: contexts and a query
        v_valid, v_q_valid, x_valid, x_q_valid = \
            sample_batch(v_data_valid, x_data_valid, args.dataset, seed=0,
                         obs_range=args.obs_range, obs_count=args.obs_count)

        # Pixel-variance annealing
        sigma = max(args.pixel_var[1] + (args.pixel_var[0] - args.pixel_var[1]) *
                    (1 - t / args.pixel_var_step), args.pixel_var[1])

        # estimate ELBO for valid
        elbo, kld, bpd = model(v_valid, v_q_valid, x_valid, x_q_valid, sigma)

        elbo_test.update(elbo)
        kld_test.update(kld)

        # reconstruct and generate scenes of queried positions
        if args.ddp:
            x_q_rec_valid = model.module.inference(v_valid, v_q_valid, x_valid, x_q_valid, sigma)
            x_q_hat_valid = model.module.generator(v_valid, v_q_valid, x_valid, sigma)
        else:
            x_q_rec_valid = model.inference(v_valid, v_q_valid, x_valid, x_q_valid, sigma)
            x_q_hat_valid = model.generator(v_valid, v_q_valid, x_valid, sigma)

        # measure elapsed time
        batch_test.update(time.time() - begin_test)

        # Logging loss values: -elbo, negative log-likelihood, kl-divergence, bits per dimension
        writer.add_scalar('Valid/elbo', elbo, t)
        writer.add_scalar('Valid/kld', kld, t)
        writer.add_scalar('Valid/bpd', bpd, t)

        # Logging comparison values (structural similarities and kl divergence)
        writer.add_scalar('Valid/ssim-inf', SSIM(data_range=1.)(x_q_valid, x_q_rec_valid), t)
        writer.add_scalar('Valid/ssim-gen', SSIM(data_range=1.)(x_q_valid, x_q_hat_valid), t)

        # Visualize results
        x_q_set = torch.cat((x_valid[:, 0],
                             x_q_valid.view(-1, 1, 3, args.image_size, args.image_size),
                             x_q_rec_valid.view(-1, 1, 3, args.image_size, args.image_size),
                             x_q_hat_valid.view(-1, 1, 3, args.image_size, args.image_size)), 1)
        writer.add_image('generation',
                         make_grid(
                             x_q_set.view(args.num_batch * (args.obs_count + 3), 3, args.image_size,
                                          args.image_size),
                             (args.obs_count + 3) * 3, pad_value=1), t)

        print('----------------------------------------------- {}'.format(args.log_title))
        print('[TEST]  STEP {0}\t'
              'TIME {batch_test.val:6.3f} ({batch_test.avg:6.3f})\t'
              'ELBO {elbo_test.val:.4f} ({elbo_test.avg:.4f})\t'
              'KLD {kld_test.val:.4f} ({kld_test.avg:.4f})'.format(
            t, batch_test=batch_test, elbo_test=elbo_test, kld_test=kld_test))
        print('--------------------------------------------------')


########################################################
# Make the log directory
########################################################
def make_log_dir(target_dir, resume):
    if resume is None:
        if os.path.isdir(target_dir):
            answer = input("The existing log directory will be removed. We cool? (y/n)\nTell me: ")
            if answer == "y" or answer == "Y":
                shutil.rmtree(target_dir)
                print("[LOG] Log directory is deleted: '{}'".format(target_dir))
            else:
                print("[ERR] See you soon!")
                sys.exit(1)
        os.mkdir(target_dir)
        print("[LOG] Log directory is created: '{}'".format(target_dir))
    else:
        print("[LOG] Keep store logs into: '{}'".format(target_dir))

    if not os.path.isdir(os.path.join(target_dir, 'models')):
        os.mkdir(os.path.join(target_dir, 'models'))
    if not os.path.isdir(os.path.join(target_dir, 'runs')):
        os.mkdir(os.path.join(target_dir, 'runs'))


########################################################
# Save and load the trained model
########################################################
def save_checkpoint(model, optimizer, epoch, filename):
    torch.save({
        'epoch': epoch,
        'optimizer': optimizer,
        'state_dict': model
    }, filename)
    print("[LOG] %s has been saved" % filename)


def load_checkpoint(model, optimizer, args):
    if args.gpu is None:
        checkpoint = torch.load(args.resume)
    else:
        # Map model to be loaded to specified single GPU
        loc = {'cuda:%d' % 0: 'cuda:%d' % args.rank}

        checkpoint = torch.load(args.resume, map_location=loc)

    # load the epoch number
    start_epoch = checkpoint['epoch'] + 1

    # load the networks
    model.load_state_dict(checkpoint['state_dict'], strict=False)

    # load the optimizer state
    optimizer.load_state_dict(checkpoint['optimizer'])

    return model, optimizer, start_epoch


# Manage to leave only fixed number of checkpoints
def manage_num_model(log_dir, save_amount):
    # sort files by numbers inside of the name of each file
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    num_key = lambda keys: [convert(c) for c in re.split('([0-9]+)', keys)]
    files = sorted(os.listdir(log_dir), key=num_key)

    # delete files exceeds the number of 'save amount'
    for i in range(len(files) - save_amount):
        try:
            os.remove(os.path.join(log_dir, files[i]))
            print("[LOG] %s has been deleted" % files[i])
        except OSError as e:
            print("[ERR] %s : %s" % (os.path.join(log_dir, files[i]), e.strerror))


########################################################
# Progress meter utilities
########################################################
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


##########################################################################################################
# Call the main() with parameter setup
##########################################################################################################
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='PyTorch Implementation of Generative Query Network')
    parser.add_argument('--num_epoch', type=int, metavar='N', default=2,
                        help='number of total epochs to run')
    parser.add_argument('--start_epoch', type=int, metavar='N', default=0,
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--num_batch', type=int, metavar='N', default=36,
                        help='mini-batch size (default: 36), this is the total '
                             'batch size of all GPUs on the current node when '
                             'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('--num_layer', type=int, metavar='N', default=12,
                        help='number of generative layers (default: 12)')
    parser.add_argument('--image_size', type=int, metavar='N', default=64,
                        help='image resizing width (default: 64)')
    parser.add_argument('--obs_train', default=False, action='store_true',
                        help='observation range to be adopted while training as well')
    parser.add_argument('--obs_range', type=int, default=None,
                        help='observation range width, 0 means that the model does not consider it. (default: None)')
    parser.add_argument('--obs_count', type=int, metavar='N', default=3,
                        help='how many observations for the test (default: 3)')
    parser.add_argument('--lr_alpha', type=float, metavar='M', default=0.0005,
                        help='initial learning rate (default: 5e-4)')
    parser.add_argument('--lr_beta', type=float, nargs='+', default=[0.9, 0.999],
                        help='exponential decay for momentum estimates (default: 0.9, 0.999)')
    parser.add_argument('--pixel_var', type=float, nargs='+', default=[2.0, 0.7],
                        help='Pixel standard deviation (default: [2.0, 0.7])')
    parser.add_argument('--pixel_var_step', type=int, default=2 * 10 ** 5,
                        help='Pixel stdev annealing step (default: 2e5)')
    parser.add_argument('--seed', type=int, default=None,
                        help='random seed (default: None)')
    parser.add_argument('--dataset', type=str, default='House',
                        help='dataset (dafault: House)')
    parser.add_argument('--data_dir', type=str, metavar='DIR', default='../dataset/house-torch',
                        help='location of dataset')
    parser.add_argument('--data_dir_train', type=str, metavar='DIR', default='train',
                        help='directory name of train dataset')
    parser.add_argument('--data_dir_test', type=str, metavar='DIR', default='test',
                        help='directory name of test dataset')
    parser.add_argument('--root_log_dir', type=str, metavar='DIR', default='../logs',
                        help='root location of log')
    parser.add_argument('--log_dir', type=str, default='Test',
                        help='name of log directory (default: Test)')
    parser.add_argument('--log_interval', type=int, default=100,
                        help='interval number of steps for logging')
    parser.add_argument('--print_interval', type=int, metavar='N', default=10,
                        help='Log print frequency (default: 10)')
    parser.add_argument('--num_saved_model', type=int, metavar='N', default=2,
                        help='the number of models to be saved')
    parser.add_argument('--model_save_interval', type=int, metavar='N', default=100000,
                        help='interval number of steps for saveing models')
    parser.add_argument('--resume', type=str, metavar='PATH', default=None,
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--workers', type=int, metavar='N', default=0,
                        help='number of data loading workers (default: 0)')
    parser.add_argument('--ddp', action='store_true',
                        help='Use multi-processing distributed training')
    parser.add_argument('--gpu', type=int, default=None,
                        help='GPU id to use.')
    parser.add_argument('--num_nodes', type=int, metavar='N', default=1,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', type=int, metavar='N', default=0,
                        help='rank of each node (whether it is the master of workers)')
    parser.add_argument('--host_url', type=str, default='tcp://127.0.0.1:23456',
                        help='url used to set up distributed training')
    parser.add_argument('--dist_backend', type=str, default='nccl',
                        help='distributed backend')
    parser.add_argument('--x_ch', type=int, metavar='N', default=3,
                        help='Dimension of the image data (default: 3 = # of channels)')
    parser.add_argument('--v_ch', type=int, metavar='N', default=7,
                        help='Dimension of the pose data (default: 7 = xyz(3)+pitch(2)+yaw(2))')
    parser.add_argument('--z_ch', type=int, metavar='N', default=3,
                        help='Dimension of the latent space (default: 3)')
    parser.add_argument('--h_ch', type=int, metavar='N', default=128,
                        help='Dimension of the LSTM channels (default: 128)')
    parser.add_argument('--r_ch', type=int, metavar='N', default=256,
                        help='Dimension of the scene encoder (default: 256; considered 4x image)')
    parser.add_argument('--attention', type=str, default=None,
                        help='type of attention')
    parser.add_argument('--att_weight', type=float, default=0.3,
                        help='Attention weight parameter (default: 0.3)')
    parser.add_argument('--att_weight_grad', default=False, action='store_true',
                        help='Let attention weight parameter be learnable or not')
    parser.add_argument('--att_weight_delay', default=False, action='store_true',
                        help='Let attention weight parameter wait to be decayed until sigma get fixed')
    args = parser.parse_args()
    main()
