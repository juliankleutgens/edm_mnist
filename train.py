# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Train diffusion-based generative model using the techniques described in the
paper "Elucidating the Design Space of Diffusion-Based Generative Models"."""

import os
import re
import json
import click
import torch
import dnnlib
from torch_utils import distributed as dist
from training import training_loop

import wandb

import warnings
warnings.filterwarnings('ignore', 'Grad strides do not match bucket view strides') # False warning printed by PyTorch 1.12.

#----------------------------------------------------------------------------
# Parse a comma separated list of numbers or ranges and return a list of ints.
# Example: '1,2,5-10' returns [1, 2, 5, 6, 7, 8, 9, 10]

def parse_int_list(s):
    if isinstance(s, list): return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2))+1))
        else:
            ranges.append(int(p))
    return ranges

#----------------------------------------------------------------------------

@click.command()

# Main options.
@click.option('--outdir',        help='Where to save the results', metavar='DIR',                   type=str, required=True)
@click.option('--data',          help='Path to the dataset', metavar='ZIP|DIR',                     type=str, required=True)
@click.option('--cond',          help='Train class-conditional model', metavar='BOOL',              type=bool, default=False, show_default=True)
@click.option('--arch',          help='Network architecture', metavar='ddpmpp|ncsnpp|adm',          type=click.Choice(['ddpmpp', 'ncsnpp', 'adm']), default='ddpmpp', show_default=True)
@click.option('--precond',       help='Preconditioning & loss function', metavar='vp|ve|edm',       type=click.Choice(['vp', 've', 'edm']), default='edm', show_default=True)

# Hyperparameters.
@click.option('--duration',      help='Training duration, in number of training iterations', metavar='MIMG',                          type=click.FloatRange(min=0, min_open=True), default=200, show_default=True)
@click.option('--batch',         help='Total batch size', metavar='INT',                            type=click.IntRange(min=1), default=2, show_default=True)
@click.option('--batch-gpu',     help='Limit batch size per GPU', metavar='INT',                    type=click.IntRange(min=1))
@click.option('--cbase',         help='Channel multiplier  [default: varies]', metavar='INT',       type=int)
@click.option('--cres',          help='Channels per resolution  [default: varies]', metavar='LIST', type=parse_int_list)
@click.option('--lr',            help='Learning rate', metavar='FLOAT',                             type=click.FloatRange(min=0, min_open=True), default=10e-4, show_default=True)
@click.option('--ema',           help='EMA half-life', metavar='MIMG',                              type=click.FloatRange(min=0), default=0.5, show_default=True)
@click.option('--dropout',       help='Dropout probability', metavar='FLOAT',                       type=click.FloatRange(min=0, max=1), default=0.13, show_default=True)
@click.option('--augment',       help='Augment probability', metavar='FLOAT',                       type=click.FloatRange(min=0, max=1), default=0.12, show_default=True)
@click.option('--xflip',         help='Enable dataset x-flips', metavar='BOOL',                     type=bool, default=False, show_default=True)

# Performance-related.
@click.option('--fp16',          help='Enable mixed-precision training', metavar='BOOL',            type=bool, default=False, show_default=True)
@click.option('--ls',            help='Loss scaling', metavar='FLOAT',                              type=click.FloatRange(min=0, min_open=True), default=1, show_default=True)
@click.option('--bench',         help='Enable cuDNN benchmarking', metavar='BOOL',                  type=bool, default=True, show_default=True)
@click.option('--cache',         help='Cache dataset in CPU memory', metavar='BOOL',                type=bool, default=True, show_default=True)
@click.option('--workers',       help='DataLoader worker processes', metavar='INT',                 type=click.IntRange(min=1), default=1, show_default=True)

# I/O-related.
@click.option('--desc',          help='String to include in result dir name', metavar='STR',        type=str)
@click.option('--nosubdir',      help='Do not create a subdirectory for results',                   is_flag=True)
@click.option('--tick',          help='How often to print progress', metavar='KIMG',                type=click.IntRange(min=1), default=30, show_default=True)
@click.option('--snap',          help='How often to save snapshots', metavar='TICKS',               type=click.IntRange(min=1), default=10, show_default=True)
@click.option('--dump',          help='How often to dump state', metavar='TICKS',                   type=click.IntRange(min=1), default=2, show_default=True)
@click.option('--seed',          help='Random seed  [default: random]', metavar='INT',              type=int)
@click.option('--transfer',      help='Transfer learning from network pickle', metavar='PKL|URL',   type=str)
@click.option('--resume',        help='Resume from previous training state', metavar='PT',          type=str)
@click.option('-n', '--dry-run', help='Print training options and exit',                            is_flag=True)

# my added options for the moving MNIST dataset
@click.option('--moving_mnist',  help='If one like to train on the moving MNIST dataset',           is_flag=True)
@click.option('--moving_mnist_path',  help='The path to the MNIST dataset',                         type=str, default='./data')
@click.option('--local_computer',  help='If you want to debug on the cpu on the local computer',    is_flag=True)
@click.option('--seq_len',       help='Th'
                                      'e length of the sequence',                                 type=int, default=64)
@click.option('--num_cond_frames', help='The number of frames to condition on. One has no condition for 0, which is set by default', type=int, default=0)
@click.option('--generate_images',help='Generate images after making the snapshot of the model',  is_flag=True)
@click.option('--digit_filter',  help='The digits to filter out from the MNIST dataset',           type=parse_int_list)

# model options
@click.option('--num_blocks',   help='The number of blocks in the model',                         type=int, default=4)
@click.option('--channel_mult_noise', help='The channel multiplier for the noise',                type=int, default=1)
@click.option('--resample_filter', help='The resample filter for the model',                      type=parse_int_list, default=[1,1])
@click.option('--model_channels', help='The number of channels in the model',                     type=int, default=32)
@click.option('--channel_mult',  help='The channel multiplier for the model',                     type=parse_int_list, default=[1,1,2])
@click.option('--move_horizontally', help='If the digits should move horizontally',              is_flag=True)
@click.option('--prob_direction_change', help='The probability of changing the direction of the digit to the right, Note: for that one has to use --move_horizontally', type=float, default=0.5)
@click.option('--let_last_frame_after_change', help='The second last frame is always in the middel and then in the last frame a dircation change was made. Note: for that one has to use --move_horizontally', is_flag=True)


def main(**kwargs):
    """Train diffusion-based generative model using the techniques described in the
    paper "Elucidating the Design Space of Diffusion-Based Generative Models".

    Examples:

    #\b
    # Train DDPM++ model for class-conditional CIFAR-10 using 8 GPUs
    torchrun --standalone --nproc_per_node=8 train.py --outdir=training-runs \\
        --data=datasets/cifar10-32x32.zip --cond=1 --arch=ddpmpp
    """
    opts = dnnlib.EasyDict(kwargs)
    torch.cuda.empty_cache()

    torch.multiprocessing.set_start_method('spawn')
    #torch.multiprocessing.set_start_method('spawn', force=True)
    dist.init()

    # Initialize config dict.
    c = dnnlib.EasyDict()
    c.dataset_kwargs = dnnlib.EasyDict(class_name='training.dataset.ImageFolderDataset', path=opts.data, use_labels=opts.cond, xflip=opts.xflip, cache=opts.cache)
    c.data_loader_kwargs = dnnlib.EasyDict(pin_memory=True, num_workers=opts.workers, prefetch_factor=2)
    c.network_kwargs = dnnlib.EasyDict()
    c.loss_kwargs = dnnlib.EasyDict()
    c.optimizer_kwargs = dnnlib.EasyDict(class_name='torch.optim.Adam', lr=opts.lr, betas=[0.9,0.999], eps=1e-8)
    c.moving_mnist = dnnlib.EasyDict(moving_mnist=opts.moving_mnist, moving_mnist_path=opts.moving_mnist_path, use_labels=opts.cond, move_horizontally=opts.move_horizontally
                                     ,prob_direction_change=opts.prob_direction_change, let_last_frame_after_change=opts.let_last_frame_after_change)
    c.local_computer = opts.local_computer
    c.seq_len = opts.seq_len
    c.num_cond_frames = opts.num_cond_frames
    c.generate_images = opts.generate_images
    c.digit_filter = opts.digit_filter

    # Validate dataset options.
    try:
        # Get the current working directory
        current_path = os.getcwd()
        if not opts.moving_mnist:
            c.dataset_kwargs['path'] = os.path.join(current_path, c.dataset_kwargs['path'])
            dataset_obj = dnnlib.util.construct_class_by_name(**c.dataset_kwargs)
            dataset_name = dataset_obj.name
            c.dataset_kwargs.resolution = dataset_obj.resolution # be explicit about dataset resolution
            c.dataset_kwargs.max_size = len(dataset_obj) # be explicit about dataset size
            if opts.cond and not dataset_obj.has_labels:
                raise click.ClickException('--cond=True requires labels specified in dataset.json')
            del dataset_obj # conserve memory
    except IOError as err:
        raise click.ClickException(f'--data: {err}')

    # Network architecture.
    if opts.arch == 'ddpmpp':
        c.network_kwargs.update(model_type='SongUNet', embedding_type='positional', encoder_type='standard', decoder_type='standard', num_blocks=opts.num_blocks)
        c.network_kwargs.update(channel_mult_noise=1, resample_filter=opts.resample_filter, model_channels=opts.model_channels, channel_mult=opts.channel_mult)
    elif opts.arch == 'ncsnpp':
        c.network_kwargs.update(model_type='SongUNet', embedding_type='fourier', encoder_type='residual', decoder_type='standard')
        c.network_kwargs.update(channel_mult_noise=2, resample_filter=[1,3,3,1], model_channels=128, channel_mult=[2,2,2])
    else:
        assert opts.arch == 'adm'
        c.network_kwargs.update(model_type='DhariwalUNet', model_channels=192, channel_mult=[1,2,3,4])

    # Preconditioning & loss function.
    if opts.precond == 'vp':
        c.network_kwargs.class_name = 'training.networks.VPPrecond'
        c.loss_kwargs.class_name = 'training.loss.VPLoss'
    elif opts.precond == 've':
        c.network_kwargs.class_name = 'training.networks.VEPrecond'
        c.loss_kwargs.class_name = 'training.loss.VELoss'
    else:
        assert opts.precond == 'edm'
        c.network_kwargs.class_name = 'training.networks.EDMPrecond'
        c.loss_kwargs.class_name = 'training.loss.EDMLoss'

    # Network options.
    if opts.cbase is not None:
        c.network_kwargs.model_channels = opts.cbase
    if opts.cres is not None:
        c.network_kwargs.channel_mult = opts.cres
    if opts.augment:
        c.augment_kwargs = dnnlib.EasyDict(class_name='training.augment.AugmentPipe', p=opts.augment)
        c.augment_kwargs.update(xflip=1e8, yflip=1, scale=1, rotate_frac=1, aniso=1, translate_frac=1)
        c.network_kwargs.augment_dim = 9
    c.network_kwargs.update(dropout=opts.dropout, use_fp16=opts.fp16)

    # Training options.
    total_img = max(int(opts.duration*opts.batch), 1)
    c.total_kimg = total_img / 1000
    if opts.moving_mnist:
        total_img = opts.duration * opts.seq_len * opts.batch
        c.total_kimg = total_img / 1000
    c.ema_halflife_kimg = int(opts.ema * 1000)
    c.update(batch_size=opts.batch, batch_gpu=opts.batch_gpu)
    c.update(loss_scaling=opts.ls, cudnn_benchmark=opts.bench)
    # those are k images!
    tick_interval = max(int(c.total_kimg // opts.tick), 1)
    snap_interval = max(int(c.total_kimg // opts.snap), 1)
    dump_interval = max(int(c.total_kimg // opts.dump), 1)
    print('---------- Iterations, Number of Images, Saving times ----------')
    print(f'kimg_per_tick: {tick_interval} every k img')
    print(f'snapshot_ticks: {snap_interval} every k img')
    print(f'state_dump_ticks: {dump_interval} every k img')
    print(f'Total kimg: {c.total_kimg}')
    num_iterations = opts.duration
    print(f'Number of iterations: {num_iterations}')
    # calculate at which iterations the snapshots are saved
    snapshot_iterations = [i for i in range(0, int(num_iterations), int(snap_interval * num_iterations//c.total_kimg))]

    #tick_iterations = [i for i in range(0, int(num_iterations), int(tick_interval * num_iterations//c.total_kimg))]
    print(f'Snapshots are saved at iterations: {snapshot_iterations[1:]}')
    print('------------------------------------------')
    c.update(kimg_per_tick=tick_interval, snapshot_ticks=snap_interval, state_dump_ticks=dump_interval, snapshot_iterations=snapshot_iterations)

    # Random seed.
    if opts.seed is not None:
        c.seed = opts.seed
    else:
        if opts.local_computer:
            seed = torch.randint(1 << 31, size=[], device=torch.device('cpu'))
        else:
            seed = torch.randint(1 << 31, size=[], device=torch.device('cuda'))
        torch.distributed.broadcast(seed, src=0)
        c.seed = int(seed)

    # Transfer learning and resume.
    if opts.transfer is not None:
        if opts.resume is not None:
            raise click.ClickException('--transfer and --resume cannot be specified at the same time')
        c.resume_pkl = opts.transfer
        c.ema_rampup_ratio = None
    elif opts.resume is not None:
        match = re.fullmatch(r'training-state-(\d+).pt', os.path.basename(opts.resume))
        if not match or not os.path.isfile(opts.resume):
            raise click.ClickException('--resume must point to training-state-*.pt from a previous training run')
        c.resume_pkl = os.path.join(os.path.dirname(opts.resume), f'network-snapshot-{match.group(1)}.pkl')
        c.resume_kimg = int(match.group(1))
        c.resume_state_dump = opts.resume

    # Description string.
    cond_str = 'cond' if c.dataset_kwargs.use_labels else 'uncond'
    dtype_str = 'fp16' if c.network_kwargs.use_fp16 else 'fp32'
    if c.moving_mnist.moving_mnist:
        dataset_name = 'moving_mnist'
    desc = f'{dataset_name:s}-gpus{dist.get_world_size():d}-batch{c.batch_size:d}-{dtype_str:s}-p{opts.prob_direction_change}-seq_len{opts.seq_len}-num_cond_frames{opts.num_cond_frames}-move_horizontally{opts.move_horizontally})'
    if opts.desc is not None:
        desc += f'-{opts.desc}'

    # Pick output directory.
    if dist.get_rank() != 0:
        c.run_dir = None
    elif opts.nosubdir:
        c.run_dir = opts.outdir
    else:
        prev_run_dirs = []
        if os.path.isdir(opts.outdir):
            prev_run_dirs = [x for x in os.listdir(opts.outdir) if os.path.isdir(os.path.join(opts.outdir, x))]
        prev_run_ids = [re.match(r'^\d+', x) for x in prev_run_dirs]
        prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
        cur_run_id = max(prev_run_ids, default=-1) + 1
        c.run_dir = os.path.join(opts.outdir, f'{cur_run_id:05d}-{desc}')
        assert not os.path.exists(c.run_dir)

    # Print options.
    dist.print0()
    dist.print0('Training options:')
    dist.print0(json.dumps(c, indent=2))
    dist.print0()
    dist.print0(f'Output directory:        {c.run_dir}')
    dist.print0(f'Dataset path:            {c.dataset_kwargs.path}')
    dist.print0(f'Class-conditional:       {c.dataset_kwargs.use_labels}')
    dist.print0(f'Network architecture:    {opts.arch}')
    dist.print0(f'Preconditioning & loss:  {opts.precond}')
    dist.print0(f'Number of GPUs:          {dist.get_world_size()}')
    dist.print0(f'Batch size:              {c.batch_size}')
    dist.print0(f'Mixed-precision:         {c.network_kwargs.use_fp16}')
    dist.print0()


    # Dry run?
    if opts.dry_run:
        dist.print0('Dry run; exiting.')
        return

    # Create output directory.
    dist.print0('Creating output directory...')
    if dist.get_rank() == 0:
        os.makedirs(c.run_dir, exist_ok=True)
        with open(os.path.join(c.run_dir, 'training_options.json'), 'wt') as f:
            json.dump(c, f, indent=2)
        dnnlib.util.Logger(file_name=os.path.join(c.run_dir, 'log.txt'), file_mode='a', should_flush=True)

    # Train.
    training_loop.training_loop(**c)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
