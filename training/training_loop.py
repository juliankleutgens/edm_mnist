# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Main training loop."""

import os
import time
import copy
import json
import pickle
import psutil
import numpy as np
import torch
import dnnlib
from torch_utils import distributed as dist
from torch_utils import training_stats
from torch_utils import misc
from torch_utils.utility import *
import wandb
from generate_helper import generate_images_during_training

def print_gpu_memory():
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"  Allocated: {torch.cuda.memory_allocated(i) / 1024 ** 2:.2f} MB")
            print(f"  Cached:    {torch.cuda.memory_reserved(i) / 1024 ** 2:.2f} MB")
            print("-" * 30)


# ----------------------------------------------------------------------------


def training_loop(
        run_dir='.',  # Output directory.
        dataset_kwargs={},  # Options for training set.
        data_loader_kwargs={},  # Options for torch.utils.data.DataLoader.
        network_kwargs={},  # Options for model and preconditioning.
        loss_kwargs={},  # Options for loss function.
        optimizer_kwargs={},  # Options for optimizer.
        augment_kwargs=None,  # Options for augmentation pipeline, None = disable.
        seed=0,  # Global random seed.
        batch_size=8,  # Total batch size for one training iteration.
        batch_gpu=None,  # Limit batch size per GPU, None = no limit.
        total_kimg=200000,  # Training duration, measured in thousands of training images.
        ema_halflife_kimg=500,  # Half-life of the exponential moving average (EMA) of model weights.
        ema_rampup_ratio=0.05,  # EMA ramp-up coefficient, None = no rampup.
        lr_rampup_kimg=10000,  # Learning rate ramp-up duration.
        loss_scaling=1,  # Loss scaling factor for reducing FP16 under/overflows.
        kimg_per_tick=50,  # Interval of progress prints.
        snapshot_ticks=50,  # How often to save network snapshots, None = disable.
        state_dump_ticks=500,  # How often to dump training state, None = disable.
        resume_pkl=None,  # Start from the given network snapshot, None = random initialization.
        resume_state_dump=None,  # Start from the given training state, None = reset training state.
        resume_kimg=0,  # Start from the given training progress.
        cudnn_benchmark=True,  # Enable torch.backends.cudnn.benchmark?
        device=torch.device('cuda'),  # cuda
        moving_mnist={},  # Use moving mnist dataset
        local_computer=False,  # Use local computer
        seq_len=32,  # Sequence length for moving mnist
        num_cond_frames=0,  # Number of conditional frames for moving mnist
        generate_images=False  # Generate images after making the snapshot
):
    if local_computer:
        device = torch.device('cpu')
        total_kimg = 1

    mnist = moving_mnist.get('moving_mnist', False)
    if mnist:
        # seq_len = 64
        batch_size_set = batch_size
        batch_size = batch_size * seq_len

    # Initialize W&B
    wandb.init(project="diffusion-model-training", config={
        'batch_size': batch_size,
        'total_kimg': total_kimg,
        'ema_halflife_kimg': ema_halflife_kimg,
        'lr_rampup_kimg': lr_rampup_kimg,
        'loss_scaling': loss_scaling,
        'kimg_per_tick': kimg_per_tick
    })
    print("Initial GPU memory state:")
    print_gpu_memory()

    # Initialize.
    start_time = time.time()
    np.random.seed((seed * dist.get_world_size() + dist.get_rank()) % (1 << 31))
    torch.manual_seed(np.random.randint(1 << 31))
    torch.backends.cudnn.benchmark = cudnn_benchmark
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False

    # Select batch size per GPU.
    print(f"batch_size: {batch_size}, world_size: {dist.get_world_size()}")
    batch_gpu_total = batch_size // dist.get_world_size()
    if batch_gpu is None or batch_gpu > batch_gpu_total:
        batch_gpu = batch_gpu_total
    num_accumulation_rounds = batch_gpu_total // batch_gpu
    print(
        f"batch_size: {batch_size}, batch_gpu: {batch_gpu}, num_accumulation_rounds: {num_accumulation_rounds}, world_size: {dist.get_world_size()}")
    assert batch_size == batch_gpu * num_accumulation_rounds * dist.get_world_size()

    # Load dataset.
    # mnist = True
    dist.print0('Loading dataset...')
    if not mnist:
        dataset_obj = dnnlib.util.construct_class_by_name(**dataset_kwargs)  # subclass of training.dataset.Dataset
        dataset_sampler = misc.InfiniteSampler(dataset=dataset_obj, rank=dist.get_rank(),
                                               num_replicas=dist.get_world_size(), seed=seed)
        dataset_iterator = iter(torch.utils.data.DataLoader(dataset=dataset_obj, sampler=dataset_sampler, batch_size=batch_gpu,**data_loader_kwargs))
    else:
        # make moving mnist dataset
        from moving_mnist import MovingMNIST
        # seq_len = 32
        image_size=32
        dataset_obj = MovingMNIST(train=True, data_root=moving_mnist.get('moving_mnist_path', './data'),
                                  seq_len=seq_len, num_digits=2, image_size=image_size, deterministic=False)
        dataset_sampler = misc.InfiniteSampler(dataset=dataset_obj, rank=dist.get_rank(),num_replicas=dist.get_world_size(), seed=seed)
        dataset_iterator = iter(torch.utils.data.DataLoader(dataset=dataset_obj, sampler=dataset_sampler, batch_size=batch_gpu,**data_loader_kwargs))
    dist.print0(f'The Batchsize for the Moving MNIST is: bs {batch_size_set} (bs) * {seq_len} (seq_len) = {batch_size}')

    # Construct network.
    dist.print0('Constructing network...')
    denoise_all_frames = False
    if denoise_all_frames:
        # if we ant to denoise all frames we simply increase the input and output channels
        network_kwargs['img_channels'] = dataset_obj.num_channels + num_cond_frames
    interface_kwargs = dict(img_resolution=dataset_obj.resolution, img_channels=dataset_obj.num_channels,
                            label_dim=dataset_obj.label_dim, num_cond_frames=num_cond_frames, denoise_all_frames=denoise_all_frames)
    net = dnnlib.util.construct_class_by_name(**network_kwargs, **interface_kwargs)  # subclass of torch.nn.Module
    net.train().requires_grad_(True).to(device)
    if dist.get_rank() == 0:
        with torch.no_grad():
            images = torch.zeros([batch_gpu, net.img_channels, net.img_resolution, net.img_resolution], device=device)
            if num_cond_frames > 0:
                images = torch.zeros([batch_gpu, dataset_obj.num_channels+num_cond_frames, dataset_obj.resolution, dataset_obj.resolution], device=device)
            sigma = torch.ones([batch_gpu], device=device)
            labels = torch.zeros([batch_gpu, net.label_dim], device=device)
            misc.print_module_summary(net, [images, sigma, labels], max_nesting=2)

    # Setup optimizer.
    dist.print0('Setting up optimizer...')
    loss_fn = dnnlib.util.construct_class_by_name(**loss_kwargs)  # training.loss.(VP|VE|EDM)Loss
    optimizer = dnnlib.util.construct_class_by_name(params=net.parameters(),
                                                    **optimizer_kwargs)  # subclass of torch.optim.Optimizer
    augment_pipe = dnnlib.util.construct_class_by_name(
        **augment_kwargs) if augment_kwargs is not None else None  # training.augment.AugmentPipe
    if local_computer:
        ddp = net  # No parallelization
    else:
        ddp = torch.nn.parallel.DistributedDataParallel(net, device_ids=[device])
    ema = copy.deepcopy(net).eval().requires_grad_(False)


    # Resume training from previous snapshot.
    if resume_pkl is not None:
        dist.print0(f'Loading network weights from "{resume_pkl}"...')
        if dist.get_rank() != 0:
            torch.distributed.barrier()  # rank 0 goes first
        with dnnlib.util.open_url(resume_pkl, verbose=(dist.get_rank() == 0)) as f:
            data = pickle.load(f)
        if dist.get_rank() == 0:
            torch.distributed.barrier()  # other ranks follow
        misc.copy_params_and_buffers(src_module=data['ema'], dst_module=net, require_all=False)
        misc.copy_params_and_buffers(src_module=data['ema'], dst_module=ema, require_all=False)
        del data  # conserve memory
    if resume_state_dump:
        dist.print0(f'Loading training state from "{resume_state_dump}"...')
        data = torch.load(resume_state_dump, map_location=torch.device('cpu'))
        if local_computer:
            data = torch.load(resume_state_dump, map_location=device)
        misc.copy_params_and_buffers(src_module=data['net'], dst_module=net, require_all=True)
        optimizer.load_state_dict(data['optimizer_state'])
        del data  # conserve memory

    # Train.
    dist.print0(f'Training for {total_kimg} kimg...')
    dist.print0()
    print("Before forward pass and After loading dataset")
    print_gpu_memory()
    cur_nimg = resume_kimg * 1000
    cur_tick = 0
    tick_start_nimg = cur_nimg
    tick_start_time = time.time()
    maintenance_time = tick_start_time - start_time
    dist.update_progress(cur_nimg // 1000, total_kimg)
    stats_jsonl = None
    i = 0
    while True:
        # Accumulate gradients.
        optimizer.zero_grad(set_to_none=True)
        for round_idx in range(num_accumulation_rounds):
            with misc.ddp_sync(ddp, (round_idx == num_accumulation_rounds - 1)):
                images, labels = next(dataset_iterator)
                if mnist:
                    # video: [batch_gpu, seq_len, img_h, img_w, gray_scale]
                    images = convert_video2images_in_batch(images)
                    # images: [batch_gpu * seq_len, img_channels, img_h, img_w]
                    if local_computer:
                        images = images[:4, :, :, :]
                # images: Tensor of shape [batch_gpu, img_channels, img_resolution, img_resolution]
                images = images.to(device).to(torch.float32) / 127.5 - 1
                labels = labels.to(device)
                loss = loss_fn(net=ddp, images=images, labels=labels, augment_pipe=augment_pipe)
                if i == 0:
                    print("After forward pass")
                    print_gpu_memory()
                training_stats.report('Loss/loss', loss)
                loss.sum().mul(loss_scaling / batch_gpu_total).backward()
                wandb.log({'loss': loss.sum().mul(loss_scaling / batch_gpu_total), 'step': cur_nimg, 'iteration': i})
                i += 1

        # Update weights.
        for g in optimizer.param_groups:
            g['lr'] = optimizer_kwargs['lr'] * min(cur_nimg / max(lr_rampup_kimg * 1000, 1e-8), 1)
        for param in net.parameters():
            if param.grad is not None:
                torch.nan_to_num(param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad)
        optimizer.step()

        # Update EMA.
        ema_halflife_nimg = ema_halflife_kimg * 1000
        if ema_rampup_ratio is not None:
            ema_halflife_nimg = min(ema_halflife_nimg, cur_nimg * ema_rampup_ratio)
        ema_beta = 0.5 ** (batch_size / max(ema_halflife_nimg, 1e-8))
        for p_ema, p_net in zip(ema.parameters(), net.parameters()):
            p_ema.copy_(p_net.detach().lerp(p_ema, ema_beta))

        # Perform maintenance tasks once per tick.
        cur_nimg += batch_size
        done = (cur_nimg >= total_kimg * 1000)
        if (not done) and (cur_tick != 0) and (cur_nimg < tick_start_nimg + kimg_per_tick * 1000):
            continue

        # Print status line, accumulating the same information in training_stats.
        tick_end_time = time.time()
        fields = []
        fields += [f"tick {training_stats.report0('Progress/tick', cur_tick):<5d}"]
        fields += [f"kimg {training_stats.report0('Progress/kimg', cur_nimg / 1e3):<9.1f}"]
        fields += [
            f"time {dnnlib.util.format_time(training_stats.report0('Timing/total_sec', tick_end_time - start_time)):<12s}"]
        fields += [f"sec/tick {training_stats.report0('Timing/sec_per_tick', tick_end_time - tick_start_time):<7.1f}"]
        fields += [
            f"sec/kimg {training_stats.report0('Timing/sec_per_kimg', (tick_end_time - tick_start_time) / (cur_nimg - tick_start_nimg) * 1e3):<7.2f}"]
        fields += [f"maintenance {training_stats.report0('Timing/maintenance_sec', maintenance_time):<6.1f}"]
        fields += [
            f"cpumem {training_stats.report0('Resources/cpu_mem_gb', psutil.Process(os.getpid()).memory_info().rss / 2 ** 30):<6.2f}"]
        fields += [
            f"gpumem {training_stats.report0('Resources/peak_gpu_mem_gb', torch.cuda.max_memory_allocated(device) / 2 ** 30):<6.2f}"]
        fields += [
            f"reserved {training_stats.report0('Resources/peak_gpu_mem_reserved_gb', torch.cuda.max_memory_reserved(device) / 2 ** 30):<6.2f}"]
        if not local_computer:
            torch.cuda.reset_peak_memory_stats()
        dist.print0(' '.join(fields))

        # Check for abort.
        if (not done) and dist.should_stop():
            done = True
            dist.print0()
            dist.print0('Aborting...')

        # Save network snapshot.
        if (snapshot_ticks is not None) and (done or cur_tick % snapshot_ticks == 0):
            data = dict(ema=ema, loss_fn=loss_fn, augment_pipe=augment_pipe, dataset_kwargs=dict(dataset_kwargs))
            for key, value in data.items():
                if isinstance(value, torch.nn.Module):
                    value = copy.deepcopy(value).eval().requires_grad_(False)
                    misc.check_ddp_consistency(value)
                    data[key] = value.cpu()
                del value  # conserve memory
            if dist.get_rank() == 0:
                snapshot_path = os.path.join(run_dir, f'network-snapshot-{i}.pkl')
                with open(os.path.join(run_dir, f'network-snapshot-{i}.pkl'), 'wb') as f:
                    pickle.dump(data, f)
                wandb.save(snapshot_path)  # Log snapshot to W&B
            if generate_images:
                try:
                    #torch.cuda.empty_cache()  # Clear cached memory
                    #torch.cuda.synchronize()  # Synchronize CUDA operations
                    generate_images_during_training( network_pkl = snapshot_path,
                                     outdir = os.path.join(run_dir, f'generated_images_{i}'),
                                     seeds = list(range(1, 17)),
                                     class_idx=None,
                                     max_batch_size=64,
                                     device=torch.device('cuda'),
                                     wandb_run_id=wandb.run.id,
                                     subdirs=False,
                                     local_computer=local_computer,
                                     dist=dist,
                                     )
                except Exception as e:
                    print(f"Error generating images: {e}")
            del data  # conserve memory

        # Save full dump of the training state.
        if (state_dump_ticks is not None) and (
                done or cur_tick % state_dump_ticks == 0) and cur_tick != 0 and dist.get_rank() == 0:
            state_path = os.path.join(run_dir, f'training-state-{cur_nimg // 1000:06d}.pt')
            torch.save(dict(net=net, optimizer_state=optimizer.state_dict()), state_path)
            wandb.save(state_path)  # Log state to W&B

        # Update logs.
        training_stats.default_collector.update()
        if dist.get_rank() == 0:
            if stats_jsonl is None:
                stats_jsonl = open(os.path.join(run_dir, 'stats.jsonl'), 'at')
            stats_jsonl.write(
                json.dumps(dict(training_stats.default_collector.as_dict(), timestamp=time.time())) + '\n')
            stats_jsonl.flush()
            wandb.save(os.path.join(run_dir, 'stats.jsonl'))  # Log stats to W&B
        dist.update_progress(cur_nimg // 1000, total_kimg)

        # Update state.
        cur_tick += 1
        tick_start_nimg = cur_nimg
        tick_start_time = time.time()
        maintenance_time = tick_start_time - tick_end_time
        if done:
            break

    # Done.
    dist.print0()
    dist.print0('Exiting...')
    wandb.finish()
# ----------------------------------------------------------------------------
