
import os, time, argparse, os.path as osp, numpy as np
import torch
import torch.distributed as dist

from utils.metric_util import MeanIoU
from utils.load_save_util import revise_ckpt, revise_ckpt_2
from dataloader.dataset import get_nuScenes_label_name
from builder import loss_builder

import mmcv
from mmcv import Config
from mmcv.runner import build_optimizer
from mmseg.utils import get_root_logger
from timm.scheduler import CosineLRScheduler
from builder import occ_occupancy_builder as model_builder
import warnings
warnings.filterwarnings("ignore")


def pass_print(*args, **kwargs):
    pass

def main(local_rank, args):
    # global settings
    torch.backends.cudnn.benchmark = True

    # load config
    cfg = Config.fromfile(args.py_config)
    cfg.work_dir = args.work_dir

    # check label_mapping, fill_label, ignore_label, pc_dataset_type
    dataset_config = cfg.dataset_params
    ignore_label = dataset_config['ignore_label']
    version = dataset_config['version']
    # check num_workers, imageset
    train_dataloader_config = cfg.train_data_loader
    val_dataloader_config = cfg.val_data_loader

    max_num_epochs = cfg.max_epochs
    grid_size = cfg.grid_size

    # init DDP
    distributed = True
    ip = os.environ.get("MASTER_ADDR", "127.0.0.1")
    port = os.environ.get("MASTER_PORT", "20508")
    hosts = int(os.environ.get("WORLD_SIZE", 1))  # number of nodes
    rank = int(os.environ.get("RANK", 0))  # node id
    gpus = torch.cuda.device_count()  # gpus per node
    print(f"tcp://{ip}:{port}")
    dist.init_process_group(
        backend="nccl", init_method=f"tcp://{ip}:{port}", 
        world_size=hosts * gpus, rank=rank * gpus + local_rank
    )
    world_size = dist.get_world_size()
    cfg.gpu_ids = range(world_size)
    torch.cuda.set_device(local_rank)

    if dist.get_rank() != 0:
        import builtins
        builtins.print = pass_print

    # configure logger
    if dist.get_rank() == 0:
        os.makedirs(args.work_dir, exist_ok=True)
        cfg.dump(osp.join(args.work_dir, osp.basename(args.py_config)))

    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(args.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level='INFO')
    logger.info(f'Config:\n{cfg.pretty_text}')

    # build model


    
    my_model = model_builder.build(cfg.model)
    n_parameters = sum(p.numel() for p in my_model.parameters() if p.requires_grad)
    logger.info(f'Number of params: {n_parameters}')
    if distributed:
        find_unused_parameters = cfg.get('find_unused_parameters', False)
        ddp_model_module = torch.nn.parallel.DistributedDataParallel
        my_model = ddp_model_module(
            my_model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters)
    else:
        my_model = my_model.cuda()
    print('done ddp model')

    # generate datasets
    SemKITTI_label_name = get_nuScenes_label_name(dataset_config["label_mapping"])
    unique_label = np.asarray(cfg.unique_label)
    unique_label_str = [SemKITTI_label_name[x] for x in unique_label]

    from builder import data_builder
    train_dataset_loader, val_dataset_loader = \
        data_builder.build(
            dataset_config,
            train_dataloader_config,
            val_dataloader_config,
            grid_size=grid_size,
            version=version,
            dist=distributed,
            scale_rate=cfg.get('scale_rate', 1)
        )


    # get optimizer, loss, scheduler
    optimizer = build_optimizer(my_model, cfg.optimizer)
    loss_func, lovasz_softmax = \
        loss_builder.build(
            wce=True, 
            lovasz=True,
            ignore_label=ignore_label)
    scheduler = CosineLRScheduler(
        optimizer,
        t_initial=len(train_dataset_loader)*max_num_epochs,
        lr_min=1e-6,
        warmup_t=500,
        warmup_lr_init=1e-5,
        t_in_epochs=False
    )
    
    CalMeanIou_vox = MeanIoU(unique_label, ignore_label, unique_label_str, 'vox')
    
    # resume and load
    epoch = 0
    best_val_miou_vox = 0, 0
    global_iter = 0

    cfg.resume_from = ''
    if osp.exists(osp.join(args.work_dir, 'latest.pth')):
        cfg.resume_from = osp.join(args.work_dir, 'latest.pth')
    if args.resume_from:
        cfg.resume_from = args.resume_from
    
    print('resume from: ', cfg.resume_from)
    print('work dir: ', args.work_dir)

    if cfg.resume_from and osp.exists(cfg.resume_from):
        map_location = 'cpu'
        ckpt = torch.load(cfg.resume_from, map_location=map_location)
        print(my_model.load_state_dict(revise_ckpt(ckpt['state_dict']), strict=False))
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        epoch = ckpt['epoch']
        if 'best_val_miou_vox' in ckpt:
            best_val_miou_vox = ckpt['best_val_miou_vox']
        global_iter = ckpt['global_iter']
        print(f'successfully resumed from epoch {epoch}')
    elif cfg.load_from:
        ckpt = torch.load(cfg.load_from, map_location='cpu')
        if 'state_dict' in ckpt:
            state_dict = ckpt['state_dict']
        else:
            state_dict = ckpt
        state_dict = revise_ckpt(state_dict)
        try:
            print(my_model.load_state_dict(state_dict, strict=False))
        except:
            state_dict = revise_ckpt_2(state_dict)
            print(my_model.load_state_dict(state_dict, strict=False))
        

    # training
    print_freq = cfg.print_freq
    while epoch < max_num_epochs:
        my_model.train()
        if hasattr(train_dataset_loader.sampler, 'set_epoch'):
            train_dataset_loader.sampler.set_epoch(epoch)
        loss_list = []
        time.sleep(10)
        data_time_s = time.time()
        time_s = time.time()
        for i_iter, (imgs, img_metas, train_vox_label, train_grid, train_pt_labs) in enumerate(train_dataset_loader):
            
            imgs = imgs.cuda()
            voxel_label = train_vox_label.type(torch.LongTensor).cuda()
            # forward + backward + optimize
            data_time_e = time.time()
            outputs_vox = my_model(img=imgs, img_metas=img_metas)


            loss = lovasz_softmax(
                torch.nn.functional.softmax(outputs_vox, dim=1), 
                voxel_label, ignore=ignore_label
            ) + loss_func(outputs_vox, voxel_label)

            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(my_model.parameters(), cfg.grad_max_norm)
            optimizer.step()
            loss_list.append(loss.item())
            scheduler.step_update(global_iter)
            time_e = time.time()

            global_iter += 1
            if i_iter % print_freq == 0 and dist.get_rank() == 0:
                lr = optimizer.param_groups[0]['lr']
                logger.info('[TRAIN] Epoch %d Iter %5d/%d: Loss: %.3f (%.3f), grad_norm: %.1f, lr: %.7f, time: %.3f (%.3f)'%(
                    epoch, i_iter, len(train_dataset_loader), 
                    loss.item(), np.mean(loss_list), grad_norm, lr,
                    time_e - time_s, data_time_e - data_time_s
                ))
            data_time_s = time.time()
            time_s = time.time()
        
        # save checkpoint
        if dist.get_rank() == 0:
            dict_to_save = {
                'state_dict': my_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch': epoch + 1,
                'global_iter': global_iter,
                'best_val_miou_vox': best_val_miou_vox
            }
            save_file_name = os.path.join(os.path.abspath(args.work_dir), f'epoch_{epoch+1}.pth')
            torch.save(dict_to_save, save_file_name)
            dst_file = osp.join(args.work_dir, 'latest.pth')
            mmcv.symlink(save_file_name, dst_file)

        epoch += 1
        
        # eval
        my_model.eval()
        val_loss_list = []
        CalMeanIou_vox.reset()

        with torch.no_grad():
            for i_iter_val, (imgs, img_metas, val_vox_label, val_grid, val_pt_labs) in enumerate(val_dataset_loader):
                
                imgs = imgs.cuda()
                vox_label = val_vox_label.type(torch.LongTensor).cuda()
                predict_labels_vox = my_model(img=imgs, img_metas=img_metas)

                
                loss = lovasz_softmax(
                    torch.nn.functional.softmax(predict_labels_vox, dim=1).detach(), 
                    vox_label, ignore=ignore_label
                ) + loss_func(predict_labels_vox.detach(), vox_label)
                
                
                predict_labels_vox = torch.argmax(predict_labels_vox, dim=1)
                predict_labels_vox = predict_labels_vox.detach().cpu()
                for count in range(len(img_metas)):
                    CalMeanIou_vox._after_step(
                        predict_labels_vox[count].flatten(),
                        val_vox_label[count].flatten())
                val_loss_list.append(loss.detach().cpu().numpy())
                if i_iter_val % print_freq == 0 and dist.get_rank() == 0:
                    logger.info('[EVAL] Epoch %d Iter %5d: Loss: %.3f (%.3f)'%(
                        epoch, i_iter_val, loss.item(), np.mean(val_loss_list)))
        

        val_miou_vox = CalMeanIou_vox._after_epoch()

        del val_vox_label

        if best_val_miou_vox < val_miou_vox:
            best_val_miou_vox = val_miou_vox

        logger.info('Current val miou vox is %.3f while the best val miou vox is %.3f' %
                (val_miou_vox, best_val_miou_vox))
        logger.info('Current val loss is %.3f' %
                (np.mean(val_loss_list)))
        

if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--py-config', default='config/occupancy.py')
    parser.add_argument('--work-dir', type=str, default='./out/occupancy')
    parser.add_argument('--resume-from', type=str, default='')

    args = parser.parse_args()
    
    args.gpus = torch.cuda.device_count()
    print(args)

    torch.multiprocessing.spawn(main, args=(args,), nprocs=args.gpus)
