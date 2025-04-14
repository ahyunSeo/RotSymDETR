# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import projects
from mmengine.config import Config, DictAction
from mmengine.registry import RUNNERS
from mmengine.runner import Runner
from mmdet.utils import setup_cache_size_limit_of_dynamo
import torch

def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--amp',
        action='store_true',
        default=False,
        help='enable automatic-mixed-precision training')
    parser.add_argument(
        '--auto-scale-lr',
        action='store_true',
        help='enable automatically scaling LR.')
    parser.add_argument(
        '--resume',
        nargs='?',
        type=str,
        const='auto',
        help='If specify checkpoint path, resume from it, while if not '
        'specify, try to auto resume from the latest checkpoint '
        'in the work directory.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():
    args = parse_args()
    import mmdet, mmengine, mmseg
    for k, v in mmdet.registry.MODELS._module_dict.items():
        mmengine.registry.MODELS.register_module(k, True, v)
    # for k, v in mmseg.registry.MODELS._module_dict.items():
    #     mmengine.registry.MODELS.register_module(k, True, v)
    for k, v in mmdet.registry.METRICS._module_dict.items():
        mmengine.registry.METRICS.register_module(k, True, v)

    # Reduce the number of repeated compilations and improve
    # training speed.
    setup_cache_size_limit_of_dynamo()

    # load config
    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    current_dir = os.getcwd()

    cfg.data_root = os.path.join(current_dir, cfg.data_root)
    if cfg.train_dataloader.dataset.get('data_root') is not None:
        cfg.train_dataloader.dataset.data_root = os.path.join(current_dir, 
                                                          cfg.train_dataloader.dataset.data_root)
    if cfg.train_dataloader.dataset.get('dataset') is not None:
        if cfg.train_dataloader.dataset.dataset.get('data_root') is not None:
            cfg.train_dataloader.dataset.dataset.data_root = os.path.join(current_dir, 
                                                            cfg.train_dataloader.dataset.dataset.data_root)
    
    cfg.val_dataloader.dataset.data_root = os.path.join(current_dir, 
                                                          cfg.val_dataloader.dataset.data_root)
    cfg.test_dataloader.dataset.data_root = os.path.join(current_dir, 
                                                          cfg.test_dataloader.dataset.data_root)
    
    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    # enable automatic-mixed-precision training
    if args.amp is True:
        cfg.optim_wrapper.type = 'AmpOptimWrapper'
        cfg.optim_wrapper.loss_scale = 'dynamic'

    # enable automatically scaling LR
    if args.auto_scale_lr:
        if 'auto_scale_lr' in cfg and \
                'enable' in cfg.auto_scale_lr and \
                'base_batch_size' in cfg.auto_scale_lr:
            cfg.auto_scale_lr.enable = True
        else:
            raise RuntimeError('Can not find "auto_scale_lr" or '
                               '"auto_scale_lr.enable" or '
                               '"auto_scale_lr.base_batch_size" in your'
                               ' configuration file.')

    # resume is determined in this priority: resume from > auto_resume
    if args.resume == 'auto':
        cfg.resume = True
        cfg.load_from = None
    elif args.resume is not None:
        cfg.resume = True
        cfg.load_from = args.resume

    # os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
    # # set random seeds
    # cfg.deterministic = True
    # cfg.randomness = dict(
    #     seed=0,
    #     # deterministic=True
    # )
    # torch.use_deterministic_algorithms(True, warn_only=True)

    if cfg.seed is not None:
        seed = cfg.seed
    else:
        seed = 0
    print('Random seed %d' % seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # On Ampere Nvidia GPUs, PyTorch can use TensorFloat32 (TF32) 
    # to speed up mathematically intensive operations,
    # in particular matrix multiplications and convolutions. 
    # When an operation is performed using TF32 tensor cores, 
    # only the first 10 bits of the input mantissa are read. This may reduce accuracy and produce surprising results 
    # (e.g., multiplying a matrix by the identity matrix may produce results that are different from the input). 
    # By default, TF32 tensor cores are disabled for matrix multiplications 
    # and enabled for convolutions, although most neural network workloads have the same convergence behavior when using TF32 as they have with fp32.
    # We recommend enabling TF32 tensor cores for matrix multiplications with torch.backends.cuda.matmul.allow_tf32 = True 
    # if your network does not need full float32 precision. 
    # If your network needs full float32 precision for both matrix multiplications and convolutions, 
    # then TF32 tensor cores can also be disabled for convolutions with torch.backends.cudnn.allow_tf32 = False.

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    from mmengine.runner.utils import set_random_seed
    set_random_seed(seed)
    import random
    import numpy as np
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    # build the runner from config
    if 'runner_type' not in cfg:
        # build the default runner
        runner = Runner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)

    # start training
    runner.train()

if __name__ == '__main__':
    main()
