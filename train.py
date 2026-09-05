import argparse
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import time
from datasets import find_dataset_def
from models.vismvsnet import VisMVSModel as BaselineModel
from models.vismvsnet import VisMVSLoss as BaselineLoss
from models.vismvsnet_oa import VisMVSModel as EnhancedModel
from models.vismvsnet_oa import VisMVSLoss as EnhancedLoss
from models.model_variants import MODEL_TYPE_CHOICES, get_model_variant
from utils import *
import gc
import sys
import datetime


# =============================================================================
# Logger
# =============================================================================
class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'a', encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        pass


cudnn.benchmark = True

# =============================================================================
# Args
# =============================================================================
parser = argparse.ArgumentParser(
    description='Vis-MVSNet final M1/M2/M3 factorial ablation training')
parser.add_argument('--mode', default='train', help='train or test', choices=['train', 'test', 'profile'])
parser.add_argument(
    '--model_type', default='vis',
    choices=MODEL_TYPE_CHOICES,
    help=(
        'complete M1/M2/M3 ablation: vis, m1_hyp, m2_visibility, '
        'm3_hybrid, m1_m2, m1_m3, m2_m3 or full'))

parser.add_argument('--dataset', default='dtu_yao', help='select dataset')
parser.add_argument('--trainpath', help='train datapath')
parser.add_argument('--testpath', help='test datapath')
parser.add_argument('--trainlist', help='train list')
parser.add_argument('--testlist', help='test list')

parser.add_argument('--epochs', type=int, default=16, help='number of epochs to train')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--lrepochs', type=str, default="10,12,14:2", help='epoch ids to downscale lr and the downscale rate')
parser.add_argument('--wd', type=float, default=0.0, help='weight decay')

parser.add_argument('--batch_size', type=int, default=4, help='train batch size')
parser.add_argument('--numdepth', type=int, default=192, help='the number of depth values')
parser.add_argument('--interval_scale', type=float, default=1.06, help='depth interval scale')

parser.add_argument('--loadckpt', default=None, help='load a specific checkpoint')
parser.add_argument('--logdir', default='./checkpoints/debug', help='the directory to save checkpoints/logs')
parser.add_argument('--resume', action='store_true', help='continue to train the model')

parser.add_argument('--summary_freq', type=int, default=20, help='print and summary frequency')
parser.add_argument('--save_freq', type=int, default=1, help='save checkpoint frequency')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed')
parser.add_argument('--train_workers', type=int, default=8)
parser.add_argument('--test_workers', type=int, default=4)
parser.add_argument('--max_train_batches', type=int, default=0,
                    help='debug limit; 0 evaluates every training batch')
parser.add_argument('--max_test_batches', type=int, default=0,
                    help='debug limit; 0 evaluates every validation batch')

# Vis-MVSNet specific
parser.add_argument('--vismode', type=str, default='soft',
                    choices=['soft', 'hard', 'average', 'uwta', 'maxpool'],
                    help='multi-view fusion mode')
parser.add_argument('--nviews', type=int, default=3, help='number of views')
parser.add_argument('--eval_nviews', type=int, default=5,
                    help='number of views used for validation/test')
parser.add_argument('--stage1_dnum', type=int, default=48, help='stage 1 depth num')
parser.add_argument('--stage2_dnum', type=int, default=32, help='stage 2 depth num')
parser.add_argument('--stage3_dnum', type=int, default=16, help='stage 3 depth num')
parser.add_argument('--stage1_iscale', type=int, default=4, help='stage 1 interval scale')
parser.add_argument('--stage2_iscale', type=int, default=2, help='stage 2 interval scale')
parser.add_argument('--stage3_iscale', type=int, default=1, help='stage 3 interval scale')

# M1 and M2 use source-specific visibility GT; M3 does not.
parser.add_argument('--visibility_gt_downsample', type=int, default=2,
                    help='downsample for source GT depth/masks; 2 matches stage 3 resolution')
parser.add_argument('--pair_l1_weight', type=float, default=1.0)
parser.add_argument('--uncertainty_weight', type=float, default=1.0)
parser.add_argument('--visibility_weight', type=float, default=0.2)
parser.add_argument('--visibility_focal_gamma', type=float, default=2.0)
parser.add_argument('--occ_abs_tol', type=float, default=2.0,
                    help='absolute source-depth agreement tolerance in mm')
parser.add_argument('--occ_rel_tol', type=float, default=0.01,
                    help='relative source-depth agreement tolerance')
parser.add_argument('--hypothesis_residual_scale', type=float, default=1.0)
parser.add_argument('--hypothesis_visibility_weight', type=float, default=0.1)
parser.add_argument('--visibility_fusion_beta', type=float, default=0.2,
                    help='M2 soft-gate strength in [0,1]')
parser.add_argument('--hybrid_stage2_wide_num', type=int, default=8,
                    help='M3 Stage-2 hypotheses reserved for expanded tails')
parser.add_argument('--hybrid_stage3_wide_num', type=int, default=4,
                    help='M3 Stage-3 hypotheses reserved for expanded tails')
parser.add_argument('--hybrid_sigma_scale', type=float, default=2.0,
                    help='M3 uncertainty-to-tail-width multiplier')
parser.add_argument('--hybrid_max_scale', type=float, default=2.0,
                    help='M3 maximum tail expansion relative to baseline')
parser.add_argument('--hybrid_clip_mode', choices=('global', 'none'), default='global',
                    help='M3 global clipping (legacy) or unclipped baseline boundary policy')

args = parser.parse_args()
variant = get_model_variant(args.model_type)

if args.visibility_gt_downsample < 1:
    raise ValueError('--visibility_gt_downsample must be at least 1')
if min(args.pair_l1_weight, args.uncertainty_weight, args.visibility_weight,
       args.hypothesis_visibility_weight) < 0.0:
    raise ValueError('loss weights must be non-negative')
if args.occ_abs_tol < 0.0 or args.occ_rel_tol < 0.0:
    raise ValueError('occlusion tolerances must be non-negative')
if args.nviews < 2 or args.eval_nviews < 2:
    raise ValueError('training and evaluation require at least two views')
if args.hypothesis_residual_scale < 0.0:
    raise ValueError('--hypothesis_residual_scale must be non-negative')
if not 0.0 <= args.visibility_fusion_beta <= 1.0:
    raise ValueError('--visibility_fusion_beta must be in [0,1]')
if args.hybrid_sigma_scale < 0.0:
    raise ValueError('--hybrid_sigma_scale must be non-negative')
if args.hybrid_max_scale < 1.0:
    raise ValueError('--hybrid_max_scale must be at least 1')
if variant.hypothesis_fusion and args.vismode != 'soft':
    raise ValueError('M1 currently requires --vismode soft')
if variant.visibility_modeling and args.vismode != 'soft':
    raise ValueError('M2 currently requires --vismode soft')

os.makedirs(args.logdir, exist_ok=True)
sys.stdout = Logger(os.path.join(args.logdir, "logs.txt"))

if args.resume:
    assert args.mode == "train"
    assert args.loadckpt is None
if args.testpath is None:
    args.testpath = args.trainpath

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

if args.mode == "train":
    try:
        from tensorboardX import SummaryWriter
    except ImportError:
        from torch.utils.tensorboard import SummaryWriter
    if not os.path.isdir(args.logdir):
        os.mkdir(args.logdir)

    current_time_str = str(datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    print("current time", current_time_str)
    print("creating new summary file")
    logger = SummaryWriter(args.logdir)

print("argv:", sys.argv[1:])
print_args(args)
print(
    "ablation factors: M1(hypothesis fusion)={} "
    "M2(visibility modeling)={} M3(hybrid sampling)={} code={}".format(
        variant.hypothesis_fusion,
        variant.visibility_modeling,
        variant.hybrid_sampling,
        variant.code,
    )
)

# =============================================================================
# Dataset, Model, Optimizer
# =============================================================================
MVSDataset = find_dataset_def(args.dataset)
dataset_kwargs = dict(
    return_visibility_gt=variant.needs_visibility_gt,
    visibility_downsample=args.visibility_gt_downsample,
)
train_dataset = MVSDataset(args.trainpath, args.trainlist, "train",
                           args.nviews, args.numdepth, args.interval_scale,
                           **dataset_kwargs)
test_dataset = MVSDataset(args.testpath, args.testlist, "test",
                          args.eval_nviews, args.numdepth, args.interval_scale,
                          **dataset_kwargs)
TrainImgLoader = DataLoader(train_dataset, args.batch_size, shuffle=True,
                            num_workers=args.train_workers, drop_last=True,
                            pin_memory=True)
TestImgLoader = DataLoader(test_dataset, args.batch_size, shuffle=False,
                           num_workers=args.test_workers, drop_last=False,
                           pin_memory=True)

model_class = BaselineModel if args.model_type == 'vis' else EnhancedModel
loss_class = BaselineLoss if args.model_type == 'vis' else EnhancedLoss
model_kwargs = dict(
    mode=args.vismode,
    stage1_depth_num=args.stage1_dnum,
    stage1_interval_scale=args.stage1_iscale,
    stage2_depth_num=args.stage2_dnum,
    stage2_interval_scale=args.stage2_iscale,
    stage3_depth_num=args.stage3_dnum,
    stage3_interval_scale=args.stage3_iscale,
)
if args.model_type != 'vis':
    model_kwargs.update(
        hypothesis_fusion=variant.hypothesis_fusion,
        hypothesis_residual_scales=(
            args.hypothesis_residual_scale,
            args.hypothesis_residual_scale,
            args.hypothesis_residual_scale,
        ),
        visibility_fusion=variant.visibility_modeling,
        visibility_fusion_betas=(
            args.visibility_fusion_beta,
            args.visibility_fusion_beta,
            args.visibility_fusion_beta,
        ),
        hybrid_sampling=variant.hybrid_sampling,
        hybrid_stage2_wide_num=args.hybrid_stage2_wide_num,
        hybrid_stage3_wide_num=args.hybrid_stage3_wide_num,
        hybrid_sigma_scale=args.hybrid_sigma_scale,
        hybrid_max_scale=args.hybrid_max_scale,
        hybrid_clip_mode=args.hybrid_clip_mode,
    )
model = model_class(**model_kwargs)

if args.mode in ["train", "test"]:
    model = nn.DataParallel(model)
model.cuda()
if args.model_type == 'vis':
    model_loss = loss_class(occ_guide=False)
else:
    model_loss = loss_class(
        visibility_supervision=variant.visibility_modeling,
        pair_l1_weight=args.pair_l1_weight,
        uncertainty_weight=args.uncertainty_weight,
        visibility_weight=args.visibility_weight,
        visibility_focal_gamma=args.visibility_focal_gamma,
        hypothesis_visibility_weight=(
            args.hypothesis_visibility_weight
            if variant.hypothesis_fusion else 0.0),
        occ_abs_tol=args.occ_abs_tol,
        occ_rel_tol=args.occ_rel_tol,
    )
optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.wd)

# =============================================================================
# Load checkpoint
# =============================================================================
start_epoch = 0


def load_model_state(state_dict):
    target_has_module = next(iter(model.state_dict())).startswith('module.')
    source_has_module = next(iter(state_dict)).startswith('module.')
    if target_has_module and not source_has_module:
        state_dict = {'module.' + key: value for key, value in state_dict.items()}
    elif source_has_module and not target_has_module:
        state_dict = {key[7:]: value for key, value in state_dict.items()}
    incompatible = model.load_state_dict(state_dict, strict=False)
    missing = list(incompatible.missing_keys)
    unexpected = list(incompatible.unexpected_keys)
    allowed_missing = []
    if variant.hypothesis_fusion:
        allowed_missing = [
            key for key in missing
            if '.hypothesis_weight_net.' in key
        ]
    invalid_missing = [key for key in missing if key not in allowed_missing]
    if unexpected or invalid_missing:
        raise RuntimeError(
            'checkpoint mismatch: unexpected={}, missing={}'.format(
                unexpected, invalid_missing))
    if allowed_missing:
        print('Initialized new hypothesis-fusion parameters:')
        for key in allowed_missing:
            print('  ', key)


if (args.mode == "train" and args.resume) or (args.mode == "test" and not args.loadckpt):
    saved_models = [
        fn for fn in os.listdir(args.logdir)
        if fn.startswith('model_') and fn.endswith('.ckpt')
    ]
    if not saved_models:
        raise FileNotFoundError('no model_*.ckpt files found in {}'.format(args.logdir))
    saved_models = sorted(saved_models, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    loadckpt = os.path.join(args.logdir, saved_models[-1])
    print("resuming", loadckpt)
    state_dict = torch.load(loadckpt, map_location='cpu')
    load_model_state(state_dict['model'])
    optimizer.load_state_dict(state_dict['optimizer'])
    start_epoch = state_dict['epoch'] + 1
elif args.loadckpt:
    print("loading model {}".format(args.loadckpt))
    state_dict = torch.load(args.loadckpt, map_location='cpu')
    load_model_state(state_dict.get('model', state_dict))
print("start at epoch {}".format(start_epoch))
print('Number of model parameters: {}'.format(
    sum([p.data.nelement() for p in model.parameters()])))


# =============================================================================
# Best-metric tracking
# =============================================================================
best_metrics = {
    "abs_depth_error": 1e9,
    "thres2mm_error": 1e9,
    "thres4mm_error": 1e9,
    "thres8mm_error": 1e9,
}


def save_best_models(avg_test, epoch_idx):
    abs_err = avg_test["abs_depth_error"]
    th2 = avg_test["thres2mm_error"]
    th4 = avg_test["thres4mm_error"]
    th8 = avg_test["thres8mm_error"]

    if abs_err < best_metrics["abs_depth_error"]:
        best_metrics["abs_depth_error"] = abs_err
        torch.save({'epoch': epoch_idx, 'model': model.state_dict(),
                     'optimizer': optimizer.state_dict()},
                   os.path.join(args.logdir, "best_abs.ckpt"))
        print(f"  Best ABS updated: {abs_err:.4f}")

    if th2 < best_metrics["thres2mm_error"]:
        best_metrics["thres2mm_error"] = th2
        torch.save({'epoch': epoch_idx, 'model': model.state_dict(),
                     'optimizer': optimizer.state_dict()},
                   os.path.join(args.logdir, "best_2mm.ckpt"))
        print(f"  Best 2mm updated: {th2:.4f}")

    if th4 < best_metrics["thres4mm_error"]:
        best_metrics["thres4mm_error"] = th4
        torch.save({'epoch': epoch_idx, 'model': model.state_dict(),
                     'optimizer': optimizer.state_dict()},
                   os.path.join(args.logdir, "best_4mm.ckpt"))
        print(f"  Best 4mm updated: {th4:.4f}")

    if th8 < best_metrics["thres8mm_error"]:
        best_metrics["thres8mm_error"] = th8
        torch.save({'epoch': epoch_idx, 'model': model.state_dict(),
                     'optimizer': optimizer.state_dict()},
                   os.path.join(args.logdir, "best_8mm.ckpt"))
        print(f"  Best 8mm updated: {th8:.4f}")


# =============================================================================
# Train / Test loops
# =============================================================================
def train():
    milestones = [int(epoch_idx) for epoch_idx in args.lrepochs.split(':')[0].split(',')]
    lr_gamma = 1 / float(args.lrepochs.split(':')[1])
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones, gamma=lr_gamma, last_epoch=start_epoch - 1)

    for epoch_idx in range(start_epoch, args.epochs):
        print('Epoch {}:'.format(epoch_idx))
        lr_scheduler.step()
        global_step = len(TrainImgLoader) * epoch_idx

        # ---- training ----
        for batch_idx, sample in enumerate(TrainImgLoader):
            if args.max_train_batches and batch_idx >= args.max_train_batches:
                break
            start_time = time.time()
            global_step = len(TrainImgLoader) * epoch_idx + batch_idx
            do_summary = global_step % args.summary_freq == 0
            loss, scalar_outputs = train_sample(sample, detailed_summary=do_summary)
            if do_summary:
                save_scalars(logger, 'train', scalar_outputs, global_step)
            del scalar_outputs
            print('Epoch {}/{}, Iter {}/{}, train loss = {:.3f}, time = {:.3f}'.format(
                epoch_idx, args.epochs, batch_idx, len(TrainImgLoader), loss, time.time() - start_time))

        # ---- save checkpoint ----
        if (epoch_idx + 1) % args.save_freq == 0:
            torch.save({'epoch': epoch_idx, 'model': model.state_dict(),
                         'optimizer': optimizer.state_dict()},
                       "{}/model_{:0>6}.ckpt".format(args.logdir, epoch_idx))

        # ---- testing ----
        avg_test_scalars = DictAverageMeter()
        for batch_idx, sample in enumerate(TestImgLoader):
            if args.max_test_batches and batch_idx >= args.max_test_batches:
                break
            start_time = time.time()
            loss, scalar_outputs = test_sample(sample, detailed_summary=True)
            avg_test_scalars.update(scalar_outputs)
            del scalar_outputs
            print('Epoch {}/{}, Iter {}/{}, test loss = {:.3f}, time = {:3f}'.format(
                epoch_idx, args.epochs, batch_idx, len(TestImgLoader), loss,
                time.time() - start_time))

        avg_test = avg_test_scalars.mean()
        save_scalars(logger, 'fulltest', avg_test, global_step)
        print("avg_test_scalars:", avg_test)

        save_best_models(avg_test, epoch_idx)

        if epoch_idx == args.epochs - 1:
            torch.save({'epoch': epoch_idx, 'model': model.state_dict(),
                         'optimizer': optimizer.state_dict()},
                       os.path.join(args.logdir, "latest.ckpt"))

        gc.collect()


def test():
    avg_test_scalars = DictAverageMeter()
    for batch_idx, sample in enumerate(TestImgLoader):
        if args.max_test_batches and batch_idx >= args.max_test_batches:
            break
        start_time = time.time()
        loss, scalar_outputs = test_sample(sample, detailed_summary=True)
        avg_test_scalars.update(scalar_outputs)
        del scalar_outputs
        print('Iter {}/{}, test loss = {:.3f}, time = {:3f}'.format(
            batch_idx, len(TestImgLoader), loss, time.time() - start_time))
        if batch_idx % 100 == 0:
            print("Iter {}/{}, test results = {}".format(
                batch_idx, len(TestImgLoader), avg_test_scalars.mean()))
    print("final", avg_test_scalars)


def train_sample(sample, detailed_summary=False):
    model.train()
    optimizer.zero_grad()

    sample_cuda = tocuda(sample)
    depth_gt = sample_cuda["depth"].unsqueeze(1)   # [B, 1, H, W]
    mask = sample_cuda["mask"].unsqueeze(1)         # [B, 1, H, W]
    depth_values = sample_cuda["depth_values"]

    # extract base depth interval for loss normalisation
    depth_interval = depth_values[:, 1] - depth_values[:, 0]  # [B]

    outputs, final_depth, prob_maps = model(
        sample_cuda["imgs"], sample_cuda["proj_matrices"], depth_values)

    loss_kwargs = {}
    if variant.needs_visibility_gt:
        loss_kwargs = dict(
            visibility_depths=sample_cuda['visibility_depths'],
            visibility_masks=sample_cuda['visibility_masks'],
            proj_matrices=sample_cuda['proj_matrices'],
        )
    loss, scalar_outputs = model_loss(
        outputs, depth_gt, mask, depth_interval, **loss_kwargs)
    loss.backward()
    optimizer.step()

    if detailed_summary:
        # use stage 3 (final) depth for metrics
        depth_est = outputs[-1][0]  # stage 3 depth [B, H_stage, W_stage]
        # upsample to gt resolution
        depth_est_full = F.interpolate(
            depth_est.unsqueeze(1),
            size=(depth_gt.shape[2], depth_gt.shape[3]),
            mode='bilinear', align_corners=False).squeeze(1)

        valid = mask.squeeze(1) > 0.5
        scalar_outputs["abs_depth_error"] = tensor2float(
            AbsDepthError_metrics(depth_est_full, depth_gt.squeeze(1), valid))
        e2 = tensor2float(Thres_metrics(depth_est_full, depth_gt.squeeze(1), valid, 2))
        e4 = tensor2float(Thres_metrics(depth_est_full, depth_gt.squeeze(1), valid, 4))
        e8 = tensor2float(Thres_metrics(depth_est_full, depth_gt.squeeze(1), valid, 8))
        scalar_outputs["thres2mm_error"] = e2
        scalar_outputs["thres4mm_error"] = e4
        scalar_outputs["thres8mm_error"] = e8
        scalar_outputs["thres2mm_acc"] = 1.0 - e2
        scalar_outputs["thres4mm_acc"] = 1.0 - e4
        scalar_outputs["thres8mm_acc"] = 1.0 - e8

    return tensor2float(loss), tensor2float(scalar_outputs)


@make_nograd_func
def test_sample(sample, detailed_summary=True):
    model.eval()

    sample_cuda = tocuda(sample)
    depth_gt = sample_cuda["depth"].unsqueeze(1)
    mask = sample_cuda["mask"].unsqueeze(1)
    depth_values = sample_cuda["depth_values"]

    depth_interval = depth_values[:, 1] - depth_values[:, 0]

    outputs, final_depth, prob_maps = model(
        sample_cuda["imgs"], sample_cuda["proj_matrices"], depth_values)

    loss_kwargs = {}
    if variant.needs_visibility_gt:
        loss_kwargs = dict(
            visibility_depths=sample_cuda['visibility_depths'],
            visibility_masks=sample_cuda['visibility_masks'],
            proj_matrices=sample_cuda['proj_matrices'],
        )
    loss, scalar_outputs = model_loss(
        outputs, depth_gt, mask, depth_interval, **loss_kwargs)

    if detailed_summary:
        depth_est = outputs[-1][0]
        depth_est_full = F.interpolate(
            depth_est.unsqueeze(1),
            size=(depth_gt.shape[2], depth_gt.shape[3]),
            mode='bilinear', align_corners=False).squeeze(1)

        valid = mask.squeeze(1) > 0.5
        scalar_outputs["abs_depth_error"] = tensor2float(
            AbsDepthError_metrics(depth_est_full, depth_gt.squeeze(1), valid))
        e2 = tensor2float(Thres_metrics(depth_est_full, depth_gt.squeeze(1), valid, 2))
        e4 = tensor2float(Thres_metrics(depth_est_full, depth_gt.squeeze(1), valid, 4))
        e8 = tensor2float(Thres_metrics(depth_est_full, depth_gt.squeeze(1), valid, 8))
        scalar_outputs["thres2mm_error"] = e2
        scalar_outputs["thres4mm_error"] = e4
        scalar_outputs["thres8mm_error"] = e8
        scalar_outputs["thres2mm_acc"] = 1.0 - e2
        scalar_outputs["thres4mm_acc"] = 1.0 - e4
        scalar_outputs["thres8mm_acc"] = 1.0 - e8

    return tensor2float(loss), tensor2float(scalar_outputs)


# =============================================================================
# Entry
# =============================================================================
if __name__ == '__main__':
    if args.mode == "train":
        train()
    elif args.mode == "test":
        test()
