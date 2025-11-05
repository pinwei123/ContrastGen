#-*- coding:utf-8 -*-
# +
from torchvision.transforms import RandomCrop, Compose, ToPILImage, Resize, ToTensor, Lambda
from diffusion_model.trainer import GaussianDiffusion, Trainer
from diffusion_model.unet import create_model
from dataset import NiftiImageGenerator, NiftiPairImageGenerator
import argparse
import torch
from torch import nn
import os 
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"


parser = argparse.ArgumentParser()
parser.add_argument('-i', '--inputfolder', type=str, default="/dataset/data/cacs")
parser.add_argument('-t', '--targetfolder', type=str, default="/dataset/data/cas")
parser.add_argument('-vi', '--val_input', type=str, default="/dataset/data/cacs_test")
parser.add_argument('-vt', '--val_target', type=str, default="/dataset/data/cas_test")
parser.add_argument('--input_size', type=int, default=128)
parser.add_argument('--depth_size', type=int, default=128)
parser.add_argument('--num_channels', type=int, default=64)
parser.add_argument('--num_res_blocks', type=int, default=1)
parser.add_argument('--num_class_labels', type=int, default=2)
parser.add_argument('--train_lr', type=float, default=1e-5)
parser.add_argument('--batchsize', type=int, default=2)
parser.add_argument('--early_stop_patience', type=int, default=20)
parser.add_argument('--val_every', type=int, default=500)
parser.add_argument('--epochs', type=int, default=50000) # epochs parameter specifies the number of training iterations
parser.add_argument('--timesteps', type=int, default=250)
parser.add_argument('--save_and_sample_every', type=int, default=5000)
parser.add_argument('--with_condition', action='store_true')
parser.add_argument('-r', '--resume_weight', type=str, default="")
args = parser.parse_args()

inputfolder = args.inputfolder
targetfolder = args.targetfolder
val_input = args.val_input
val_target = args.val_target
input_size = args.input_size
depth_size = args.depth_size
num_channels = args.num_channels
num_res_blocks = args.num_res_blocks
num_class_labels = args.num_class_labels
save_and_sample_every = args.save_and_sample_every
with_condition = args.with_condition
resume_weight = args.resume_weight
train_lr = args.train_lr
early_stop_patience = args.early_stop_patience
val_every = args.val_every

# input tensor: (B, 1, H, W, D)  value range: [-1, 1]
transform = Compose([
    Lambda(lambda t: torch.tensor(t).float()),             # numpy → tensor
    Lambda(lambda t: (t * 2) - 1),                          # normalize to [-1, 1]
    Lambda(lambda t: t.permute(2, 0, 1)),                  # (H, W, D) → (D, H, W)
    Lambda(lambda t: t.unsqueeze(0)),                      # → (1, D, H, W)
])


input_transform = Compose([
    Lambda(lambda t: torch.tensor(t).float()),             # numpy → tensor
    Lambda(lambda t: (t * 2) - 1),                          # normalize to [-1, 1]
    Lambda(lambda t: t.permute(2, 0, 1)),                  # (H, W, D) → (D, H, W)
    Lambda(lambda t: t.unsqueeze(0)),                      # → (1, D, H, W)
])



if with_condition:
    dataset = NiftiPairImageGenerator(
        inputfolder,
        targetfolder,
        input_size=input_size,
        depth_size=depth_size,
        transform=input_transform if with_condition else transform,
        target_transform=transform,
        augment=False
    )
    val_dataset = NiftiPairImageGenerator(
        val_input,
        val_target,
        input_size=input_size,
        depth_size=depth_size,
        transform=input_transform if with_condition else transform,
        target_transform=transform,
        augment=False
    )
else:
    dataset = NiftiImageGenerator(
        inputfolder,
        input_size=input_size,
        depth_size=depth_size,
        transform=transform
    )

print(len(dataset))

in_channels = num_class_labels if with_condition else 1
out_channels = 1

print(f"🔥 Using device: {torch.cuda.current_device()} - {torch.cuda.get_device_name(0)}")

model = create_model(input_size, num_channels, num_res_blocks, in_channels=in_channels, out_channels=out_channels)

# 使用 DataParallel 進行多 GPU 訓練
if torch.cuda.device_count() > 1:
    print(f"🔥 使用 {torch.cuda.device_count()} 張 GPU")
    model = nn.DataParallel(model)  # 包裝成 DataParallel
else:
    print("🔥 使用單張 GPU")

# 然後將模型移動到 GPU
model = model.cuda()  # 模型會自動分配到多個 GPU

diffusion = GaussianDiffusion(
    model,
    image_size = input_size,
    depth_size = depth_size,
    timesteps = args.timesteps,   # number of steps
    loss_type = 'l1',    # L1 or L2
    with_condition=with_condition,
    channels=out_channels
).cuda()

if len(resume_weight) > 0:
    weight = torch.load(resume_weight, map_location='cuda')
    diffusion.load_state_dict(weight['ema'])
    print("Model Loaded!")

trainer = Trainer(
    diffusion,
    dataset,
    val_dataset,
    early_stop_patience=early_stop_patience,  # 連續幾次沒進步就停止
    val_every=val_every,  # 每幾步驗證一次
    image_size = input_size,
    depth_size = depth_size,
    train_batch_size = args.batchsize,
    train_lr = train_lr,
    train_num_steps = args.epochs,         # total training steps
    gradient_accumulate_every = 2,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    fp16 = False,#True,                       # turn on mixed precision training with apex
    with_condition=with_condition,
    save_and_sample_every = save_and_sample_every,
)

trainer.train()
