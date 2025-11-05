#-*- coding:utf-8 -*-
from diffusion_model.trainer import GaussianDiffusion, num_to_groups
from diffusion_model.unet import create_model
from torchvision.transforms import Compose, Lambda
import nibabel as nib
import torchio as tio
import numpy as np
import argparse
import torch
import os
import glob
from sklearn.preprocessing import MinMaxScaler

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--inputfolder', type=str, default="dataset/no_contrast_ct")  # 改成 CT 資料夾
parser.add_argument('-e', '--exportfolder', type=str, default="exports/")
parser.add_argument('--input_size', type=int, default=128)
parser.add_argument('--depth_size', type=int, default=128)
parser.add_argument('--num_channels', type=int, default=64)
parser.add_argument('--num_res_blocks', type=int, default=1)
parser.add_argument('--batchsize', type=int, default=1)
parser.add_argument('--num_samples', type=int, default=1)
parser.add_argument('--num_class_labels', type=int, default=2)  # [CT + noise]
parser.add_argument('--timesteps', type=int, default=250)
parser.add_argument('-w', '--weightfile', type=str, default="model/model_128.pt")
args = parser.parse_args()

exportfolder = args.exportfolder
inputfolder = args.inputfolder
input_size = args.input_size
depth_size = args.depth_size
batchsize = args.batchsize
weightfile = args.weightfile
num_channels = args.num_channels
num_res_blocks = args.num_res_blocks
num_samples = args.num_samples
in_channels = args.num_class_labels
out_channels = 1
device = "cuda"

ct_list = sorted(glob.glob(f"{inputfolder}/*.nii*"))
print(f"找到 {len(ct_list)} 筆輸入 CT 影像")

scaler = MinMaxScaler()  # 放在外面，避免每張都 new
def read_image(file_path, use_scaler=True):
    img = nib.load(file_path).get_fdata()
    if use_scaler:
        orig_shape = img.shape
        img = scaler.fit_transform(img.reshape(-1, 1)).reshape(orig_shape)
    return img

def resize_img(img, input_size, depth_size):
    h, w, d = img.shape
    if h != input_size or w != input_size or d != depth_size:
        img_t = tio.ScalarImage(tensor=img[np.newaxis, ...])
        resize_transform = tio.Resize((input_size, input_size, depth_size))
        img_resized = np.asarray(resize_transform(img_t))[0]
        return img_resized
    return img

# ===== 轉換影像格式 =====
input_transform = Compose([
    Lambda(lambda t: torch.tensor(t).float()),
    Lambda(lambda t: (t * 2) - 1),
    Lambda(lambda t: t.permute(2, 0, 1)),     # HWD → D H W
    Lambda(lambda t: t.unsqueeze(0)),         # → (1, D, H, W)
])

# ===== 建立模型 =====
model = create_model(input_size, num_channels, num_res_blocks, in_channels=in_channels, out_channels=out_channels).cuda()

diffusion = GaussianDiffusion(
    model,
    image_size=input_size,
    depth_size=depth_size,
    timesteps=args.timesteps,
    loss_type='L1',
    with_condition=True,
).cuda()

state_dict = torch.load(weightfile, map_location=device)['ema']

# 移除多餘的 `module.` 前綴
new_state_dict = {}
for k, v in state_dict.items():
    new_k = k.replace('module.', '')  # 如果有 DataParallel 包裝
    new_state_dict[new_k] = v

diffusion.load_state_dict(new_state_dict)
print("✅ 模型載入成功")

# ===== 輸出資料夾 =====
img_dir = os.path.join(exportfolder, "image")
input_dir = os.path.join(exportfolder, "input")
os.makedirs(img_dir, exist_ok=True)
os.makedirs(input_dir, exist_ok=True)


# ===== 推論過程 =====
for k, inputfile in enumerate(ct_list):
    print(f"剩餘 {len(ct_list) - (k + 1)} 張")
    
    ct_nii = nib.load(inputfile)
    ct_img = read_image(inputfile, use_scaler=True)
    img_resized = resize_img(ct_img, input_size, depth_size)

    input_tensor = input_transform(img_resized).unsqueeze(0).to(device)
    batches = num_to_groups(num_samples, batchsize)
    steps = len(batches)
    sample_count = 0
    counter = 0

    for i, bsize in enumerate(batches):
        print(f"Step [{i+1}/{steps}]")
        condition_tensors, counted_samples = [], []

        for _ in range(bsize):
            condition_tensors.append(input_tensor)
            counted_samples.append(sample_count)
            sample_count += 1

        condition_tensors = torch.cat(condition_tensors, 0).cuda()

        # 生成影像
        all_images = diffusion.sample(batch_size=bsize, condition_tensors=condition_tensors)
        sampleImages = all_images.cpu()

        for b, c in enumerate(counted_samples):
            counter += 1
            sampleImage = sampleImages[b][0].numpy()
            sampleImage = sampleImage.transpose(1, 2, 0)  # 回復 HWD
            out_name = os.path.basename(inputfile)

            # 儲存生成影像
            out_path = os.path.join(img_dir, f"{counter}_pred_{out_name}")
            nib.save(nib.Nifti1Image(sampleImage, affine=ct_nii.affine), out_path)

            # 同時儲存輸入影像（方便比對）
            in_path = os.path.join(input_dir, f"{counter}_input_{out_name}")
            nib.save(nib.Nifti1Image(ct_img, affine=ct_nii.affine), in_path)

        torch.cuda.empty_cache()
    print("✅ 處理完成！")
