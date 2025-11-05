#-*- coding:utf-8 -*-
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor, Lambda
from glob import glob
from utils.dtypes import LabelEnum
import matplotlib.pyplot as plt
import nibabel as nib
import torchio as tio
import numpy as np
import torch
import re
import os
from datetime import datetime

class NiftiImageGenerator(Dataset):
    def __init__(self, imagefolder, input_size, depth_size, transform=None):
        self.imagefolder = imagefolder
        self.input_size = input_size
        self.depth_size = depth_size
        self.inputfiles = glob(os.path.join(imagefolder, '*.nii.gz'))
        self.scaler = MinMaxScaler()
        self.transform = transform

    def read_image(self, file_path):
        img = nib.load(file_path).get_fdata()
        img = self.scaler.fit_transform(img.reshape(-1, img.shape[-1])).reshape(img.shape) # 0 -> 1 scale
        return img

    def plot_samples(self, n_slice=15, n_row=4):
        samples = [self[index] for index in np.random.randint(0, len(self), n_row*n_row)]
        for i in range(n_row):
            for j in range(n_row):
                sample = samples[n_row*i+j]
                sample = sample[0]
                plt.subplot(n_row, n_row, n_row*i+j+1)
                plt.imshow(sample[:, :, n_slice])
        plt.show()

    def __len__(self):
        return len(self.inputfiles)

    def __getitem__(self, index):
        inputfile = self.inputfiles[index]
        img = self.read_image(inputfile)
        h, w, d= img.shape
        if h != self.input_size or w != self.input_size or d != self.depth_size:
            img = tio.ScalarImage(inputfile)
            cop = tio.Resize((self.input_size, self.input_size, self.depth_size))
            img = np.asarray(cop(img))[0]

        if self.transform is not None:
            img = self.transform(img)
        return img

class NiftiPairImageGenerator(Dataset):
    def __init__(self,
            input_folder: str,
            target_folder: str,
            input_size: int,
            depth_size: int,
            transform=None,
            target_transform=None,
            augment=False  
        ):
        self.input_folder = input_folder
        self.target_folder = target_folder
        self.pair_files = self.pair_file()
        self.input_size = input_size
        self.depth_size = depth_size
        self.scaler = MinMaxScaler()
        self.transform = transform
        self.target_transform = target_transform
        self.augment = augment

        if self.augment:
            self.augment_transform = tio.Compose([
                tio.RandomFlip(axes=('LR',), p=0.5),
                tio.RandomGamma(log_gamma=(-0.3, 0.3), p=0.3),
                tio.RandomNoise(std=(0, 0.05), p=0.3)
            ])
        else:
            self.augment_transform = None

    def pair_file(self):
        # 讀取兩個資料夾內所有 nii 或 nii.gz 檔案，並依檔名排序
        input_files = sorted(glob(os.path.join(self.input_folder, '*.nii*')))
        target_files = sorted(glob(os.path.join(self.target_folder, '*.nii*')))

        # 此處假設兩邊檔案數量與排序皆一致，否則可進一步加入檔名比對邏輯
        pairs = list(zip(input_files, target_files))
        print(f"Found {len(pairs)} paired files.")
        return pairs

    def read_image(self, file_path, use_scaler=True):
        # 讀取 .nii 檔案，並用 MinMaxScaler 將數值縮放到 [0,1]
        img = nib.load(file_path).get_fdata()
        if use_scaler:
            # 將原始數據 reshape 成 (N, feature) 再轉換
            orig_shape = img.shape
            img = self.scaler.fit_transform(img.reshape(-1, 1)).reshape(orig_shape)
        return img

    def resize_img(self, img):
        h, w, d = img.shape
        if h != self.input_size or w != self.input_size or d != self.depth_size:
            # 利用 torchio 將影像縮放到指定的尺寸 (input_size, input_size, depth_size)
            img_t = tio.ScalarImage(tensor=img[np.newaxis, ...])
            resize_transform = tio.Resize((self.input_size, self.input_size, self.depth_size))
            img_resized = np.asarray(resize_transform(img_t))[0]
            return img_resized   
        return img


    def sample_conditions(self, batch_size: int):

        if not hasattr(self, "sample_count"):
            self.sample_count = 0
        self.sample_count += 1  # 增加 sample 次數

        indexes = np.random.randint(0, len(self.pair_files), batch_size)
        input_files = [self.pair_files[index][0] for index in indexes]
        input_tensors = []

        # 設定 log 檔案路徑與名稱
        log_dir = "./results"
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, "sample_log.txt")

        with open(log_file, "a") as f:  # "a" 代表 append 模式，不會覆蓋
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"\n[{timestamp}] Sampling times {self.sample_count}:\n")

            for input_file in input_files:
                log_msg = f"Sampling from input file: {input_file}"
                print(log_msg)  # 終端機顯示
                f.write(log_msg + "\n")  # 寫入 log 檔案

                input_img = self.read_image(input_file)
                input_img = self.resize_img(input_img)
                if self.transform is not None:
                    input_img = self.transform(input_img).unsqueeze(0)
                    input_tensors.append(input_img)

        return torch.cat(input_tensors, 0).cuda()


    def __len__(self):
        return len(self.pair_files) * (2 if self.augment else 1)

    def __getitem__(self, index):

        if not self.augment:
            base_index = index
            augment_flag = False
        else:
            base_index = index // 2
            augment_flag = index % 2 == 1

        input_file, target_file = self.pair_files[base_index]

        input_img = self.read_image(input_file)
        target_img = self.read_image(target_file)

        input_img = self.resize_img(input_img)
        target_img = self.resize_img(target_img)

        # ✅ 如果是增強版本，才進行增強
        if augment_flag and self.augment_transform is not None:
            subject = tio.Subject(
                input=tio.ScalarImage(tensor=input_img[np.newaxis, ...]),
                target=tio.ScalarImage(tensor=target_img[np.newaxis, ...])
            )
            transformed = self.augment_transform(subject)
            input_img = transformed['input'].numpy()[0]
            target_img = transformed['target'].numpy()[0]

        # 後處理轉成 Tensor
        if self.transform is not None:
            input_img = self.transform(input_img)
        if self.target_transform is not None:
            target_img = self.target_transform(target_img)

        return {'input': input_img, 'target': target_img}
