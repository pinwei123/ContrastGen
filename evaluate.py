import os
import re
import nibabel as nib
import numpy as np
import pandas as pd
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

def load_nifti(path):
    img = nib.load(path)
    return img.get_fdata().astype(np.float32)

def compute_mae(pred, gt):
    return np.mean(np.abs(pred - gt))

def compute_ssim(pred, gt):
    vals = []
    for i in range(pred.shape[2]):  # 逐切片算 SSIM
        vals.append(
            ssim(pred[:,:,i], gt[:,:,i], data_range=gt[:,:,i].max()-gt[:,:,i].min())
        )
    return float(np.mean(vals))

def compute_psnr(pred, gt):
    return psnr(gt, pred, data_range=gt.max() - gt.min())

def extract_id_from_pred(fname):
    # 例: "1_pred_18389097_CACS.nii" → "18389097"
    match = re.search(r"pred_(\d+)_", fname)
    return match.group(1) if match else None

def extract_id_from_gt(fname):
    # 例: "18389097_CAS.nii" → "18389097"
    match = re.match(r"(\d+)_", fname)
    return match.group(1) if match else None

def evaluate_pair(pred_path, gt_path):
    pred = load_nifti(pred_path)
    gt   = load_nifti(gt_path)
    assert pred.shape == gt.shape, f"Shape mismatch: {pred.shape} vs {gt.shape}"

    mae_val  = compute_mae(pred, gt)
    ssim_val = compute_ssim(pred, gt)
    psnr_val = compute_psnr(pred, gt)
    return mae_val, ssim_val, psnr_val

if __name__ == "__main__":
    pred_dir = "/wei/med-ddpm/sample_output/image"
    gt_dir   = "/dataset/data/cas_test"

    # 建立 gt 檔名 → ID 對照表
    gt_map = {}
    for fname in os.listdir(gt_dir):
        if fname.endswith(".nii") or fname.endswith(".nii.gz"):
            pid = extract_id_from_gt(fname)
            if pid:
                gt_map[pid] = os.path.join(gt_dir, fname)

    results = []
    for fname in sorted(os.listdir(pred_dir)):
        if not (fname.endswith(".nii") or fname.endswith(".nii.gz")):
            continue

        pid = extract_id_from_pred(fname)
        if not pid:
            print(f"⚠️ 無法從 {fname} 解析病人 ID")
            continue

        pred_path = os.path.join(pred_dir, fname)
        gt_path   = gt_map.get(pid, None)
        if not gt_path:
            print(f"⚠️ 找不到對應的 GT for {fname} (ID={pid})")
            continue

        mae, ssim_val, psnr_val = evaluate_pair(pred_path, gt_path)
        results.append([pid, fname, os.path.basename(gt_path), mae, ssim_val, psnr_val])
        print(f"ID {pid}: MAE={mae:.4f}, SSIM={ssim_val:.4f}, PSNR={psnr_val:.2f} dB")

    # 存 CSV
    df = pd.DataFrame(results, columns=["PatientID", "PredFile", "GTFile", "MAE", "SSIM", "PSNR"])
    df.to_csv("evaluation_results.csv", index=False)

    # 印出平均值
    if len(results) > 0:
        print("\n===== 平均結果 =====")
        print(f"MAE : {df['MAE'].mean():.4f}")
        print(f"SSIM: {df['SSIM'].mean():.4f}")
        print(f"PSNR: {df['PSNR'].mean():.2f} dB")
        print("✅ 已將結果存到 evaluation_results.csv")


