import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def rgb2y(img_rgb):
    """RGB → Y 채널 (float32, 0-255 range)"""
    return 0.257 * img_rgb[..., 0] + 0.504 * img_rgb[..., 1] + 0.098 * img_rgb[..., 2] + 16

def load_image(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def compare_images(gt_path, input_path, restored_path):
    gt = load_image(gt_path)
    input_img = load_image(input_path)
    restored = load_image(restored_path)

    gt_y = rgb2y(gt)
    input_y = rgb2y(input_img)
    restored_y = rgb2y(restored)

    # Resize if needed
    h, w = gt_y.shape
    input_y = cv2.resize(input_y, (w, h), interpolation=cv2.INTER_LINEAR)
    restored_y = cv2.resize(restored_y, (w, h), interpolation=cv2.INTER_LINEAR)

    psnr_input = peak_signal_noise_ratio(gt_y, input_y, data_range=255)
    ssim_input = structural_similarity(gt_y, input_y, data_range=255)

    psnr_restored = peak_signal_noise_ratio(gt_y, restored_y, data_range=255)
    ssim_restored = structural_similarity(gt_y, restored_y, data_range=255)

    print("\n📌 단일 이미지 PSNR / SSIM (Y 채널 기준)")
    print(f"📎 입력 vs GT      → PSNR: {psnr_input:.2f}  SSIM: {ssim_input:.4f}")
    print(f"📎 복원결과 vs GT → PSNR: {psnr_restored:.2f}  SSIM: {ssim_restored:.4f}")

# -----------------------------
# ✅ 사용 예시 (수정 가능)
gt      = "E:/Restormer/Deraining/Datasets/rain100H/test/norain/norain-22.png"
input_  = "E:/Restormer/Deraining/Datasets/rain100H/test/rain/norain-22.png"
output  = "E:/Restormer/Deraining/results/rain100H/norain-22.png"

compare_images(gt, input_, output)
