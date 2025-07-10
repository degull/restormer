# python ver
""" import os
import numpy as np
import cv2
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from natsort import natsorted
from glob import glob

def read_image(fp):
    img = cv2.imread(fp)
    if img is None:
        raise FileNotFoundError(f"파일 없음: {fp}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def compute_metrics(ref_dir, cmp_dir):
    ref_files = natsorted(glob(os.path.join(ref_dir, "*.png")))
    cmp_files = natsorted(glob(os.path.join(cmp_dir, "*.png")))

    assert len(ref_files) == len(cmp_files), "파일 수가 다릅니다."

    psnr_list = []
    ssim_list = []

    for ref_fp, cmp_fp in zip(ref_files, cmp_files):
        ref = read_image(ref_fp)
        cmp = read_image(cmp_fp)

        if ref.shape != cmp.shape:
            print(f"[WARN] 크기 다름: {os.path.basename(ref_fp)} → 리사이즈됨")
            cmp = cv2.resize(cmp, (ref.shape[1], ref.shape[0]))

        psnr_list.append(psnr(ref, cmp, data_range=255))
        ssim_list.append(ssim(ref, cmp, channel_axis=2, data_range=255))

    return np.mean(psnr_list), np.mean(ssim_list)


def main():
    base = "E:/Restormer/Deraining"
    datasets = ["rain100H", "rain100L"]

    for d in datasets:
        gt_dir   = os.path.join(base, "Datasets", d, "test", "norain")
        inp_dir  = os.path.join(base, "Datasets", d, "test", "rain")
        out_dir  = os.path.join(base, "results", d)

        psnr_inp, ssim_inp = compute_metrics(gt_dir, inp_dir)
        psnr_out, ssim_out = compute_metrics(gt_dir, out_dir)

        print(f"\n📊 [{d.upper()}] 복원 전 vs 복원 후 PSNR / SSIM:")
        print(f"  ✗ 입력   : PSNR={psnr_inp:.2f}, SSIM={ssim_inp:.4f}")
        print(f"  ✓ 복원결과: PSNR={psnr_out:.2f}, SSIM={ssim_out:.4f}")

if __name__ == "__main__":
    main() """


# matlab ver
import os
import numpy as np
import cv2
from glob import glob
from natsort import natsorted
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def rgb2y(img_rgb):
    """CV2 BGR→RGB 후 YCbCr 변환하여 Y 채널만 반환 (실수형 0-255)."""
    img_rgb = img_rgb.astype(np.float32)
    y = 0.257 * img_rgb[..., 2] + 0.504 * img_rgb[..., 1] + 0.098 * img_rgb[..., 0] + 16
    return y

def read_img(fp):
    bgr = cv2.imread(fp, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(fp)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb

def avg_metrics(ref_dir, cmp_dir):
    ref_files = natsorted(glob(os.path.join(ref_dir, "*.png")) +
                          glob(os.path.join(ref_dir, "*.jpg")))
    cmp_files = natsorted(glob(os.path.join(cmp_dir, "*.png")) +
                          glob(os.path.join(cmp_dir, "*.jpg")))
    assert len(ref_files) == len(cmp_files), "ref/cmp 개수 불일치"

    psnr_list, ssim_list = [], []
    for rf, cf in zip(ref_files, cmp_files):
        ref = rgb2y(read_img(rf))
        cmp = rgb2y(read_img(cf))
        if ref.shape != cmp.shape:
            cmp = cv2.resize(cmp, (ref.shape[1], ref.shape[0]), interpolation=cv2.INTER_LINEAR)

        psnr_list.append(peak_signal_noise_ratio(ref, cmp, data_range=255))
        ssim_list.append(structural_similarity(ref, cmp, data_range=255))

    return np.mean(psnr_list), np.mean(ssim_list)

def main():
    root = "E:/Restormer/Deraining"
    sets = ["rain100H", "rain100L"]

    for d in sets:
        gt   = f"{root}/Datasets/{d}/test/norain"
        inp  = f"{root}/Datasets/{d}/test/rain"
        out  = f"{root}/results/{d}"

        psnr_in, ssim_in = avg_metrics(gt, inp)
        psnr_out, ssim_out = avg_metrics(gt, out)

        print(f"\n[{d.upper()}]  입력→GT  PSNR {psnr_in:.2f}  SSIM {ssim_in:.4f}")
        print(f"[{d.upper()}]  복원→GT PSNR {psnr_out:.2f}  SSIM {ssim_out:.4f}")

if __name__ == "__main__":
    main()
