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
    """RGB → Y 채널 추출 (float32, 0~255 기준)"""
    img_rgb = img_rgb.astype(np.float32)
    y = 0.257 * img_rgb[..., 2] + 0.504 * img_rgb[..., 1] + 0.098 * img_rgb[..., 0] + 16
    return y

def read_img(fp):
    bgr = cv2.imread(fp, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"이미지를 불러올 수 없습니다: {fp}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb

def avg_metrics(ref_dir, cmp_dir):
    ref_files = natsorted(glob(os.path.join(ref_dir, "*.png")) + glob(os.path.join(ref_dir, "*.jpg")))
    cmp_files = natsorted(glob(os.path.join(cmp_dir, "*.png")) + glob(os.path.join(cmp_dir, "*.jpg")))

    # 🔍 파일명 기준 일치 여부 확인
    ref_names = sorted([os.path.basename(f) for f in ref_files])
    cmp_names = sorted([os.path.basename(f) for f in cmp_files])

    if ref_names != cmp_names:
        print(f"\n❌ 파일 수 또는 이름이 일치하지 않습니다!")
        print(f"GT ({len(ref_names)}개): {ref_names[:5]} ...")
        print(f"CMP({len(cmp_names)}개): {cmp_names[:5]} ...")
        missing_in_cmp = set(ref_names) - set(cmp_names)
        missing_in_ref = set(cmp_names) - set(ref_names)
        if missing_in_cmp:
            print("⚠️ 복원 결과에 빠진 파일:", missing_in_cmp)
        if missing_in_ref:
            print("⚠️ GT에 빠진 파일:", missing_in_ref)
        raise AssertionError("ref/cmp 개수 또는 이름 불일치")

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

        print(f"\n🔍 [{d.upper()}] 평가 시작:")

        try:
            psnr_in, ssim_in = avg_metrics(gt, inp)
            print(f"📥 입력 → GT  PSNR: {psnr_in:.2f} dB  SSIM: {ssim_in:.4f}")
        except Exception as e:
            print(f"⚠️ 입력 평가 중 오류 발생: {e}")

        try:
            psnr_out, ssim_out = avg_metrics(gt, out)
            print(f"📤 복원 → GT PSNR: {psnr_out:.2f} dB  SSIM: {ssim_out:.4f}")
        except Exception as e:
            print(f"⚠️ 복원 평가 중 오류 발생: {e}")

if __name__ == "__main__":
    main()
