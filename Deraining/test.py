#!/usr/bin/env python
# test_derain.py
# Restormer: Efficient Transformer for High-Resolution Image Restoration
# Paper: https://arxiv.org/abs/2111.09881

import os, sys, argparse, yaml
from glob import glob
from natsort import natsorted
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage import img_as_ubyte

# ---------------------------------------------------------------------
# 경로 설정: 프로젝트 루트(<script>/..)를 PYTHONPATH에 추가
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.append(repo_root)

# Restormer 아키텍처 & utils
from basicsr.models.archs.restormer_arch import Restormer
import utils  # utils.load_img / utils.save_img
# ---------------------------------------------------------------------


def load_checkpoint(model: nn.Module, ckpt_path: str, strict: bool = True) -> None:
    """다양한 key 이름을 모두 지원해 checkpoint 로딩."""
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict):
        state = ckpt.get("params") or ckpt.get("state_dict") or ckpt.get("model") or ckpt
    else:
        state = ckpt
    model.load_state_dict(state, strict=strict)


def parse_args():
    p = argparse.ArgumentParser(description="Restormer Deraining Inference")
    p.add_argument("--input_dir", type=str, default="E:/Restormer/Deraining/Datasets",
                help="Root folder that contains each dataset sub-folder")
    p.add_argument("--result_dir", type=str, default="./results",
                   help="Folder to save restoration results")
    p.add_argument("--weights", type=str, default="E:/Restormer/Deraining/pretrained_models/deraining.pth",

                   help="Path to pretrained Restormer weights")
    p.add_argument("--yaml",       type=str, default=None,
                   help="Network config YAML; if omitted, <script>/Options/Deraining_Restormer.yml 사용")
    p.add_argument("--gpu_id",     type=str, default="0",
                   help="CUDA_VISIBLE_DEVICES setting, e.g. 0 or 0,1; set -1 for CPU")
    p.add_argument("--strict",     action="store_true",
                   help="Strict checkpoint loading (default False for 편의성)")
    return p.parse_args()


def auto_discover_datasets(root: str):
    """root 하위 폴더 중 이미지가 들어있는 폴더명을 모두 반환 (대소문자 구분 無)."""
    dirs = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
    return sorted(dirs, key=str.lower)


def find_input_folder(base_dir: str):
    """
    base_dir 한 곳에서 흔히 쓰는 5가지 패턴을 순서대로 탐색:
    """
    cand = [
        os.path.join(base_dir, "test", "rain"),     # ✅ 최우선 탐색 경로
        os.path.join(base_dir, "test", "input"),
        os.path.join(base_dir, "test"),
        os.path.join(base_dir, "input", "test"),
        os.path.join(base_dir, "input"),
    ]
    return next((d for d in cand if os.path.isdir(d)), None)



def main():
    args = parse_args()

    # ------------- GPU / CPU 선택 -------------
    if args.gpu_id != "-1":
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    device = torch.device("cuda" if torch.cuda.is_available() and args.gpu_id != "-1" else "cpu")
    if device.type == "cpu":
        print("[INFO] CUDA 사용 불가 또는 --gpu_id -1 지정 → CPU 모드로 실행합니다.")

    # ------------- YAML 로드 -------------------
    yaml_path = (args.yaml if args.yaml
                 else os.path.join(os.path.dirname(__file__), "Options", "Deraining_Restormer.yml"))
    if not os.path.isfile(yaml_path):
        raise FileNotFoundError(f"YAML 파일을 찾지 못했습니다: {yaml_path}")
    with open(yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    _ = cfg["network_g"].pop("type")  # 'type' 제거
    model = Restormer(**cfg["network_g"])

    # ------------- 체크포인트 로드 -------------
    if not os.path.isfile(args.weights):
        raise FileNotFoundError(f"weights가 없습니다: {args.weights}")
    load_checkpoint(model, args.weights, strict=args.strict)
    print(f"[INFO] weights 로드 완료: {args.weights}")

    model = model.to(device)
    if torch.cuda.device_count() > 1 and device.type == "cuda":
        model = nn.DataParallel(model)
    model.eval()

    # ------------- 데이터셋 목록 자동 추출 ------
    datasets = auto_discover_datasets(args.input_dir)
    if not datasets:
        raise RuntimeError(f"--input_dir `{args.input_dir}` 안에 유효한 폴더가 없습니다.")
    print("[INFO] 탐색된 데이터셋:", ", ".join(datasets))

    # ------------- 추론 루프 -------------------
    factor = 8  # for 8-pixel padding
    for dname in datasets:
        base_dir = os.path.join(args.input_dir, dname)
        inp_dir  = find_input_folder(base_dir)
        if inp_dir is None:
            print(f"[WARN] {dname}: 입력 폴더 패턴을 찾지 못해 건너뜀")
            continue

        files = natsorted(glob(os.path.join(inp_dir, "*.png")) +
                          glob(os.path.join(inp_dir, "*.jpg")))
        if not files:
            print(f"[WARN] {dname}: 이미지 파일이 없습니다 → 건너뜀")
            continue

        out_dir = os.path.join(args.result_dir, dname)
        os.makedirs(out_dir, exist_ok=True)

        with torch.no_grad():
            for fp in tqdm(files, desc=f"Deraining {dname}", leave=False):
                img = np.float32(utils.load_img(fp)) / 255.0
                img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)

                # pad to multiple of 8
                h, w = img.shape[2:]
                H = (h + factor) // factor * factor
                W = (w + factor) // factor * factor
                img_pad = F.pad(img, (0, W - w, 0, H - h), mode="reflect")

                # inference
                restored = model(img_pad)

                # unpad & clamp
                restored = restored[..., :h, :w]
                restored = torch.clamp(restored, 0, 1).cpu().squeeze(0).permute(1, 2, 0).numpy()

                # save
                fname = os.path.splitext(os.path.basename(fp))[0] + ".png"
                utils.save_img(os.path.join(out_dir, fname), img_as_ubyte(restored))

        print(f"[INFO] {dname}: {len(files)}장 복원 완료 → {out_dir}")

    print("\n✅ 모든 데이터셋 처리 완료. 결과는:", os.path.abspath(args.result_dir))


if __name__ == "__main__":
    main()
