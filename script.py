#python image_project.py --mode build --raw_dir raw/ --target_dir goal/ --params_path params.pkl

#python image_project.py --mode restore --image path/to/underwater.png --params_path params.pkl

import os
import argparse
import pickle
import numpy as np
import cv2
from tqdm import tqdm

def dark_channel(img, patch_size=15):
    min_channel = np.min(img, axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch_size, patch_size))
    dark = cv2.erode(min_channel, kernel)
    return dark

def estimate_atmospheric_light(img, dark, top_percent=0.001):
    h, w = dark.shape
    npx = int(max(h * w * top_percent, 1))
    dark_vec = dark.reshape(-1)
    img_vec = img.reshape(-1, 3)
    idx = np.argsort(dark_vec)[-npx:]
    A = np.max(img_vec[idx], axis=0)
    return A

def estimate_transmission(img, A, patch_size=15, omega=0.95):
    norm = img.astype(np.float64) / A
    norm = np.clip(norm, 0, 1)
    dark = dark_channel((norm * 255).astype(np.uint8), patch_size)
    t = 1 - omega * (dark.astype(np.float64) / 255)
    return np.clip(t, 0.01, 1)

def recover_scene(img, t, A):
    J = np.empty_like(img, dtype=np.float64)
    for c in range(3):
        J[:, :, c] = (img[:, :, c].astype(np.float64) - A[c]) / t + A[c]
    return np.clip(J, 0, 255).astype(np.uint8)

def compute_pairwise_lut(raw_dir, target_dir):
    sum_map = np.zeros((3, 256), dtype=np.float64)
    count_map = np.zeros((3, 256), dtype=np.int64)
    files = sorted(os.listdir(raw_dir))
    assert files == sorted(os.listdir(target_dir)), \
        "raw_dir and target_dir must have same filenames"
    for fname in tqdm(files, desc='Compute LUT'):
        raw_bgr = cv2.imread(os.path.join(raw_dir, fname))
        tgt_bgr = cv2.imread(os.path.join(target_dir, fname))
        if raw_bgr is None or tgt_bgr is None:
            continue
        raw = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)
        tgt = cv2.cvtColor(tgt_bgr, cv2.COLOR_BGR2RGB)
        for ch in range(3):
            raw_ch = raw[..., ch].ravel()
            tgt_ch = tgt[..., ch].ravel().astype(np.float64)
            sums = np.bincount(raw_ch, weights=tgt_ch, minlength=256)
            cnts = np.bincount(raw_ch, minlength=256)
            sum_map[ch] += sums
            count_map[ch] += cnts
    lut = np.zeros((3, 256), dtype=np.uint8)
    x = np.arange(256)
    for ch in range(3):
        avg = np.zeros(256, dtype=np.float64)
        nonzero = count_map[ch] > 0
        avg[nonzero] = sum_map[ch][nonzero] / count_map[ch][nonzero]
        avg = np.interp(x, x[nonzero], avg[nonzero])
        lut[ch] = np.clip(np.round(avg), 0, 255).astype(np.uint8)
    return lut

def compute_channel_stats(dir_path):
    sum_c = np.zeros(3, dtype=np.float64)
    sum_sq_c = np.zeros(3, dtype=np.float64)
    count = 0
    for fname in tqdm(sorted(os.listdir(dir_path)), desc='Compute Stats'):
        img = cv2.imread(os.path.join(dir_path, fname))
        if img is None: continue
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float64)
        mean = rgb.mean(axis=(0, 1))
        sum_c += mean
        sum_sq_c += (rgb**2).mean(axis=(0, 1))
        count += 1
    mean_c = sum_c / count
    var_c = sum_sq_c / count - mean_c**2
    std_c = np.sqrt(np.maximum(var_c, 1e-6))
    return mean_c, std_c

def unsharp_mask(img, kernel_size=(9, 9), sigma=1.0, amount=1.0):
    blurred = cv2.GaussianBlur(img, kernel_size, sigma)
    return cv2.addWeighted(img, 1 + amount, blurred, -amount, 0)

def restore_single_image(image_path, params, dehaze=False, sharp_amount=1.0):
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot read {image_path}")
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    #Dehazing
    if dehaze:
        dark = dark_channel(rgb)
        A = estimate_atmospheric_light(rgb, dark)
        t = estimate_transmission(rgb, A)
        rgb = recover_scene(rgb, t, A)
    #Dataset normalization
    raw_mean, raw_std = params['raw_mean'], params['raw_std']
    goal_mean, goal_std = params['goal_mean'], params['goal_std']
    norm = (rgb.astype(np.float64) - raw_mean) * (goal_std / raw_std) + goal_mean
    norm = np.clip(norm, 0, 255).astype(np.uint8)
    #LUT remapping
    lut = params['lut']
    remap = np.zeros_like(norm)
    for ch in range(3): remap[..., ch] = lut[ch][norm[..., ch]]
    #CLAHE on L channel
    lab = cv2.cvtColor(remap, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_eq = clahe.apply(l)
    rgb_eq = cv2.cvtColor(cv2.merge([l_eq, a, b]), cv2.COLOR_LAB2RGB)
    #Mild bilateral filter
    bf = np.empty_like(rgb_eq)
    for ch in range(3):
        bf[..., ch] = cv2.bilateralFilter(rgb_eq[..., ch], d=5, sigmaColor=50, sigmaSpace=50)
    #Unsharp mask for sharpening
    sharp = unsharp_mask(bf, amount=sharp_amount)
    #Save result
    out_bgr = cv2.cvtColor(sharp, cv2.COLOR_RGB2BGR)
    base, ext = os.path.splitext(image_path)
    out_path = f"{base}_restored{ext}"
    cv2.imwrite(out_path, out_bgr)
    print(f"Saved restored image to {out_path}")

def main():
    p = argparse.ArgumentParser(description='Single-Image Underwater Restoration')
    p.add_argument('--mode', choices=['build', 'restore'], required=True)
    p.add_argument('--raw_dir')
    p.add_argument('--target_dir')
    p.add_argument('--params_path', required=True)
    p.add_argument('--image')
    p.add_argument('--dehaze', action='store_true')
    p.add_argument('--sharp_amount', type=float, default=1.0)
    args = p.parse_args()

    if args.mode == 'build':
        assert args.raw_dir and args.target_dir, 'raw_dir and target_dir required for build'
        lut = compute_pairwise_lut(args.raw_dir, args.target_dir)
        raw_mean, raw_std = compute_channel_stats(args.raw_dir)
        goal_mean, goal_std = compute_channel_stats(args.target_dir)
        params = {
            'lut': lut,
            'raw_mean': raw_mean,
            'raw_std': raw_std,
            'goal_mean': goal_mean,
            'goal_std': goal_std
        }
        with open(args.params_path, 'wb') as f:
            pickle.dump(params, f)
        print(f"Parameters saved to {args.params_path}")
    else:
        assert args.image, 'image path required for restore'
        with open(args.params_path, 'rb') as f:
            params = pickle.load(f)
        restore_single_image(args.image, params, dehaze=args.dehaze,
                             sharp_amount=args.sharp_amount)

if __name__ == '__main__':
    main()
