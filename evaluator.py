"""Image-quality evaluation and feature-weight search helpers."""

import os
from functools import lru_cache
from itertools import product
from typing import Dict, Tuple

import lpips
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torchvision import transforms

from mosaic_pipeline import (
    extract_features_from_folder,
    extract_tile_features,
    match_tiles_to_gallery_faiss,
    reconstruct_mosaic_image,
)


EVALUATION_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
    ]
)


@lru_cache(maxsize=3)
def get_lpips_model(net_type: str) -> torch.nn.Module:
    """Cache one LPIPS model per supported backbone."""

    return lpips.LPIPS(net=net_type).eval()

# 1. Compute PSNR
def compute_psnr(img1_path: str, img2_path: str) -> float:
    img1 = Image.open(img1_path).convert("RGB")
    img2 = Image.open(img2_path).convert("RGB")
    img1 = img1.resize(img2.size)
    arr1 = np.array(img1)
    arr2 = np.array(img2)
    return peak_signal_noise_ratio(arr1, arr2, data_range=255)

# 2. Compute SSIM
def compute_ssim(img1_path: str, img2_path: str) -> float:
    img1 = Image.open(img1_path).convert("RGB")
    img2 = Image.open(img2_path).convert("RGB")
    img1 = img1.resize(img2.size)
    arr1 = np.array(img1)
    arr2 = np.array(img2)
    return structural_similarity(arr1, arr2, channel_axis=-1)

# 3. Compute LPIPS
def compute_lpips(img1_path: str, img2_path: str, net_type: str = 'alex') -> float:
    loss_fn = get_lpips_model(net_type)
    img1 = EVALUATION_TRANSFORM(Image.open(img1_path).convert("RGB")).unsqueeze(0)
    img2 = EVALUATION_TRANSFORM(Image.open(img2_path).convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        score = loss_fn(img1, img2).item()
    return score

# 4. Unified evaluation function
def evaluate_all(img1_path: str, img2_path: str) -> dict:
    return {
        "PSNR": compute_psnr(img1_path, img2_path),
        "SSIM": compute_ssim(img1_path, img2_path),
        "LPIPS": compute_lpips(img1_path, img2_path)
    }

# 5. Print evaluation result
def print_evaluation_report(original_path: str, mosaic_path: str) -> None:
    results = evaluate_all(original_path, mosaic_path)
    print("[Evaluation Report]")
    print(f"PSNR  : {results['PSNR']:.2f}")
    print(f"SSIM  : {results['SSIM']:.4f}")
    print(f"LPIPS : {results['LPIPS']:.4f} (lower is better)")

# Evaluate a single weight combination
def evaluate_weights(weights: Dict[str, float], 
                     image_path: str,
                     gallery_image_folder: str,
                     tile_size: int,
                     output_dir: str,
                     run_name: str) -> Tuple[float, Dict[str, float]]:
    os.makedirs(output_dir, exist_ok=True)

    # Normalize weights
    total = sum(weights.values())
    if total == 0:
        return -1.0, weights
    weights = {k: v / total for k, v in weights.items()}

    gallery_feat_path = os.path.join(output_dir, f"{run_name}_gallery_features.pkl")
    tile_feat_path = os.path.join(output_dir, f"{run_name}_tile_features.pkl")
    match_path = os.path.join(output_dir, f"{run_name}_match.pkl")
    mosaic_path = os.path.join(output_dir, f"{run_name}.jpg")

    # Step 1: Extract gallery features (using current weights)
    extract_features_from_folder(
        folder_path=gallery_image_folder,
        method="combined",
        output_path=gallery_feat_path,
        **weights
    )

    # Step 2: Extract tile features (using current weights)
    extract_tile_features(
        image_path=image_path,
        tile_size=tile_size,
        method="combined",
        output_path=tile_feat_path,
        **weights
    )

    # Step 3: Match tiles using FAISS
    match_tiles_to_gallery_faiss(tile_feat_path, gallery_feat_path, match_path)

    # Step 4: Reconstruct and save the mosaic image
    reconstruct_mosaic_image(match_path, gallery_folder=gallery_image_folder, 
                             tile_size=tile_size, output_image_path=mosaic_path)

    # Step 5: Evaluate using PSNR, SSIM, and LPIPS
    metrics = evaluate_all(image_path, mosaic_path)
    score = metrics["SSIM"]  # Use SSIM as the optimization criterion

    # Save metrics to text file
    metrics_path = os.path.join(output_dir, f"{run_name}_metrics.txt")
    with open(metrics_path, "w") as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v:.4f}\n")

    return score, weights

# Perform grid search over weight combinations
def grid_search_weights(image_path: str,
                        gallery_image_folder: str,
                        tile_size: int,
                        output_dir: str,
                        search_space: Dict[str, list]) -> Tuple[float, Dict[str, float] | None]:

    os.makedirs(output_dir, exist_ok=True)
    keys = list(search_space.keys())
    grid = list(product(*[search_space[k] for k in keys]))

    best_score = -1.0
    best_weights = None

    for i, combo in enumerate(tqdm(grid, desc="Tuning combinations")):
        weights = dict(zip(keys, combo))
        run_id = "_".join(f"{k}{v:.1f}" for k, v in weights.items())
        run_name = f"mosaic_{i:03d}_{run_id}"

        try:
            score, normalized_weights = evaluate_weights(
                weights=weights,
                image_path=image_path,
                gallery_image_folder=gallery_image_folder,
                tile_size=tile_size,
                output_dir=output_dir,
                run_name=run_name
            )

            print(f"[{i+1}/{len(grid)}] SSIM: {score:.4f} | Weights: {normalized_weights}")
            if score > best_score:
                best_score = score
                best_weights = normalized_weights

        except Exception as e:
            print(f"Error in run {run_name}: {e}")

    print(f"\nBest SSIM: {best_score:.4f}")
    print(f"Best Weights: {best_weights}")
    return best_score, best_weights
