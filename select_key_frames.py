#!/usr/bin/env python3
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as T


def parse_args():
    p = argparse.ArgumentParser(
        description="Select diverse keyframes from a folder of JPG drone images."
    )
    p.add_argument("input_dir", type=str, help="Folder containing input .jpg images")
    p.add_argument("output_dir", type=str, help="Folder to save selected keyframes")
    p.add_argument(
        "-k", "--num_frames", type=int, default=70,
        help="Number of keyframes to select (default: 70)",
    )
    p.add_argument(
        "--segments", type=int, default=None,
        help="Number of temporal segments (default: 3 * K, capped by N)",
    )
    p.add_argument(
        "--blur_thresh", type=float, default=80.0,
        help="Minimum blur score to keep a frame (variance of Laplacian)",
    )
    return p.parse_args()


def list_images(input_dir: Path):
    # Only jpg/jpeg as you said, but easy to extend
    exts = (".jpg", ".jpeg", ".JPG", ".JPEG")
    paths = sorted(
        [p for p in input_dir.iterdir() if p.suffix in exts],
        key=lambda p: p.name,
    )
    return paths


def blur_score(gray_img: np.ndarray) -> float:
    return cv2.Laplacian(gray_img, cv2.CV_64F).var()


def build_resnet18(device):
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    # remove final classification layer, keep features
    model.fc = torch.nn.Identity()
    model.to(device)
    model.eval()
    return model


def get_transform():
    return T.Compose(
        [
            T.ToPILImage(),
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def compute_quality_scores(paths, blur_thresh):
    valid_indices = []
    blur_scores = {}

    print(f"[1/4] Quality filtering on {len(paths)} images...")
    for i, p in enumerate(paths):
        gray = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue  # unreadable image
        b = blur_score(gray)
        if b >= blur_thresh:
            valid_indices.append(i)
            blur_scores[i] = b

    print(f"  -> {len(valid_indices)} frames passed blur threshold {blur_thresh}")
    return valid_indices, blur_scores


def pick_temporal_candidates(N, valid_indices, blur_scores, K, segments):
    if segments is None:
        segments = 3 * K
    segments = min(segments, N)  # not more segments than frames

    valid_set = set(valid_indices)
    candidates = []

    print(f"[2/4] Temporal segmentation into {segments} segments...")
    for s in range(segments):
        start = int(s * N / segments)
        end = int((s + 1) * N / segments)
        best_i, best_b = None, -1.0

        for i in range(start, min(end, N)):
            if i in valid_set:
                b = blur_scores.get(i, -1.0)
                if b > best_b:
                    best_b = b
                    best_i = i
        if best_i is not None:
            candidates.append(best_i)

    candidates = sorted(set(candidates))
    print(f"  -> {len(candidates)} candidate frames after temporal selection")
    return candidates


def compute_embeddings(paths, candidates, device):
    print(f"[3/4] Computing embeddings for {len(candidates)} candidates...")
    model = build_resnet18(device)
    transform = get_transform()

    feats = {}
    with torch.no_grad():
        for idx in candidates:
            img_bgr = cv2.imread(str(paths[idx]))
            if img_bgr is None:
                continue
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img_t = transform(img_rgb).unsqueeze(0).to(device)
            f = model(img_t).squeeze(0)
            f = f / (f.norm() + 1e-8)
            feats[idx] = f.cpu()  # keep on CPU for distance calc

    print(f"  -> {len(feats)} embeddings computed")
    return feats


def farthest_point_sampling(candidates, feats, K):
    print(f"[4/4] Farthest-point sampling to select {K} keyframes...")

    # Filter candidates to those with embeddings
    candidates = [i for i in candidates if i in feats]
    if not candidates:
        print("  !! No candidates with embeddings; returning empty selection.")
        return []

    K = min(K, len(candidates))

    selected = []

    # Initialize with central candidate in time
    first = candidates[len(candidates) // 2]
    selected.append(first)

    # Pre-assemble feature tensor for speed (optional, but nice)
    # index -> position in tensor
    idx_to_pos = {idx: pos for pos, idx in enumerate(candidates)}
    feat_mat = torch.stack([feats[idx] for idx in candidates], dim=0)  # [C, D]

    # map selected indices to their positions in feat_mat
    selected_pos = [idx_to_pos[first]]

    while len(selected) < K:
        # compute distances to nearest selected for all candidates
        # feat_mat: [C, D], selected_feats: [S, D]
        selected_feats = feat_mat[selected_pos]  # [S, D]
        # expand to [C, S, D]; compute squared dist
        diff = feat_mat.unsqueeze(1) - selected_feats.unsqueeze(0)  # [C, S, D]
        dist_sq = (diff ** 2).sum(-1)  # [C, S]
        min_dist_sq, _ = dist_sq.min(dim=1)  # [C]

        # mask out already selected (set distance to -inf)
        for pos in selected_pos:
            min_dist_sq[pos] = -1.0

        best_pos = int(min_dist_sq.argmax().item())
        if best_pos in selected_pos:
            # All remaining are effectively equivalent
            break
        selected_pos.append(best_pos)
        selected.append(candidates[best_pos])

    selected = sorted(selected)
    print(f"  -> Selected {len(selected)} keyframes.")
    return selected


def copy_selected_frames(paths, selected_indices, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving selected frames to: {output_dir}")
    for rank, idx in enumerate(selected_indices):
        src = paths[idx]
        # prefix with rank for easy ordering
        dst = output_dir / f"{rank:04d}_{src.name}"
        # Use imread + imwrite for portability
        img = cv2.imread(str(src))
        if img is not None:
            cv2.imwrite(str(dst), img)


def main():
    args = parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.is_dir():
        raise SystemExit(f"Input directory does not exist: {input_dir}")

    paths = list_images(input_dir)
    if not paths:
        raise SystemExit(f"No .jpg images found in {input_dir}")

    N = len(paths)
    print(f"Found {N} images.")

    # Step 1: quality filter
    valid_indices, blur_scores = compute_quality_scores(paths, args.blur_thresh)

    if not valid_indices:
        raise SystemExit("No frames passed the blur threshold; try lowering --blur_thresh.")

    # Step 2: temporal segmentation -> candidates
    candidates = pick_temporal_candidates(
        N, valid_indices, blur_scores, args.num_frames, args.segments
    )
    if not candidates:
        raise SystemExit("No temporal candidates selected; check thresholds.")

    # Device selection (use GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Step 3: embeddings
    feats = compute_embeddings(paths, candidates, device)
    if not feats:
        raise SystemExit("Failed to compute any embeddings; check images / dependencies.")

    # Step 4: farthest-point sampling
    selected_indices = farthest_point_sampling(candidates, feats, args.num_frames)

    if not selected_indices:
        raise SystemExit("No keyframes selected; something went wrong in sampling.")

    # Save outputs
    copy_selected_frames(paths, selected_indices, output_dir)


if __name__ == "__main__":
    main()
