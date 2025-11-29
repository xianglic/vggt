#!/usr/bin/env python3
import os
import glob
import argparse
import numpy as np
import torch

# -----------------------------------------------------------------------------
# VGGT imports (assumes you're running from the repo root OR vggt is on PYTHONPATH)
# -----------------------------------------------------------------------------
from vggt_needle.models.vggt import VGGT as VGGT_needle
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt_needle.helper import sd_torch2needle
import needle


# -----------------------------------------------------------------------------
# Model loading
# -----------------------------------------------------------------------------
def load_model(device: str) -> VGGT_needle:
    """
    Load the VGGT_needle model and initialize it with the original PyTorch state dict.
    `device` is a torch/needle device string, e.g., "cuda".
    """
    _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
    vggt_sd = torch.hub.load_state_dict_from_url(
        _URL, map_location="cpu", model_dir="/data/hf_cache/"
    )
    model = VGGT_needle()
    # Convert torch state dict to Needle-compatible weights
    sd_torch2needle(model, vggt_sd)
    # Load any remaining weights in a non-strict fashion
    model.load_state_dict(vggt_sd, strict=False)
    model.eval()
    model = model.to(device)
    return model


# -----------------------------------------------------------------------------
# Rotation matrix (3x3) -> quaternion [qw, qx, qy, qz], batched
# -----------------------------------------------------------------------------
def rotmat_to_quat_batch(R: np.ndarray) -> np.ndarray:
    """
    Convert a batch of 3x3 rotation matrices to quaternions.

    R: (N, 3, 3)
    Returns: (N, 4) quaternions in [qw, qx, qy, qz] format
    """
    assert R.ndim == 3 and R.shape[1:] == (3, 3)
    N = R.shape[0]
    q = np.empty((N, 4), dtype=np.float64)

    for i in range(N):
        m = R[i]
        trace = m[0, 0] + m[1, 1] + m[2, 2]

        if trace > 0.0:
            S = np.sqrt(trace + 1.0) * 2.0  # S = 4 * qw
            qw = 0.25 * S
            qx = (m[2, 1] - m[1, 2]) / S
            qy = (m[0, 2] - m[2, 0]) / S
            qz = (m[1, 0] - m[0, 1]) / S
        elif (m[0, 0] > m[1, 1]) and (m[0, 0] > m[2, 2]):
            S = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
            qw = (m[2, 1] - m[1, 2]) / S
            qx = 0.25 * S
            qy = (m[0, 1] + m[1, 0]) / S
            qz = (m[0, 2] + m[2, 0]) / S
        elif m[1, 1] > m[2, 2]:
            S = np.sqrt(1.0 - m[0, 0] + m[1, 1] - m[2, 2]) * 2.0
            qw = (m[0, 2] - m[2, 0]) / S
            qx = (m[0, 1] + m[1, 0]) / S
            qy = 0.25 * S
            qz = (m[1, 2] + m[2, 1]) / S
        else:
            S = np.sqrt(1.0 - m[0, 0] - m[1, 1] + m[2, 2]) * 2.0
            qw = (m[1, 0] - m[0, 1]) / S
            qx = (m[0, 2] + m[2, 0]) / S
            qy = (m[1, 2] + m[2, 1]) / S
            qz = 0.25 * S

        q[i] = [qw, qx, qy, qz]

    return q.astype(np.float32)


# -----------------------------------------------------------------------------
# Compute camera pose as (x, y, z, qw, qx, qy, qz)
# -----------------------------------------------------------------------------
def compute_camera_pose_quat(extrinsic: np.ndarray) -> np.ndarray:
    """
    extrinsic: (N, 3, 4) or (N, 4, 4) world->camera matrices of the form [R | t].

    We want camera *position* and *orientation* in the WORLD frame.

    world -> cam:  x_cam = R_cam * x_world + t_cam
    => R_world = R_cam^T
       C = -R_world * t_cam

    Returns:
        poses: (N, 7) with columns [x, y, z, qw, qx, qy, qz]
    """
    if extrinsic.ndim != 3 or extrinsic.shape[1] not in (3, 4):
        raise RuntimeError(
            f"Unexpected extrinsic shape in pose computation: {extrinsic.shape}"
        )

    # Handle both (N,3,4) and (N,4,4) by always taking the top 3 rows
    R_cam = extrinsic[:, :3, :3]  # (N, 3, 3)
    t_cam = extrinsic[:, :3, 3]   # (N, 3)

    # Invert rotation: R_world = R_cam^T
    R_world = np.transpose(R_cam, (0, 2, 1))  # (N, 3, 3)

    # Camera center in world coords: C = -R_world @ t_cam
    C = -np.einsum("nij,nj->ni", R_world, t_cam)  # (N, 3)

    # Quaternions for R_world
    q = rotmat_to_quat_batch(R_world)  # (N, 4) [qw, qx, qy, qz]

    poses = np.concatenate([C, q], axis=-1)  # (N, 7)
    return poses


# -----------------------------------------------------------------------------
# Main pipeline: images -> VGGT (Needle) -> pose_enc (torch) -> poses -> CSV
# -----------------------------------------------------------------------------
def main(images_dir: str, output_csv: str, max_images: int | None):
    # --- Device selection (torch-side utilities) ---
    if not torch.cuda.is_available():
        raise ValueError("CUDA is not available. VGGT_needle currently expects CUDA.")

    device = "cuda"
    print(f"[Device] Using {device}")

    # Load Needle model (weights from original torch checkpoint)
    model = load_model(device)

    # Collect images
    exts = ["*.png", "*.jpg", "*.jpeg", "*.bmp"]
    image_paths = []
    for ext in exts:
        image_paths.extend(glob.glob(os.path.join(images_dir, ext)))
    image_paths = sorted(image_paths)

    if len(image_paths) == 0:
        raise ValueError(f"No images found in {images_dir} (looked for {exts})")

    print(f"[Data] Found {len(image_paths)} images in: {images_dir}")

    # --- Optional subsampling to avoid OOM ---
    if max_images is not None and len(image_paths) > max_images:
        step = len(image_paths) / float(max_images)
        indices = [int(i * step) for i in range(max_images)]
        image_paths = [image_paths[i] for i in indices]
        print(
            f"[Data] Subsampled to {len(image_paths)} images (max_images={max_images}) "
            f"to reduce GPU memory usage."
        )
    else:
        print(f"[Data] Using all {len(image_paths)} images (no subsampling).")

    # >>> PRINT SELECTED IMAGES <<<
    print("[Data] Images selected for processing:")
    for p in image_paths:
        print("   -", p)
    print("-" * 60)

    # --- Load & preprocess images with the original torch loader ---
    images_torch = load_and_preprocess_images(image_paths).to(device)
    print(f"[Data] Preprocessed image tensor shape (torch): {images_torch.shape}")

    # --- Convert torch images -> Needle Tensor ---
    # VGGT_needle expects shape [S, 3, H, W] or [B, S, 3, H, W], same as torch.
    images_np = images_torch.detach().cpu().numpy().astype("float32")
    images_needle = needle.Tensor(images_np).to(device=device)

    # --- Run inference with Needle model ---
    print("[VGGT] Running inference with VGGT_needle...")
    with torch.no_grad():  # harmless for Needle; keeps torch ops out of grad mode
        predictions_needle = model(images_needle)

    # --- Extract pose_enc and convert Needle -> torch for downstream utilities ---
    if "pose_enc" not in predictions_needle:
        raise KeyError("Model predictions do not contain 'pose_enc' key.")

    pose_enc_needle = predictions_needle["pose_enc"]

    if isinstance(pose_enc_needle, needle.Tensor):
        nd = pose_enc_needle.realize_cached_data()  # NDArray
        pose_enc_np = nd.numpy()                    # numpy array
        pose_enc_torch = torch.from_numpy(pose_enc_np).to(device)
    else:
        # In case the model already returns a torch.Tensor
        pose_enc_torch = pose_enc_needle

    # --- Convert pose encoding to extrinsic and intrinsic matrices (torch utils) ---
    print("[VGGT] Converting pose encoding to extrinsic and intrinsic matrices...")
    extrinsic, _ = pose_encoding_to_extri_intri(
        pose_enc_torch, images_torch.shape[-2:]
    )

    # --- To numpy, drop batch dimension if present: e.g. (1, N, 3, 4) -> (N, 3, 4) ---
    extrinsic_np = extrinsic.detach().cpu().numpy()
    if extrinsic_np.ndim == 4:
        # assume (1, N, 3, 4) or (1, N, 4, 4)
        extrinsic_np = np.squeeze(extrinsic_np, axis=0)

    print(f"[VGGT] Extrinsic shape: {extrinsic_np.shape}")

    if extrinsic_np.ndim != 3 or extrinsic_np.shape[1] not in (3, 4):
        raise RuntimeError(f"Unexpected extrinsic shape: {extrinsic_np.shape}")

    # --- Compute camera poses (x, y, z, qw, qx, qy, qz) ---
    poses = compute_camera_pose_quat(extrinsic_np)  # (N, 7)
    assert poses.shape[0] == len(image_paths)

    # --- Save CSV ---
    header = "x,y,z,qw,qx,qy,qz"
    np.savetxt(
        output_csv,
        poses,
        delimiter=",",
        header=header,
        comments="",
    )

    print(f"[Output] Saved {poses.shape[0]} poses to: {output_csv}")
    print(f"[Output] CSV header: {header}")
    print("[Note] Row i corresponds to sorted(image_paths)[i].")

    # Clean up CUDA memory
    torch.cuda.empty_cache()


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run VGGT_needle on a folder of images and export camera poses as (x,y,z,qw,qx,qy,qz)."
    )
    parser.add_argument(
        "--images",
        type=str,
        required=True,
        help="Folder containing input images (png/jpg/jpeg/bmp).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="camera_poses_xyz_quat.csv",
        help="Output CSV file path.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=24,
        help="Max number of images to use (subsampled if more are present) to avoid GPU OOM.",
    )

    args = parser.parse_args()

    main(args.images, args.out, args.max_images)
