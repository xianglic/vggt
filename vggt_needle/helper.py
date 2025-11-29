# helper.py
import math
import torch
from needle import Tensor, cuda, cpu


def _reshape_with_leading_singletons(torch_tensor, expected_shape):
    """
    Try to reshape torch_tensor to expected_shape by *only* adding leading
    singleton dimensions (1, ...), e.g. (3072,) -> (1, 3072) or (1,1,3072).

    Returns a reshaped tensor if possible, else None.
    """
    t_shape = tuple(torch_tensor.shape)
    e_shape = tuple(expected_shape)

    # Same shape: trivial
    if t_shape == e_shape:
        return torch_tensor

    # Different number of elements: impossible
    if torch_tensor.numel() != math.prod(e_shape):
        return None

    # Only handle the case where expected has more dims than torch
    if len(e_shape) > len(t_shape):
        k = len(e_shape) - len(t_shape)
        prefix = e_shape[:k]
        suffix = e_shape[k:]

        # We only allow extra leading dims that are all 1
        if all(d == 1 for d in prefix) and suffix == t_shape:
            return torch_tensor.reshape(e_shape)

    # Otherwise, don't try to be clever
    return None


def sd_torch2needle(model, torch_sd: dict):
    """
    In-place: convert all torch.Tensor in torch_sd into needle.Tensor
    with shapes compatible with model.state_dict().

    - Exact same shape is accepted.
    - If Needle expects leading singleton dims (e.g. (1, D) vs (D,)),
      we reshape the torch tensor accordingly.
    """
    dev = cuda() if cuda().enabled() else cpu()
    needle_sd = model.state_dict()

    for name, torch_tensor in list(torch_sd.items()):
        if not isinstance(torch_tensor, torch.Tensor):
            # skip non-tensor entries (e.g. metadata)
            continue

        if name not in needle_sd:
            print(f"⚠️  Skipping unknown key in torch_sd: {name}")
            continue

        expected_shape = tuple(needle_sd[name].shape)

        # Try direct match or leading-singleton reshape
        reshaped = _reshape_with_leading_singletons(torch_tensor, expected_shape)
        if reshaped is None:
            print(
                f"❌ Shape mismatch for {name}: "
                f"torch {tuple(torch_tensor.shape)} vs needle {expected_shape}"
            )
            raise ValueError(f"Weight shape mismatch: {name}")

        # Convert torch → numpy → Needle Tensor
        arr = reshaped.detach().cpu().float().numpy()
        torch_sd[name] = Tensor(arr, device=dev)

    return torch_sd
