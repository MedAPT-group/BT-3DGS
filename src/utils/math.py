import torch
from torch import Tensor


def invert_rigid_transform(transform: Tensor) -> Tensor:
    """Invert homogeneous rigid transforms without a general matrix inverse.

    ``transform`` may contain any number of leading batch dimensions and must
    end in ``(4, 4)``.  Camera poses in this project are rigid transforms, so
    their inverse is ``[R^T | -R^T t]``.  Besides being cheaper than
    ``torch.linalg.inv``, this avoids launching a generic linear-algebra
    routine for every small camera batch.
    """
    if transform.shape[-2:] != (4, 4):
        raise ValueError(
            f"Expected transform shape [..., 4, 4], got {tuple(transform.shape)}"
        )

    rotation_t = transform[..., :3, :3].transpose(-1, -2)
    translation = transform[..., :3, 3]
    inverse_translation = -torch.matmul(
        rotation_t, translation.unsqueeze(-1)
    ).squeeze(-1)

    top = torch.cat((rotation_t, inverse_translation.unsqueeze(-1)), dim=-1)
    bottom = torch.zeros(
        transform.shape[:-2] + (1, 4),
        dtype=transform.dtype,
        device=transform.device,
    )
    bottom[..., 0, 3] = 1
    return torch.cat((top, bottom), dim=-2).contiguous()


def normalized_quat_to_rotmat(quat: Tensor) -> Tensor:
    """Convert normalized quaternion in wxyz to rotation matrix."""
    assert quat.shape[-1] == 4, quat.shape
    w, x, y, z = torch.unbind(quat, dim=-1)
    mat = torch.stack(
        [
            1 - 2 * (y**2 + z**2),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
            2 * (x * y + w * z),
            1 - 2 * (x**2 + z**2),
            2 * (y * z - w * x),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x**2 + y**2),
        ],
        dim=-1,
    )
    return mat.reshape(quat.shape[:-1] + (3, 3))
