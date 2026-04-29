import torch


def load_checkpoint(path, map_location=None):
    """Load project checkpoints across PyTorch versions.

    PyTorch 2.6 changed torch.load's default to weights_only=True. GaussianAvatars
    checkpoints store a tuple, so request the legacy behavior explicitly.
    """
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)
