"""MNIST data loading, Poisson spike encoding, and fitness scoring for SNN training.

The training loop follows the same "generation -> fitness -> noiser update" pattern used
elsewhere in HyperscaleES; this module only supplies the data + reward, and is agnostic
to the underlying SNN model.
"""

import gzip
import os

import numpy as np
import jax
import jax.numpy as jnp

try:
    from datasets import load_dataset as _huggingface_load
    _HAS_DATASETS = True
except Exception:  # pragma: no cover - graceful fallback
    _HAS_DATASETS = False

_GLOBAL_MNIST = {}

# Local IDX-format MNIST data (preferred, offline-friendly).
#   images: magic(4) n(4) rows(4) cols(4) then n*rows*cols bytes
#   labels: magic(4) n(4) then n bytes
IDX_IMAGES = {
    "train": "train-images-idx3-ubyte.gz",
    "test": "t10k-images-idx3-ubyte.gz",
}
IDX_LABELS = {
    "train": "train-labels-idx1-ubyte.gz",
    "test": "t10k-labels-idx1-ubyte.gz",
}


def _read_idx(path):
    """Read an IDX-format file (possibly gzipped) at ``path`` into a numpy array."""
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rb") as f:
        magic = int.from_bytes(f.read(4), "big")
        n = int.from_bytes(f.read(4), "big")
        if magic == 0x803:  # images: magic, n, rows, cols
            rows = int.from_bytes(f.read(4), "big")
            cols = int.from_bytes(f.read(4), "big")
            return np.frombuffer(f.read(), dtype=np.uint8).reshape(n, rows * cols)
        elif magic == 0x801:  # labels: magic, n
            return np.frombuffer(f.read(), dtype=np.uint8)
        else:
            raise ValueError(f"Unknown IDX magic number 0x{magic:08x} in {path}")


def _load_mnist_from_dir(data_dir):
    """Load MNIST from a local directory containing the four IDX gz files."""
    imgs = _read_idx(os.path.join(data_dir, IDX_IMAGES["train"]))
    lbls = _read_idx(os.path.join(data_dir, IDX_LABELS["train"]))
    timgs = _read_idx(os.path.join(data_dir, IDX_IMAGES["test"]))
    tlbls = _read_idx(os.path.join(data_dir, IDX_LABELS["test"]))
    return (imgs.astype(np.float32) / 255.0, lbls.astype(np.int64)), \
           (timgs.astype(np.float32) / 255.0, tlbls.astype(np.int64))


def set_mnist_data_dir(data_dir):
    """Set a local MNIST directory as the data source for ``get_mnist_arrays``.

    The directory must contain train-images-idx3-ubyte.gz, train-labels-idx1-ubyte.gz,
    t10k-images-idx3-ubyte.gz and t10k-labels-idx1-ubyte.gz.
    """
    train, test = _load_mnist_from_dir(data_dir)
    _GLOBAL_MNIST["train"] = train
    _GLOBAL_MNIST["test"] = test


def get_mnist_arrays(split="train", data_dir=None):
    """Return (images, labels) numpy arrays for MNIST.

    images: (n, 784) float32 normalized to [0, 1].
    labels: (n,) int64.
    Data is cached in module state so repeated calls are free.

    If ``data_dir`` is given it is loaded from the local IDX-format files there;
    otherwise it falls back to ``set_mnist_data_dir()`` if that was called, else to
    HuggingFace `datasets` (requires network).
    """
    if data_dir is not None:
        set_mnist_data_dir(data_dir)
    if split not in _GLOBAL_MNIST:
        if not _HAS_DATASETS:  # pragma: no cover
            raise RuntimeError(
                "No local MNIST loaded and HuggingFace `datasets` is unavailable. "
                "Either call set_mnist_data_dir(<dir>) or install `datasets`."
            )
        ds = _huggingface_load("mnist", split=split)
        imgs = np.asarray(ds["image"], dtype=np.float32) / 255.0
        imgs = imgs.reshape(imgs.shape[0], -1)  # (n, 784)
        labels = np.asarray(ds["label"], dtype=np.int64)
        _GLOBAL_MNIST[split] = (imgs, labels)
    return _GLOBAL_MNIST[split]


def poisson_encode(images, T, key):
    """Convert a batch of images in [0,1] to Poisson spike trains.

    Args:
        images: (batch, in_dim) float in [0, 1].
        T: number of timesteps.
        key: PRNG key for the Bernoulli sampling (independent across timesteps).
    Returns:
        (T, batch, in_dim) 0/1 float spikes.
    """
    keys = jax.random.split(key, T)
    spikes = jax.vmap(
        lambda k: jax.random.bernoulli(k, p=images).astype(images.dtype)
    )(keys)
    return spikes


def fitness_from_logits(logits, labels):
    """Per-sample hard reward: 1.0 if argmax(logits) == label else 0.0.

    Note: across reward designs tested (hard 0/1 vs log-likelihood vs sigmoid margin),
    hard 0/1 proved the most stable and gave the highest terminal accuracy in this
    pure evolutionary (ES + LoRA + z-score) framework (see docs section 7.5), so it is
    the default reward used for learning-rate experiments.

    Args:
        logits: (batch, num_classes).
        labels: (batch,) int.
    Returns:
        (batch,) float32 rewards in {0, 1}.
    """
    pred = jnp.argmax(logits, axis=-1)
    return (pred == labels).astype(jnp.float32)


def accuracy_from_logits(logits, labels):
    pred = jnp.argmax(logits, axis=-1)
    return jnp.mean((pred == labels).astype(jnp.float32))
