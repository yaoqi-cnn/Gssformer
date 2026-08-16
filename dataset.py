from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree
from torch.utils.data import DataLoader, Dataset

ATTRIBUTE_K = 16
BOUNDARY_K = 16
ATTRIBUTE_CACHE_VERSION = 2


def scene_files(data_root, areas):
    root = Path(data_root)
    files = []
    for area in areas:
        folder = root / area
        if not folder.is_dir():
            raise FileNotFoundError(f"Area directory not found: {folder}")
        for path in sorted(folder.glob("*.npy")):
            if not any(token in path.name for token in ("_pred", "_boundary", "_attrs")):
                files.append(path)
    if not files:
        raise FileNotFoundError(f"No scene files found under {root}")
    return files


def load_raw_scene(path):
    path = Path(path)
    data = np.load(path).astype(np.float32)
    if data.ndim != 2 or data.shape[0] == 0 or data.shape[1] < 7:
        raise ValueError(f"Expected a non-empty N x 7 array: {path}")
    xyz = data[:, :3]
    rgb = data[:, 3:6]
    labels = data[:, 6].astype(np.int64)
    if not np.isfinite(xyz).all() or not np.isfinite(rgb).all():
        raise ValueError(f"Non-finite coordinates or colors: {path}")
    if rgb.min() < 0:
        raise ValueError(f"Negative RGB value found: {path}")
    if rgb.max() > 1.0:
        rgb = rgb / 255.0
    if rgb.max() > 1.0 + 1e-6:
        raise ValueError(f"RGB values are outside [0, 255] or [0, 1]: {path}")
    return xyz, rgb.astype(np.float32), labels


def local_attributes(xyz, rgb, k=ATTRIBUTE_K):


    n = len(xyz)
    if n == 0:
        raise ValueError("Cannot compute attributes for an empty point cloud")
    k_eff = min(max(1, int(k)), n)
    tree = cKDTree(xyz)
    _, indices = tree.query(xyz, k=k_eff)
    if k_eff == 1:
        indices = indices[:, None]
    neighbors = xyz[indices]
    centered = neighbors - neighbors.mean(axis=1, keepdims=True)
    covariance = np.einsum("nki,nkj->nij", centered, centered) / float(k_eff)
    _, eigenvectors = np.linalg.eigh(covariance)
    normals = eigenvectors[:, :, 0].astype(np.float32)
    rgb_neighbors = rgb[indices]
    color_gradient = np.linalg.norm(
        rgb_neighbors - rgb[:, None], axis=-1
    ).mean(axis=1).astype(np.float32)
    return normals, color_gradient


def boundary_labels(xyz, labels, k=BOUNDARY_K):

    n = len(xyz)
    if n <= 1:
        return np.zeros(n, dtype=np.float32)
    k_eff = min(max(1, int(k)), n - 1)
    tree = cKDTree(xyz)
    _, indices = tree.query(xyz, k=k_eff + 1)
    if indices.ndim == 1:
        indices = indices[:, None]
    indices = indices[:, 1:]
    return (labels[indices] != labels[:, None]).any(axis=1).astype(np.float32)


def cache_path(path):
    path = Path(path)
    return path.with_name(path.stem + "_attrs.npz")


def _source_signature(path):
    stat = Path(path).stat()
    return int(stat.st_size), int(stat.st_mtime_ns)


def _cache_is_valid(values, path, length):
    required = {
        "cache_version",
        "attribute_k",
        "boundary_k",
        "source_size",
        "source_mtime_ns",
        "normals",
        "color_grad",
    }
    if not required.issubset(values.files):
        return False
    source_size, source_mtime_ns = _source_signature(path)
    return (
        int(values["cache_version"]) == ATTRIBUTE_CACHE_VERSION
        and int(values["attribute_k"]) == ATTRIBUTE_K
        and int(values["boundary_k"]) == BOUNDARY_K
        and int(values["source_size"]) == source_size
        and int(values["source_mtime_ns"]) == source_mtime_ns
        and len(values["normals"]) == length
        and len(values["color_grad"]) == length
    )


def _save_attribute_cache(target, path, normals, color_gradient, boundary=None):
    source_size, source_mtime_ns = _source_signature(path)
    values = {
        "cache_version": np.asarray(ATTRIBUTE_CACHE_VERSION, dtype=np.int64),
        "attribute_k": np.asarray(ATTRIBUTE_K, dtype=np.int64),
        "boundary_k": np.asarray(BOUNDARY_K, dtype=np.int64),
        "source_size": np.asarray(source_size, dtype=np.int64),
        "source_mtime_ns": np.asarray(source_mtime_ns, dtype=np.int64),
        "normals": normals.astype(np.float32),
        "color_grad": color_gradient.astype(np.float32),
    }
    if boundary is not None:
        values["boundary"] = boundary.astype(np.float32)
    np.savez_compressed(target, **values)


def load_attributes(path, xyz, rgb, labels=None):


    path = Path(path)
    target = cache_path(path)
    normals = color_gradient = boundary = None
    cache_valid = False
    boundary_missing = labels is not None

    if target.exists():
        try:
            with np.load(target) as values:
                cache_valid = _cache_is_valid(values, path, len(xyz))
                if cache_valid:
                    normals = values["normals"].astype(np.float32)
                    color_gradient = values["color_grad"].astype(np.float32)
                    if labels is not None and "boundary" in values.files:
                        boundary = values["boundary"].astype(np.float32)
                        boundary_missing = False
        except (OSError, ValueError, KeyError):
            cache_valid = False
            normals = color_gradient = boundary = None

    if normals is None or color_gradient is None:
        normals, color_gradient = local_attributes(xyz, rgb)
    if labels is not None and boundary is None:
        boundary = boundary_labels(xyz, labels)

    if not cache_valid or boundary_missing:
        _save_attribute_cache(target, path, normals, color_gradient, boundary)

    return normals, color_gradient, boundary


def normalize_coordinates(xyz, center, block_size):
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    coordinates = xyz.astype(np.float32).copy()
    coordinates[:, 0] = (coordinates[:, 0] - center[0]) / block_size
    coordinates[:, 1] = (coordinates[:, 1] - center[1]) / block_size
    coordinates[:, 2] = (coordinates[:, 2] - xyz[:, 2].min()) / block_size
    return coordinates


def make_features(xyz, rgb, normals, color_gradient, center, block_size):
    coordinates = normalize_coordinates(xyz, center, block_size)
    features = np.concatenate(
        (coordinates, rgb, normals, color_gradient[:, None]), axis=1
    ).astype(np.float32)
    if features.shape[1] != 10:
        raise RuntimeError(f"Expected 10 input features, got {features.shape[1]}")
    return features, coordinates.astype(np.float32)


class SpatialBlockDataset(Dataset):
    def __init__(
        self,
        data_root,
        areas,
        class_names,
        block_size,
        num_points=-1,
        minimum_points=2,
        attempts=32,
    ):
        self.block_size = float(block_size)
        self.num_points = int(num_points)
        self.minimum_points = max(1, int(minimum_points))
        self.attempts = max(1, int(attempts))
        self.class_names = list(class_names)
        self.scenes = []
        self.scene_indices = []
        for path in scene_files(data_root, areas):
            xyz, rgb, labels = load_raw_scene(path)
            if labels.min() < 0 or labels.max() >= len(self.class_names):
                raise ValueError(
                    f"Label outside [0, {len(self.class_names) - 1}]: {path}"
                )
            normals, color_gradient, boundary = load_attributes(
                path, xyz, rgb, labels
            )
            self.scenes.append(
                (path, xyz, rgb, labels, normals, color_gradient, boundary)
            )
            effective = 50000 if self.num_points == -1 else self.num_points
            self.scene_indices.extend(
                [len(self.scenes) - 1] * max(1, len(xyz) // effective)
            )
        print(
            f"Scenes={len(self.scenes)}, samples_per_epoch={len(self.scene_indices)}, "
            f"block_size={self.block_size}, all_points={self.num_points == -1}"
        )

    def __len__(self):
        return len(self.scene_indices)

    def _indices(self, xyz, center):
        half = self.block_size / 2.0
        mask = (
            (xyz[:, 0] >= center[0] - half)
            & (xyz[:, 0] <= center[0] + half)
            & (xyz[:, 1] >= center[1] - half)
            & (xyz[:, 1] <= center[1] + half)
        )
        return np.flatnonzero(mask)

    def _sample(self, xyz):
        best_center = xyz[0, :2].copy()
        best_indices = self._indices(xyz, best_center)
        for _ in range(self.attempts):
            center = xyz[np.random.randint(len(xyz)), :2].copy()
            indices = self._indices(xyz, center)
            if len(indices) > len(best_indices):
                best_center, best_indices = center, indices
            if len(indices) >= self.minimum_points:
                return center, indices
        return best_center, best_indices

    def __getitem__(self, index):
        _, xyz, rgb, labels, normals, color_gradient, boundary = self.scenes[
            self.scene_indices[index]
        ]
        center, indices = self._sample(xyz)
        total = len(indices)
        if total == 0:
            raise RuntimeError("The sampled spatial block is empty")
        if self.num_points == -1:
            selected = np.arange(total)
        elif total >= self.num_points:
            selected = np.random.choice(total, self.num_points, replace=False)
        else:
            selected = np.random.choice(total, self.num_points, replace=True)
        indices = indices[selected]
        features, coordinates = make_features(
            xyz[indices],
            rgb[indices],
            normals[indices],
            color_gradient[indices],
            center,
            self.block_size,
        )
        return (
            torch.from_numpy(features),
            torch.from_numpy(coordinates),
            torch.from_numpy(labels[indices]),
            torch.from_numpy(boundary[indices]),
        )


def worker_seed(worker_id):
    del worker_id
    np.random.seed(torch.initial_seed() % (2**32))


def make_loader(dataset, num_workers=6, shuffle=True, seed=None):
    kwargs = {
        "dataset": dataset,
        "batch_size": 1,
        "shuffle": shuffle,
        "num_workers": int(num_workers),
        "pin_memory": True,
        "drop_last": False,
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 4
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(int(seed))
        kwargs["generator"] = generator
        kwargs["worker_init_fn"] = worker_seed
    return DataLoader(**kwargs)
