"""Read official BlendedMVS inputs and GT without requiring CUDA/PyTorch.

The geometry normalization uses cameras only. GT is used for evaluation, never
for model inputs, view selection, or the scene scale.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from datasets.data_io import read_pfm


@dataclass
class Camera:
    intrinsics: np.ndarray
    extrinsics: np.ndarray
    depth_min: float
    depth_max: float


def read_camera(filename, fallback_ndepths=192):
    lines = Path(filename).read_text(encoding="utf-8").splitlines()
    try:
        ext = np.asarray([[float(x) for x in line.split()]
                          for line in lines[1:5]], dtype=np.float64)
        intr = np.asarray([[float(x) for x in line.split()]
                           for line in lines[7:10]], dtype=np.float64)
        bounds = [float(x) for x in lines[11].split()]
        near, interval = bounds[:2]
        count = bounds[2] if len(bounds) >= 3 else fallback_ndepths
        far = bounds[3] if len(bounds) >= 4 else near + interval * (count - 1)
    except (ValueError, IndexError) as exc:
        raise ValueError(f"Malformed MVSNet camera: {filename}") from exc
    if (ext.shape != (4, 4) or intr.shape != (3, 3)
            or not np.isfinite(ext).all() or not np.isfinite(intr).all()
            or abs(np.linalg.det(intr)) < 1e-12
            or abs(np.linalg.det(ext)) < 1e-12):
        raise ValueError(f"Invalid camera matrices: {filename}")
    return Camera(intr, ext, near, far)


def read_pairs(filename):
    lines = [line.strip() for line in Path(filename).read_text().splitlines()
             if line.strip()]
    if not lines:
        raise ValueError(f"Empty pair file: {filename}")
    count = int(lines[0])
    if len(lines) != 1 + 2 * count:
        raise ValueError(f"Pair-file count differs from content: {filename}")
    pairs = []
    seen = set()
    for index in range(count):
        ref = int(lines[1 + 2 * index])
        tokens = lines[2 + 2 * index].split()
        nsource = int(tokens[0])
        if len(tokens) != 1 + 2 * nsource:
            raise ValueError(f"Malformed source row for view {ref}: {filename}")
        sources = [int(tokens[1 + 2 * j]) for j in range(nsource)]
        if ref in seen or ref in sources or len(set(sources)) != len(sources):
            raise ValueError(f"Duplicate/self views in pair file: {filename}")
        seen.add(ref)
        pairs.append((ref, sources))
    return pairs


def resized_shape(width, height, max_width=768, max_height=576):
    ratio = min(1.0, max_width / width, max_height / height)
    # /8 features and three spatial downsamplings require multiples of 64.
    w, h = int(width * ratio) // 64 * 64, int(height * ratio) // 64 * 64
    if min(w, h) < 64:
        raise ValueError("Image/resize bounds must allow at least 64x64 pixels")
    return w, h


class MVSDataset:
    """Evaluation-only dataset, accepted by torch DataLoader through duck typing."""

    def __init__(self, datapath, listfile=None, mode="test", nviews=5,
                 ndepths=192, interval_scale=1.0, **kwargs):
        if mode not in ("test", "val"):
            raise ValueError("This BlendedMVS adapter is for evaluation, not training")
        if nviews < 2 or ndepths < 2 or interval_scale <= 0:
            raise ValueError("Require nviews/ndepths >= 2 and positive interval_scale")
        self.root = Path(datapath).resolve()
        if not self.root.is_dir():
            raise FileNotFoundError(f"BlendedMVS root absent: {self.root}")
        self.nviews, self.ndepths = nviews, ndepths
        self.interval_scale = interval_scale
        self.max_width = int(kwargs.get("max_width", 768))
        self.max_height = int(kwargs.get("max_height", 576))
        self.normalize_depth_max = float(kwargs.get("normalize_depth_max", 1000.0))
        if self.normalize_depth_max <= 0:
            raise ValueError("normalize_depth_max must be positive")
        selected_scene = kwargs.get("scene")
        if listfile:
            scenes = [line.strip() for line in Path(listfile).read_text().splitlines()
                      if line.strip() and not line.lstrip().startswith("#")]
            self.split_source = str(Path(listfile).resolve())
        else:
            scenes = sorted(p.name for p in self.root.iterdir()
                            if p.is_dir() and (p / "cams" / "pair.txt").is_file())
            self.split_source = "all scene directories with cams/pair.txt"
        if not scenes or len(scenes) != len(set(scenes)):
            raise ValueError("No scenes or duplicate scene entries")
        if any(Path(s).name != s or s in (".", "..") for s in scenes):
            raise ValueError("Scene list must contain directory names, not paths")
        if selected_scene:
            if selected_scene not in scenes:
                raise ValueError(f"Scene not found in selected split: {selected_scene}")
            scenes = [selected_scene]
        self.scenes = scenes
        self.cameras, self.scene_scales, self.skipped = {}, {}, []
        self.metas = []
        for scene in scenes:
            folder = self.root / scene
            for subdir in ("blended_images", "cams", "rendered_depth_maps"):
                if not (folder / subdir).is_dir():
                    raise FileNotFoundError(f"Required directory absent: {folder / subdir}")
            pairs = read_pairs(folder / "cams" / "pair.txt")
            # All camera upper bounds in the scene, independent of predictions,
            # GT validity, requested sample subset, or selected reference views.
            camera_paths = sorted((folder / "cams").glob("*_cam.txt"))
            uppers = []
            for path in camera_paths:
                cam = read_camera(path, ndepths)
                self.cameras[(scene, int(path.stem.split("_")[0]))] = cam
                if np.isfinite(cam.depth_max) and 0 < cam.depth_min < cam.depth_max:
                    uppers.append(cam.depth_max)
            if not uppers:
                raise ValueError(f"No valid camera depth ranges in scene {scene}")
            self.scene_scales[scene] = self.normalize_depth_max / float(np.median(uppers))
            if not np.isfinite(self.scene_scales[scene]) or self.scene_scales[scene] <= 0:
                raise ValueError(f"Invalid normalized camera scale in scene {scene}")
            for ref, sources in pairs:
                if len(sources) < nviews - 1:
                    self.skipped.append(dict(scene=scene, view=ref,
                                             reason=f"fewer than {nviews - 1} source views"))
                    continue
                selected = [ref] + sources[:nviews - 1]
                # Do not replace the source views according to GT availability.
                for view in selected:
                    for path in self.paths(scene, view):
                        if not path.is_file():
                            raise FileNotFoundError(f"Selected view lacks input/GT: {path}")
                camera = self.cameras[(scene, ref)]
                if not (np.isfinite(camera.depth_min) and np.isfinite(camera.depth_max)
                        and 0 < camera.depth_min < camera.depth_max):
                    self.skipped.append(dict(scene=scene, view=ref,
                                             reason="invalid reference camera depth range"))
                    continue
                self.metas.append((scene, ref, sources[:nviews - 1]))
        if not self.metas:
            raise ValueError("No eligible samples; inspect view counts and camera ranges")

    def paths(self, scene, view):
        folder = self.root / scene
        return (folder / "blended_images" / f"{view:08d}.jpg",
                folder / "cams" / f"{view:08d}_cam.txt",
                folder / "rendered_depth_maps" / f"{view:08d}.pfm")

    def __len__(self):
        return len(self.metas)

    def __getitem__(self, index):
        scene, ref, sources = self.metas[index]
        scale = self.scene_scales[scene]
        with Image.open(self.paths(scene, ref)[0]) as image:
            size = resized_shape(*image.size, self.max_width, self.max_height)
        images, raw_images, depths, masks, intrinsics, extrinsics, projections = (
            [] for _ in range(7))
        resampling = getattr(Image, "Resampling", Image)
        for view in [ref] + sources:
            image_path, _, depth_path = self.paths(scene, view)
            with Image.open(image_path) as image:
                original_size = image.size
                rgb = np.asarray(image.convert("RGB").resize(size, resampling.BILINEAR))
            native, _ = read_pfm(str(depth_path))
            if native.ndim != 2 or native.shape != original_size[::-1]:
                raise ValueError(f"GT/image dimensions differ; alignment must be checked: {depth_path}")
            valid = np.isfinite(native) & (native > 0)
            normalized = np.where(valid, native * scale, 0).astype(np.float32)
            if not np.isfinite(normalized).all():
                raise ValueError(f"GT overflows after camera normalization: {depth_path}")
            depth = np.asarray(Image.fromarray(normalized).resize(
                size, resampling.NEAREST), dtype=np.float32)
            mask = np.asarray(Image.fromarray(valid.astype(np.uint8)).resize(
                size, resampling.NEAREST), dtype=np.uint8).astype(bool)
            cam = self.cameras[(scene, view)]
            k = cam.intrinsics.copy()
            k[0] *= size[0] / original_size[0]
            k[1] *= size[1] / original_size[1]
            ext = cam.extrinsics.copy()
            ext[:3, 3] *= scale
            proj = ext.copy()
            proj[:3, :4] = k @ ext[:3, :4]
            # This repository's Vis-MVSNet expects 1/4-scale projections,
            # while GT geometry below deliberately retains full input K.
            proj[:2, :4] *= 0.25
            raw_images.append(rgb)
            images.append(rgb.astype(np.float32) / 255.0)
            depths.append(depth)
            masks.append(mask)
            intrinsics.append(k)
            extrinsics.append(ext)
            projections.append(proj.astype(np.float32))
        rgb_stack = np.stack(images).transpose(0, 3, 1, 2)
        mean = np.array([.485, .456, .406], np.float32)[None, :, None, None]
        std = np.array([.229, .224, .225], np.float32)[None, :, None, None]
        cam = self.cameras[(scene, ref)]
        near, far = cam.depth_min * scale, cam.depth_max * scale
        candidates = np.linspace(near, near + (far - near) * self.interval_scale,
                                 self.ndepths, dtype=np.float32)
        return dict(imgs=(rgb_stack - mean) / std,
                    proj_matrices=np.stack(projections), depth=depths[0],
                    mask=masks[0], depth_values=candidates,
                    gt_depths=np.stack(depths), gt_masks=np.stack(masks),
                    intrinsics=np.stack(intrinsics), extrinsics=np.stack(extrinsics),
                    rgb=raw_images[0], scene=scene, view=ref,
                    source_views=sources, scene_scale=scale)
