"""
NeRF-style dataset loader for Gaussian Splatting.

This module provides a dataloader for NeRF synthetic datasets (Blender format)
that use transforms_train.json and transforms_test.json files.

Dataset structure expected:
    data_dir/
        transforms_train.json
        transforms_test.json  (optional)
        images/  (or images referenced in transforms)
        ...

The transforms JSON files should follow the NeRF format:
    {
        "camera_angle_x": <float>,
        "frames": [
            {
                "file_path": "<relative_path_to_image>",
                "transform_matrix": <4x4 camera-to-world matrix>
            },
            ...
        ]
    }

Reference: https://github.com/graphdeco-inria/gaussian-splatting/blob/main/scene/dataset_readers.py
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from .normalize import (
    align_principal_axes,
    similarity_from_cameras,
    transform_cameras,
    transform_points,
)


def _focal2fov(focal: float, pixels: float) -> float:
    """Convert focal length to field of view."""
    return 2 * np.arctan(pixels / (2 * focal))


def _fov2focal(fov: float, pixels: float) -> float:
    """Convert field of view to focal length."""
    return pixels / (2 * np.tan(fov / 2))


class Parser:
    """NeRF-style dataset parser (Blender synthetic format)."""

    def __init__(
        self,
        data_dir: str,
        factor: int = 1,
        normalize: bool = False,
        test_every: int = 8,
        white_background: bool = False,
        extension: str = ".png",
        custom_train_json: Optional[str] = None,
    ):
        """
        Args:
            data_dir: Path to the dataset directory
            factor: Downsampling factor for images
            normalize: Whether to normalize the scene
            test_every: Every N images is a test image (if test split not explicitly defined)
            white_background: Whether to use white background for RGBA images
            extension: Image file extension (.png, .jpg, etc.)
            custom_train_json: Optional path to custom training JSON file (if None, uses transforms_train.json)
        """
        self.data_dir = data_dir
        self.factor = factor
        self.normalize = normalize
        self.test_every = test_every
        self.white_background = white_background
        self.extension = extension

        # Load training transforms
        if custom_train_json is not None:
            # Use custom JSON file (can be absolute or relative path)
            if os.path.isabs(custom_train_json):
                transforms_train_path = custom_train_json
            else:
                # Relative path: resolve relative to data_dir
                transforms_train_path = os.path.join(data_dir, custom_train_json)
            print(f"Using custom training JSON: {transforms_train_path}")
        else:
            # Use default transforms_train.json
            transforms_train_path = os.path.join(data_dir, "transforms_train.json")
        
        if not os.path.exists(transforms_train_path):
            raise ValueError(
                f"Training JSON not found at {transforms_train_path}"
            )

        # Load test transforms (optional)
        transforms_test_path = os.path.join(data_dir, "transforms_test.json")
        has_test_split = os.path.exists(transforms_test_path)

        # Parse training cameras
        train_cam_infos = self._read_cameras_from_transforms(
            transforms_train_path, is_test=False
        )
        print(f"[Parser] Loaded {len(train_cam_infos)} training cameras")

        # Parse test cameras
        test_cam_infos = []
        if has_test_split:
            test_cam_infos = self._read_cameras_from_transforms(
                transforms_test_path, is_test=True
            )
            print(f"[Parser] Loaded {len(test_cam_infos)} test cameras")
        else:
            # Use every Nth image as test (if test_every > 0)
            if test_every > 0:
                test_indices = list(range(0, len(train_cam_infos), test_every))
                test_cam_infos = [train_cam_infos[i] for i in test_indices]
                train_cam_infos = [
                    train_cam_infos[i]
                    for i in range(len(train_cam_infos))
                    if i not in test_indices
                ]
                print(
                    f"[Parser] Split into {len(train_cam_infos)} train and {len(test_cam_infos)} test cameras"
                )

        # Combine all cameras for processing
        all_cam_infos = train_cam_infos + test_cam_infos

        # Extract camera-to-world matrices
        camtoworlds = np.array([cam_info["camtoworld"] for cam_info in all_cam_infos])
        image_paths = [cam_info["image_path"] for cam_info in all_cam_infos]
        image_names = [cam_info["image_name"] for cam_info in all_cam_infos]
        Ks = [cam_info["K"] for cam_info in all_cam_infos]
        widths = [cam_info["width"] for cam_info in all_cam_infos]
        heights = [cam_info["height"] for cam_info in all_cam_infos]
        is_test_flags = [cam_info["is_test"] for cam_info in all_cam_infos]

        # Normalize the world space
        if normalize:
            T1 = similarity_from_cameras(camtoworlds)
            camtoworlds = transform_cameras(T1, camtoworlds)
            transform = T1

            # Align principal axes (requires points, which we'll generate)
            # For NeRF synthetic, we create a simple bounding box
            camera_locations = camtoworlds[:, :3, 3]
            scene_center = np.mean(camera_locations, axis=0)
            scene_scale = np.max(np.linalg.norm(camera_locations - scene_center, axis=1))
            # Create dummy points around the scene
            dummy_points = scene_center + (np.random.rand(1000, 3) - 0.5) * scene_scale * 2
            dummy_points = transform_points(T1, dummy_points)

            T2 = align_principal_axes(dummy_points)
            camtoworlds = transform_cameras(T2, camtoworlds)
            transform = T2 @ T1

            # Fix for upside down scenes
            if np.median(dummy_points[:, 2]) > np.mean(dummy_points[:, 2]):
                T3 = np.array(
                    [
                        [1.0, 0.0, 0.0, 0.0],
                        [0.0, -1.0, 0.0, 0.0],
                        [0.0, 0.0, -1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ]
                )
                camtoworlds = transform_cameras(T3, camtoworlds)
                transform = T3 @ transform
        else:
            transform = np.eye(4)

        # Store camera information
        self.image_names = image_names
        self.image_paths = image_paths
        self.camtoworlds = camtoworlds
        self.is_test = np.array(is_test_flags)
        self.Ks = Ks
        self.widths = np.array(widths)
        self.heights = np.array(heights)
        self.transform = transform

        # Create a unified K dictionary (all cameras use same intrinsics typically)
        # Use camera_id 0 for all since NeRF datasets usually have uniform intrinsics
        self.Ks_dict = {0: Ks[0]}  # All cameras share the same intrinsics
        self.params_dict = {0: np.empty(0, dtype=np.float32)}  # No distortion
        self.imsize_dict = {0: (widths[0] // factor, heights[0] // factor)}
        self.mask_dict = {0: None}
        self.camera_ids = [0] * len(all_cam_infos)

        # Update intrinsics for downsampling
        if factor > 1:
            for cam_id in self.Ks_dict:
                K = self.Ks_dict[cam_id].copy()
                K[:2, :] /= factor
                self.Ks_dict[cam_id] = K
                w, h = self.imsize_dict[cam_id]
                self.imsize_dict[cam_id] = (w, h)

        # Generate dummy 3D points for compatibility
        # In NeRF synthetic, we don't have COLMAP points, so we create a sparse set
        camera_locations = camtoworlds[:, :3, 3]
        scene_center = np.mean(camera_locations, axis=0)
        scene_scale = np.max(np.linalg.norm(camera_locations - scene_center, axis=1))
        
        # Create a grid of points in the scene (used for initialization)
        # These are transformed if normalization is applied
        num_points = 100_000
        points = scene_center + (np.random.rand(num_points, 3) - 0.5) * scene_scale * 2.5
        if normalize:
            points = transform_points(transform, points)
        points_rgb = np.random.randint(0, 256, size=(num_points, 3), dtype=np.uint8)
        points_err = np.ones(num_points, dtype=np.float32) * 0.01

        self.points = points.astype(np.float32)
        self.points_rgb = points_rgb
        self.points_err = points_err
        self.point_indices = {}  # Empty for NeRF datasets

        # Scene scale (after normalization)
        camera_locations_normalized = camtoworlds[:, :3, 3]
        scene_center_normalized = np.mean(camera_locations_normalized, axis=0)
        self.scene_scale = np.max(
            np.linalg.norm(camera_locations_normalized - scene_center_normalized, axis=1)
        )

        # Load extended metadata if available
        self.extconf = {
            "spiral_radius_scale": 1.0,
            "no_factor_suffix": False,
        }
        extconf_file = os.path.join(data_dir, "ext_metadata.json")
        if os.path.exists(extconf_file):
            with open(extconf_file) as f:
                self.extconf.update(json.load(f))

        # Load bounds if available
        self.bounds = np.array([0.01, 1.0])
        posefile = os.path.join(data_dir, "poses_bounds.npy")
        if os.path.exists(posefile):
            self.bounds = np.load(posefile)[:, -2:]

    def _read_cameras_from_transforms(
        self, transforms_path: str, is_test: bool
    ) -> List[Dict[str, Any]]:
        """Read camera information from transforms JSON file."""
        cam_infos = []

        with open(transforms_path, "r") as json_file:
            contents = json.load(json_file)
            fovx = contents["camera_angle_x"]

            frames = contents["frames"]
            for idx, frame in enumerate(frames):
                cam_name = os.path.join(self.data_dir, frame["file_path"] + self.extension)

                # NeRF 'transform_matrix' is a camera-to-world transform
                c2w = np.array(frame["transform_matrix"]).astype(np.float32)
                # Change from OpenGL/Blender camera axes (Z up, Y back) to COLMAP (Y down, Z forward)
                c2w[:3, 1:3] *= -1

                # Load image to get dimensions
                if not os.path.exists(cam_name):
                    # Try alternative paths
                    alt_paths = [
                        os.path.join(self.data_dir, "images", os.path.basename(cam_name)),
                        cam_name.replace(self.extension, ".jpg"),
                        cam_name.replace(self.extension, ".JPG"),
                    ]
                    found = False
                    for alt_path in alt_paths:
                        if os.path.exists(alt_path):
                            cam_name = alt_path
                            found = True
                            break
                    if not found:
                        raise FileNotFoundError(
                            f"Image not found: {cam_name}. Tried: {alt_paths}"
                        )

                image = Image.open(cam_name)
                width, height = image.size

                # Handle RGBA images with alpha channel
                if image.mode == "RGBA":
                    im_data = np.array(image.convert("RGBA"))
                    bg = np.array([1, 1, 1]) if self.white_background else np.array([0, 0, 0])
                    norm_data = im_data / 255.0
                    arr = (
                        norm_data[:, :, :3] * norm_data[:, :, 3:4]
                        + bg * (1 - norm_data[:, :, 3:4])
                    )
                    image = Image.fromarray(np.array(arr * 255.0, dtype=np.uint8), "RGB")
                elif image.mode != "RGB":
                    image = image.convert("RGB")

                # Calculate FOV
                fovy = _focal2fov(_fov2focal(fovx, width), height)
                FovY = fovy
                FovX = fovx

                # Calculate intrinsics
                focal_length = _fov2focal(fovx, width)
                fx = focal_length
                fy = _fov2focal(fovy, height)
                cx = width / 2.0
                cy = height / 2.0

                K = np.array(
                    [[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32
                )

                # Apply downsampling factor
                if self.factor > 1:
                    width = width // self.factor
                    height = height // self.factor
                    K[:2, :] /= self.factor

                image_name = Path(cam_name).stem
                image_path = cam_name

                cam_info = {
                    "camtoworld": c2w,
                    "K": K,
                    "width": width,
                    "height": height,
                    "image_path": image_path,
                    "image_name": image_name,
                    "is_test": is_test,
                }

                cam_infos.append(cam_info)

        return cam_infos


class Dataset:
    """Dataset class for NeRF-style datasets."""

    def __init__(
        self,
        parser: Parser,
        split: str = "train",
        patch_size: Optional[int] = None,
        load_depths: bool = False,
    ):
        """
        Args:
            parser: Parser instance
            split: "train" or "val" (test set)
            patch_size: Optional random crop size for training
            load_depths: Whether to load depth information (not available for NeRF synthetic)
        """
        self.parser = parser
        self.split = split
        self.patch_size = patch_size
        self.load_depths = load_depths
        self._image_cache = {}
        self._mask_cache = {}
        self._downsample_cache = {}

        # Get indices based on split
        if split == "train":
            self.indices = np.where(~parser.is_test)[0]
        else:  # val/test
            self.indices = np.where(parser.is_test)[0]

        if len(self.indices) == 0:
            raise ValueError(f"No {split} images found in dataset")

    def __len__(self):
        return len(self.indices)
    
    def cache_image(
        self,
        image_id: int,
        image: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> None:
        if self.patch_size is not None:
            return
        if image_id not in self._image_cache:
            self._image_cache[image_id] = image.detach().cpu()
        if mask is not None and image_id not in self._mask_cache:
            self._mask_cache[image_id] = mask.detach().cpu()
    
    def get_downsampled(
        self, image_id: int, downsample_factor: int
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if self.patch_size is not None or downsample_factor <= 1:
            return None, None
        key = (image_id, int(downsample_factor))
        cached = self._downsample_cache.get(key)
        if cached is not None:
            return cached
        image = self._image_cache.get(image_id)
        if image is None:
            return None, None
        height = max(1, int(image.shape[0] // downsample_factor))
        width = max(1, int(image.shape[1] // downsample_factor))
        image_bchw = image.permute(2, 0, 1).unsqueeze(0).float()
        image_down_bchw = F.interpolate(
            image_bchw,
            size=(height, width),
            mode="bicubic",
            align_corners=False,
        )
        image_down = image_down_bchw.squeeze(0).permute(1, 2, 0).contiguous()
        mask_down = None
        mask = self._mask_cache.get(image_id)
        if mask is not None:
            mask_bchw = mask.unsqueeze(0).unsqueeze(0).float()
            mask_down_bchw = F.interpolate(
                mask_bchw,
                size=(height, width),
                mode="nearest",
            )
            mask_down = mask_down_bchw.squeeze(0).squeeze(0).bool()
        self._downsample_cache[key] = (image_down, mask_down)
        return image_down, mask_down

    def __getitem__(self, item: int) -> Dict[str, Any]:
        index = self.indices[item]
        
        # Load image (preserve alpha channel if present)
        image_path = self.parser.image_paths[index]
        image = imageio.imread(image_path)  # Load full image (may be RGB or RGBA)
        
        # Apply white_background if specified and image has alpha channel
        if image.shape[-1] == 4:  # RGBA image
            if self.parser.white_background:
                # Composite with white background
                im_data = image.astype(np.float32) / 255.0
                bg = np.array([1, 1, 1], dtype=np.float32)  # White background
                arr = (
                    im_data[:, :, :3] * im_data[:, :, 3:4]
                    + bg * (1 - im_data[:, :, 3:4])
                )
                image = (arr * 255.0).astype(np.uint8)
                # Keep as RGB (remove alpha channel after compositing)
                image = image[:, :, :3]
            # If white_background is False, keep RGBA as is

        # Get camera parameters
        camera_id = self.parser.camera_ids[index]
        K = self.parser.Ks_dict[camera_id].copy()
        camtoworld = self.parser.camtoworlds[index]
        mask = self.parser.mask_dict[camera_id]

        # Resize image if needed (if factor > 1)
        # Preserve alpha channel if present
        if self.parser.factor > 1:
            height, width = image.shape[:2]
            new_height = height // self.parser.factor
            new_width = width // self.parser.factor
            # Determine image mode based on number of channels
            if image.shape[-1] == 4:
                pil_image = Image.fromarray(image, mode="RGBA")
            else:
                pil_image = Image.fromarray(image, mode="RGB")
            pil_image = pil_image.resize((new_width, new_height), Image.BICUBIC)
            image = np.array(pil_image)

        # Random crop for patch training
        if self.patch_size is not None:
            h, w = image.shape[:2]
            x = np.random.randint(0, max(w - self.patch_size, 1))
            y = np.random.randint(0, max(h - self.patch_size, 1))
            image = image[y : y + self.patch_size, x : x + self.patch_size]
            K[0, 2] -= x
            K[1, 2] -= y

        data = {
            "K": torch.from_numpy(K).float(),
            "camtoworld": torch.from_numpy(camtoworld).float(),
            "image": torch.from_numpy(image).float(),
            "image_id": item,  # the index of the image in the dataset
        }
        if mask is not None:
            data["mask"] = torch.from_numpy(mask).bool()

        if self.load_depths:
            # For NeRF synthetic, we don't have depth information
            # Return empty tensors for compatibility (same type as colmap.py)
            data["points"] = torch.empty((0, 2), dtype=torch.float32)
            data["depths"] = torch.empty((0,), dtype=torch.float32)

        return data


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--factor", type=int, default=1)
    parser.add_argument("--white_background", action="store_true")
    args = parser.parse_args()

    # Parse NeRF dataset
    parser = Parser(
        data_dir=args.data_dir,
        factor=args.factor,
        normalize=True,
        test_every=8,
        white_background=args.white_background,
    )
    
    train_dataset = Dataset(parser, split="train")
    val_dataset = Dataset(parser, split="val")
    
    print(f"Train dataset: {len(train_dataset)} images")
    print(f"Val dataset: {len(val_dataset)} images")
    print(f"Scene scale: {parser.scene_scale:.3f}")
    
    # Test loading a sample
    sample = train_dataset[0]
    print(f"Sample image shape: {sample['image'].shape}")
    print(f"K matrix:\n{sample['K']}")
    print(f"Camera-to-world shape: {sample['camtoworld'].shape}")

