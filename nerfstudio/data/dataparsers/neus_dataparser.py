# Copyright 2022 The Nerfstudio Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Data parser for nerfstudio datasets. """

from __future__ import annotations

import os
from dataclasses import dataclass, field
from glob import glob
from pathlib import Path
from typing import Optional, Type

import cv2
import numpy as np
import torch
from rich.console import Console
from typing_extensions import Literal

from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.data.dataparsers.base_dataparser import (
    DataParser,
    DataParserConfig,
    DataparserOutputs,
)
from nerfstudio.data.scene_box import SceneBox

CONSOLE = Console(width=120)
MAX_AUTO_RESOLUTION = 1600


@dataclass
class NeuS2DataParserConfig(DataParserConfig):
    """Nerfstudio dataset config"""

    _target: Type = field(default_factory=lambda: NeuS2Parser)
    """target class to instantiate"""
    data: Path = Path("/home/jizong/workspace/volrend/data/neus_dragon_2fps")
    """Directory or explicit json file path specifying location of data."""
    cameras_name: str = "cameras_sphere.npz"
    """Name of the camera_sphere.npz"""
    scale_factor: float = 1.0
    """How much to scale the camera origins by."""
    downscale_factor: Optional[int] = 8
    """How much to downscale images. If not set, images are chosen such that the max dimension is <1600px."""
    scene_scale: float = 1.0
    """How much to scale the region of interest by."""
    orientation_method: Literal["pca", "up", "vertical", "none"] = "up"
    """The method to use for orientation."""
    center_method: Literal["poses", "focus", "none"] = "poses"
    """The method to use to center the poses."""
    auto_scale_poses: bool = False
    """Whether to automatically scale the poses to fit in +/- 1 bounding box."""
    train_split_fraction: float = 0.9
    """The fraction of images to use for training. The remaining images are for eval."""
    depth_unit_scale_factor: float = 1e-3
    """Scales the depth values to meters. Default value is 0.001 for a millimeter to meter conversion."""


@dataclass
class NeuS2Parser(DataParser):
    """Nerfstudio DatasetParser"""

    config: NeuS2DataParserConfig
    downscale_factor: Optional[int] = None

    def _generate_dataparser_outputs(self, split="train"):

        self.data_dir = self.config.data
        self.render_cameras_name = self.config.cameras_name
        self.object_cameras_name = self.config.cameras_name

        camera_dict = np.load(os.path.join(self.data_dir, self.render_cameras_name))
        self.camera_dict = camera_dict
        folder_name = "images" if self.config.downscale_factor == 1 else f"images_{int(self.config.downscale_factor)}"
        self.images_lis = [Path(x) for x in sorted(glob(os.path.join(self.data_dir, f'{folder_name}/*.png')))]
        all_indices = list(range(len(self.images_lis)))
        test_indices = all_indices[::10]
        train_indices = [x for x in all_indices if x not in test_indices]

        assert split in ("train", "val", "test"), split
        if split in ("val", "test"):
            cur_index = test_indices
        else:
            cur_index = train_indices

        self.images_lis = [x for i, x in enumerate(self.images_lis) if i in cur_index]

        self.n_images = len(self.images_lis)
        h, w, *_ = cv2.imread(str(self.images_lis[0])).shape
        h, w = int(h * self.config.downscale_factor), int(w * self.config.downscale_factor)

        # world_mat is a projection matrix from world to image
        self.world_mats_np = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in cur_index]
        self.scale_mats_np = []

        # scale_mat: used for coordinate normalization, we assume the scene to render is inside a unit sphere at origin.
        self.scale_mats_np = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in cur_index]

        self.intrinsics_all = []
        self.pose_all = []

        for scale_mat, world_mat in zip(self.scale_mats_np, self.world_mats_np):
            P = world_mat @ scale_mat
            P = P[:3, :4]
            intrinsics, pose = self.load_K_Rt_from_P(None, P)
            self.intrinsics_all.append(torch.from_numpy(intrinsics).float())
            self.pose_all.append(torch.from_numpy(pose).float())

        # self.images = torch.from_numpy(self.images_np.astype(np.float32)).cpu()  # [n_images, H, W, 3]
        # self.masks = torch.from_numpy(self.masks_np.astype(np.float32)).cpu()  # [n_images, H, W, 3]
        self.intrinsics_all = torch.stack(self.intrinsics_all)  # [n_images, 4, 4]
        self.intrinsics_all_inv = torch.inverse(self.intrinsics_all)  # [n_images, 4, 4]
        self.focal = self.intrinsics_all[0][0, 0]
        self.pose_all = torch.stack(self.pose_all)  # [n_images, 4, 4]
        # self.H, self.W = self.images.shape[1], self.images.shape[2]
        # self.image_pixels = self.H * self.W

        # todo: what does this mean?
        object_bbox_min = np.array([-1.01, -1.01, -1.01, 1.0])
        object_bbox_max = np.array([1.01, 1.01, 1.01, 1.0])
        # Object scale mat: region of interest to **extract mesh**
        object_scale_mat = np.load(os.path.join(self.data_dir, self.object_cameras_name))['scale_mat_0']
        object_bbox_min = np.linalg.inv(self.scale_mats_np[0]) @ object_scale_mat @ object_bbox_min[:, None]
        object_bbox_max = np.linalg.inv(self.scale_mats_np[0]) @ object_scale_mat @ object_bbox_max[:, None]
        self.object_bbox_min = object_bbox_min[:3, 0]
        self.object_bbox_max = object_bbox_max[:3, 0]
        fx = self.intrinsics_all[:, 0, 0]
        fy = self.intrinsics_all[:, 1, 1]
        camera_type = CameraType.PERSPECTIVE
        pose = self.pose_all

        # pose[:, :3, :3] = torch.inverse(pose[:, :3, :3])
        # pose[:, :3, 3] = torch.bmm(pose[:, :3, :3], pose[:, :3, 3:4]).squeeze(-1)

        # pose = pose[:, np.array([1, 0, 2, 3]), :]
        # pose[:, 2, :] *= -1
        #
        # pose[:, 2, :] *= -1
        # pose = pose[:, np.array([1, 0, 2, 3]), :]
        # pose[:, 0:3, 1:3] *= -1
        #
        # pose = pose[:, :3, ]

        # pose[:, 0:3, 1:3] *= -1
        # pose = pose[:, np.array([1, 0, 2, 3]), :]

        pose[:, 0:3, 1:3] *= -1  # switch cam coord x,y
        pose = pose[:, [1, 0, 2], :]  # switch world x,y
        pose[:, 2, :] *= -1  # invert world z
        # for aabb bbox usage
        pose = pose[:, [1, 2, 0], :]  # switch world xyz to zxy

        # pose[..., 0:3, 1:3] *= -1
        # pose = pose[:, :3, ]
        # Scale poses
        scale_factor = 1.0
        if self.config.auto_scale_poses:
            scale_factor /= float(torch.max(torch.abs(pose[:, :3, 3])))
        scale_factor *= self.config.scale_factor
        pose[:, :3, 3] *= scale_factor

        cameras = Cameras(
            fx=fx[..., None],
            fy=fy[..., None],
            cx=self.intrinsics_all[:, 0, 2][..., None],
            cy=self.intrinsics_all[:, 1, 2][..., None],
            distortion_params=None,
            height=(h * torch.ones(self.n_images, 1)).long(),
            width=(w * torch.ones(self.n_images, 1)).long(),
            camera_to_worlds=pose,
            camera_type=camera_type,
            times=None,
        )
        cameras.rescale_output_resolution(1 / (self.config.downscale_factor * scale_factor))

        aabb_scale = self.config.scene_scale
        scene_box = SceneBox(
            aabb=torch.tensor(
                [[-aabb_scale, -aabb_scale, -aabb_scale], [aabb_scale, aabb_scale, aabb_scale]], dtype=torch.float32
            )
        )
        dataparser_outputs = DataparserOutputs(
            image_filenames=self.images_lis,
            cameras=cameras,
            scene_box=scene_box,
            mask_filenames=None,
            # dataparser_transform=applied_transform,
            # dataparser_scale=applied_scale,
            # metadata=metadata,
        )
        return dataparser_outputs

    # This function is borrowed from IDR: https://github.com/lioryariv/idr
    @staticmethod
    def load_K_Rt_from_P(filename, P=None):
        """
        return camera intrinsic matrix and
        """
        if P is None:
            lines = open(filename).read().splitlines()
            if len(lines) == 4:
                lines = lines[1:]
            lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
            P = np.asarray(lines).astype(np.float32).squeeze()

        r"""
        [cameraMatrix,rotMatrix,transVect] = cv.decomposeProjectionMatrix(projMatrix)

        cameraMatrix 3x3 camera matrix K.
        rotMatrix 3x3 external rotation matrix R.
        transVect 4x1 translation vector T.
        S Optional output struct with the following fields:
            rotMatrX 3x3 rotation matrix around x-axis.
            rotMatrY 3x3 rotation matrix around y-axis.
            rotMatrZ 3x3 rotation matrix around z-axis.
            eulerAngles 3-element vector containing three Euler angles of rotation in degrees.
        """
        # https://stackoverflow.com/questions/62686618/opencv-decompose-projection-matrix
        K, R, t, *_ = cv2.decomposeProjectionMatrix(P)
        # here t =? -RC, yes, as the translation of the real world coordinate.
        K = K / K[2, 2]
        intrinsics = np.eye(4)
        intrinsics[:3, :3] = K

        pose = np.eye(4, dtype=np.float32)
        pose[:3, :3] = R.transpose()  # The transpose of a rotation matrix is its inverse.
        pose[:3, 3] = (t[:3] / t[3])[:, 0]

        # pose[:3, 3] = np.linalg.inv(P[:3, :3]) @ P[:3, 3]
        return intrinsics, pose


if __name__ == "__main__":
    import tyro

    dataset = tyro.cli(NeuS2Parser)._generate_dataparaser_outputs_neus()
