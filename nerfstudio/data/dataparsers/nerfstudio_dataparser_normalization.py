from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import trimesh
from loguru import logger

from nerfstudio.configs.base_config import PrintableConfig


def plot_origin_mesh(origin, vertices, center):
    x_o, y_o, z_o = origin.T
    x_m, y_m, z_m = vertices.T
    c_x, c_y, c_z = center.T

    import plotly.graph_objects as go
    fig = go.Figure(data=[
        go.Scatter3d(
            x=x_m,
            y=y_m,
            z=z_m,
            opacity=0.5,
            mode='markers',
            marker=dict(
                size=1,
                color='blue',
                opacity=0.8
            )
        ),
        go.Scatter3d(
            x=(x_m.mean(),),
            y=(y_m.mean(),),
            z=(z_m.mean(),),
            opacity=0.5,
            mode='markers',
            marker=dict(
                size=10,
                color='red',
                opacity=0.8
            )
        ),
        go.Scatter3d(
            x=x_o,
            y=y_o,
            z=z_o,
            mode='markers',
            marker=dict(
                size=2,
                color='red',
                opacity=0.8
            )
        ),
        go.Scatter3d(
            x=(c_x,),
            y=(c_y,),
            z=(c_z,),
            mode='markers',
            marker=dict(
                size=20,
                color='green',
                opacity=0.8
            )
        )
    ])
    fig.show()


def plot3d(x, y, z):
    import plotly.express as px
    df = px.data.iris()
    fig = px.scatter_3d(df, x=x, y=y, z=z,
                        )
    fig.show()


@dataclass
class NerfStudioDataParserNormalizationConfig(PrintableConfig):
    """normalize the camera pose based on a point cloud"""

    pcd_path: Path = None
    """random seed initialization"""
    pcd_scale: float = 1.0
    """scale for the point cloud"""

    def auto_orient_and_center_poses(self, poses: torch.Tensor):
        """Auto orient and center poses based on point cloud"""

        pcd = trimesh.load(str(self.pcd_path))

        # pcd in t1 space
        t1 = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]])

        vertices = pcd.vertices
        vertices = np.einsum("ij,kj->ki", t1, vertices)
        origin = poses[:, :3, 3].numpy()
        # breakpoint()

        bbox_max = np.max(vertices, axis=0)
        bbox_min = np.min(vertices, axis=0)
        center = (bbox_max + bbox_min) * 0.5
        radius = np.linalg.norm(vertices - center, ord=2, axis=-1).max()
        rotation = np.diag([1, 1, 1])
        translation = -center
        scale_mat = np.eye(4)
        scale_mat[:3, :3] = rotation
        scale_mat[:3, 3] = translation
        scale_mat = torch.from_numpy(scale_mat).float()
        logger.debug(f"radius: {radius}")

        z_flip_matrix = np.zeros((4, 4))
        z_flip_matrix[0, 1] = 1.0
        z_flip_matrix[1, 2] = 1.0
        z_flip_matrix[2, 0] = -1.0
        z_flip_matrix[3, 3] = 1.0
        z_flip_matrix = torch.from_numpy(z_flip_matrix).float()
        return z_flip_matrix @ scale_mat @ poses, scale_mat, 1 / radius * self.pcd_scale
