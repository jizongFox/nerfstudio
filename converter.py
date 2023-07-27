import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import open3d as o3d


@dataclass
class ConvertPCD2OpenCV:
    pcd_path: Path
    output_path: Path
    transform_json_path: Path

    def main(self):
        pcd_path = self.pcd_path
        assert pcd_path.exists() and pcd_path.is_file(), pcd_path
        point_cloud: o3d.geometry.PointCloud = o3d.io.read_point_cloud(str(pcd_path))
        points = np.array(point_cloud.points)
        colors = np.array(point_cloud.colors)
        points_homo = np.concatenate([points, np.ones_like(points[:, 0:1])], axis=-1)

        transform, scale = self.read_transform(self.transform_json_path)
        M = np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

        invers_transform = np.linalg.inv(transform @ M)

        opencv_points = np.einsum("ij,nj->ni", invers_transform, points_homo)[:, :3]
        # opencv_points /= scale
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(opencv_points)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        o3d.io.write_point_cloud(str(self.output_path), pcd)

    def read_transform(self, path):
        with open(path) as f:
            transforms = json.load(f)
        transform = np.eye(4)
        transform[:3] = np.array(transforms["transform"])
        return transform, float(transforms["scale"])


# o3d.io.write_point_cloud(str(self.output_dir / "point_cloud.ply"), pcd)
ConvertPCD2OpenCV(Path("outputs/data-flower_undist3/nerfacto/2023-07-26_224617/point_cloud.ply"),
                  Path("outputs/data-flower_undist3/nerfacto/2023-07-26_224617/point_cloud_cv.ply"),
                  Path(
                      "/home/jizong/Workspace/nerfstudio/outputs/data-flower_undist3/nerfacto/2023-07-26_224617/dataparser_transforms.json")
                  ).main()
