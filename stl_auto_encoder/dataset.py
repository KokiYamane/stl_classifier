import torch
from torch.utils.data import Dataset
import glob
import open3d as o3d
import numpy as np


class STLDataset(Dataset):
    def __init__(self, datafolder):
        paths = sorted(
            glob.glob(
                '{}/*.stl'.format(datafolder),
                recursive=True))
        self.data = []
        for path in paths:
            mesh = o3d.io.read_triangle_mesh(path)
            print(mesh)
            if np.array(mesh.triangles).shape[0] == 0:
                continue
            pcd = mesh.sample_points_uniformly(number_of_points=500)
            # voxel = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, 0.03)
            pcd_points = np.array(pcd.points)
            pcd_points = pcd_points.flatten()
            pcd_points = torch.from_numpy(pcd_points).float()
            self.data.append(pcd_points)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
