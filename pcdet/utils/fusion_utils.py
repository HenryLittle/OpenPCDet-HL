import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from kornia.utils.grid import create_meshgrid3d
    from kornia.geometry.linalg import transform_points
except Exception as e:
    # Note: Kornia team will fix this import issue to try to allow the usage of lower torch versions.
    # print('Warning: kornia is not installed correctly, please ignore this warning if you do not use CaDDN. Otherwise, it is recommended to use torch version greater than 1.2 to use kornia properly.')
    pass

from pcdet.utils import transform_utils
from pcdet.utils.common_utils import rotate_points_along_z

class ImageGridGenerator(nn.Module):

    def __init__(self, voxel_indices, grid_size, pc_range, noise_rotation, model_cfg):
        """
        Initializes Grid Generator for frustum features
        Args:
            voxel_indices: [N, D], indices of the sparse tensor to be processed
            grid_size: [X, Y, Z], Voxel grid size, aka spTensor spatial_shape
            pc_range: [x_min, y_min, z_min, x_max, y_max, z_max], Voxelization point cloud range (m)
        """
        super().__init__()
        try:
            import kornia
        except Exception as e:
            # Note: Kornia team will fix this import issue to try to allow the usage of lower torch versions.
            print('Error: kornia is not installed correctly, please ignore this warning if you do not use CaDDN. '
                  'Otherwise, it is recommended to use torch version greater than 1.2 to use kornia properly.')
            exit(-1)

        self.dtype = torch.float32
        self.grid_size = torch.as_tensor(grid_size, dtype=self.dtype)
        self.grid_size = self.grid_size[[2, 1, 0]]
        self.pc_range = pc_range
        self.model_cfg = model_cfg
        self.noise_rot = noise_rotation # used to counter rotation and restore aligment between Image and Voxel grid
        
        # Calculate voxel size
        pc_range = torch.as_tensor(pc_range).reshape(2, 3)
        self.pc_min = pc_range[0]
        self.pc_max = pc_range[1]
        self.voxel_size = (self.pc_max - self.pc_min) / self.grid_size

        # self.voxel_grid = self.voxel_grid.permute(0, 1, 3, 2, 4)  # XZY-> XYZ
        self.voxel_grid = voxel_indices.type(self.dtype) # [N, BXYZ]
        # Add offsets to center of voxel
        self.voxel_grid[:, 1:] += 0.5
        
        self.grid_to_lidar = self.grid_to_lidar_unproject(pc_min=self.pc_min,
                                                          voxel_size=self.voxel_size)

    def grid_to_lidar_unproject(self, pc_min, voxel_size):
        """
        Calculate grid to LiDAR unprojection for each plane
        Args:
            pc_min: [x_min, y_min, z_min], Minimum of point cloud range (m)
            voxel_size: [x, y, z], Size of each voxel (m)
        Returns:
            unproject: (4, 4), Voxel grid to LiDAR unprojection matrix
        """
        x_size, y_size, z_size = voxel_size
        x_min, y_min, z_min = pc_min
        unproject = torch.tensor([[x_size, 0, 0, x_min],
                                  [0, y_size, 0, y_min],
                                  [0,  0, z_size, z_min],
                                  [0,  0, 0, 1]],
                                 dtype=self.dtype)  # (4, 4)

        return unproject

    def transform_grid(self, image_shape, grid_to_lidar, lidar_to_cam, cam_to_img):
        """
        Transforms voxel sampling grid into image sampling grid
        Args:
            grid: B x (XYZ, 3), Voxel sampling grid
            grid_to_lidar: (4, 4), Voxel grid to LiDAR unprojection matrix
            lidar_to_cam: (B, 4, 4), LiDAR to camera frame transformation
            cam_to_img: (B, 3, 4), Camera projection matrix
        Returns:
            frustum_grid: (XYZ, 4), Frustum sampling grid
        """
        B = lidar_to_cam.shape[0]
        
        # Create transformation matricies
        V_G = grid_to_lidar  # Voxel Grid -> LiDAR (4, 4)
        C_V = lidar_to_cam  # LiDAR -> Camera (B, 4, 4)
        I_C = cam_to_img  # Camera -> Image (B, 3, 4)
        trans = C_V @ V_G # Matrix Multiplication

        image_grid_list = []

        for i in range(0, B):
            #  xi[(xi[:, 0]==1).nonzero().squeeze(1)] https://discuss.pytorch.org/t/select-rows-of-the-tensor-whose-first-element-is-equal-to-some-value/1718
            voxel_grid_sample = self.voxel_grid[(self.voxel_grid[:, 0]==i).nonzero().squeeze(1)]
            voxel_grid_sample_xyz = voxel_grid_sample[:, 1:].unsqueeze(0) # [1, N, 3]
            t_V2L = V_G.unsqueeze(0) # [1, 4, 4]
            # VoxelGrid to LiDAR
            voxel_grid_sample_xyz = transform_points(trans_01=t_V2L, points_1=voxel_grid_sample_xyz)
            # counter the rotation noise in LiDAR Coord
            if self.noise_rot is not None:
                voxel_grid_sample_xyz = rotate_points_along_z(voxel_grid_sample_xyz, -self.noise_rot[i].unsqueeze(0))

            # t_trans = trans[i].unsqueeze(0) # [1, 4, 4]
            t_L2C = C_V[i].unsqueeze(0) # [1, 4, 4]
            cam_grid = transform_points(trans_01=t_L2C, points_1=voxel_grid_sample_xyz)

            t_I_C = I_C[i].unsqueeze(0) # [1, 3, 4]
            image_grid = transform_utils.project_to_image_no_depth(project=t_I_C, points=cam_grid) # [1, N, 2]
            image_grid = transform_utils.normalize_coords(coords=image_grid, shape=image_shape)

            image_grid_list.append(image_grid.squeeze(0))


        # Reshape to match dimensions
        # trans = trans.reshape(B, 4, 4)
        # breakpoint()
        # voxel_grid = voxel_grid.repeat_interleave(repeats=B, dim=0)

        # Transform to camera frame
        # function provided by [Kornia]
        # camera_grid = transform_points(trans_01=trans, points_1=voxel_grid)

        # Project to image
        # I_C = I_C.reshape(B, 3, 4)
        # image_grid, image_depths = transform_utils.project_to_image(project=I_C, points=camera_grid)

        # Convert depths to depth bins
        # image_depths = transform_utils.bin_depths(depth_map=image_depths, **self.disc_cfg)

        # Stack to form frustum grid
        # image_depths = image_depths.unsqueeze(-1)
        # frustum_grid = torch.cat((image_grid, image_depths), dim=-1)
        return image_grid_list

    def forward(self, lidar_to_cam, cam_to_img, image_shape):
        """
        Generates sampling grid for frustum features
        Args:
            lidar_to_cam: (B, 4, 4), LiDAR to camera frame transformation
            cam_to_img: (B, 3, 4), Camera projection matrix
            image_shape: (B, 2), Image shape [H, W]
        Returns:
            frustum_grid B (N, 2), Sampling grids for frustum features
        """

        # image_shape = torch.as_tensor(image_shape).to(lidar_to_cam.device)

        image_shape, _ = torch.max(image_shape, dim=0)


        frustum_grid = self.transform_grid(image_shape=image_shape,
                                           grid_to_lidar=self.grid_to_lidar.to(lidar_to_cam.device),
                                           lidar_to_cam=lidar_to_cam,
                                           cam_to_img=cam_to_img)

        # Normalize grid
        # image_depth = torch.tensor([self.disc_cfg["num_bins"]],
        #                            device=image_shape.device,
        #                            dtype=image_shape.dtype)
        # frustum_shape = torch.cat((image_depth, image_shape))
        # frustum_grid = transform_utils.normalize_coords(coords=frustum_grid, shape=image_shape)

        # Replace any NaNs or infinites with out of bounds
        # mask = ~torch.isfinite(frustum_grid)
        # frustum_grid[mask] = self.out_of_bounds_val

        return frustum_grid


class Sampler(nn.Module):

    def __init__(self, mode="bilinear", padding_mode="zeros"):
        """
        Initializes module
        Args:
            mode: string, Sampling mode [bilinear/nearest]
            padding_mode: string, Padding mode for outside grid values [zeros/border/reflection]
        """
        super().__init__()
        self.mode = mode
        self.padding_mode = padding_mode

    def forward(self, input_features, grid):
        """
        Samples input using sampling grid
        Args:
            input_features: [B, C, H, W], Input Image features
            grid: B [N, 2], Sampling grids for input features
        Returns
            output_features: B [N, C] Output voxel features
        """
        # Sample from grid
        output = []
        B = input_features.shape[0]
        for i in range(0, B):
            # 1xCx1xN feats
            sampled = F.grid_sample(
                input=input_features[i].unsqueeze(0), 
                grid=grid[i].unsqueeze(0).unsqueeze(0), 
                mode=self.mode, 
                align_corners=False,
                padding_mode=self.padding_mode)
            output.append(sampled.squeeze().t()) # [N, C]
        return output