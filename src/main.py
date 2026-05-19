import cv2
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt
from pathlib import Path

class DepthToPointCloud:
    def __init__(self, camera_intrinsics: np.ndarray, max_depth: float = 10.0):
        """
        Initialize the depth to point cloud converter.
        
        Args:
            camera_intrinsics: 3x3 camera intrinsic matrix [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
            max_depth: Maximum valid depth value in meters
        """
        self.K = camera_intrinsics
        self.fx = camera_intrinsics[0, 0]
        self.fy = camera_intrinsics[1, 1]
        self.cx = camera_intrinsics[0, 2]
        self.cy = camera_intrinsics[1, 2]
        self.max_depth = max_depth
        
        # For storing the global point cloud
        self.global_point_cloud = None
        self.frame_count = 0
        
    def depth_to_local_pointcloud(self, depth_map: np.ndarray, 
                                color_image: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Convert a single depth map to local 3D point cloud.
        
        Args:
            depth_map: HxW depth map with metric values
            color_image: Optional HxWx3 color image
            
        Returns:
            points: Nx3 array of 3D points in camera coordinates
            colors: Nx3 array of RGB colors (if color_image provided)
        """
        height, width = depth_map.shape
        
        # Create pixel coordinate grids
        u, v = np.meshgrid(np.arange(width), np.arange(height))
        u = u.flatten()
        v = v.flatten()
        depth = depth_map.flatten()
        
        # Filter out invalid depths
        valid_mask = (depth > 0) & (depth < self.max_depth) & np.isfinite(depth)
        u = u[valid_mask]
        v = v[valid_mask]
        depth = depth[valid_mask]
        
        # Convert to 3D points in camera coordinates
        x = (u - self.cx) * depth / self.fx
        y = (v - self.cy) * depth / self.fy
        z = depth
        
        points = np.stack([x, y, z], axis=1)
        
        colors = None
        if color_image is not None:
            color_flat = color_image.reshape(-1, 3)[valid_mask]
            colors = color_flat / 255.0  # Normalize to [0,1]
            
        return points, colors
    
    def transform_pointcloud(self, points: np.ndarray, pose: np.ndarray) -> np.ndarray:
        """
        Transform point cloud from camera coordinates to world coordinates.
        
        Args:
            points: Nx3 array of points in camera coordinates
            pose: 4x4 transformation matrix (camera to world)
            
        Returns:
            transformed_points: Nx3 array of points in world coordinates
        """
        if points.shape[0] == 0:
            return points
            
        # Convert to homogeneous coordinates
        points_homogeneous = np.hstack([points, np.ones((points.shape[0], 1))])
        
        # Transform to world coordinates
        world_points = (pose @ points_homogeneous.T).T
        
        # Convert back to 3D
        return world_points[:, :3]
    
    def add_frame(self, depth_map: np.ndarray, pose: np.ndarray, 
                  color_image: Optional[np.ndarray] = None, 
                  voxel_size: float = 0.05) -> None:
        """
        Add a new frame's depth map to the global point cloud.
        
        Args:
            depth_map: HxW metric depth map
            pose: 4x4 camera pose (camera to world transform)
            color_image: Optional HxWx3 color image
            voxel_size: Voxel size for downsampling
        """
        # Convert depth to local point cloud
        local_points, colors = self.depth_to_local_pointcloud(depth_map, color_image)
        
        if local_points.shape[0] == 0:
            print(f"Frame {self.frame_count}: No valid points")
            return
            
        # Transform to world coordinates
        world_points = self.transform_pointcloud(local_points, pose)
        
        # Create Open3D point cloud for processing
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(world_points)
        
        if colors is not None:
            pcd.colors = o3d.utility.Vector3dVector(colors)
        
        # Remove statistical outliers
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        
        # Downsample
        if voxel_size > 0:
            pcd = pcd.voxel_down_sample(voxel_size)
        
        # Add to global point cloud
        if self.global_point_cloud is None:
            self.global_point_cloud = pcd
        else:
            self.global_point_cloud += pcd
            # Periodically downsample the global point cloud to manage memory
            if self.frame_count % 10 == 0:
                self.global_point_cloud = self.global_point_cloud.voxel_down_sample(voxel_size)
        
        self.frame_count += 1
        print(f"Frame {self.frame_count}: Added {len(pcd.points)} points, "
              f"Total: {len(self.global_point_cloud.points) if self.global_point_cloud else 0}")
    
    def get_point_cloud(self) -> o3d.geometry.PointCloud:
        """Get the current global point cloud."""
        return self.global_point_cloud
    
    def save_point_cloud(self, filename: str) -> None:
        """Save the point cloud to file."""
        if self.global_point_cloud is not None:
            o3d.io.write_point_cloud(filename, self.global_point_cloud)
            print(f"Saved point cloud with {len(self.global_point_cloud.points)} points to {filename}")
        else:
            print("No point cloud to save")

def load_poses_from_file(pose_file: str) -> List[np.ndarray]:
    """
    Load camera poses from file (similar to KITTI format).
    Each line contains 12 values representing a 3x4 transformation matrix.
    """
    poses = []
    with open(pose_file, 'r') as f:
        for line in f:
            values = [float(x) for x in line.strip().split()]
            if len(values) == 12:
                # Reshape to 3x4 and add bottom row to make 4x4
                pose_3x4 = np.array(values).reshape(3, 4)
                pose_4x4 = np.vstack([pose_3x4, [0, 0, 0, 1]])
                poses.append(pose_4x4)
    return poses

def integrate_slam_poses(slam_poses: List[np.ndarray]) -> List[np.ndarray]:
    """
    Convert relative SLAM poses to absolute world poses.
    Assumes slam_poses are relative transformations between consecutive frames.
    """
    absolute_poses = []
    current_pose = np.eye(4)
    
    for relative_pose in slam_poses:
        current_pose = current_pose @ relative_pose
        absolute_poses.append(current_pose.copy())
    
    return absolute_poses

def example_usage():
    """Example of how to use the DepthToPointCloud class."""
    
    # Example camera intrinsics (adjust for your camera)
    K = np.array([
        [525.0, 0.0, 320.0],
        [0.0, 525.0, 240.0],
        [0.0, 0.0, 1.0]
    ])
    
    # Initialize converter
    converter = DepthToPointCloud(K, max_depth=15.0)
    
    # Example data paths (adjust for your setup)
    depth_folder = "/path/to/depth/maps/"  # Folder with depth images
    color_folder = "/path/to/color/images/"  # Optional color images
    pose_file = "/path/to/poses.txt"  # Camera poses
    
    # Load poses
    poses = load_poses_from_file(pose_file)
    
    # Process each frame
    depth_files = sorted(Path(depth_folder).glob("*.png"))  # or *.exr, *.tiff
    color_files = sorted(Path(color_folder).glob("*.jpg")) if Path(color_folder).exists() else []
    
    for i, depth_file in enumerate(depth_files[:100]):  # Process first 100 frames
        # Load depth map (assuming 16-bit PNG with depth in mm)
        depth_map = cv2.imread(str(depth_file), cv2.IMREAD_UNCHANGED)
        depth_map = depth_map.astype(np.float32) / 1000.0  # Convert mm to meters
        
        # Load color image if available
        color_image = None
        if i < len(color_files):
            color_image = cv2.imread(str(color_files[i]))
            color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        
        # Get corresponding pose
        if i < len(poses):
            pose = poses[i]
            
            # Add frame to point cloud
            converter.add_frame(depth_map, pose, color_image, voxel_size=0.05)
        
        # Visualize progress
        if i % 20 == 0:
            print(f"Processed {i+1} frames")
    
    # Get final point cloud
    final_pcd = converter.get_point_cloud()
    
    # Save point cloud
    converter.save_point_cloud("output_pointcloud.ply")
    
    # Visualize
    if final_pcd is not None:
        o3d.visualization.draw_geometries([final_pcd])

def integrate_with_your_slam():
    """
    Example showing how to integrate with your existing SLAM pipeline.
    """
    # Your existing SLAM setup
    K = np.array([
        [525.0, 0.0, 320.0],
        [0.0, 525.0, 240.0],
        [0.0, 0.0, 1.0]
    ])
    
    converter = DepthToPointCloud(K)
    
    # In your main SLAM loop, after computing pose:
    # (This would replace part of your existing while loop)
    
    pose_abs = np.eye(4)  # Your accumulated pose
    
    for i in range(100):  # Your frame loop
        # Get your depth map (from ML model)
        depth_map = get_metric_depth_map(i)  # Your ML depth estimation
        
        # Get color image
        color_image = get_color_image(i)  # Your color image
        
        # Your existing pose computation...
        # R, t = extract_pose(...)
        # pose_abs = integrate_pose(pose_abs, R, t)
        
        # Add to point cloud
        converter.add_frame(depth_map, pose_abs, color_image)
        
        # Continue with your visualization...
    
    # Save final result
    converter.save_point_cloud("slam_pointcloud.ply")

if __name__ == "__main__":
    # Run example
    example_usage()
    
    # Or integrate with your SLAM
    # integrate_with_your_slam()