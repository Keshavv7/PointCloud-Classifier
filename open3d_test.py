import open3d as o3d

# Load a point cloud file or create a simple one
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector([[0, 0, 0], [1, 1, 1], [2, 2, 2]])

# Save it to a file
o3d.io.write_point_cloud("simple_point_cloud.ply", pcd)
