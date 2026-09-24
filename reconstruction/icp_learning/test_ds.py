import open3d as o3d

demo_icp_pcds = o3d.data.DemoICPPointClouds()

# 查看数据集存放的根目录
print("数据集根目录:", demo_icp_pcds.data_root)

# 查看具体文件的完整路径
print("源点云路径:", demo_icp_pcds.paths[0])
print("目标点云路径:", demo_icp_pcds.paths[1])
