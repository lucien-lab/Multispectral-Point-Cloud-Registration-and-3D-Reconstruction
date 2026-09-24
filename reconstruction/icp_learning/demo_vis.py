
import open3d as o3d
import numpy as np
import copy  # 必须导入 copy 库才能使用 deepcopy

def draw_registration_result(source, target, transformation):
    # 使用 deepcopy 确保不会修改原始的点云对象
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    
    # 为源点云（黄色）和目标点云（青色）上色
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    
    # 对源点云应用变换矩阵
    source_temp.transform(transformation)
    
    # 可视化展示
    # 注意：zoom, front, lookat, up 是预设的视角参数，方便直接看到最佳观察位置
    o3d.visualization.draw_geometries([source_temp, target_temp],
                                      zoom=0.4459,
                                      front=[0.9288, -0.2951, -0.2242],
                                      lookat=[1.6784, 2.0612, 1.4451],
                                      up=[-0.3402, -0.9189, -0.1996])



# 2. 读取点云文件
source_path = "../data/open3d_demo_icp/cloud_bin_0.pcd"
target_path = "../data/open3d_demo_icp/cloud_bin_1.pcd"

source = o3d.io.read_point_cloud(source_path)
target = o3d.io.read_point_cloud(target_path)

# 3. 设置搜索阈值和初始变换矩阵
threshold = 0.02
trans_init = np.asarray([[0.862, 0.011, -0.507, 0.5],
                         [-0.139, 0.967, -0.215, 0.7],
                         [0.487, 0.255, 0.835, -1.4], 
                         [0.0, 0.0, 0.0, 1.0]])

# 4. 展示初始对齐效果
print("正在展示初始对齐效果（按 'Q' 键关闭窗口并退出）...")
draw_registration_result(source, target, trans_init)

print("Initial alignment")
evaluation = o3d.pipelines.registration.evaluate_registration(
    source, target, threshold, trans_init)
print(evaluation)

print("Apply point-to-point ICP")
reg_p2p = o3d.pipelines.registration.registration_icp(
    source, target, threshold, trans_init,
    o3d.pipelines.registration.TransformationEstimationPointToPoint())
print(reg_p2p)
print("Transformation is:")
print(reg_p2p.transformation)
draw_registration_result(source, target, reg_p2p.transformation)


reg_p2p = o3d.pipelines.registration.registration_icp(
    source, target, threshold, trans_init,
    o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000))
print(reg_p2p)
print("Transformation is:")
print(reg_p2p.transformation)
draw_registration_result(source, target, reg_p2p.transformation)

print("Apply point-to-plane ICP")
reg_p2l = o3d.pipelines.registration.registration_icp(
    source, target, threshold, trans_init,
    o3d.pipelines.registration.TransformationEstimationPointToPlane())
print(reg_p2l)
print("Transformation is:")
print(reg_p2l.transformation)
draw_registration_result(source, target, reg_p2l.transformation)

# # 5. 执行 ICP 配准
# # threshold 是对应点之间允许的最大距离，超过该距离的点对不参与优化。
# registration_result = o3d.pipelines.registration.registration_icp(
#     source,
#     target,
#     threshold,
#     trans_init,
#     o3d.pipelines.registration.TransformationEstimationPointToPoint(),
# )

# print("配准结果：")
# print(registration_result)
# print("变换矩阵：")
# print(registration_result.transformation)

# # 6. 展示 ICP 配准后的效果
# print("正在展示 ICP 配准结果（按 'Q' 键关闭窗口并退出）...")
# draw_registration_result(source, target, registration_result.transformation)
