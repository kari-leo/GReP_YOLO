# 可视化抓取角度的筛选标准
import open3d as o3d
import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

# ---------- 通用函数 ----------

def angle_between_rotations(R1, R2):
    R_diff = R1.T @ R2
    cos_theta = np.clip((np.trace(R_diff) - 1) / 2, -1.0, 1.0)
    return np.arccos(cos_theta)

# ---------- Part 1: 坐标系可视化（柱形箭头） ----------

def create_arrow_axes(R_matrix, origin=np.zeros(3), length=0.1, radius=0.003):
    """用柱形箭头展示一个坐标系：X红 Y绿 Z蓝"""
    colors = [np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])]
    axes = []
    for i in range(3):
        arrow = o3d.geometry.TriangleMesh.create_arrow(
            cylinder_radius=radius, cone_radius=radius * 1.5,
            cylinder_height=length * 0.8, cone_height=length * 0.2
        )
        arrow.paint_uniform_color(colors[i])
        axis_dir = R_matrix[:, i]
        rot = R.align_vectors([axis_dir], [[0, 0, 1]])[0].as_matrix()
        arrow.rotate(rot, center=np.zeros(3))
        arrow.translate(origin)
        axes.append(arrow)
    return axes

def visualize_coordinate_axes():
    R_ref = np.eye(3)
    threshold = np.pi / 4
    geoms = []
    
    # 参考姿态
    ref_axes = create_arrow_axes(R_ref, length=0.15)
    for arrow in ref_axes:
        arrow.paint_uniform_color([0.7, 0.7, 0.7])
    geoms.extend(ref_axes)

    # 合法姿态
    for _ in range(80):
        R_sample = R.random().as_matrix()
        if angle_between_rotations(R_ref, R_sample) < threshold:
            geoms.extend(create_arrow_axes(R_sample))

    # 边界姿态
    boundary_axes = [
        np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1]),
        np.array([1, 1, 0]) / np.sqrt(2),
        np.array([1, 0, 1]) / np.sqrt(2),
        np.array([0, 1, 1]) / np.sqrt(2),
        np.array([1, 1, 1]) / np.sqrt(3)
    ]
    for axis in boundary_axes:
        R_b = R.from_rotvec(axis * threshold).as_matrix()
        geoms.extend(create_arrow_axes(R_b))
    
    o3d.visualization.draw_geometries(geoms, window_name="合法姿态坐标系（柱形箭头）")


# ---------- Part 2: 姿态角度热图 ----------

def fibonacci_sphere(samples=1000):
    points = []
    phi = np.pi * (3. - np.sqrt(5.))
    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2
        radius = np.sqrt(1 - y * y)
        theta = phi * i
        x = np.cos(theta) * radius
        z = np.sin(theta) * radius
        points.append([x, y, z])
    return np.array(points)

def visualize_heatmap():
    points = fibonacci_sphere(2000)
    R_ref = np.eye(3)
    angles = []

    for axis in points:
        R_test = R.from_rotvec(axis * np.pi / 4).as_matrix()
        diff = angle_between_rotations(R_ref, R_test)
        angles.append(diff)

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=angles, cmap='plasma', s=8)
    plt.colorbar(sc, label="Angle to Reference (rad)")
    ax.set_title("姿态空间夹角热图")
    ax.set_box_aspect([1,1,1])
    plt.show()


# ---------- Part 3: 姿态夹角锥体 ----------

def visualize_cone(threshold=np.pi / 4):
    height = 0.2
    radius = np.tan(threshold) * height
    cone = o3d.geometry.TriangleMesh.create_cone(radius=radius, height=height, resolution=60)
    cone.compute_vertex_normals()
    cone.paint_uniform_color([1.0, 0.6, 0.0])  # 橙色
    cone.translate([0, 0, 0])
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    o3d.visualization.draw_geometries([cone, frame], window_name="姿态空间合法锥体")

# ---------- 运行全部 ----------

if __name__ == "__main__":
    visualize_coordinate_axes()  # 图1：姿态坐标系
    visualize_heatmap()          # 图2：角度热图
    visualize_cone()             # 图3：合法夹角锥体





# # 校准标定的到的TF旋转阵
# import numpy as np

# # 原始矩阵
# T_cam_tool = np.array([
#     [-0.05460029, -0.99840556,  0.01432292,  0.04845816],
#     [ 0.99788222, -0.05506832, -0.03462028, -0.02659892],
#     [ 0.03535382,  0.01240231,  0.9992979,   0.04      ],
#     [ 0.,          0.,          0.,          1.        ]
# ])

# # 角度（弧度）
# theta_x = np.radians(0)
# theta_y = np.radians(3)
# theta_z = np.radians(-3)

# # 绕x轴旋转
# Rx = np.array([
#     [1, 0, 0],
#     [0, np.cos(theta_x), -np.sin(theta_x)],
#     [0, np.sin(theta_x),  np.cos(theta_x)]
# ])

# # 绕y轴旋转
# Ry = np.array([
#     [ np.cos(theta_y), 0, np.sin(theta_y)],
#     [0, 1, 0],
#     [-np.sin(theta_y), 0, np.cos(theta_y)]
# ])

# # 绕z轴旋转
# Rz = np.array([
#     [np.cos(theta_z), -np.sin(theta_z), 0],
#     [np.sin(theta_z),  np.cos(theta_z), 0],
#     [0, 0, 1]
# ])

# # 总旋转（右乘顺序：Rx -> Ry -> Rz）
# R_delta = Rz @ Ry @ Rx

# # 构造T_delta
# T_delta = np.eye(4)
# T_delta[:3, :3] = R_delta

# # 新的变换矩阵
# T_new = T_cam_tool @ T_delta

# # 打印为嵌套列表格式
# print("[")
# for row in T_new:
#     print("    [", ", ".join(f"{v:.8f}" for v in row), "],")
# print("]")


# import torch

# def random_rotation_matrix():
#     """ 生成一个随机旋转矩阵 (3x3) """
#     angle = torch.rand(1, device="cuda") * 2 * torch.pi  # 生成 0 到 2π 之间的随机角度
#     axis = torch.rand(1, 3, device="cuda") - 0.5  # 形状为 (1, 3)
#     axis = torch.nn.functional.normalize(axis, dim=1).squeeze(0)  # 归一化并去掉 batch 维度

#     cos_a = torch.cos(angle)
#     sin_a = torch.sin(angle)
#     one_minus_cos = 1 - cos_a

#     x, y, z = axis
#     R = torch.tensor([
#         [cos_a + x*x*one_minus_cos, x*y*one_minus_cos - z*sin_a, x*z*one_minus_cos + y*sin_a],
#         [y*x*one_minus_cos + z*sin_a, cos_a + y*y*one_minus_cos, y*z*one_minus_cos - x*sin_a],
#         [z*x*one_minus_cos - y*sin_a, z*y*one_minus_cos + x*sin_a, cos_a + z*z*one_minus_cos]
#     ], dtype=torch.float32, device="cuda")
    
#     return R

# # 生成至少 5 个随机点
# num_points = 5
# points = torch.rand((1, num_points, 3), dtype=torch.float32, device="cuda")  # (1, N, 3)

# # 生成 2 组随机旋转矩阵 R 和平移向量 T
# num_transforms = 2
# T = torch.rand((num_transforms, 3), dtype=torch.float32, device="cuda")  # (K, 3)
# R = torch.stack([random_rotation_matrix() for _ in range(num_transforms)])  # (K, 3, 3)

# # 第一次变换: 先减去 T, 然后乘以 R
# transformed_points = torch.matmul(points - T[:, None, :], R)  # (K, N, 3)

# # 反变换: 先乘 R.T, 然后加回 T
# R_inv = R.transpose(1, 2)  # R 的转置
# recovered_points = torch.bmm(transformed_points, R_inv) + T[:, None, :]  # (K, N, 3)

# # 再次变换: 重新执行 torch.matmul
# transformed_again = torch.matmul(recovered_points - T[:, None, :], R)  # (K, N, 3)

# # 比较第一次变换和第三次变换的结果
# is_equal = torch.allclose(transformed_points, transformed_again, atol=1e-6)
# print("变换一致:", is_equal)
# print('transformed_again:', transformed_again)
# print('transformed_points:', transformed_points)



# 可视化转换测试
# import numpy as np
# import torch
# import torch.nn.functional as F
# import matplotlib.pyplot as plt
# from sklearn.neighbors import NearestNeighbors
# from mpl_toolkits.mplot3d import Axes3D

# # 创建简单的点云数据
# N = 500  # 点的数量
# scene_points = np.random.rand(N, 3) * 0.1  # 随机生成点云数据

# # 模拟抓取位姿参数
# class GraspGroup:
#     def __init__(self):
#         self.translations = np.random.rand(10, 3) * 0.1  # 随机的10个平移向量
#         self.rotations = np.random.rand(10, 3, 3)  # 随机的10个旋转矩阵
#         self.widths = np.random.rand(10) * 0.02 + 0.02  # 随机的宽度值
#         self.scores = np.random.rand(10)  # 随机的置信度值

# grasp_group = GraspGroup()

# # 计算抓取点和法向量
# class GraspPointFinder:
#     def __init__(self, scene_points, voxel_size=0.005):
#         self.voxel_size = voxel_size
#         self.height = 0.02
#         self.finger_width = 0.01
#         self.delta_width = 0
#         self.finger_length = 0.04
#         self.depth = 0.02
#         self.scene_points = scene_points
#         self.k_neighbors = 10

#     def find_nearest_points(self, grasp_group):
#         T = torch.from_numpy(grasp_group.translations).float().cuda()
#         R = torch.from_numpy(grasp_group.rotations).float().cuda()
#         widths = torch.from_numpy(grasp_group.widths[:, None]).float().cuda()
#         widths += self.delta_width

#         left_finger_center = torch.cat([
#             torch.full_like(widths, self.depth - self.finger_length),
#             -(widths / 2 + self.finger_width / 2),
#             torch.full_like(widths, self.height / 2)
#         ], dim=-1)

#         right_finger_center = torch.cat([
#             torch.full_like(widths, self.depth - self.finger_length),
#             (widths / 2 + self.finger_width / 2),
#             torch.full_like(widths, self.height / 2)
#         ], dim=-1)

#         points = torch.from_numpy(self.scene_points[None, ...]).float().cuda()
#         transformed_points = torch.matmul(points - T[:, None, :], R)
        
#         left_distances = torch.norm(transformed_points - left_finger_center[:, None, :], dim=-1)
#         right_distances = torch.norm(transformed_points - right_finger_center[:, None, :], dim=-1)

#         nearest_left_indices = torch.argmin(left_distances, dim=1)
#         nearest_right_indices = torch.argmin(right_distances, dim=1)

#         contact_left_points = transformed_points[torch.arange(T.size(0)), nearest_left_indices]
#         contact_right_points = transformed_points[torch.arange(T.size(0)), nearest_right_indices]

#         original_contact_left_points = torch.bmm(contact_left_points.unsqueeze(1), R).squeeze(1) + T
#         original_contact_right_points = torch.bmm(contact_right_points.unsqueeze(1), R).squeeze(1) + T

#         nbrs = NearestNeighbors(n_neighbors=self.k_neighbors).fit(self.scene_points)
#         _, left_indices = nbrs.kneighbors(original_contact_left_points.cpu().numpy())
#         _, right_indices = nbrs.kneighbors(original_contact_right_points.cpu().numpy())

#         nearest_m_left_points = self.scene_points[left_indices]
#         nearest_m_right_points = self.scene_points[right_indices]

#         nearest_m_left_points = torch.from_numpy(nearest_m_left_points).float().cuda()
#         nearest_m_right_points = torch.from_numpy(nearest_m_right_points).float().cuda()

#         T_expanded = T[:, None, :]
#         T_expanded = T_expanded.expand(-1, nearest_m_left_points.size(1), -1)

#         R_expanded = R[:, None, :, :]
#         R_expanded = R_expanded.expand(-1, nearest_m_left_points.size(1), -1, -1)

#         nearest_m_left_points_transformed = torch.matmul(nearest_m_left_points[:, :, None, :], R_expanded)
#         nearest_m_left_points_transformed = nearest_m_left_points_transformed.squeeze(2)
#         nearest_m_left_points_transformed = nearest_m_left_points_transformed + T_expanded

#         nearest_m_right_points_transformed = torch.matmul(nearest_m_right_points[:, :, None, :], R_expanded)
#         nearest_m_right_points_transformed = nearest_m_right_points_transformed.squeeze(2)
#         nearest_m_right_points_transformed = nearest_m_right_points_transformed + T_expanded

#         left_normals = self.compute_normals(nearest_m_left_points_transformed)
#         right_normals = self.compute_normals(nearest_m_right_points_transformed)

#         return contact_left_points.cpu().numpy(), contact_right_points.cpu().numpy(), left_normals, right_normals

#     @staticmethod
#     def compute_normals(points):
#         normals = []
#         for i in range(points.shape[0]):
#             point_cloud = points[i]
#             if point_cloud.shape[0] < 3:
#                 normals.append(np.zeros((1, 3)))
#             else:
#                 point_cloud_cpu = point_cloud.cpu().numpy()
#                 pca = np.linalg.svd(point_cloud_cpu - point_cloud_cpu.mean(axis=0), full_matrices=False)
#                 normals.append(pca[2][-1])
#         return np.array(normals)

# # 创建 GraspPointFinder 实例
# grasp_finder = GraspPointFinder(scene_points)

# # 计算最近点和法向量
# left_points, right_points, left_normals, right_normals = grasp_finder.find_nearest_points(grasp_group)

# # 可视化
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(scene_points[:, 0], scene_points[:, 1], scene_points[:, 2], c='gray', label="Scene Points")

# # 可视化左夹爪接触点
# ax.quiver(left_points[:, 0], left_points[:, 1], left_points[:, 2],
#           left_normals[:, 0], left_normals[:, 1], left_normals[:, 2],
#           length=0.01, color='r', label="Left Grasp Contact Points")

# # 可视化右夹爪接触点
# ax.quiver(right_points[:, 0], right_points[:, 1], right_points[:, 2],
#           right_normals[:, 0], right_normals[:, 1], right_normals[:, 2],
#           length=0.01, color='b', label="Right Grasp Contact Points")

# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# ax.legend()
# plt.show()


# import torch

# def transform_points(points, R, T):
#     """从抓取坐标系转换到相机坐标系"""
#     return torch.matmul(points, R) + T

# def inverse_transform_points(points, R, T):
#     """从相机坐标系转换回抓取坐标系"""
#     return torch.matmul(points - T, R.T)

# # 定义两个 3D 点（在抓取坐标系下）
# points = torch.tensor([
#     [0.02, -0.01, 0.015],  # 点1
#     [0.01, 0.02, 0.025]   # 点2
# ], dtype=torch.float32)

# # 定义旋转矩阵（绕Z轴旋转30°）
# theta = torch.tensor(30.0).deg2rad()
# R = torch.tensor([
#     [torch.cos(theta), -torch.sin(theta), 0],
#     [torch.sin(theta),  torch.cos(theta), 0],
#     [0, 0, 1]
# ], dtype=torch.float32)

# # 定义平移向量
# T = torch.tensor([0.05, -0.02, 0.03], dtype=torch.float32)

# # **坐标变换**
# transformed_points = transform_points(points, R, T)
# print("转换到相机坐标系的点:\n", transformed_points)

# # **逆变换**
# recovered_points = inverse_transform_points(transformed_points, R, T)
# print("转换回抓取坐标系的点:\n", recovered_points)

# # 检查误差
# error = torch.norm(recovered_points - points, dim=1)
# print("变换误差:\n", error)


# import torch

# def load_checkpoint(file_path):
#     checkpoint = torch.load(file_path, map_location='cpu')  # 加载到CPU，避免显存问题
#     print("Checkpoint Keys:", checkpoint.keys())  # 打印所有键
#     for key, value in checkpoint.items():
#         print(f"\nKey: {key}")
#         if isinstance(value, dict):
#             print(f"  Contains {len(value)} keys: {list(value.keys())[:5]}...")  # 显示前5个键
#         elif isinstance(value, torch.Tensor):
#             print(f"  Tensor Shape: {value.shape}")  # 如果是Tensor，打印形状
#         else:
#             print(f"  Type: {type(value)}")  # 其他类型，打印类型
#     return checkpoint

# # 示例调用
# file_path = '/home/johnny/goal_grasp_projects/mine_HGGD_ws/src/HGGD/logs/full/250301_141628_realsense/epoch_4_iou_0.992_cover_0.663'  # 修改为你的checkpoint路径
# checkpoint_data = load_checkpoint(file_path)
