import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.dlpack import to_dlpack
import cupoch
from sklearn.neighbors import NearestNeighbors
import open3d as o3d


class GraspPointFinder:
    def __init__(self, scene_points, voxel_size=0.005):
        self.voxel_size = voxel_size

        # 夹爪参数
        self.height = 0.02
        self.finger_width = 0.01
        self.delta_width = 0
        self.finger_length = 0.04
        self.depth = 0.02

        # 降采样点云
        scene_cloud = cupoch.geometry.PointCloud()
        scene_cloud.from_points_dlpack(to_dlpack(scene_points))
        # scene_cloud = scene_cloud.voxel_down_sample(voxel_size)
        self.scene_points = np.array(scene_cloud.points.cpu(),
                                     dtype=np.float32)
        
        self.k_neighbors = 10
    
    def find_nearest_points(self, grasp_group):
        """找到抓取位姿处，距离左右夹爪形心坐标最近的点及其法向量"""
      
        # 获取抓取位姿参数
        T = torch.from_numpy(grasp_group.translations).to(dtype=torch.float32, device='cuda')  # (K, 3)
        R = torch.from_numpy(grasp_group.rotations.reshape(-1, 3, 3)).to(dtype=torch.float32, device='cuda')  # (K, 3, 3)
        widths = torch.from_numpy(grasp_group.widths[:, None]).to(dtype=torch.float32, device='cuda')  # (K, 1)
        # T = torch.from_numpy(grasp_group.translations).to(dtype=torch.float32, device='cpu')
        # R = torch.from_numpy(grasp_group.rotations.reshape(-1, 3, 3)).to(dtype=torch.float32, device='cpu')
        # widths = torch.from_numpy(grasp_group.widths[:, None]).to(dtype=torch.float32, device='cpu')
        widths += self.delta_width  # Adjust width

        # 计算左右夹爪的形心坐标 (最终修正)
        left_finger_center = torch.cat([
            torch.full_like(widths, self.depth - self.finger_length),  # x: depth - finger_length
            -(widths / 2 + self.finger_width / 2),  # y: -(width / 2 + finger_width / 2)
            torch.full_like(widths, self.height / 2)  # z: height / 2
        ], dim=-1)  # (K, 3)

        right_finger_center = torch.cat([
            torch.full_like(widths, self.depth - self.finger_length),  # x: depth - finger_length
            (widths / 2 + self.finger_width / 2),  # y: (width / 2 + finger_width / 2)
            torch.full_like(widths, self.height / 2)  # z: height / 2
        ], dim=-1)  # (K, 3)

        # 将点云转换到相机坐标系
        points = torch.from_numpy(self.scene_points[None, ...]).to(dtype=torch.float32, device='cuda')  # (1, N, 3)
        # points = torch.from_numpy(self.scene_points[None, ...]).to(dtype=torch.float32, device='cpu')
        transformed_points = torch.matmul(points - T[:, None, :], R)  # (K, N, 3)

        # 计算每个点到左右手指形心的距离
        left_distances = torch.norm(transformed_points - left_finger_center[:, None, :], dim=-1)  # (K, N)
        right_distances = torch.norm(transformed_points - right_finger_center[:, None, :], dim=-1)  # (K, N)

        # 找到每个姿态中距离左手和右手形心最近的点
        nearest_left_indices = torch.argmin(left_distances, dim=1)  # (K,)
        nearest_right_indices = torch.argmin(right_distances, dim=1)  # (K,)

        # 提取最接近的点
        contact_left_points = transformed_points[torch.arange(T.size(0)), nearest_left_indices]  # (K, 3)
        contact_right_points = transformed_points[torch.arange(T.size(0)), nearest_right_indices]  # (K, 3)

        # 将接触点转换回原始坐标系
        R_transposed = R.transpose(1, 2)  # (K, 3, 3)
        original_contact_left_points = torch.bmm(contact_left_points.unsqueeze(1), R_transposed).squeeze(1) + T  # (K, 3)
        original_contact_right_points = torch.bmm(contact_right_points.unsqueeze(1), R_transposed).squeeze(1) + T  # (K, 3)

        # 使用最近邻算法在原始坐标系中查找最近的点
        nbrs = NearestNeighbors(n_neighbors=self.k_neighbors, algorithm='auto').fit(self.scene_points)
        _, left_indices = nbrs.kneighbors(original_contact_left_points.cpu().numpy())  # (K, m)
        _, right_indices = nbrs.kneighbors(original_contact_right_points.cpu().numpy())  # (K, m)

        # 提取最近邻点
        nearest_m_left_points = self.scene_points[left_indices]  # (K, m, 3)
        nearest_m_right_points = self.scene_points[right_indices]  # (K, m, 3)

        # 将 nearest_m_left_points 和 nearest_m_right_points 转换为 torch.Tensor 类型
        nearest_m_left_points = torch.from_numpy(nearest_m_left_points).to(dtype=torch.float32, device='cuda')  # (K, m, 3)
        nearest_m_right_points = torch.from_numpy(nearest_m_right_points).to(dtype=torch.float32, device='cuda')  # (K, m, 3)
        # nearest_m_left_points = torch.from_numpy(nearest_m_left_points).to(dtype=torch.float32, device='cpu')
        # nearest_m_right_points = torch.from_numpy(nearest_m_right_points).to(dtype=torch.float32, device='cpu')

        # 计算每个点的法向量
        left_normals = self.compute_normals(nearest_m_left_points)  # (K, 3)
        right_normals = self.compute_normals(nearest_m_right_points)  # (K, 3)

        return original_contact_left_points, original_contact_right_points, left_normals, right_normals


    def compute_loss(self, grasp_group):
        """计算抓取损失 R_PCR"""
        left_points, right_points, left_normals, right_normals = self.find_nearest_points(grasp_group)

        # 计算c1到c2的向量
        c1_to_c2 = right_points - left_points  # c1 -> c2
        c2_to_c1 = left_points - right_points  # c2 -> c1

        right_normals = torch.tensor(right_normals).to(c1_to_c2.device)   # 将right_normals转换为tensor
        left_normals = torch.tensor(left_normals).to(c2_to_c1.device)  # 将left_normals转换为tensor

        # 计算余弦相似度，取绝对值
        cos1 = F.cosine_similarity(c1_to_c2, right_normals, dim=-1)  # c1->c2 与 c2 法向量
        cos2 = F.cosine_similarity(c2_to_c1, left_normals, dim=-1)   # c2->c1 与 c1 法向量
        RA = 1 - 0.5 * (torch.abs(cos1) + torch.abs(cos2))  # 方向相似性损失，取绝对值
        RA = RA.squeeze()

        # 计算投影距离，取绝对值
        d1 = torch.sum(c1_to_c2 * right_normals, dim=-1, keepdim=True)  # c1->c2 在 c2 法向量方向上的投影
        d2 = torch.sum(c2_to_c1 * left_normals, dim=-1, keepdim=True)   # c2->c1 在 c1 法向量方向上的投影
        d1 = torch.abs(d1)  # 取绝对值
        d2 = torch.abs(d2)  # 取绝对值
        max_d = torch.max(d1, d2)  # 选取较大的投影距离
        widths = torch.from_numpy(grasp_group.widths[:, None]).to(dtype=torch.float32, device='cuda')
        # max_d = max_d.to('cpu')
        # widths = torch.from_numpy(grasp_group.widths[:, None]).to(dtype=torch.float32, device='cpu')

        RD = F.smooth_l1_loss(max_d, widths, reduction='none')  # Smooth L1 Loss 计算 RD
        RD = torch.sigmoid(RD)
        RD = RD.squeeze()

        # 计算最终加权损失
        RG = 0.7 * RA + 0.3 * RD

        # 置信度加权求和
        scores = torch.from_numpy(grasp_group.scores[:, None]).to(dtype=torch.float32, device='cuda').squeeze()
        # scores = torch.from_numpy(grasp_group.scores[:, None]).to(dtype=torch.float32, device='cpu').squeeze()
        R_PCR = torch.sum(scores * RG) / torch.sum(scores)

        return R_PCR.cpu().item()  # 返回最终损失值

    
    def visualize_grasp_points(self, grasp_group):
        """可视化场景点云，夹爪，最近点和法向量"""
        
        # 计算最近的点和法向量
        contact_left_points, contact_right_points, left_normals, right_normals = self.find_nearest_points(grasp_group)

        # 将场景点云转换为 Open3D 点云对象
        pcd_scene = o3d.geometry.PointCloud()
        pcd_scene.points = o3d.utility.Vector3dVector(self.scene_points)

        # 创建抓取夹爪和接触点的点云
        grasp_left_points = contact_left_points.cpu().numpy()
        grasp_right_points = contact_right_points.cpu().numpy()

        pcd_left_grasp = o3d.geometry.PointCloud()
        pcd_left_grasp.points = o3d.utility.Vector3dVector(grasp_left_points)
        pcd_left_grasp.paint_uniform_color([1, 0, 0])  # 红色

        pcd_right_grasp = o3d.geometry.PointCloud()
        pcd_right_grasp.points = o3d.utility.Vector3dVector(grasp_right_points)
        pcd_right_grasp.paint_uniform_color([0, 0, 1])  # 蓝色

       # 创建法向量的线框
        lines_left = []
        for i, point in enumerate(grasp_left_points):
            lines_left.append([i, i + len(grasp_left_points)])  # 从点到法向量端点
        lines_left = np.array(lines_left)

        lines_left_set = o3d.geometry.LineSet()
        lines_left_set.points = o3d.utility.Vector3dVector(np.concatenate([grasp_left_points, grasp_left_points + left_normals * 0.05], axis=0))
        lines_left_set.lines = o3d.utility.Vector2iVector(lines_left)  # 正确赋值

        # 同样处理右侧的法向量
        lines_right = []
        for i, point in enumerate(grasp_right_points):
            lines_right.append([i, i + len(grasp_right_points)])  # 从点到法向量端点
        lines_right = np.array(lines_right)

        lines_right_set = o3d.geometry.LineSet()
        lines_right_set.points = o3d.utility.Vector3dVector(np.concatenate([grasp_right_points, grasp_right_points + right_normals * 0.05], axis=0))
        lines_right_set.lines = o3d.utility.Vector2iVector(lines_right)  # 正确赋值

        # 绘制场景点云和抓取点云
        o3d.visualization.draw_geometries([pcd_scene, pcd_left_grasp, pcd_right_grasp, lines_left_set, lines_right_set])

    @staticmethod
    def compute_normals(points):
        """计算点云的法向量"""
        normals = []
        for i in range(points.shape[0]):  # 对每个抓取姿态的点云计算法向量
            point_cloud = points[i]  # (m, 3)
            if point_cloud.shape[0] < 3:
                normals.append(np.zeros((1, 3)))  # 如果点数少于3，法向量为空
            else:
                # 计算法向量，使用SVD进行PCA
                point_cloud_cpu = point_cloud.cpu().numpy()  # 将tensor转换为numpy
                pca = np.linalg.svd(point_cloud_cpu - point_cloud_cpu.mean(axis=0), full_matrices=False)
                normals.append(pca[2][-1])  # 取最小特征值对应的特征向量
        return np.array(normals)  # 返回(K, 3)的法向量


    
def PC_Regularization(points_all: torch.Tensor, pred_gg):
    """
    计算基于点云的抓取正则化损失 (PC Regularization)。
    
    参数:
    - points_all: (torch.Tensor) 输入点云数据，形状为 (N, C)，至少包含前三个通道 (x, y, z)。
    - pred_gg: 预测的抓取信息 (grasp group)。
    - mode: (str) 选择抓取模式 (默认 'regnet')，决定夹爪参数。

    返回:
    - R_PCR: 计算得到的抓取损失值。
    """
    T = torch.from_numpy(pred_gg.translations).to(dtype=torch.float32, device='cuda')  # (K, 3)
    R = torch.from_numpy(pred_gg.rotations.reshape(-1, 3, 3)).to(dtype=torch.float32, device='cuda')  # (K, 3, 3)
    # 检查 T R 是否包含 NaN 或 Inf
    valid_mask = ~torch.isnan(T).any(dim=1) & ~torch.isinf(T).any(dim=1)
    valid_mask &= (~torch.isnan(R).view(R.shape[0], -1).any(dim=1)) & (~torch.isinf(R).view(R.shape[0], -1).any(dim=1))
    if not valid_mask.all():
        pred_gg = pred_gg[valid_mask.cpu().numpy()]
    
    # 仅保留(x, y, z)坐标
    if points_all.shape[1] == 35:

        cloud = points_all[:, :3].clone()

        # 创建 GraspPointFinder 实例
        grasp_finder = GraspPointFinder(cloud, voxel_size=0.005)

        # 计算 R_PCR 损失
        R_PCR = grasp_finder.compute_loss(pred_gg) 

    elif points_all.shape[2] == 35:

        R_PCR = 0

        for i in range(points_all.shape[0]):

            cloud = points_all[i, :, :3].clone().detach()

            # 创建 GraspPointFinder 实例
            grasp_finder = GraspPointFinder(cloud, voxel_size=0.01)

            if len(pred_gg) <= 60:
                # 计算 R_PCR 损失
                R_PCR += grasp_finder.compute_loss(pred_gg) 
            else:
                R_PCR += grasp_finder.compute_loss(pred_gg[:60]) 
                # grasp_finder.visualize_grasp_points(pred_gg[:5])

        R_PCR /= points_all.shape[0]

        torch.cuda.empty_cache()

    return 0.4 * R_PCR
