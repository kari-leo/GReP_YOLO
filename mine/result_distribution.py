import os
import numpy as np
import torch


def generate_anchor_grid(anchor_points, img_width=1280, img_height=720, grid_size=(6, 8)):
    # 定义子区域的大小
    grid_width = img_width // grid_size[1]
    grid_height = img_height // grid_size[0]
    
    # 创建一个全为0的结果张量，大小为 (6, 8)
    result_grid = torch.zeros(grid_size, dtype=torch.int)
    
    # 遍历所有锚点
    for point in anchor_points:
        x, y = point
        
        # 计算该锚点属于哪个子区域
        grid_x = int(x // grid_width)
        grid_y = int(y // grid_height)
        
        # 设定该子区域的值为1
        result_grid[grid_y, grid_x] = 1
    
    return result_grid

def process_npz_file(npz_file):
    """
    处理每个npz文件，返回一个(6, 8)形状的张量
    假设你要对npz文件进行的处理放在这里
    """
    # 加载npz文件
    data = np.load(npz_file)
    
    # 假设对数据进行处理，返回一个(6, 8)的张量
    # 生成结果张量
    result = generate_anchor_grid(data['centers_2d'])
    return result

def process_all_npz_files(root_dir):
    """
    遍历6d_dataset目录，处理所有的npz文件，并保存为.npy文件
    """
    # 遍历所有scene文件夹
    for scene_name in os.listdir(root_dir):
        scene_path = os.path.join(root_dir, scene_name)
        
        if os.path.isdir(scene_path):
            grasp_labels_path = os.path.join(scene_path, 'grasp_labels')
            result_distribution_path = os.path.join(scene_path, 'resault_distribution')
            
            # 确保结果目录存在，如果不存在则创建
            if not os.path.exists(result_distribution_path):
                os.makedirs(result_distribution_path)
            
            # 遍历grasp_labels文件夹下的所有.npz文件
            for file_name in os.listdir(grasp_labels_path):
                if file_name.endswith('.npz'):
                    npz_file = os.path.join(grasp_labels_path, file_name)
                    
                    # 处理npz文件
                    result = process_npz_file(npz_file)
                    
                    # 生成保存结果的.npy文件路径
                    result_file = os.path.join(result_distribution_path, file_name.replace('.npz', '.npy'))
                    
                    # 保存结果为.npy文件
                    np.save(result_file, result)
                    print(f"Saved: {result_file}")

# 设置根目录
root_dir = '/home/johnny/goal_grasp_projects/mine_HGGD_ws/src/HGGD/6dto2drefine_realsense/6d_dataset'

# 执行文件处理
process_all_npz_files(root_dir)
