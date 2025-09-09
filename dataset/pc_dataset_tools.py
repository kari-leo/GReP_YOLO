from functools import wraps
from time import time
from typing import List

import numpy as np
import torch
import torch.nn.functional as nnf
from numba import njit
from pytorch3d.ops import ball_query, knn_points, sample_farthest_points
from pytorch3d.ops.utils import masked_gather

from .config import get_camera_intrinsic
from .utils import convert_2d_to_3d, euclid_distance


def timing(f):

    @wraps(f)
    def wrap(*args, **kw):
        ts = time.time()
        result = f(*args, **kw)
        te = time.time()
        print(f'func:{f.__name__} took: {te-ts} sec')
        return result

    return wrap


def feature_fusion(points, perpoint_features, xyzs, mode='knn'):
    # points: [B, N, 3]  perpoint_features: [B, C, H, W]  xyzs: [B, 3, H, W]
    perpoint_features = torch.concat([xyzs, perpoint_features], 1)
    B, C = points.shape[0], perpoint_features.shape[1]
    perpoint_features = perpoint_features.reshape((B, C, -1)).transpose(1, 2)
    # knn neigbor selection
    if mode == 'knn':
        _, nn_idxs, _ = knn_points(points[..., :3],
                                   perpoint_features[..., :3],
                                   K=8)
    else:
        _, nn_idxs, _ = ball_query(points[..., :3],
                                   perpoint_features[..., :3],
                                   radius=0.2,
                                   K=32,
                                   return_nn=False)
    nn_features = masked_gather(perpoint_features[..., 3:], nn_idxs)
    # max pooling
    nn_features = nn_features.max(2)[0]
    # concat
    points_all = torch.concat([points, nn_features], axis=2)
    return points_all


def get_group_pc(pc: torch.Tensor,
                 local_centers: List,
                 group_num,
                 grasp_widths,
                 min_points=32,
                 is_training=True,
                 is_wide=False):
    batch_size, feature_len = pc.shape[0], pc.shape[2]
    pc_group = torch.zeros((0, group_num, feature_len),
                           dtype=torch.float32,
                           device='cuda')
    wide_group_num = group_num * 2
    wide_pc_group = torch.zeros((0, wide_group_num, feature_len),
                           dtype=torch.float32,
                           device='cuda') if is_wide else None
    valid_local_centers = []
    valid_center_masks = []
    # get the points around one scored center
    for i in range(batch_size):
        # deal with empty input
        if len(local_centers[i]) == 0:
            # no need to append pc_group
            valid_local_centers.append(local_centers[i])
            valid_center_masks.append(
                torch.ones((0, ), dtype=torch.bool, device='cuda'))
            continue
        # cal distance and get masks (for all centers)
        dis = euclid_distance(local_centers[i], pc[i])
        # using grasp width for ball segment
        grasp_widths_tensor = torch.from_numpy(grasp_widths[i]).to(
            device='cuda', dtype=torch.float32)[..., None]
        # add noise when trainning
        width_scale = 1
        if is_training:
            # 0.8 ~ 1.2
            width_scale = 0.8 + 0.4 * torch.rand(
                (len(grasp_widths_tensor), 1), device='cuda')
            # width_scale = 1.1 * torch.rand(
            #     (len(grasp_widths_tensor), 1), device='cuda')
        masks = (dis < grasp_widths_tensor * width_scale)
        # select valid center from all center
        center_cnt = len(local_centers[i])
        valid_mask = torch.ones((center_cnt, ), dtype=torch.bool).cuda()
        # concat pc first
        max_pc_cnt = max(group_num, masks.sum(1).max())
        partial_pcs = torch.zeros((center_cnt, max_pc_cnt, feature_len),
                                  device='cuda')
        lengths = torch.zeros((center_cnt, ), device='cuda')

        # 为wide采样的准备
        if is_wide:
            wide_scale = 2
            wide_masks = (dis < grasp_widths_tensor * width_scale * wide_scale) #在2倍抓取宽度的范围内进行点采样
            wide_min_points = min_points * 3
            wide_max_pc_cnt = max(wide_group_num, wide_masks.sum(1).max())
            wide_partial_pcs = torch.zeros((center_cnt, wide_max_pc_cnt, feature_len),
                                    device='cuda')
            wide_lengths = torch.zeros((center_cnt, ), device='cuda')
        
        for j in range(center_cnt):
            # seg points
            partial_points = pc[i, masks[j]]
            point_cnt = partial_points.shape[0]
            if point_cnt < group_num:
                if point_cnt > min_points:
                    idxs = torch.randint(point_cnt, (group_num, ),
                                         device='cuda')
                    # idxs = np.random.choice(point_cnt, group_num, replace=True)
                    partial_points = partial_points[idxs]
                    point_cnt = group_num
                else:
                    valid_mask[j] = False
                    lengths[j] = group_num
                    continue
            partial_pcs[j, :point_cnt] = partial_points
            lengths[j] = point_cnt
            
        if is_wide:
            for j in range(center_cnt):
                # wide部分
                wide_partial_points = pc[i, wide_masks[j]]
                wide_point_cnt = wide_partial_points.shape[0]
                if wide_point_cnt < wide_group_num:
                    if wide_point_cnt > wide_min_points:
                        idxs = torch.randint(wide_point_cnt, (wide_group_num, ),
                                            device='cuda')
                        wide_partial_points = wide_partial_points[idxs]
                        wide_point_cnt = wide_group_num
                    else:
                        valid_mask[j] = False
                        wide_lengths[j] = wide_group_num
                        continue
                wide_partial_pcs[j, :wide_point_cnt] = wide_partial_points
                wide_lengths[j] = wide_point_cnt

        # add a little noise to avoid repeated points
        partial_pcs[..., :3] += torch.randn(partial_pcs.shape[:-1] + (3, ),
                                            device='cuda') * 5e-4
        # doing fps
        _, idxs = sample_farthest_points(partial_pcs[..., :3],
                                         lengths=lengths,
                                         K=group_num,
                                         random_start_point=True)
        # mv center of pc to (0, 0, 0), stack to pc_group
        temp_idxs = idxs[..., None].repeat(1, 1, feature_len)
        cur_pc = torch.gather(partial_pcs, 1, temp_idxs)
        cur_pc = cur_pc[valid_mask]
        cur_pc[..., :3] = cur_pc[..., :3] - local_centers[i][valid_mask][:,
                                                                         None]
        pc_group = torch.concat([pc_group, cur_pc], 0)

        # wide同样加噪声，最远点采样和平移
        if is_wide:
            # print("wide: ", wide_lengths.shape)
            # print("origin: ", lengths.shape)
            # wide也加噪声
            wide_partial_pcs[..., :3] += torch.randn(wide_partial_pcs.shape[:-1] + (3, ),
                                                device='cuda') * 5e-4
            # doing fps
            _, idxs = sample_farthest_points(wide_partial_pcs[..., :3],
                                            lengths=wide_lengths,
                                            K=wide_group_num,
                                            random_start_point=True)
            # mv center of pc to (0, 0, 0), stack to pc_group
            temp_idxs = idxs[..., None].repeat(1, 1, feature_len)
            cur_pc = torch.gather(wide_partial_pcs, 1, temp_idxs)
            cur_pc = cur_pc[valid_mask]
            cur_pc[..., :3] = cur_pc[..., :3] - local_centers[i][valid_mask][:,
                                                                            None]
            wide_pc_group = torch.concat([wide_pc_group, cur_pc], 0)

        # stack pc and get valid center list
        valid_local_centers.append(local_centers[i][valid_mask])
        valid_center_masks.append(valid_mask)

    if is_wide:
        return pc_group, wide_pc_group, valid_local_centers, valid_center_masks
    else:
        return pc_group, valid_local_centers, valid_center_masks


def center2dtopc(rect_ggs: List,
                 center_num,
                 depths: torch.Tensor,
                 output_size,
                 append_random_center=True,
                 is_training=True):
    # add extra axis when valid, avoid dim errors
    batch_size = depths.shape[0]
    # print(depths.shape)
    center_batch_pc = []

    scale_x, scale_y = 1280 / output_size[0], 720 / output_size[1]
    for i in range(batch_size):
        center_2d = rect_ggs[i].centers.copy()
        center_depth = rect_ggs[i].depths.copy()

        # add random center when local max count not enough
        if append_random_center and len(center_2d) < center_num:
            # print(f'current center_2d == {len(center_2d)}. using random center')
            random_local_max = np.random.rand(center_num - len(center_2d), 2)
            random_local_max = np.vstack([
                (random_local_max[:, 0] * output_size[0]).astype(np.int32),
                (random_local_max[:, 1] * output_size[1]).astype(np.int32)
            ]).T
            center_2d = np.vstack([center_2d, random_local_max])

        # scale
        center_2d[:, 0] = center_2d[:, 0] * scale_x
        center_2d[:, 1] = center_2d[:, 1] * scale_y
        # mask d != 0
        d = depths[i, center_2d[:, 0], center_2d[:, 1]]
        mask = (d != 0)
        # convert
        intrinsics = get_camera_intrinsic()
        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        cx, cy = intrinsics[0, 2], intrinsics[1, 2]
        center_tensor = torch.from_numpy(center_2d).float().cuda()
        # add delta depth
        delta_d = torch.from_numpy(center_depth).cuda()
        z = (d[mask] + delta_d[mask]) / 1000.0
        x = z / fx * (center_tensor[mask, 0] - cx)
        y = z / fy * (center_tensor[mask, 1] - cy)
        cur_pc_tensor = torch.vstack([x, y, z]).T
        # deal with d == 0
        idxs = torch.nonzero(~mask).cpu().numpy().squeeze(-1)
        for j in idxs:
            x, y = center_2d[j, 0], center_2d[j, 1]
            # choose neighbor average to fix zero depth
            neighbor = 4
            x_range = slice(max(0, x - neighbor), min(1279, x + neighbor))
            y_range = slice(max(0, y - neighbor), min(719, y + neighbor))
            neighbor_depths = depths[i, x_range, y_range]
            depth_mask = (neighbor_depths > 0)
            if depth_mask.sum() == 0:
                # continue
                # this will use all centers
                cur_d = depths[i].mean()
            else:
                cur_d = neighbor_depths[depth_mask].float().median(
                ) + delta_d[j]
            # set valid mask
            mask[j] = True
            # convert
            new_center = torch.from_numpy(convert_2d_to_3d(
                x, y, cur_d.cpu())).cuda()
            cur_pc_tensor = torch.concat([cur_pc_tensor, new_center[None]], 0)

        # modify rect_ggs and append
        rect_ggs[i] = rect_ggs[i][mask.cpu().numpy()]
        # convert delta depth to actual depth for further width conversion
        rect_ggs[i].actual_depths = cur_pc_tensor[:, 2].cpu().numpy() * 1000.0
        # attention: rescale here
        rect_ggs[i].actual_depths *= (1280 // output_size[0])
        # add small noise to local centers (when train)
        if is_training:
            cur_pc_tensor += torch.randn(*cur_pc_tensor.shape,
                                         device='cuda') * 5e-3
        center_batch_pc.append(cur_pc_tensor)
    return center_batch_pc


def get_ori_grasp_label(grasppath):
    # load grasp
    grasp_label = np.load(grasppath[0])  # B=1
    grasp_num = grasp_label['centers_2d'].shape[0]
    gg_ori_labels = -np.ones((grasp_num, 8), dtype=np.float32)

    # get grasp original labels
    centers_2d = grasp_label['centers_2d']
    grasp_num = centers_2d.shape[0]
    gg_ori_labels[:, :3] = convert_2d_to_3d(centers_2d[:, 0], centers_2d[:, 1],
                                            grasp_label['center_z_depths'])
    gg_ori_labels[:, 3] = grasp_label['thetas_rad']
    gg_ori_labels[:, 4] = grasp_label['gammas_rad']
    gg_ori_labels[:, 5] = grasp_label['betas_rad']
    gg_ori_labels[:, 6] = grasp_label['widths_2d']
    gg_ori_labels[:, 7] = grasp_label['scores_from_6d']
    gg_ori_labels = torch.from_numpy(gg_ori_labels).cuda()

    return gg_ori_labels


def get_center_group_label(local_center: List, grasp_labels: List,
                           local_grasp_num) -> List:
    batch_size = len(local_center)
    gg_group_labels = []
    total_labels = torch.zeros((0, 8), dtype=torch.float32, device='cuda')
    for i in range(batch_size):
        # get grasp
        grasp_label = grasp_labels[i]
        # set up numpy grasp label
        centers_2d = grasp_label['centers_2d']
        grasp_num = centers_2d.shape[0]
        gg_label = -np.ones((grasp_num, 8), dtype=np.float32)
        gg_label[:, :3] = convert_2d_to_3d(centers_2d[:, 0], centers_2d[:, 1],
                                           grasp_label['center_z_depths'])
        gg_label[:, 3] = grasp_label['thetas_rad']
        gg_label[:, 4] = grasp_label['gammas_rad']
        gg_label[:, 5] = grasp_label['betas_rad']
        gg_label[:, 6] = grasp_label['widths_2d']
        gg_label[:, 7] = grasp_label['scores_from_6d']

        # convert to cuda tensor
        gg_label = torch.from_numpy(gg_label).cuda()

        # cal distance to valid center
        valid_center = local_center[i]
        distance = euclid_distance(
            valid_center, gg_label)  # distance: (center_num, grasp_num)

        # select nearest grasp labels for all center
        mask = (distance < 0.02)
        for j in range(len(distance)):
            # mask with min dis
            mask_gg = gg_label[mask[j]]
            mask_distance = distance[j][mask[j]]
            if len(mask_distance) == 0:
                gg_group_labels.append(torch.zeros((0, 8)).cuda())
            else:
                # sorted and select nearest
                _, topk_idxs = torch.topk(mask_distance,
                                          k=min(local_grasp_num,
                                                mask_distance.shape[0]),
                                          largest=False)
                gg_nearest = mask_gg[topk_idxs]
                # move to (0, 0, 0)
                gg_nearest[:, :3] = gg_nearest[:, :3] - valid_center[j]
                gg_group_labels.append(gg_nearest)
                total_labels = torch.cat([total_labels, gg_nearest], 0)
    return gg_group_labels, total_labels


@njit
def select_area(loc_map, top, bottom, left, right, grid_size, overlap):
    center_num = len(top)
    local_areas = np.zeros((center_num, (grid_size + overlap * 2)**2))
    for j in range(center_num):
        # extend to make overlap
        local_area = loc_map[top[j]:bottom[j], left[j]:right[j]]
        local_area = np.ascontiguousarray(local_area).reshape((-1, ))
        local_areas[j, :len(local_area)] = local_area
    return local_areas


def farthest_grid_sampling(points, num_samples):
    """
    对通过网格采样得到的像素点进行最远点采样
    
    参数:
        points: 通过网格采样得到的像素点数组，形状为 (N, 2)
        num_samples: 最终要选取的像素点数量
        
    返回:
        最远点采样后的像素点数组，形状为 (num_samples, 2)
    """
    n = points.shape[0]
    if n <= num_samples:
        return points
    
    # 1. 初始化
    selected_points = np.zeros((num_samples, 2), dtype=points.dtype)
    min_distances = np.full(n, np.inf)  # 存储每个点到已选点集的最小距离
    
    # 2. 选择起始点 (随机选择或根据热力图值)
    start_idx = np.random.randint(n)
    selected_points[0] = points[start_idx]
    
    # 3. 计算所有点到第一个点的距离
    dist_to_start = np.linalg.norm(points - points[start_idx], axis=1)
    min_distances = np.minimum(min_distances, dist_to_start)
    
    # 4. 迭代选择剩余点
    for i in range(1, num_samples):
        # 找到距离当前已选点集最远的点
        farthest_idx = np.argmax(min_distances)
        selected_points[i] = points[farthest_idx]
        
        # 计算所有点到新选点的距离
        dist_to_new = np.linalg.norm(points - points[farthest_idx], axis=1)
        
        # 更新最小距离
        min_distances = np.minimum(min_distances, dist_to_new)
    
    return selected_points

def select_2d_center(loc_maps, center_num, reduce='max', grid_size=8, farthest=False) -> List:
    # deal with validation stage
    if isinstance(loc_maps, np.ndarray):
        loc_maps = loc_maps.copy()
    else:
        loc_maps = loc_maps.clone()
    if len(loc_maps.shape) == 2:
        loc_maps = loc_maps[None]
    # using torch to downsample
    if isinstance(loc_maps, np.ndarray):
        loc_maps = torch.from_numpy(loc_maps).cuda()
    print(loc_maps.shape)
    batch_size = loc_maps.shape[0]
    center_2ds = []
    # using downsampled grid to avoid center too near
    new_size = (loc_maps.shape[1] // grid_size, loc_maps.shape[2] // grid_size)
    if reduce == 'avg':
        heat_grids = nnf.avg_pool2d(loc_maps[None], grid_size).squeeze()
    elif reduce == 'max':
        heat_grids = nnf.max_pool2d(loc_maps[None], grid_size).squeeze()
    else:
        raise RuntimeError(f'Unrecognized reduce: {reduce}')
    heat_grids = heat_grids.view((batch_size, -1))
    # get topk grid point
    for i in range(batch_size):
        if not farthest:
            local_idx = torch.topk(heat_grids[i],
                                k=min(heat_grids.shape[1], center_num),
                                dim=0)[1]
            local_max = np.zeros((len(local_idx), 2), dtype=np.int64)
            local_max[:, 0] = torch.div(local_idx,
                                        new_size[1],
                                        rounding_mode='floor').cpu().numpy()
            local_max[:, 1] = (local_idx % new_size[1]).cpu().numpy()
        if farthest:
            local_idx = torch.topk(heat_grids[i],
                                k=min(heat_grids.shape[1], int(center_num * 1.5)),
                                dim=0)[1]
            # print("值：",heat_grids[i][local_idx])
            local_max = np.zeros((len(local_idx), 2), dtype=np.int64)
            local_max[:, 0] = torch.div(local_idx,
                                        new_size[1],
                                        rounding_mode='floor').cpu().numpy()
            local_max[:, 1] = (local_idx % new_size[1]).cpu().numpy()
            local_max = farthest_grid_sampling(local_max, center_num)
        # get local max in this grid point
        overlap = 1
        top, bottom = local_max[:, 0] * grid_size - overlap, (
            local_max[:, 0] + 1) * grid_size + overlap
        top, bottom = np.maximum(0, top), np.minimum(bottom,
                                                     loc_maps.shape[1] - 1)
        left, right = local_max[:, 1] * grid_size - overlap, (
            local_max[:, 1] + 1) * grid_size + overlap
        left, right = np.maximum(0, left), np.minimum(right,
                                                      loc_maps.shape[2] - 1)
        # using jit to faster get local areas
        local_areas = select_area(loc_maps[i].cpu().numpy(), top, bottom, left,
                                  right, grid_size, overlap)
        local_areas = torch.from_numpy(local_areas).float().cuda()
        # batch calculate
        grid_idxs = torch.argmax(local_areas, dim=1).cpu().numpy()
        local_max[:, 0] = top + grid_idxs // (right - left)
        local_max[:, 1] = left + grid_idxs % (right - left)
        center_2ds.append(local_max)
    return center_2ds


def bayes_gain(heat_grids, obj_grids, exclude=False):
    """
    Args:
        heat_grids: torch.Tensor, shape (1, W, H), 先验概率
        obj_grids:  torch.Tensor, shape (1, W, H), 似然概率
        exclude:  为True则将蒙板外其他位置设为零; False则只进行蒙板内的增益。
    Returns:
        heat_gained_grids: torch.Tensor, shape (1, W, H), 后验概率
    """
    heat_grids = heat_grids.double()
    obj_grids = obj_grids.double()

    # 基础概率（可以修改）
    base_rate = 0.25  

    prior = heat_grids
    likelihood = obj_grids

    # 计算边际概率
    marginal = base_rate * (1 - prior) + likelihood * prior

    # 避免除零
    eps = 1e-8
    marginal = marginal + eps

    posterior = likelihood * prior / marginal

    if exclude:
        # where选择: 如果likelihood=0，则取0，否则取posterior，即蒙板外其他位置均设为0
        heat_gained_grids = torch.where(likelihood > 0, posterior, 0.)
    else:
        # where选择: 如果likelihood=0，则取prior，否则取posterior
        heat_gained_grids = torch.where(likelihood > 0, posterior, prior)

    return heat_gained_grids


def select_2d_center_yolo(loc_maps, center_num, obj_mask, reduce='max', grid_size=8, farthest=True, goal_only=False) -> List:
    # arg:  obj_mask (W, H) is conf_mask tensor
    #       loc_maps (1, W, H)
    # deal with validation stage
    if isinstance(loc_maps, np.ndarray):
        loc_maps = loc_maps.copy()
    else:
        loc_maps = loc_maps.clone()
    if len(loc_maps.shape) == 2:
        loc_maps = loc_maps[None]
    # using torch to downsample
    if isinstance(loc_maps, np.ndarray):
        loc_maps = torch.from_numpy(loc_maps).cuda()
    batch_size = loc_maps.shape[0]
    center_2ds = []
    # using downsampled grid to avoid center too near
    new_size = (loc_maps.shape[1] // grid_size, loc_maps.shape[2] // grid_size)
    if reduce == 'avg':
        heat_grids = nnf.avg_pool2d(loc_maps[None], grid_size).squeeze()
    elif reduce == 'max':
        heat_grids = nnf.max_pool2d(loc_maps[None], grid_size).squeeze()
    else:
        raise RuntimeError(f'Unrecognized reduce: {reduce}')
    # 处理obj_mask
    obj_mask = obj_mask.cuda()
    if len(obj_mask.shape) == 2:
        obj_grids = nnf.max_pool2d(obj_mask[None, None], grid_size).squeeze()
    else:
        raise RuntimeError(f'need obj_mask dim 2, but got {len(obj_mask.shape)}')
    # print("loc",loc_maps.shape," mask",obj_mask.shape)
    # print("heat",heat_grids.shape," obj",obj_grids.shape)
    # 置信度增益，arg：exclude为True则不在蒙板外采样；为false则依然全局采样，依据goal_only选择是否在蒙板内过滤
    # 问题：当采用exclude时，可能该物体蒙板内置信度过低，使得采样虽然只在蒙板内进行，但最后回归得到靠近蒙板的其他物体的抓取。
    heat_grids = bayes_gain(heat_grids, obj_grids, exclude=False)

    heat_grids = heat_grids.view((batch_size, -1))
    # get topk grid point
    for i in range(batch_size):
        if not farthest:
            local_idx = torch.topk(heat_grids[i],
                                k=min(heat_grids.shape[1], center_num),
                                dim=0)[1]
            local_max = np.zeros((len(local_idx), 2), dtype=np.int64)
            local_max[:, 0] = torch.div(local_idx,
                                        new_size[1],
                                        rounding_mode='floor').cpu().numpy()
            local_max[:, 1] = (local_idx % new_size[1]).cpu().numpy()
        if farthest:
            local_idx = torch.topk(heat_grids[i],
                                k=min(heat_grids.shape[1], int(center_num * 1.5)),
                                dim=0)[1]
            # print("值：",heat_grids[i][local_idx])
            local_max = np.zeros((len(local_idx), 2), dtype=np.int64)
            local_max[:, 0] = torch.div(local_idx,
                                        new_size[1],
                                        rounding_mode='floor').cpu().numpy()
            local_max[:, 1] = (local_idx % new_size[1]).cpu().numpy()
            local_max = farthest_grid_sampling(local_max, center_num)
        # === goal_only过滤逻辑 ===
        if goal_only:
            obj_grid_np = obj_grids.squeeze(0).cpu().numpy()  # (W, H)
            keep_mask = obj_grid_np[local_max[:, 0], local_max[:, 1]] > 0
            local_max = local_max[keep_mask]
        # =======================
        # get local max in this grid point
        overlap = 1
        top, bottom = local_max[:, 0] * grid_size - overlap, (
            local_max[:, 0] + 1) * grid_size + overlap
        top, bottom = np.maximum(0, top), np.minimum(bottom,
                                                     loc_maps.shape[1] - 1)
        left, right = local_max[:, 1] * grid_size - overlap, (
            local_max[:, 1] + 1) * grid_size + overlap
        left, right = np.maximum(0, left), np.minimum(right,
                                                      loc_maps.shape[2] - 1)
        # using jit to faster get local areas
        local_areas = select_area(loc_maps[i].cpu().numpy(), top, bottom, left,
                                  right, grid_size, overlap)
        local_areas = torch.from_numpy(local_areas).float().cuda()
        # batch calculate
        grid_idxs = torch.argmax(local_areas, dim=1).cpu().numpy()
        local_max[:, 0] = top + grid_idxs // (right - left)
        local_max[:, 1] = left + grid_idxs % (right - left)

        center_2ds.append(local_max)
    return center_2ds

def data_process(points: torch.Tensor,
                 depths: torch.Tensor,
                 rect_ggs: List,
                 center_num,
                 group_num,
                 output_size,
                 min_points=32,
                 is_training=True,
                 is_wide=False):
    # select partial pc centers
    local_center = center2dtopc(rect_ggs,
                                center_num,
                                depths,
                                output_size,
                                append_random_center=False,
                                is_training=is_training)
    # get grasp width for pc segmentation
    grasp_widths = []
    if is_wide:
        grasp_wide_widths = []
        for rect_gg in rect_ggs:
            grasp_wide_widths.append(2 * rect_gg.get_6d_width())

    for rect_gg in rect_ggs:
        grasp_widths.append(rect_gg.get_6d_width())
    # print(grasp_widths)
    # seg point cloud
    if is_wide:
        pc_group, wide_pc_group, valid_local_centers, valid_center_masks = get_group_pc(
            points,
            local_center,
            group_num,
            grasp_widths,
            min_points=min_points,
            is_training=is_training,
            is_wide=is_wide)
    else:
        pc_group, valid_local_centers, valid_center_masks = get_group_pc(
            points,
            local_center,
            group_num,
            grasp_widths,
            min_points=min_points,
            is_training=is_training)        
    # modify rect_ggs
    for i, mask in enumerate(valid_center_masks):
        rect_ggs[i] = rect_ggs[i][mask.cpu().numpy()]

    # visualize_grasp_and_pointcloud(rect_ggs, pc_group, wide_pc_group=wide_pc_group)
    if is_wide:
        return pc_group, wide_pc_group, valid_local_centers
    else:
        return pc_group, valid_local_centers

import open3d as o3d

def visualize_grasp_and_pointcloud(rect_ggs, pc_group, wide_pc_group=None):
    rect_grasp = rect_ggs[0]  #取第一个batch
    print("pc: ",pc_group.shape)
    print("pc_wide: ",wide_pc_group.shape)
    pointcloud_data = pc_group[0, :, :3]  # 取第一个点云并提取前三个维度 (x, y, z)
    if isinstance(pointcloud_data, torch.Tensor):
        pointcloud_data = pointcloud_data.cpu().numpy()  # 确保转换为 NumPy

    # 确保是 float64 类型
    pointcloud_data = pointcloud_data.astype(np.float64)

    grasps = rect_grasp.to_6d_grasp_group()
    # print(grasps.widths)

    # 生成抓取框的 3D 形状
    gripper_mesh = grasps.to_open3d_geometry_list()[0]

    # 生成点云
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pointcloud_data)

    if wide_pc_group is not None:
        wide_pointcloud = wide_pc_group[0, :, :3]
        wide_pointcloud = wide_pointcloud.cpu().numpy()
        pcd_wide = o3d.geometry.PointCloud()
        pcd_wide.points = o3d.utility.Vector3dVector(wide_pointcloud)

        # 设置颜色：小范围红色，大范围红加绿，先画大范围，再小范围
        color = np.array([[0.0, 0.0, 0.7]])
        colors = np.tile(color, (wide_pointcloud.shape[0], 1))
        pcd_wide.colors = o3d.utility.Vector3dVector(colors)

        # 设置颜色：统一红色
        color = np.array([[1.0, 0.0, 0.0]])
        colors = np.tile(color, (pointcloud_data.shape[0], 1))
        pcd.colors = o3d.utility.Vector3dVector(colors)

        # 可视化
        o3d.visualization.draw_geometries([pcd, pcd_wide])
    
    else:
        # 可视化
        o3d.visualization.draw_geometries([gripper_mesh, pcd])
