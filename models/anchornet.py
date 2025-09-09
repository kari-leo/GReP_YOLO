import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .resnet_model import BasicBlock, BottleNeck, ResNet


def set_bn_momentum_default(bn_momentum):

    def fn(m):
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.momentum = bn_momentum

    return fn


class BNMomentumScheduler(object):

    def __init__(self,
                 model,
                 bn_lambda,
                 last_epoch=-1,
                 setter=set_bn_momentum_default):
        if not isinstance(model, nn.Module):
            raise RuntimeError("Class '{}' is not a PyTorch nn Module".format(
                type(model).__name__))

        self.model = model
        self.setter = setter
        self.lmbd = bn_lambda

        self.step(last_epoch + 1)
        self.last_epoch = last_epoch

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1

        self.last_epoch = epoch
        self.model.apply(self.setter(self.lmbd(epoch)))


class Backbone(nn.Module):

    def __init__(self, in_dim, planes=16, mode='34'):
        # if out_dim is None then return features
        super().__init__()
        if mode == '18':
            self.net = ResNet(BasicBlock, [2, 2, 2, 2],
                              in_dim=in_dim,
                              planes=planes)
        elif mode == '34':
            self.net = ResNet(BasicBlock, [3, 4, 6, 3],
                              in_dim=in_dim,
                              planes=planes)
        elif mode == '50':
            # need to div 4 for same channels
            self.net = ResNet(BottleNeck, [3, 4, 6, 3],
                              in_dim=in_dim,
                              planes=planes // 4)
        else:
            raise RuntimeError(f'Backbone mode {mode} not found!')

    def forward(self, x):
        x = self.net(x)
        return x


class convolution(nn.Module):

    def __init__(self,
                 in_dim,
                 out_dim,
                 kernel,
                 bias=False,
                 stride=1,
                 act=nn.ReLU):
        super(convolution, self).__init__()
        pad = (kernel - 1) // 2
        self.conv = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel, stride, pad, bias=bias),
            nn.BatchNorm2d(out_dim), act(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x


class trconvolution(nn.Module):

    def __init__(self,
                 in_dim,
                 out_dim,
                 padding=1,
                 output_padding=0,
                 act=nn.ReLU):
        super().__init__()
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(in_dim,
                               out_dim,
                               kernel_size=4,
                               stride=2,
                               padding=padding,
                               output_padding=output_padding,
                               bias=False), nn.BatchNorm2d(out_dim),
            act(inplace=True))

    def forward(self, x):
        x = self.deconv(x)
        return x


class upsampleconvolution(nn.Module):

    def __init__(self, in_dim, out_dim, output_size=None, act=nn.ReLU):
        super().__init__()
        self.deconv = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2) if output_size is None else
            nn.UpsamplingBilinear2d(size=output_size),
            nn.Conv2d(in_channels=in_dim,
                      out_channels=out_dim,
                      kernel_size=5,
                      padding=2,
                      bias=False), nn.BatchNorm2d(out_dim), act(inplace=True))

    def forward(self, x):
        x = self.deconv(x)
        return x


def conv_with_dim_reshape(in_dim, mid_dim, out_dim, bias=True):
    return nn.Sequential(convolution(in_dim, mid_dim, 5, bias=bias),
                         nn.Conv2d(mid_dim, out_dim, 1, bias=bias))


# 使用SE（Squeeze-and-Excitation）模块，增加通道注意力机制（Channel Attention）

class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()

        # 应对输入通道小于16的情况
        if in_channels < reduction:
            reduction = in_channels

        # 对每个池化结果分别应用全连接层
        self.fc1_avg = nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False)
        self.fc1_max = nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Squeeze: Global Average Pooling
        avg_pool = F.adaptive_avg_pool2d(x, (1, 1))
        # Squeeze: Global Max Pooling
        max_pool = F.adaptive_max_pool2d(x, (1, 1))

        # 分别通过全连接层
        avg_out = self.fc1_avg(avg_pool)
        max_out = self.fc1_max(max_pool)

        # 合并两个全连接层的输出
        combined = avg_out + max_out  # 将两者相加

        # 再通过一个全连接层
        y = self.fc2(self.relu(combined))

        # 使用Sigmoid生成注意力权重
        attention = self.sigmoid(y)

        # 将注意力权重应用到输入特征图上
        return x * attention

# 空间注意力机制（Spatial Attention）

class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        # 1x1卷积用于生成空间注意力
        self.conv1 = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 计算平均池化和最大池化
        avg_out = torch.mean(x, dim=1, keepdim=True)  # 平均池化
        max_out, _ = torch.max(x, dim=1, keepdim=True) # 最大池化

        # 拼接平均池化和最大池化结果
        x_cat = torch.cat([avg_out, max_out], dim=1)

        # 通过卷积层生成空间注意力
        attention_map = self.conv1(x_cat)
        attention_map = self.sigmoid(attention_map)

        # 乘上注意力图，增强重要区域
        return x * attention_map

class AnchorGraspNet(nn.Module):

    def __init__(self,
                 ratio=1,
                 in_dim=3,
                 anchor_k=6,
                 mid_dim=32,
                 use_upsampling=False):
        super(AnchorGraspNet, self).__init__()

        # Using imagenet pre-trained model as feature extractor
        self.ratio = ratio
        self.trconv = nn.ModuleList()
        # backbone
        self.feature_dim = 128
        self.backbone = Backbone(in_dim, self.feature_dim // 16)

        # SE blocks for CA
        self.se_blocks = nn.ModuleList()

        # Space Attention Blocks
        self.spatial_attention = SpatialAttention()

        # transconv
        self.depth = 4
        channels = [
            max(8, self.feature_dim // (2**(i + 1)))
            for i in range(self.depth + 1)
        ]
        cur_dim = self.feature_dim
        output_sizes = [(40, 23), (80, 45)]
        for i, dim in enumerate(channels):
            if use_upsampling:
                if i < min(5 - np.log2(ratio), 2):
                    self.trconv.append(
                        upsampleconvolution(cur_dim, dim, output_sizes[i]))
                else:
                    self.trconv.append(upsampleconvolution(cur_dim, dim))
            else:
                if i < min(5 - np.log2(ratio), 2):
                    self.trconv.append(
                        trconvolution(cur_dim,
                                      dim,
                                      padding=(1, 2),
                                      output_padding=(0, 1)))
                else:
                    self.trconv.append(trconvolution(cur_dim, dim))

            # Add SEBlock after each trconv layer
            self.se_blocks.append(SEBlock(dim))

            cur_dim = dim

        # Heatmap predictor
        cur_dim = channels[self.depth - int(np.log2(self.ratio))]
        self.hmap = conv_with_dim_reshape(channels[-1], mid_dim, 1)
        self.cls_mask_conv = conv_with_dim_reshape(cur_dim, mid_dim, anchor_k)
        self.theta_offset_conv = conv_with_dim_reshape(cur_dim, mid_dim,
                                                       anchor_k)
        self.width_offset_conv = conv_with_dim_reshape(cur_dim, mid_dim,
                                                       anchor_k)
        self.depth_offset_conv = conv_with_dim_reshape(cur_dim, mid_dim,
                                                       anchor_k)

        # init weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # print(m)
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # use backbone to get downscaled features
        xs = self.backbone(x)
        # use transposeconve or upsampling + conv to get perpoint features
        x = xs[-1]
        for i, layer in enumerate(self.trconv):
            # skip connection
            x = layer(x + xs[self.depth - i])
            # Apply SE Block (Attention Mechanism)
            x = self.se_blocks[i](x)
            # Apply Spatial Attention
            x = self.spatial_attention(x)
            # down sample classification mask
            if x.shape[2] == 80:
                features = x.detach()
            if int(np.log2(self.ratio)) == self.depth - i:
                cls_mask = self.cls_mask_conv(x)
                theta_offset = self.theta_offset_conv(x)
                width_offset = self.width_offset_conv(x)
                depth_offset = self.depth_offset_conv(x)
        # full scale location map
        loc_map = self.hmap(x)
        return (loc_map, cls_mask, theta_offset, depth_offset,
                width_offset), features


from .pointnet import PointNetfeat
from dataset.config import get_camera_intrinsic
# 基于点云的抓取可行性检测
class PointNetConf(nn.Module):
    def __init__(self, extra_feature_len=0, all_points_num=1000, need_stn=True):
        super().__init__()
        self.extra_feature_len = extra_feature_len
        self.all_points_num = all_points_num
        
        # PointNet feature extraction
        self.pointnet = PointNetfeat(feature_len=3, extra_feature_len=self.extra_feature_len, need_stn=need_stn)

        # MLP for point features (512 hidden units)
        self.point_layer = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.Dropout(0.3),
            nn.ReLU(inplace=True)
        )
        
        # MLP with adjusted layers: (512, 256) -> (256, 32) -> (32, 1)
        self.mlp = nn.Sequential(
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 32),
            nn.LayerNorm(32),
            nn.ReLU(inplace=True)
        )

        # # MLP with adjusted layers: (512, 256) -> (256, 64) -> (64, 1)
        # self.mlp = nn.Sequential(
        #     nn.Linear(512, 256),
        #     nn.LayerNorm(256),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(256, 64),
        #     nn.LayerNorm(64),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(64, 1)  # Binary classification output (0 or 1)
        # )

        # # MLP with adjusted layers: (512, 256) -> (256, 128) -> (128, 1)
        # self.mlp = nn.Sequential(
        #     nn.Linear(512, 256),
        #     nn.LayerNorm(256),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(256, 128),
        #     nn.LayerNorm(128),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(128, 1)  # Binary classification output (0 or 1)
        # )

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # 先处理点云信息
        points= self.process_depth_to_point_cloud(x, points_per_patch=self.all_points_num // 48)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        points = points.to(device)

        # Output storage for classification results and features
        batch_size, grid_rows, grid_cols, num_points, _ = points.shape
        features = torch.zeros(batch_size, grid_rows, grid_cols, 32).to(points.device)  # 32 is feature size from MLP

        # Transpose points to match input shape (batch, channels, points)
        points = points.view(batch_size * grid_rows * grid_cols, num_points, 3)  # Flatten grid
        points = points.permute(0, 2, 1)  # Swap the last two dimensions

        # PointNet feature extraction
        point_features = self.pointnet(points)  # Shape: (batch_size * grid_size, feature_len)

        # MLP to process point features
        point_features = self.point_layer(point_features)  # Shape: (batch_size * grid_size, 512)

        # Directly apply the MLP to point_features without reshaping
        x = self.mlp[0](point_features)  # (batch_size * grid_rows * grid_cols, 512 -> 256)
        x = self.mlp[1](x)  # LayerNorm
        x = self.mlp[2](x)  # ReLU
        x = self.mlp[3](x)  # (256 -> 32)
        x = self.mlp[4](x)  # LayerNorm
        x = self.mlp[5](x)  # ReLU, 32dim feature

        # Store the feature vector (x after final MLP layer)
        features = x.view(batch_size, grid_rows, grid_cols, 32)
        # features = x.view(batch_size, grid_rows, grid_cols, 64)
        # features = x.view(batch_size, grid_rows, grid_cols, 128)

        # 将conf_features调整形状并上采样
        features = features.permute(0, 3, 2, 1)  # [2, 6, 8, 64] -> [2, 64, 8, 6]
        features = F.interpolate(features, size=(80, 45), mode='nearest')  # [2, 64, 80, 45]

        return features

    # def forward(self, points):
    #     # Output storage for classification results and features
    #     batch_size, grid_rows, grid_cols, num_points, _ = points.shape
    #     classification_result = torch.zeros(batch_size, grid_rows, grid_cols, 1).to(points.device)
    #     features = torch.zeros(batch_size, grid_rows, grid_cols, 32).to(points.device)  # 32 is feature size from MLP

    #     # Transpose points to match input shape (batch, channels, points)
    #     points = points.view(batch_size * grid_rows * grid_cols, num_points, 3)  # Flatten grid
    #     points = points.permute(0, 2, 1)  # Swap the last two dimensions

    #     # PointNet feature extraction
    #     point_features = self.pointnet(points)  # Shape: (batch_size * grid_size, feature_len)

    #     # MLP to process point features
    #     point_features = self.point_layer(point_features)  # Shape: (batch_size * grid_size, 512)

    #     # Directly apply the MLP to point_features without reshaping
    #     x = self.mlp[0](point_features)  # (batch_size * grid_rows * grid_cols, 512 -> 256)
    #     x = self.mlp[1](x)  # LayerNorm
    #     x = self.mlp[2](x)  # ReLU
    #     x = self.mlp[3](x)  # (256 -> 32)
    #     x = self.mlp[4](x)  # LayerNorm
    #     x = self.mlp[5](x)  # ReLU, 32dim feature
    #     classification_result_flat = self.mlp[6](x)  # (32 -> 1), binary classification

    #     # Reshape classification result to match the grid
    #     classification_result = classification_result_flat.view(batch_size, grid_rows, grid_cols, 1)

    #     # Store the feature vector (x after final MLP layer)
    #     # features = x.view(batch_size, grid_rows, grid_cols, 32)
    #     features = x.view(batch_size, grid_rows, grid_cols, 64)
    #     # features = x.view(batch_size, grid_rows, grid_cols, 128)

    #     return classification_result, features
    
    def process_depth_to_point_cloud(self, x, patch_rows=8, patch_cols=6, points_per_patch=1000):
        """
        处理输入形状为 (batch_size, 4, input_w, input_h) 的张量 x，提取其深度通道，分块并转换为点云。
        
        参数:
            x: 形状为 (batch_size, 4, input_w, input_h) 的张量，包含深度数据和RGB数据。
            patch_rows: 分块的行数，默认值为 8。
            patch_cols: 分块的列数，默认值为 6。
            points_per_patch: 每个patch需要的点数。
        
        返回:
            point_cloud_tensor: 转换后的点云张量，形状为 (batch_size, patch_rows, patch_cols, points_per_patch, 3)。
        """
        x = x.permute(0, 1, 3, 2)  # 交换 input_w 和 input_h 维度

        # 获取batch大小和深度图像的尺寸
        batch_size = x.shape[0]
        depth_image = x[:, 0]  # 取出所有batch的深度图像 (batch_size, input_h, input_w)
        height, width = depth_image.shape[1], depth_image.shape[2]

        # 计算每个patch的大小
        patch_height = height // patch_rows
        patch_width = width // patch_cols

        # 生成patch的索引
        row_indices = np.repeat(np.arange(patch_rows), patch_cols)  # (patch_rows * patch_cols)
        col_indices = np.tile(np.arange(patch_cols), patch_rows)  # (patch_rows * patch_cols)

        # 存储所有batch的点云
        all_point_clouds = []

        # 处理每个batch
        for b in range(batch_size):
            # 获取当前batch的深度图像
            depth_image = x[b, 0].cpu().numpy()

            # 存储当前batch的所有patch点云
            point_clouds = []

            # 计算每个patch的点云
            for i in range(patch_rows * patch_cols):
                patch_row, patch_col = row_indices[i], col_indices[i]

                # 提取当前patch
                patch = depth_image[patch_row * patch_height: (patch_row + 1) * patch_height,
                                    patch_col * patch_width: (patch_col + 1) * patch_width]
                
                # 转换为点云
                point_cloud = self.depth_to_point_cloud(patch, points_per_patch, patch_row, patch_col)
                point_clouds.append(point_cloud)

            # 将所有patch的点云堆叠成一个Tensor，形状为 (patch_rows, patch_cols, points_per_patch, 3)
            point_cloud_tensor = torch.stack(point_clouds).view(patch_rows, patch_cols, points_per_patch, 3)

            # 添加到batch的点云集合
            all_point_clouds.append(point_cloud_tensor)

        # 最终返回形状为 (batch_size, patch_rows, patch_cols, points_per_patch, 3) 的张量
        return torch.stack(all_point_clouds)


    def depth_to_point_cloud(self, depth_patch, points_per_patch, patch_row, patch_col):
        """
        将深度图像patch转换为点云，并返回指定数量的点。
        
        参数:
            depth_patch: 当前patch的深度图像，形状为 (patch_height, patch_width)。
            points_per_patch: 每个patch需要的点数。
            patch_row: 当前patch的行索引。
            patch_col: 当前patch的列索引。
        
        返回:
            point_cloud: 当前patch的点云，形状为 (points_per_patch, 3)。
        """
        # 获取相机内参 (这里假设K已经定义)
        K = get_camera_intrinsic()

        # 获取patch的高度和宽度
        patch_height, patch_width = depth_patch.shape

        factor = (1280 / 6) / patch_height

        # 获取深度图像的有效坐标（非零深度值）
        valid_mask = depth_patch != 0
        z_values = depth_patch[valid_mask]
        
        # 计算有效像素的图像坐标
        x_img = patch_col * patch_width + np.repeat(np.arange(patch_width), patch_height)[valid_mask.flatten()]
        y_img = patch_row * patch_height + np.tile(np.arange(patch_height), patch_width)[valid_mask.flatten()]

        # # 使用相机内参将像素坐标转换为世界坐标
        # x = (x_img - K[0, 2] / factor) * z_values / (K[0, 0] / factor)
        # y = (y_img - K[1, 2] / factor) * z_values / (K[1, 1] / factor)

        # # 组合成点云
        # valid_points = np.vstack((x, y, z_values)).T

        # 将计算的坐标和深度值转换为 PyTorch tensor
        x_tensor = torch.tensor((x_img - K[0, 2] / factor) * z_values / (K[0, 0] / factor), dtype=torch.float32)
        y_tensor = torch.tensor((y_img - K[1, 2] / factor) * z_values / (K[1, 1] / factor), dtype=torch.float32)
        z_tensor = torch.tensor(z_values, dtype=torch.float32)

        # 拼接 x, y, z 的点云数据，并转置为 (N, 3) 的形状
        valid_points = torch.stack((x_tensor, y_tensor, z_tensor), dim=1)

        # 确保每个patch有固定数量的点
        num_valid_points = len(valid_points)
        if num_valid_points > points_per_patch:
            # sampled_indices = np.random.choice(num_valid_points, points_per_patch, replace=False)
            # valid_points = [valid_points[i] for i in sampled_indices]
            sampled_indices = torch.randint(0, num_valid_points, (points_per_patch,))
            valid_points = valid_points[sampled_indices]
        elif num_valid_points < points_per_patch:
            if num_valid_points == 0:
                # valid_points = np.zeros((points_per_patch, 3))  # 填充0
                valid_points = torch.zeros((points_per_patch, 3), dtype=torch.float32)  # 填充0
            while len(valid_points) < points_per_patch:
                # 追加新点
                # valid_points = np.vstack([valid_points, valid_points[np.random.randint(num_valid_points)]])
                valid_points = torch.cat([valid_points, valid_points[torch.randint(0, num_valid_points, (1,))]])

        # 返回点云
        # return torch.tensor(valid_points, dtype=torch.float32)
        # return valid_points.cuda().cpu()
        return valid_points


class AnchorGraspNet_ver2_Simplified(nn.Module):
    def __init__(self,
                all_points_num,
                ratio=1,
                in_dim=3,
                anchor_k=6,
                mid_dim=32,
                use_upsampling=False):
        super(AnchorGraspNet_ver2_Simplified, self).__init__()

        # 使用AnchorGraspNet的结构
        self.anchor_grasp_net = AnchorGraspNet(ratio, in_dim, anchor_k, mid_dim, use_upsampling)

        # 使用PointNetConf处理点云信息
        self.point_conf_model = PointNetConf()
        self.all_points_num = all_points_num

    def forward(self, x):
        # 先处理点云信息
        depths = self.point_conf_model.process_depth_to_point_cloud(x, points_per_patch=self.all_points_num // 48)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        depths = depths.to(device)
        conf_map, conf_features = self.point_conf_model(depths)  # conf_map, conf_features形状为 [2, 6, 8, 1] 和 [2, 6, 8, 64]

        # 将conf_features调整形状并上采样
        # conf_features = conf_features.detach()
        conf_features = conf_features
        conf_features = conf_features.permute(0, 3, 2, 1)  # [2, 6, 8, 64] -> [2, 64, 8, 6]
        conf_features = F.interpolate(conf_features, size=(80, 45), mode='nearest')  # [2, 64, 80, 45]

        # 使用AnchorGraspNet的结构
        (loc_map, cls_mask, theta_offset, depth_offset, width_offset), features = self.anchor_grasp_net(x)

        # 将AnchorGraspNet的features与conf_features拼接
        features = torch.cat([features, conf_features], dim=1)  # [2, 96, 80, 45]

        return (loc_map, cls_mask, theta_offset, depth_offset, width_offset), features, conf_map