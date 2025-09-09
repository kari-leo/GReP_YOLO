import cv2
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image as Ros_Image

import argparse
import os
import random
from time import time

import numpy as np
import open3d as o3d
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from PIL import Image

from dataset.config import get_camera_intrinsic
from dataset.evaluation import (anchor_output_process, collision_detect,
                                detect_2d_grasp, detect_6d_grasp_multi)
from dataset.pc_dataset_tools import data_process, feature_fusion
from models.anchornet import AnchorGraspNet
from models.localgraspnet import PointMultiGraspNet
from train_utils import *

# for grasp experiment
from ros_utils.panda_control import PandaCommander
import tf2_ros
import franka_msgs.msg
import sensor_msgs.msg
import geometry_msgs.msg
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs.point_cloud2 as pcl2

from ros_utils import ros_utils
from ros_utils.transform import Rotation, Transform
from ros_utils.panda_control import PandaCommander

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint-path', default=None)

# 2d
parser.add_argument('--input-h', type=int)
parser.add_argument('--input-w', type=int)
parser.add_argument('--sigma', type=int, default=10)
parser.add_argument('--use-depth', type=int, default=1)
parser.add_argument('--use-rgb', type=int, default=1)
parser.add_argument('--ratio', type=int, default=8)
parser.add_argument('--anchor-k', type=int, default=6)
parser.add_argument('--anchor-w', type=float, default=50.0)
parser.add_argument('--anchor-z', type=float, default=20.0)
parser.add_argument('--grid-size', type=int, default=8)

# pc
parser.add_argument('--anchor-num', type=int)
parser.add_argument('--all-points-num', type=int)
parser.add_argument('--center-num', type=int)
parser.add_argument('--group-num', type=int)

# grasp detection
parser.add_argument('--heatmap-thres', type=float, default=0.01)
parser.add_argument('--local-k', type=int, default=10)
parser.add_argument('--local-thres', type=float, default=0.01)
parser.add_argument('--rotation-num', type=int, default=1)

# others
parser.add_argument('--random-seed', type=int, default=123, help='Random seed')

args = parser.parse_args()


class PointCloudHelper:

    def __init__(self, all_points_num) -> None:
        # precalculate x,y map
        self.all_points_num = all_points_num
        self.output_shape = (80, 45)
        # get intrinsics
        # intrinsics = get_camera_intrinsic()
        intrinsics = np.array([ [914.76495, 0.0, 639.85388],
                                [0.0, 914.05573, 356.87897],
                                [0.0, 0.0, 1.0]])
        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        cx, cy = intrinsics[0, 2], intrinsics[1, 2]
        # cal x, y
        ymap, xmap = np.meshgrid(np.arange(720), np.arange(1280))
        points_x = (xmap - cx) / fx
        points_y = (ymap - cy) / fy
        self.points_x = torch.from_numpy(points_x).float()
        self.points_y = torch.from_numpy(points_y).float()
        # for get downsampled xyz map
        ymap, xmap = np.meshgrid(np.arange(self.output_shape[1]),
                                 np.arange(self.output_shape[0]))
        factor = 1280 / self.output_shape[0]
        points_x = (xmap - cx / factor) / (fx / factor)
        points_y = (ymap - cy / factor) / (fy / factor)
        self.points_x_downscale = torch.from_numpy(points_x).float()
        self.points_y_downscale = torch.from_numpy(points_y).float()

    def to_scene_points(self,
                        rgbs: torch.Tensor,
                        depths: torch.Tensor,
                        include_rgb=True):
        batch_size = rgbs.shape[0]
        feature_len = 3 + 3 * include_rgb
        points_all = -torch.ones(
            (batch_size, self.all_points_num, feature_len),
            dtype=torch.float32).cuda()
        # cal z
        idxs = []
        masks = (depths > 0)
        cur_zs = depths / 1000.0

        print("to_scene_points")


        cur_xs = self.points_x.cuda() * cur_zs
        cur_ys = self.points_y.cuda() * cur_zs

        print("normalized")


        for i in range(batch_size):
            # convert point cloud to xyz maps
            points = torch.stack([cur_xs[i], cur_ys[i], cur_zs[i]], axis=-1)
            # remove zero depth
            mask = masks[i]
            points = points[mask]
            colors = rgbs[i][:, mask].T

            # random sample if points more than required
            if len(points) >= self.all_points_num:
                cur_idxs = random.sample(range(len(points)),
                                         self.all_points_num)
                points = points[cur_idxs]
                colors = colors[cur_idxs]
                # save idxs for concat fusion
                idxs.append(cur_idxs)

            # concat rgb and features after translation
            if include_rgb:
                points_all[i] = torch.concat([points, colors], axis=1)
            else:
                points_all[i] = points
        return points_all, idxs, masks

    def to_xyz_maps(self, depths):
        # downsample
        downsample_depths = F.interpolate(depths[:, None],
                                          size=self.output_shape,
                                          mode='nearest').squeeze(1).cuda()
        # convert xyzs
        cur_zs = downsample_depths / 1000.0
        cur_xs = self.points_x_downscale.cuda() * cur_zs
        cur_ys = self.points_y_downscale.cuda() * cur_zs
        xyzs = torch.stack([cur_xs, cur_ys, cur_zs], axis=-1)
        return xyzs.permute(0, 3, 1, 2)


def inference(view_points,
              xyzs,
              x,
              ori_depth,
              vis_heatmap=False,
              vis_grasp=True):
    with torch.no_grad():
        # 2d prediction
        pred_2d, perpoint_features = anchornet(x)

        loc_map, cls_mask, theta_offset, height_offset, width_offset = \
            anchor_output_process(*pred_2d, sigma=args.sigma)

        # detect 2d grasp (x, y, theta)
        rect_gg = detect_2d_grasp(loc_map,
                                  cls_mask,
                                  theta_offset,
                                  height_offset,
                                  width_offset,
                                  ratio=args.ratio,
                                  anchor_k=args.anchor_k,
                                  anchor_w=args.anchor_w,
                                  anchor_z=args.anchor_z,
                                  mask_thre=args.heatmap_thres,
                                  center_num=args.center_num,
                                  grid_size=args.grid_size,
                                  grasp_nms=args.grid_size,
                                  reduce='max')

        # check 2d result
        if rect_gg.size == 0:
            print('No 2d grasp found')
            return

        # show heatmap
        if vis_heatmap:
            # plt.clf()   # 清空当前 figure
            rgb_t = x[0, 1:].cpu().numpy().squeeze().transpose(2, 1, 0)
            resized_rgb = Image.fromarray((rgb_t * 255.0).astype(np.uint8))
            resized_rgb = np.array(
                resized_rgb.resize((args.input_w, args.input_h))) / 255.0
            depth_t = ori_depth.cpu().numpy().squeeze().T
            plt.subplot(221)
            plt.imshow(rgb_t)
            plt.subplot(222)
            plt.imshow(depth_t);
            plt.subplot(223)
            plt.imshow(loc_map.squeeze().T, cmap='jet')
            plt.subplot(224)
            rect_rgb = rect_gg.plot_rect_grasp_group(resized_rgb, 0)
            plt.imshow(rect_rgb)
            plt.tight_layout()
            plt.show()

        # feature fusion
        points_all = feature_fusion(view_points[..., :3], perpoint_features,
                                    xyzs)
        rect_ggs = [rect_gg]
        pc_group, valid_local_centers = data_process(
            points_all,
            ori_depth,
            rect_ggs,
            args.center_num,
            args.group_num, (args.input_w, args.input_h),
            min_points=32,
            is_training=False)
        rect_gg = rect_ggs[0]
        # batch_size == 1 when valid
        points_all = points_all.squeeze()

        # get 2d grasp info (not grasp itself) for trainning
        grasp_info = np.zeros((0, 3), dtype=np.float32)
        g_thetas = rect_gg.thetas[None]
        g_ws = rect_gg.widths[None]
        g_ds = rect_gg.depths[None]
        cur_info = np.vstack([g_thetas, g_ws, g_ds])
        grasp_info = np.vstack([grasp_info, cur_info.T])
        grasp_info = torch.from_numpy(grasp_info).to(dtype=torch.float32,
                                                     device='cuda')

        # localnet
        _, pred, offset = localnet(pc_group, grasp_info)

        # detect 6d grasp from 2d output and 6d output
        _, pred_rect_gg = detect_6d_grasp_multi(rect_gg,
                                                pred,
                                                offset,
                                                valid_local_centers,
                                                (args.input_w, args.input_h),
                                                anchors,
                                                k=args.local_k)

        # collision detect
        pred_grasp_from_rect = pred_rect_gg.to_6d_grasp_group(depth=0.02)
        pred_gg, _ = collision_detect(points_all,
                                      pred_grasp_from_rect,
                                      mode='graspnet')

        # nms
        pred_gg = pred_gg.nms()

        # show grasp
        if vis_grasp:
            print('pred grasp num ==', len(pred_gg))
            grasp_geo = pred_gg.to_open3d_geometry_list()
            points = view_points[..., :3].cpu().numpy().squeeze()
            colors = view_points[..., 3:6].cpu().numpy().squeeze()
            vispc = o3d.geometry.PointCloud()
            vispc.points = o3d.utility.Vector3dVector(points)
            vispc.colors = o3d.utility.Vector3dVector(colors)
            # 创建世界坐标系，调整size以确保足够粗
            world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])
            # 可视化点云、抓取几何体和世界坐标系
            o3d.visualization.draw_geometries([vispc, world_frame] + sum(grasp_geo, []))
            # o3d.visualization.draw_geometries([vispc] + grasp_geo)
            # o3d.visualization.draw_geometries([vispc] + sum(grasp_geo, []))
        return pred_gg
    
# 自加 抓取检测流程函数
def process_images(color_image, depth_image, pc_helper, point_cloud_is_needed=False):
    # read image and conver to tensor
    ori_depth = np.array(depth_image)
    ori_rgb = np.array(color_image)
    ori_depth = np.clip(ori_depth, 0, 1000)
    ori_rgb = torch.from_numpy(ori_rgb).permute(2, 1, 0)[None]
    ori_rgb = ori_rgb.to(device='cuda', dtype=torch.float32)
    ori_depth = torch.from_numpy(ori_depth.astype(np.float32)).T[None]
    ori_depth = ori_depth.to(device='cuda', dtype=torch.float32)

    # get scene points
    view_points, _, _ = pc_helper.to_scene_points(ori_rgb,
                                                  ori_depth,
                                                  include_rgb=True)
    # get xyz maps
    xyzs = pc_helper.to_xyz_maps(ori_depth)

    # pre-process
    rgb = F.interpolate(ori_rgb, (args.input_w, args.input_h))
    depth = F.interpolate(ori_depth[None], (args.input_w, args.input_h))[0]
    depth = depth / 1000.0
    depth = torch.clip((depth - depth.mean()), -1, 1)
    # generate 2d input
    x = torch.concat([depth[None], rgb], 1)
    x = x.to(device='cuda', dtype=torch.float32)

    # inference
    pred_gg = inference(view_points,
                        xyzs,
                        x,
                        ori_depth,
                        vis_heatmap=False,
                        vis_grasp=False)
    if point_cloud_is_needed:
        return pred_gg.sort(), view_points.squeeze()
    else:
        return pred_gg.sort()

# 定义订阅者节点
class HGGD_Node:
    def __init__(self):
        # 初始化节点
        rospy.init_node('HGGD_node', anonymous=True)

        # set up pc transform helper
        self.pc_helper = PointCloudHelper(all_points_num=args.all_points_num)

        # 标志位，分别用于深度图像和RGB图像
        self.rgb_flag = False
        self.depth_flag = False

        # 存储图像数据
        self.rgb_image = None
        self.depth_image = None

        # 创建CvBridge对象
        self.bridge = CvBridge()

        # 定义订阅者（放到最后，以免回调函数打断初始化）

        # 抓取实验相关
        self.executing_grasp = False    #表征是否在抓取动作中
        self.base_frame_id = "panda_link0"
        # self.tool0_frame_id = "panda_hand"
        self.tool0_frame_id = "panda_hand_tcp"
        self.setup_panda_control()
        print("建立buffer")
        self.tf_buffer = tf2_ros.Buffer()
        listener = tf2_ros.TransformListener(self.tf_buffer)
        print("开始lookup")
        rospy.sleep(1.0)  # 等待 TF 缓存
        temp = self.tf_buffer.lookup_transform(self.base_frame_id, self.tool0_frame_id, rospy.Time(0))
        print("顺利lookup")
        self.T_base_tool = Transform.from_dict({
            "translation": [temp.transform.translation.x, temp.transform.translation.y, temp.transform.translation.z],
            "rotation": [temp.transform.rotation.x, temp.transform.rotation.y, temp.transform.rotation.z, temp.transform.rotation.w]
        })

        # tool_camera 变换由标定得到
#         self.T_base_camera = Transform.from_matrix(np.array([
#             [-0.01416513, -0.99945893,  0.02968485,  0.04220781],
#  [ 0.99936383, -0.01512309, -0.0322989,  -0.02859493],
#  [ 0.03273035,  0.02920844,  0.99903733,  0.03199675],
#  [ 0.  ,        0.  ,        0.     ,     1.        ]]))

        self.T_tool_camera = Transform.from_matrix(np.array([[ 0.02409077, -0.99966291,  0.00968029,  0.05977349],
                                                            [ 0.98316255,  0.0254454,   0.18095284, -0.04056437],
                                                            [-0.18113816,  0.00515801,  0.98344413,  0.01788398],
                                                            [ 0.        , 0.          ,0.          ,1.        ]]))

        # T_fake_grasp 因为算法对抓取的定义和夹爪工具坐标系的定义不同而产生
        self.T_fake_grasp = Transform.from_matrix(np.array([
            [ 0, 0, 1, 0],
            [ 0, -1, 0, 0],
            [ 1, 0, 0, 0],
            [ 0, 0, 0, 1]
        ]))

        print("到发布相关")
        # 发布相关
        self.tf_broadcaster = tf2_ros.StaticTransformBroadcaster()
        self.publish_grasp_pose(self.T_tool_camera, "panda_hand", "camera_link")
        # self.publish_grasp_pose(self.T_base_camera, "panda_link0", "camera_link")
        self.pc_pub = rospy.Publisher('/point_cloud', PointCloud2, queue_size=1)

        # 深度相机坐标系到基坐标系
        temp = self.tf_buffer.lookup_transform(self.base_frame_id, "camera_aligned_depth_to_color_frame", rospy.Time(0), rospy.Duration(4.0))
        self.T_base_depth = Transform.from_dict({
            "translation": [temp.transform.translation.x, temp.transform.translation.y, temp.transform.translation.z],
            "rotation": [temp.transform.rotation.x, temp.transform.rotation.y, temp.transform.rotation.z, temp.transform.rotation.w]
        })

        # 抓取位姿过滤条件
        self.T_base_threshold = Transform.from_matrix(np.array([
            [ 1, 0, 0, 0],
            [ 0, -1, 0, 0],
            [ 0, 0, -1, 0.01],
            [ 0, 0, 0, 1]
        ]))

        # 定义订阅者（放到最后，以免回调函数打断初始化）
        self.depth_sub = rospy.Subscriber('/camera/aligned_depth_to_color/image_raw', Ros_Image, self.depth_callback)
        self.rgb_sub = rospy.Subscriber('/camera/color/image_raw', Ros_Image, self.rgb_callback)

        print("ImageSubscriberNode initialized\n")

    def depth_callback(self, depth_msg):
        """
        处理深度图像的回调函数
        """

        # 处理深度图像
        depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
        self.depth_image = np.array(depth_image, dtype=np.float32)
        self.depth_flag = True  # 设置深度图像标志位
        # rospy.loginfo("Depth image received and processed.")

        # 检查两个标志位，如果都为True，则执行处理
        if self.rgb_flag:
            self.process_images()

    def rgb_callback(self, rgb_msg):
        """
        处理RGB图像的回调函数
        """

        # 处理RGB图像
        rgb_image = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="bgr8")
        self.rgb_image = np.array(rgb_image, dtype=np.float32) / 255.0
        self.rgb_flag = True  # 设置RGB图像标志位
        # rospy.loginfo("RGB image received and processed.")

        # 检查两个标志位，如果都为True，则执行处理
        if self.depth_flag:
            self.process_images()

    def process_images(self):
        """
        当RGB图像和深度图像都收到后，处理这两张图像
        """
        if self.executing_grasp:
            # rospy.logwarn("Grasp in progress, skipping current image processing.")
            return
        
        # rospy.loginfo("Both images received, processing...")
        self.executing_grasp = True  # 锁定
        
        try:
            gg, points_tensor = process_images(self.rgb_image, self.depth_image, self.pc_helper, point_cloud_is_needed=True)
            self.pc_pub.publish(self.create_PC_msg(points_tensor))

            # 在这里进行抓取的选择和执行
            # 选择抓取并转换格式得到位置，并判断其可达性
            trans = np.zeros((4, 4))
            width = 0
            highest = -1
            real = None
            T_base_grasp = None
            # 取抓取宽度在10cm之内的抓取配置
            for i in range(len(gg)):
                if (gg.widths[i] <= 0.1) & (gg.widths[i] >= 0.005) & (gg.scores[i] >= 0.05):                
                    trans[:3, :3] = gg[i].rotation[:3, :3]
                    trans[:3, 3] = gg[i].translation
                    width = gg.widths[i]

                    # 将抓取位姿映射至基坐标系下，做进一步判断
                    T_depth_fake = Transform.from_matrix(trans)
                    T_base_fake = self.T_base_depth * T_depth_fake
                    T_base_grasp = T_base_fake * self.T_fake_grasp

                    R_ref = self.T_base_threshold.rotation.as_matrix()
                    R_grasp = T_base_grasp.rotation.as_matrix()

                    # 计算两个旋转矩阵之间的角度（使用旋转矩阵相对姿态差的公式）
                    R_diff = R_ref.T @ R_grasp
                    angle_diff = np.arccos(np.clip((np.trace(R_diff) - 1) / 2, -1.0, 1.0))  # 旋转角度（单位：弧度）

                    print("z: ", T_base_grasp.translation[2], "angle_diff: ", angle_diff)
                    
                    if (T_base_grasp.translation[2] >= self.T_base_threshold.translation[2]) & (angle_diff < np.pi / 4 ):
                        print("score: ", gg.scores[i])
                        # break
                        if T_base_grasp.translation[2] > highest:
                            real = T_base_grasp
                            highest = T_base_grasp.translation[2]

                # if i == len(gg) - 1:
                if (i == len(gg) - 1) & (highest < 0):
                    print("No suitable grasp configuration found.")
                    return

            T_base_grasp = real
            # 发布抓取变换（或可视化抓取结果）和点云数据
            self.publish_grasp_pose(T_base_grasp, self.base_frame_id, "grasp_pose")
            # self.pc_pub.publish(self.create_PC_msg(points_tensor))
            # print("pc shape: ", points_tensor.shape)

            self.run_grasp(T_base_grasp)

            rospy.loginfo("Finished.")

        except Exception as e:
            rospy.logerr(f"Grasp process failed: {e}")

        # 处理完后，重置标志位
        finally:
            # 无论成功失败，都解锁
            self.rgb_flag = False
            self.depth_flag = False
            self.executing_grasp = False

    def publish_grasp_pose(self, T, parent, sub):
        static_transform_stamped = geometry_msgs.msg.TransformStamped()

        static_transform_stamped.header.stamp = rospy.Time.now()
        static_transform_stamped.header.frame_id = parent
        static_transform_stamped.child_frame_id = sub

        static_transform_stamped.transform.translation.x = T.translation[0]
        static_transform_stamped.transform.translation.y = T.translation[1]
        static_transform_stamped.transform.translation.z = T.translation[2]

        quat = T.rotation.as_quat()
        static_transform_stamped.transform.rotation.x = quat[0]
        static_transform_stamped.transform.rotation.y = quat[1]
        static_transform_stamped.transform.rotation.z = quat[2]
        static_transform_stamped.transform.rotation.w = quat[3]

        self.tf_broadcaster.sendTransform(static_transform_stamped)

    def create_PC_msg(self, points_tensor, frame_ID="camera_aligned_depth_to_color_frame"):
        """
        将 PyTorch 张量转换为 ROS PointCloud2 消息格式
        :param points_tensor: (N, 6) 的 PyTorch 张量，前3列为 xyz，后3列为 rgb
        :return: PointCloud2 消息
        """
        # 确保数据在 CPU 上，并转换为 NumPy 数组
        points_numpy = points_tensor.cpu().numpy()

        # 提取 xyz 和 rgb 数据
        xyz = points_numpy[:, :3]  # (N, 3)
        rgb = points_numpy[:, 3:6]  # (N, 3)

        rgb = (rgb * 255).astype(np.uint8)

        # 计算 RGB 值并转换为 float32
        rgb_uint32 = (rgb[:, 0].astype(np.uint32) << 16) | (rgb[:, 1].astype(np.uint32) << 8) | rgb[:, 2].astype(np.uint32)
        rgb_float = rgb_uint32.view(np.float32).reshape(-1, 1)

        # 组合 XYZ 和 RGB
        cloud_data = np.hstack((xyz, rgb_float))

        # 定义 PointCloud2 字段
        fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="rgb", offset=12, datatype=PointField.FLOAT32, count=1),
        ]

        # 生成 PointCloud2 消息
        header = rospy.Header()
        header.stamp = rospy.Time.now()
        header.frame_id = frame_ID  # 根据实际情况修改

        return pcl2.create_cloud(header, fields, cloud_data)

    def run_grasp(self, T):
        rospy.loginfo("Executing the Grasp...")

        # 达到预抓取位置
        self.execute_grasp(T)

        # 移动到指定位置, drop
        self.drop()

        # 回到观察位
        self.pc.home()
        # self.pc.goto_joints([0.22543255500980752, -1.1640961296761363, -0.6900010590818441, -2.4830706984332176, -0.12215280903249538, 1.5819833138944133, 0.6934830441433523],0.2,0.2)
        # time.sleep(0.5)

        return
    
    def setup_panda_control(self):
        # rospy.Subscriber(
        #     "/franka_state_controller/franka_states",
        #     franka_msgs.msg.FrankaState,
        #     self.robot_state_cb,
        #     queue_size=1,
        # )
        rospy.Subscriber(
            "/joint_states", sensor_msgs.msg.JointState, self.joints_cb, queue_size=1
        )
        self.pc = PandaCommander()
        self.pc.move_group.set_end_effector_link(self.tool0_frame_id)

    def joints_cb(self, msg):
        self.gripper_width = msg.position[7] + msg.position[8]

    def execute_grasp(self, pose:Transform):
        T_base_grasp = pose

        T_grasp_pregrasp = Transform(Rotation.identity(), [0.0, 0.0, -0.05])
        T_grasp_deep = Transform(Rotation.identity(), [0.0, 0.0, 0.035])
        T_grasp_retreat = Transform(Rotation.identity(), [0.0, 0.0, -0.05])
        T_base_pregrasp = T_base_grasp * T_grasp_pregrasp
        T_base_deep = T_base_grasp * T_grasp_deep
        T_base_retreat = T_base_grasp * T_grasp_retreat

        self.pc.goto_pose(T_base_pregrasp, velocity_scaling=0.2)
        
        # 加约束
        self.pc.set_approach_constraints(T_base_grasp.rotation.as_quat())

        self.pc.goto_pose(T_base_deep)

        self.pc.grasp(width=0.0, force=20.0)

        self.pc.goto_pose(T_base_retreat)

        # 解除约束
        self.pc.clear_constraints()

        # lift hand
        T_retreat_lift_base = Transform(Rotation.identity(), [0.0, 0.0, 0.1])
        T_base_lift = T_retreat_lift_base * T_base_retreat
        self.pc.goto_pose(T_base_lift)

        if self.gripper_width > 0.004:
            return True
        else:
            return False

    def drop(self):
        self.pc.goto_joints(
            [0.678, 0.097, 0.237, -1.63, -0.031, 1.756, 0.931], 0.2, 0.2
        )
        self.pc.move_gripper(0.08)

if __name__ == '__main__':
    # set torch and gpu setting
    np.set_printoptions(precision=4, suppress=True)
    torch.set_printoptions(precision=4, sci_mode=False)
    if torch.cuda.is_available():
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = False
    else:
        raise RuntimeError('CUDA not available')

    # random seed
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)

    # Init the model
    anchornet = AnchorGraspNet(in_dim=4,
                               ratio=args.ratio,
                               anchor_k=args.anchor_k)
    localnet = PointMultiGraspNet(info_size=3, k_cls=args.anchor_num**2)

    # gpu
    anchornet = anchornet.cuda()
    localnet = localnet.cuda()

    # Load checkpoint
    check_point = torch.load(args.checkpoint_path)
    anchornet.load_state_dict(check_point['anchor'])
    localnet.load_state_dict(check_point['local'])
    # load checkpoint
    basic_ranges = torch.linspace(-1, 1, args.anchor_num + 1).cuda()
    basic_anchors = (basic_ranges[1:] + basic_ranges[:-1]) / 2
    anchors = {'gamma': basic_anchors, 'beta': basic_anchors}
    anchors['gamma'] = check_point['gamma']
    anchors['beta'] = check_point['beta']
    logging.info('Using saved anchors')
    print('-> loaded checkpoint %s ' % (args.checkpoint_path))

    # network eval mode
    anchornet.eval()
    localnet.eval()

    # Initial the Node
    node = HGGD_Node()
    # HGGD_Node.pc.home()

    rospy.spin()