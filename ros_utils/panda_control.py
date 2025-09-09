import actionlib
import franka_gripper.msg
import moveit_commander
from moveit_msgs.msg import Constraints, OrientationConstraint
import rospy
import time
import numpy as np
import geometry_msgs.msg

import ros_utils.ros_utils as ros_utils


class PandaCommander(object):
    def __init__(self):
        self.name = "panda_arm"
        self._connect_to_move_group()
        self._connect_to_gripper()
        self.create_planning_scene()
        rospy.loginfo("PandaCommander ready")

    def _connect_to_move_group(self):
        self.robot = moveit_commander.RobotCommander()
        self.scene = moveit_commander.PlanningSceneInterface()
        self.move_group = moveit_commander.MoveGroupCommander(self.name)

    def _connect_to_gripper(self):
        self.grasp_client = actionlib.SimpleActionClient(
            "/franka_gripper/grasp", franka_gripper.msg.GraspAction
        )
        self.grasp_client.wait_for_server()
        rospy.loginfo("Connected to grasp action server")
        self.move_client = actionlib.SimpleActionClient(
            "/franka_gripper/move", franka_gripper.msg.MoveAction
        )
        self.move_client.wait_for_server()
        rospy.loginfo("Connected to move action server")

    def home(self):
        self.goto_joints([0, -0.785, 0, -2.356, 0, 1.57, 0.785], 0.2, 0.2)
        time.sleep(0.5)
        # self.goto_joints([0, -0.785, 0, -2.356, 0, 1.57, 0.785, 0.01, 0.01], 0.2, 0.2)

    def goto_joints(self, joints, velocity_scaling=0.1, acceleration_scaling=0.1):
        self.move_group.set_max_velocity_scaling_factor(velocity_scaling)
        self.move_group.set_max_acceleration_scaling_factor(acceleration_scaling)
        self.move_group.set_joint_value_target(joints)
        plan = self.move_group.plan()[1]
        success = self.move_group.execute(plan, wait=True)
        self.move_group.stop()

        # # 等待关节位置达到目标，不好使，经查卡在最后
        # if success:
        #     current_joints = self.move_group.get_current_joint_values()
        #     while not np.allclose(current_joints, joints, atol=0.01):
        #         time.sleep(0.05)  # 等待 50 毫秒
        #         current_joints = self.move_group.get_current_joint_values()

        return success

    def goto_pose(self, pose, velocity_scaling=0.1, acceleration_scaling=0.1):
        pose_msg = ros_utils.to_pose_msg(pose)
        self.move_group.set_max_velocity_scaling_factor(velocity_scaling)
        self.move_group.set_max_acceleration_scaling_factor(acceleration_scaling)
        self.move_group.set_pose_target(pose_msg)
        plan = self.move_group.plan()[1]
        success = self.move_group.execute(plan, wait=True)
        self.move_group.stop()
        self.move_group.clear_pose_targets()
        return success

    def grasp(self, width=0.0, e_inner=0.1, e_outer=0.1, speed=0.1, force=10.0):
        epsilon = franka_gripper.msg.GraspEpsilon(e_inner, e_outer)
        goal = franka_gripper.msg.GraspGoal(width, epsilon, speed, force)
        self.grasp_client.send_goal(goal)
        return self.grasp_client.wait_for_result(rospy.Duration(2.0))

    def move_gripper(self, width, speed=0.1):
        goal = franka_gripper.msg.MoveGoal(width, speed)
        self.move_client.send_goal(goal)
        return self.move_client.wait_for_result(rospy.Duration(2.0))
    
    def create_planning_scene(self):
        # collision box for table
        msg = geometry_msgs.msg.PoseStamped()
        msg.header.frame_id = "world"
        msg.pose.orientation.w = 1.0
        msg.pose.orientation.x = 0
        msg.pose.orientation.y = 0
        msg.pose.orientation.z = 0
        # msg.pose.position.x = 0
        # msg.pose.position.y = 0
        # msg.pose.position.z = 0
        # msg.pose.position.z -= 0.02
        # self.scene.add_box("table", msg, size=(3, 3, 0.05))
        msg.pose.position.x = 0.35
        msg.pose.position.y = 0
        msg.pose.position.z = 0.04
        msg.pose.position.z -= 0
        self.scene.add_box("table", msg, size=(0.6, 3, 0.16))

    def set_approach_constraints(self, quat):
        q_msg = geometry_msgs.msg.Quaternion()
        q_msg.x = quat[0]
        q_msg.y = quat[1]
        q_msg.z = quat[2]
        q_msg.w = quat[3]

        # 1. 创建姿态约束
        orientation_constraint = OrientationConstraint()
        orientation_constraint.link_name = "panda_hand_tcp"  # 你的末端执行器链接名
        orientation_constraint.header.frame_id = "panda_link0"  # 参考坐标系
        orientation_constraint.orientation = q_msg  # 固定姿态
        orientation_constraint.absolute_x_axis_tolerance = 0.05
        orientation_constraint.absolute_y_axis_tolerance = 0.05
        orientation_constraint.absolute_z_axis_tolerance = 0.05
        orientation_constraint.weight = 1.0

        # 2. 添加到整体约束
        constraints = Constraints()
        constraints.orientation_constraints.append(orientation_constraint)

        # 3. 应用约束
        self.move_group.set_path_constraints(constraints)

    def clear_constraints(self):
        self.move_group.clear_path_constraints()