from ros_utils.panda_control import PandaCommander
import rospy
import tf2_ros
import franka_msgs.msg
import sensor_msgs.msg
import numpy as np

from ros_utils import ros_utils
from ros_utils.transform import Rotation, Transform
from ros_utils.panda_control import PandaCommander

# tag lies on the table in the center of the workspace
T_base_tag = Transform(Rotation.identity(), [0.42, 0.02, 0.21])
round_id = 0


class PandaGraspController():
    def __init__(self):
        self.robot_error = False

        self.base_frame_id = "panda_link0"
        self.tool0_frame_id = "panda_hand"
        # self.T_tool0_tcp = Transform.from_dict(rospy.get_param("~T_tool0_tcp"))  # TODO
        # self.T_tcp_tool0 = self.T_tool0_tcp.inverse()
        # self.finger_depth = rospy.get_param("~finger_depth")
        # self.scan_joints = rospy.get_param("~scan_joints")

        self.setup_panda_control()
        # self.tf_tree = ros_utils.TransformTree()
        # self.define_workspace()
        # self.create_planning_scene()

        rospy.loginfo("Ready to take action")

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

    def execute_grasp(self, grasp):
        T_task_grasp = grasp.pose
        T_base_grasp = self.T_base_task * T_task_grasp

        T_grasp_pregrasp = Transform(Rotation.identity(), [0.0, 0.0, -0.05])
        T_grasp_retreat = Transform(Rotation.identity(), [0.0, 0.0, -0.05])
        T_base_pregrasp = T_base_grasp * T_grasp_pregrasp
        T_base_retreat = T_base_grasp * T_grasp_retreat

        self.pc.goto_pose(T_base_pregrasp * self.T_tcp_tool0, velocity_scaling=0.2)
        self.approach_grasp(T_base_grasp)

        if self.robot_error:
            return False

        self.pc.grasp(width=0.0, force=20.0)

        if self.robot_error:
            return False

        self.pc.goto_pose(T_base_retreat * self.T_tcp_tool0)

        # lift hand
        T_retreat_lift_base = Transform(Rotation.identity(), [0.0, 0.0, 0.1])
        T_base_lift = T_retreat_lift_base * T_base_retreat
        self.pc.goto_pose(T_base_lift * self.T_tcp_tool0)

        if self.gripper_width > 0.004:
            return True
        else:
            return False

    def approach_grasp(self, T_base_grasp):
        self.pc.goto_pose(T_base_grasp * self.T_tcp_tool0)

    def drop(self):
        self.pc.goto_joints(
            [0.678, 0.097, 0.237, -1.63, -0.031, 1.756, 0.931], 0.2, 0.2
        )
        self.pc.move_gripper(0.08)




if __name__ == "__main__":
    rospy.init_node("panda_grasp")
    panda_grasp = PandaGraspController()

    tf_buffer = tf2_ros.Buffer()
    listener = tf2_ros.TransformListener(tf_buffer)
    trans = tf_buffer.lookup_transform(panda_grasp.base_frame_id, panda_grasp.tool0_frame_id, rospy.Time(0), rospy.Duration(4.0))

    print(trans)

    trans_dict = {
        "translation": [trans.transform.translation.x, trans.transform.translation.y, trans.transform.translation.z],
        "rotation": [trans.transform.rotation.x, trans.transform.rotation.y, trans.transform.rotation.z, trans.transform.rotation.w]
    }

    T = Transform.from_dict(trans_dict)

    print(T)

