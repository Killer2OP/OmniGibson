import carb
import numpy as np
from omni.kit.xr.core import XRCore, XRDeviceClass, XRCoreEventType, XRProfileEventType
from omni.kit.xr.ui.stage.common import XRAvatarManager
from pxr import Gf
from typing import Iterable, List, Optional

import omnigibson as og
import omnigibson.utils.transform_utils as T
from omnigibson.robots.robot_base import BaseRobot
from omnigibson.robots.manipulation_robot import ManipulationRobot

class VRSys():
    def __init__(
        self, 
        system: str="OpenXR",
        vr_robot: Optional[BaseRobot]=None,
        show_controller: bool=False,
        disable_display_output: bool=False,
        enable_anchor_movement: bool=False,
        use_hand_tracking: bool=False,
    ) -> None:
        """
        Initializes the VR system
        Args:
            vr_robot (None of BaseRobot): the robot that VR will control.
            system (str): the VR system to use, one of ["OpenXR", "SteamVR"], default is "OpenXR".
            show_controller (bool): whether to show the controller model in the scene, default is False.
            disable_display_output (bool): whether we will not display output to the VR headset (only use controller tracking), default is False.
            enable_anchor_movement (bool): whether to enable VR system anchor movement, default is False.
            use_hand_tracking (bool): whether to use hand tracking instead of controllers, default is False.
        """
        self.xr_core = XRCore.get_singleton()
        self.vr_profile = self.xr_core.get_profile("vr")
        self.enable_anchor_movement = enable_anchor_movement
        # set avatar
        if show_controller:
            self.vr_profile.set_avatar(XRAvatarManager.get_singleton().create_avatar("basic_avatar", {}))
        else:
            self.vr_profile.set_avatar(XRAvatarManager.get_singleton().create_avatar("empty_avatar", {}))
        # set anchor mode to be scene origin
        carb.settings.get_settings().set(self.vr_profile.get_scene_persistent_path() + "anchorMode", "scene origin")
        # set vr system
        carb.settings.get_settings().set(self.vr_profile.get_persistent_path() + "system/display", system)
        # set display mode
        carb.settings.get_settings().set(self.vr_profile.get_persistent_path() + "disableDisplayOutput", disable_display_output)
        carb.settings.get_settings().set('/rtx/rendermode', "RaytracedLighting")
        # robot info
        self.vr_robot = vr_robot
        self.robot_attached = False
        self.reset_button_pressed = False
        # devices info
        self.hmd = None
        self.controllers = {}
        self.trackers = {}
        # setup event subscriptions
        self.use_hand_tracking = use_hand_tracking
        if use_hand_tracking:
            self.hand_data = {}
            self._hand_tracking_subscription = self.xr_core.get_event_stream().create_subscription_to_pop_by_type(
                XRCoreEventType.hand_joints, self.get_hand_tracking_data, name="hand tracking"
            )
            


    def xr2og(self, transform: np.ndarray) -> np.ndarray:
        """
        Apply the transform matrix from the Omniverse XR coordinate system to the OmniGibson coordinate system
        Args:
            transform (np.ndarray): the transform matrix in the Omniverse XR coordinate system
        Returns:
            np.ndarray: the transform matrix in the OmniGibson coordinate system
        """
        return transform.T @ T.pose2mat((np.zeros(3), T.euler2quat(np.array([0, np.pi / 2, np.pi / 2]))))
    
    def og2xr(self, transform: np.ndarray) -> np.ndarray:
        """
        Apply the transform matrix from the OmniGibson coordinate system to the Omniverse XR coordinate system
        Args:
            transform (np.ndarray): the transform matrix in the OmniGibson coordinate system
        Returns:
            np.ndarray: the transform matrix in the Omniverse XR coordinate system
        """
        return transform.T @ T.pose2mat((np.zeros(3), T.euler2quat(np.array([0, -np.pi / 2, -np.pi / 2]))))

    @property
    def is_enabled(self) -> bool:
        """
        Checks whether the VR system is enabled
        Returns:
            bool: whether the VR system is enabled
        """
        return self.vr_profile.is_enabled()

    def start(self) -> None:
        """
        Starts the VR system by enabling the VR profile
        """
        if self.vr_robot:
            self.vr_robot.set_position_orientation([0, 0, 0], [0, 0, 0, 1])
        self.vr_profile.request_enable_profile()
        for _ in range(100):
            og.sim.step()
        self.update_devices()
        assert self.vr_profile.is_enabled(), "[VRSys] VR profile not enabled!"
    
    def step(self) -> dict:
        """
        Steps the VR system
        Returns:
            dict: a dictionary of VR data containing device transforms, controller button data, and (optionally) hand tracking data
        """
        vr_data = {}
        # update device list
        self.update_devices()
        # update anchor
        self.update_anchor()
        # get transforms
        vr_data["transforms"] = self.get_device_transforms()
        # get controller button data
        vr_data["button_data"] = self.get_controller_button_data()
        # update robot attachment info
        if vr_data["button_data"][1]["press"]["grip"]:
            if not self.reset_button_pressed:
                self.reset_button_pressed = True
                self.robot_attached = not self.robot_attached
                # teleport controller to match robot end effector (ManipulationRobot only)
                if self.robot_attached and self.vr_robot and isinstance(self.vr_robot, ManipulationRobot):
                    robot_eef_position = self.vr_robot.links[self.vr_robot.eef_link_names[self.vr_robot.default_arm]].get_position()
                    controller_position = self.xr2og(np.array(self.controllers[1].get_transform()))[:3, 3]
                    self.update_anchor(robot_eef_position - controller_position)
        else:
            self.reset_button_pressed = False
        vr_data["robot_attached"] = self.robot_attached
        # Optionally get hand tracking data
        if self.use_hand_tracking:
            vr_data["hand_data"] = self.hand_data
        return vr_data

    def stop(self) -> None:
        """
        disable VR profile
        """
        self.xr_core.request_disable_profile()
        self.sim.step()
        assert not self.vr_profile.is_enabled(), "[VRSys] VR profile not disabled!"

    def update_devices(self) -> None:
        """
        Update the VR device list
        """
        for device in self.vr_profile.get_device_list():
            if device.get_class() == XRDeviceClass.xrdisplaydevice:
                self.hmd = device
            elif device.get_class() == XRDeviceClass.xrcontroller:
                self.controllers[device.get_index()] = device
            elif device.get_class() == XRDeviceClass.xrtracker:
                self.trackers[device.get_index()] = device
        assert self.hmd is not None, "[VRSys] HMD not detected! Please make sure you have a VR headset connected to your computer."

    def update_anchor(self, offset: Optional[Iterable[float]]=None) -> None:
        """
        Updates the anchor of the xr system in the virtual world
        Args:
            offset (Iterable[float]): the offset to apply to the anchor. If None, will conpute the offset based on controller input and current hmd pose.
        """
        # get controller axis input
        if offset is None:
            offset = np.zeros(3)
            if self.enable_anchor_movement:
                if 1 in self.controllers:
                    right_axis_state = self.controllers[1].get_axis_state()
                    # calculate right and forward vectors based on hmd transform
                    hmd_transform = self.hmd.get_transform()
                    right, forward = hmd_transform[0][:3], hmd_transform[2][:3]
                    right = right / np.linalg.norm(right)
                    forward = forward / np.linalg.norm(forward)
                    # calculate offset based on controller input
                    offset = np.array([right[i] * right_axis_state["touchpad_x"] - forward[i] * right_axis_state["touchpad_y"] for i in range(3)])
                    offset[2] = 0
                if 0 in self.controllers:
                    offset[2] = self.controllers[0].get_axis_state()["touchpad_y"]
                # normalize offset
                length = np.linalg.norm(offset)
                if length != 0:
                    offset *= 0.03 / length
        # set new anchor transform
        offset = Gf.Vec3d(offset[0], offset[2], offset[1])
        self.vr_profile.add_move_physical_world_relative_to_device(offset)  # this move action is applied instantly

    def get_device_transforms(self) -> dict:
        """
        Get the transform matrix of each VR device
        Returns:
            dict: a dictionary of device transforms, with keys "hmd", "controllers", "trackers"
        """
        transforms = {}
        transforms["hmd"] = self.xr2og(np.array(self.hmd.get_transform()))
        transforms["controllers"] = {}
        transforms["trackers"] = {}
        for controller_index in self.controllers:
            transforms["controllers"][controller_index] = self.xr2og(np.array(self.controllers[controller_index].get_transform()))
        for tracker_index in self.trackers:
            transforms["trackers"][tracker_index] = self.xr2og(np.array(self.trackers[tracker_index].get_transform()))
        return transforms

    def get_controller_button_data(self) -> dict:
        """
        Get the button data for each controller
        Returns:
            dict: a dictionary of whether each button is pressed or touched, and the axis state for touchpad and joysticks
        """
        button_data = {}
        for controller_index in self.controllers:
            button_data[controller_index] = {}
            button_data[controller_index]["press"] = self.controllers[controller_index].get_button_press_state()
            button_data[controller_index]["touch"] = self.controllers[controller_index].get_button_touch_state()
            button_data[controller_index]["axis"] = self.controllers[controller_index].get_axis_state()
        return button_data       
    
    def get_hand_tracking_data(self, e: carb.events.IEvent) -> None:
        """
        Get hand tracking data, see https://registry.khronos.org/OpenXR/specs/1.0/html/xrspec.html#convention-of-hand-joints for joint indices
        Args:
            e (carb.events.IEvent): event that contains hand tracking data as payload
        """
        e.consume()
        data_dict = e.payload
        for hand in ["left", "right"]:
            self.hand_data[hand] = {}
            if data_dict[f"joint_count_{hand}"] != 0:
                cur_hand_data = []
                hand_joint_matrices = data_dict[f"joint_matrices_{hand}"]
                for i in range(26):
                    cur_hand_data.append(self.xr2og(np.reshape(hand_joint_matrices[16 * i: 16 * (i + 1)], (4, 4))))
                self.hand_data[hand]["raw"] = np.array(cur_hand_data)
                self.hand_data[hand]["angles"] = self.get_joint_angle_from_hand_data(np.array(cur_hand_data))



    def get_joint_angle_from_hand_data(self, raw_hand_data: List[np.ndarray]) -> np.ndarray:
        """
        Get each finger joint's rotation angle from hand tracking data
        Each finger has 3 joints
        Args:
            raw_hand_data (List[np.ndarray]): a list of 26 matrices representing the hand tracking data
        Returns:
            np.ndarray: a 5 x 3 array of joint rotations (from thumb to pinky, from base to tip)
        """
        joint_angles = np.zeros((5, 3))
        for i in range(5):
            for j in range(3):
                # get the 3 related joints
                prev_joint_idx, cur_joint_idx, next_joint_idx = i * 5 + j + 1, i * 5 + j + 2, i * 5 + j + 3
                # get the 3 related joints' positions
                prev_joint_pos = raw_hand_data[prev_joint_idx][:3, 3]
                cur_joint_pos = raw_hand_data[cur_joint_idx][:3, 3]
                next_joint_pos = raw_hand_data[next_joint_idx][:3, 3]
                # calculate the angle formed by 3 points
                v1 = cur_joint_pos - prev_joint_pos
                v2 = next_joint_pos - cur_joint_pos
                v1 /= np.linalg.norm(v1)
                v2 /= np.linalg.norm(v2)
                joint_angles[i, j] = np.arccos(v1 @ v2)
        return joint_angles
