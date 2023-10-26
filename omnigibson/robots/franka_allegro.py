import os
import numpy as np
from typing import Iterable
from omni.isaac.motion_generation import LulaKinematicsSolver

import omnigibson.utils.transform_utils as T
from omnigibson.macros import gm
from omnigibson.robots.manipulation_robot import ManipulationRobot


class FrankaAllegro(ManipulationRobot):
    """
    Franka Robot with Allegro hand
    """

    def __init__(
        self,
        # Shared kwargs in hierarchy
        name,
        prim_path=None,
        class_id=None,
        uuid=None,
        scale=None,
        visible=True,
        visual_only=False,
        self_collisions=True,
        load_config=None,
        fixed_base=True,

        # Unique to USDObject hierarchy
        abilities=None,

        # Unique to ControllableObject hierarchy
        control_freq=None,
        controller_config=None,
        action_type="continuous",
        action_normalize=True,
        reset_joint_pos=None,

        # Unique to BaseRobot
        obs_modalities="all",
        proprio_obs="default",
        sensor_config=None,

        # Unique to ManipulationRobot
        grasping_mode="physical",

        **kwargs,
    ):
        """
        Args:
            name (str): Name for the object. Names need to be unique per scene
            prim_path (None or str): global path in the stage to this object. If not specified, will automatically be
                created at /World/<name>
            class_id (None or int): What class ID the object should be assigned in semantic segmentation rendering mode.
                If None, the ID will be inferred from this object's category.
            uuid (None or int): Unique unsigned-integer identifier to assign to this object (max 8-numbers).
                If None is specified, then it will be auto-generated
            scale (None or float or 3-array): if specified, sets either the uniform (float) or x,y,z (3-array) scale
                for this object. A single number corresponds to uniform scaling along the x,y,z axes, whereas a
                3-array specifies per-axis scaling.
            visible (bool): whether to render this object or not in the stage
            visual_only (bool): Whether this object should be visual only (and not collide with any other objects)
            self_collisions (bool): Whether to enable self collisions for this object
            load_config (None or dict): If specified, should contain keyword-mapped values that are relevant for
                loading this prim at runtime.
            abilities (None or dict): If specified, manually adds specific object states to this object. It should be
                a dict in the form of {ability: {param: value}} containing object abilities and parameters to pass to
                the object state instance constructor.
            control_freq (float): control frequency (in Hz) at which to control the object. If set to be None,
                simulator.import_object will automatically set the control frequency to be 1 / render_timestep by default.
            controller_config (None or dict): nested dictionary mapping controller name(s) to specific controller
                configurations for this object. This will override any default values specified by this class.
            action_type (str): one of {discrete, continuous} - what type of action space to use
            action_normalize (bool): whether to normalize inputted actions. This will override any default values
                specified by this class.
            reset_joint_pos (None or n-array): if specified, should be the joint positions that the object should
                be set to during a reset. If None (default), self.default_joint_pos will be used instead.
            obs_modalities (str or list of str): Observation modalities to use for this robot. Default is "all", which
                corresponds to all modalities being used.
                Otherwise, valid options should be part of omnigibson.sensors.ALL_SENSOR_MODALITIES.
                Note: If @sensor_config explicitly specifies `modalities` for a given sensor class, it will
                    override any values specified from @obs_modalities!
            proprio_obs (str or list of str): proprioception observation key(s) to use for generating proprioceptive
                observations. If str, should be exactly "default" -- this results in the default proprioception
                observations being used, as defined by self.default_proprio_obs. See self._get_proprioception_dict
                for valid key choices
            sensor_config (None or dict): nested dictionary mapping sensor class name(s) to specific sensor
                configurations for this object. This will override any default values specified by this class.
            grasping_mode (str): One of {"physical", "assisted", "sticky"}.
                If "physical", no assistive grasping will be applied (relies on contact friction + finger force).
                If "assisted", will magnetize any object touching and within the gripper's fingers.
                If "sticky", will magnetize any object touching the gripper's fingers.
            kwargs (dict): Additional keyword arguments that are used for other super() calls from subclasses, allowing
                for flexible compositions of various object subclasses (e.g.: Robot is USDObject + ControllableObject).
        """

        # Run super init
        super().__init__(
            prim_path=prim_path,
            name=name,
            class_id=class_id,
            uuid=uuid,
            scale=scale,
            visible=visible,
            fixed_base=fixed_base,
            visual_only=visual_only,
            self_collisions=self_collisions,
            load_config=load_config,
            abilities=abilities,
            control_freq=control_freq,
            controller_config=controller_config,
            action_type=action_type,
            action_normalize=action_normalize,
            reset_joint_pos=reset_joint_pos,
            obs_modalities=obs_modalities,
            proprio_obs=proprio_obs,
            sensor_config=sensor_config,
            grasping_mode=grasping_mode,
            **kwargs,
        )

    @property
    def model_name(self):
        return "FrankaAllegro"

    @property
    def discrete_action_list(self):
        # Not supported for this robot
        raise NotImplementedError()

    def _create_discrete_action_space(self):
        # Fetch does not support discrete actions
        raise ValueError("Franka does not support discrete actions!")

    def update_controller_mode(self):
        super().update_controller_mode()
        # overwrite joint params here
        for i in range(7):
            self.joints[f"panda_joint{i+1}"].damping = 10
            self.joints[f"panda_joint{i+1}"].stiffness = 1000
        for i in range(16):
            self.joints[f"joint_{i}_0"].damping = 10
            self.joints[f"joint_{i}_0"].stiffness = 1000

    @property
    def controller_order(self):
        return ["arm_{}".format(self.default_arm), "gripper_{}".format(self.default_arm)]

    @property
    def _default_controllers(self):
        controllers = super()._default_controllers
        controllers["arm_{}".format(self.default_arm)] = "InverseKinematicsController"
        controllers["gripper_{}".format(self.default_arm)] = "MultiFingerGripperController"
        return controllers
    
    @property
    def _default_gripper_multi_finger_controller_configs(self):
        return {self.default_arm: {
            "name": "MultiFingerGripperController",
            "control_freq": self._control_freq,
            "motor_type": "position",
            "control_limits": self.control_limits,
            "dof_idx": self.gripper_control_idx[self.default_arm],
            "command_input_limits": None,
            "mode": "independent",
        }}

    @property
    def default_joint_pos(self):
        # position where the hand is parallel to the ground
        return np.r_[[0.86, -0.27, -0.68, -1.52, -0.18, 1.29, 1.72], np.zeros(16)]

    @property
    def finger_lengths(self):
        return {self.default_arm: 0.1}

    @property
    def arm_control_idx(self):
        return {self.default_arm: np.arange(7)}

    @property
    def gripper_control_idx(self):
        # thumb.proximal, ..., thumb.tip, ..., ring.tip
        return {self.default_arm: np.array([8, 12, 16, 20, 10, 14, 18, 22, 9, 13, 17, 21, 7, 11, 15, 19])}

    @property
    def arm_link_names(self):
        return {self.default_arm: [f"panda_link{i}" for i in range(8)]}

    @property
    def arm_joint_names(self):
        return {self.default_arm: [f"panda_joint_{i+1}" for i in range(7)]}

    @property
    def eef_link_names(self):
        return {self.default_arm: "base_link"}

    @property
    def finger_link_names(self):
        return {self.default_arm: [f"link_{i}_0" for i in range(16)]}

    @property
    def finger_joint_names(self):
        return {self.default_arm: [f"joint_{i}_0" for i in range(16)]}

    @property
    def usd_path(self):
        return os.path.join(gm.ASSET_PATH, "models/franka/franka_allegro.usd")
    
    @property
    def robot_arm_descriptor_yamls(self):
        return {self.default_arm: os.path.join(gm.ASSET_PATH, "models/franka/franka_allegro_description.yaml")}
    
    @property
    def robot_gripper_descriptor_yamls(self):
        return {
            finger: os.path.join(gm.ASSET_PATH, f"models/franka/allegro_{finger}_description.yaml")
            for finger in ["thumb", "index", "middle", "ring"]
        }

    @property
    def urdf_path(self):
        return os.path.join(gm.ASSET_PATH, "models/franka/franka_allegro.urdf")
    
    @property
    def gripper_urdf_path(self):
        return os.path.join(gm.ASSET_PATH, "models/franka/allegro_hand.urdf")
    
    @property
    def disabled_collision_pairs(self):
        return [
            ["link_12_0", "part_studio_link"],
        ]
    
    @property
    def vr_rotation_offset(self):
        return {self.default_arm: T.euler2quat(np.array([0, np.pi / 2, 0]))}

    def gen_action_from_vr_data(self, vr_data: dict):
        hand_data = vr_data["hand_data"]
        action = np.zeros(22)
        if "right" in hand_data and "raw" in hand_data["right"]:
            target_pos, target_orn = T.mat2pose(hand_data["right"]["raw"][0])
            target_pos = [0.6, 0, 0.7]
            target_orn = T.quat_multiply(target_orn, self.vr_rotation_offset[self.default_arm])
            cur_robot_eef_pos = self.links[self.eef_link_names[self.default_arm]].get_position()
            action[:6] = np.r_[target_pos - cur_robot_eef_pos, T.quat2axisangle(target_orn)]
            # set finger root joint positions to 0
            action[[6, 10, 14, 18]] = np.zeros(4)
            # joint order: thumb, index, middle, pinky
            angles = hand_data["right"]["angles"]
            for f_idx in range(4):
                for j_idx in range(3):
                    action[7 + f_idx * 4 + j_idx] = angles[f_idx][j_idx]
        return action


class AllegroIKController:
    """
    IK controller for Allegro hand, based on the LulaKinematicsSolver
    """
    def __init__(self, robot: FrankaAllegro, max_iter=100) -> None:
        """
        Initializes the IK controller
        Args:
            robot (FrankaAllegro): the Franka Allegro robot
            max_iter (int): maximum number of iterations for the IK solver, default is 100.
        """
        self.robot = robot
        self.fingers = {
            "ring":    ("link_3_0_tip",   np.array([0, 4, 8, 12])),
            "middle":   ("link_7_0_tip",    np.array([2, 6, 10, 14])),
            "index":     ("link_11_0_tip",    np.array([3, 7, 11, 15])),
            "thumb":    ("link_15_0_tip",   np.array([1, 5, 9, 13])), 
        }
        self.finger_ik_solvers = {}
        for finger in self.fingers.keys():
            self.finger_ik_solvers[finger]  = LulaKinematicsSolver(
                robot_description_path = robot.robot_gripper_descriptor_yamls[finger],
                urdf_path = robot.gripper_urdf_path
            )
            self.finger_ik_solvers[finger].ccd_max_iterations = max_iter

    def solve(self, target_gripper_pos:Iterable[float]) -> np.ndarray:
        """
        compute the joint positions given the position of each finger tip
        Args:
            target_gripper_pos (Iterable[float]): 12-array of target positions of the finger tips, in order for ring, middle, index, thumb
        Returns:
            np.ndarray: 16-array of joint positions (corresponding to robot gripper control indices)
        """
        # get the current finger joint positions
        finger_joint_positions = self.robot.get_joint_positions()[7:]
        if target_gripper_pos is not None:
            # get current hand base pose
            hand_base_pos, hand_base_orn = self.robot.links["base_link"].get_position_orientation()
            # Grab the finger joint positions in order to reach the desired finger pose
            for i, finger in enumerate(self.fingers.keys()):
                self.finger_ik_solvers[finger].set_robot_base_pose(hand_base_pos, T.convert_quat(hand_base_orn, "wxyz"))
                finger_joint_pos, success = self.finger_ik_solvers[finger].compute_inverse_kinematics(
                    frame_name=self.fingers[finger][0],
                    target_position=target_gripper_pos[3 * i: 3 * (i + 1)],
                    target_orientation=None,
                    warm_start=finger_joint_positions[self.fingers[finger][1]]
                )
                if success:
                    finger_joint_positions[self.fingers[finger][1]] = finger_joint_pos
            
        return finger_joint_positions
