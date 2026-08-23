import argparse
import mujoco
import numpy as np
import time
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.pinocchio_kinematic import Kinematics
from src.utils import transform2mat, mat2transform
from src.state_machine import StateMachine
from src.contact_detection import ContactDetector
from src.target_smoother import RateLimiter, SecondOrderTargetSmoother
from src.offscreen_renderer import OffscreenCameraRenderer
from src.episode_recorder import EpisodeRecorder


class TaskStateMachine(StateMachine):
    """抓取放置任务状态机"""

    STATE_IDLE = "idle"
    STATE_APPROACH = "approach"          # 移动到方块上方
    STATE_DESCEND = "descend"            # 下降接近方块
    STATE_GRASP = "grasp"                # 闭合夹爪
    STATE_LIFT = "lift"                  # 提升方块
    STATE_MOVE = "move"                  # 移动到目标位置
    STATE_PLACE_DESCEND = "place_descend"  # 下降放置
    STATE_RELEASE = "release"            # 打开夹爪
    STATE_SETTLE = "settle"              # 夹爪全开、机械臂静止, 等方块落稳
    STATE_RETURN = "return"              # 返回 home (先升到 lifting 高度再回到home位置,避免手指扫过刚放下的方块)
    STATE_FAILED = "failed"              # 任务失败

    # 各状态超时(秒)
    TIMEOUTS = {
        STATE_APPROACH: 10.0,
        STATE_DESCEND: 8.0,
        STATE_GRASP: 3.0,
        STATE_LIFT: 8.0,
        STATE_MOVE: 10.0,
        STATE_PLACE_DESCEND: 8.0,
        STATE_RELEASE: 5.0,
        STATE_SETTLE: 3.0,
        STATE_RETURN: 12.0,
    }

    def __init__(self):
        super().__init__(initial_state=self.STATE_IDLE, timeouts=self.TIMEOUTS)


class AutomatedPickAndPlaceAdvanced:
    GRIP_OFFSET = 0.100
    APPROACH_HEIGHT = 0.08 
    LIFT_HEIGHT = 0.08
    GRIP_XY_OFFSET = 0.020
    PLACE_XY_OFFSET = 0.017
    DESCEND_SPEED = 0.03
    TARGET_SMOOTH_FREQ = 0.6   # 二阶临界阻尼滤波器把加减速变成S曲线
    CARTESIAN_VEL_LIMIT = 0.15  # 笛卡尔目标速度上限

    GRIPPER_OPEN = 1.0
    GRIPPER_CLOSE = 0.30    # 0.30时夹持力足够抓起方块
    JAW_RATE_LIMIT = 2.0    # 夹爪开合速度限制, 防止撞击方块
    GRASP_TOL = 0.006

    CUBE_REST_Z = 0.02

    SOURCE_BOX_CENTER = np.array([0.21, -0.10, 0.0])
    SOURCE_BOX_HALF = 0.05
    TARGET_BOX_CENTER = np.array([0.10, -0.10, 0.0])
    TARGET_BOX_HALF = 0.05
    CUBE_JITTER_HALF = 0.03
    CUBE_DEFAULT_POS = np.array([0.21, -0.10, 0.02])

    def __init__(
        self,
        scene_path: str = "model/trs_so_arm100/scene_with_cube_and_cameras.xml",
        arm_model_path: str = "model/trs_so_arm100/so_arm100.xml",
        fps: int = 30,
        image_size: tuple = (480, 640),
        gui: bool = True,
        randomize: bool = False,
        seed: int = 42,
        output_dir: str = "data/automated_pick_place_dataset",
    ):
        self.scene_path = scene_path
        self.arm_model_path = arm_model_path
        self.fps = fps
        self.image_size = image_size
        self.randomize = randomize
        self.rng = np.random.default_rng(seed)
        self.output_dir = output_dir

        self.model = mujoco.MjModel.from_xml_path(scene_path)
        self.data = mujoco.MjData(self.model)

        self.model.opt.timestep = 0.005
        self.control_dt = 1.0 / fps
        self.n_sim_steps = int(self.control_dt / self.model.opt.timestep)

        self.ee_body_name = "Fixed_Jaw"
        self.ee_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, self.ee_body_name)
        self.cube_body_name = "cube"
        self.cube_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, self.cube_body_name)
        self.cube_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "cube")
        self.cube_qadr = self.model.jnt_qposadr[self.cube_joint_id]
        self.cube_dof_adr = self.model.jnt_dofadr[self.cube_joint_id]
        self.jaw_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "Jaw")

        self.arm_joint_names = ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw"]
        self.arm_joint_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in self.arm_joint_names]

        self.gripper_open_val = self.GRIPPER_OPEN
        self.gripper_close_val = self.GRIPPER_CLOSE
        self.gripper_target = self.gripper_open_val
        self.gripper_cmd = self.gripper_open_val  # 限速后的实际指令 (防高速撞击)
        self._gripper_limiter = RateLimiter(self.gripper_open_val, self.JAW_RATE_LIMIT)

        self.ik_solver = Kinematics(self.ee_body_name)
        self.ik_solver.buildFromMJCF(arm_model_path)
        self.last_q = None  # IK 热启动

        self.home_q = np.array([0, -1.57, 1.57, 1.57, -1.57, 0], dtype=np.float32)
        self.cube_default_pos = self.CUBE_DEFAULT_POS.copy()
        self.cube_pos = self.cube_default_pos.copy()
        self._reset_to_home()
        self.home_ee_pos = self.data.body(self.ee_body_id).xpos.copy().astype(np.float32)
        R = self.data.body(self.ee_body_id).xmat.reshape(3, 3)
        M = np.eye(4)
        M[:3, :3] = R
        M[:3, 3] = self.home_ee_pos
        # 全程保持 home 朝向 (夹爪竖直向下)
        self.ee_target_rot = np.array(mat2transform(M)[3:], dtype=np.float32)

        # 笛卡尔目标平滑
        self._ee_smoother = SecondOrderTargetSmoother(
            dim=3, freq=self.TARGET_SMOOTH_FREQ,
            vel_limit=self.CARTESIAN_VEL_LIMIT, initial=self.home_ee_pos)
        self._sm_ee = self._ee_smoother.x.copy()

        # 目标框中心
        self.place_pos = np.array([
            self.TARGET_BOX_CENTER[0], self.TARGET_BOX_CENTER[1], self.CUBE_REST_Z,
        ], dtype=np.float32)

        self.cam_wrist_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist_camera")
        self.cam_top_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "top_camera")
        self.cam_side_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "side_camera")

        self.recording = False
        self.recorder = EpisodeRecorder(
            self.output_dir,
            base_metadata={
                "dataset_name": "automated_pick_place_so100",
                "robot": "so_arm100",
                "controller": "automated_pick_place_advanced",
                "fps": self.fps,
                "observation_features": {
                    "arm_q": (6,), "arm_dq": (6,), "gripper": (1,),
                    "ee_pos": (3,), "cube_pos": (3,),
                },
                "image_features": {
                    "wrist": (self.image_size[0], self.image_size[1], 3),
                    "top": (self.image_size[0], self.image_size[1], 3),
                    "side": (self.image_size[0], self.image_size[1], 3),
                },
            })
        self.start_time = 0
        self.stats = {"attempts": 0, "successes": 0, "failures": 0}

        self.handle = None
        self.images_enabled = True
        self._init_rendering(gui=gui)

        self.state_machine = TaskStateMachine()
        self.contact_detector = ContactDetector(
            self.model, self.data,
            bodies_a=["Fixed_Jaw", "Moving_Jaw"],
            bodies_b=["cube"],
            force_threshold=0.01,
            debounce_steps=3)
        self.task_ok = False
        self.fail_reason = None

        self.t_approach = None
        self.t_grasp = None
        self.t_lift = None
        self.t_move = None
        self.t_place = None
        self.cube_start_pos = None
        self.descend_z = None

    def _init_rendering(self, gui=True):
        if gui:
            from mujoco import viewer as _mj_viewer
            self.handle = _mj_viewer.launch_passive(self.model, self.data)
            self.handle.cam.distance = 1.2
            self.handle.cam.azimuth = 140
            self.handle.cam.elevation = -30

        # 离屏渲染 (相机图像)
        self._renderer = OffscreenCameraRenderer(self.model, image_size=self.image_size)
        self.images_enabled = self._renderer.enabled

    def _reset_to_home(self):
        mujoco.mj_resetData(self.model, self.data)
        for i, jid in enumerate(self.arm_joint_ids):
            qpos_idx = self.model.jnt_qposadr[jid]
            self.data.qpos[qpos_idx] = self.home_q[i]
            self.data.ctrl[i] = self.home_q[i]
        self.gripper_target = self.gripper_open_val
        self._gripper_limiter.reset(self.gripper_open_val)
        self.gripper_cmd = self.gripper_open_val
        self.data.ctrl[5] = self.gripper_open_val
        self._reset_cube()
        mujoco.mj_forward(self.model, self.data)

    def _reset_cube(self, pos=None):
        pos = self.cube_default_pos if pos is None else pos
        self.data.qpos[self.cube_qadr:self.cube_qadr + 3] = pos
        self.data.qpos[self.cube_qadr + 3:self.cube_qadr + 7] = [1, 0, 0, 0]
        self.data.qvel[self.cube_dof_adr:self.cube_dof_adr + 6] = 0
        self.cube_pos = np.array(pos, dtype=np.float32)

    def _sample_cube_pos(self):
        for _ in range(10):
            pos = np.array([
                self.rng.uniform(
                    self.SOURCE_BOX_CENTER[0] - self.CUBE_JITTER_HALF,
                    self.SOURCE_BOX_CENTER[0] + self.CUBE_JITTER_HALF),
                self.rng.uniform(
                    self.SOURCE_BOX_CENTER[1] - self.CUBE_JITTER_HALF,
                    self.SOURCE_BOX_CENTER[1] + self.CUBE_JITTER_HALF),
                self.CUBE_REST_Z,
            ], dtype=np.float32)
            target = pos + np.array([self.GRIP_XY_OFFSET, 0, self.GRIP_OFFSET], dtype=np.float32)
            T = transform2mat(*target, *self.ee_target_rot)
            try:
                q, info = self.ik_solver.ik(T, current_arm_motor_q=self.home_q)
                fk = self.ik_solver.fk(q)
                if np.linalg.norm(fk[:3, 3] - target) < 0.008:
                    return pos
            except Exception:
                continue
        print("[Warn] 随机位置 IK 预检失败, 使用默认位置")
        return self.CUBE_DEFAULT_POS.copy()

    def _get_arm_state(self):
        arm_q = np.array([self.data.qpos[self.model.jnt_qposadr[jid]] for jid in self.arm_joint_ids], dtype=np.float32)
        arm_dq = np.array([self.data.qvel[self.model.jnt_dofadr[jid]] for jid in self.arm_joint_ids], dtype=np.float32)
        return arm_q, arm_dq

    def _capture_camera_image(self, camera_id):
        """渲染单个固定相机为 RGB (H, W, 3) uint8; 不可用时返回 None"""
        return self._renderer.render(self.data, camera_id)

    def _capture_images(self):
        images = {}
        if self.cam_wrist_id >= 0:
            images["wrist"] = self._capture_camera_image(self.cam_wrist_id)
        if self.cam_top_id >= 0:
            images["top"] = self._capture_camera_image(self.cam_top_id)
        if self.cam_side_id >= 0:
            images["side"] = self._capture_camera_image(self.cam_side_id)
        return images

    def _get_observation_dict(self):
        arm_q, arm_dq = self._get_arm_state()
        gripper = np.array([self.data.qpos[self.model.jnt_qposadr[self.jaw_joint_id]]], dtype=np.float32)
        ee_pos = self.data.body(self.ee_body_id).xpos.copy().astype(np.float32)
        cube_pos = self.data.body(self.cube_body_id).xpos.copy().astype(np.float32)
        return {
            "arm_q": arm_q.tolist(),
            "arm_dq": arm_dq.tolist(),
            "gripper": gripper.tolist(),
            "ee_pos": ee_pos.tolist(),
            "cube_pos": cube_pos.tolist(),
            "ee_target": (self.ee_target_pos.tolist() if self.ee_target_pos is not None else None),
            "task_state": self.state_machine.state,
            "task_progress": self.state_machine.progress,
        }

    def _update_gripper_cmd(self):
        self.gripper_cmd = self._gripper_limiter.step(self.gripper_target, self.control_dt)

    def _smooth_target(self):
        self._sm_ee = self._ee_smoother.step(self.ee_target_pos, self.control_dt)

    def _apply_ik_control(self):
        self._update_gripper_cmd()
        if self.ee_target_pos is None:
            self.data.ctrl[5] = self.gripper_cmd
            return
        self._smooth_target()
        target_mat = transform2mat(self._sm_ee[0], self._sm_ee[1], self._sm_ee[2],
                                   self.ee_target_rot[0], self.ee_target_rot[1], self.ee_target_rot[2])
        current_q = np.array([self.data.qpos[self.model.jnt_qposadr[jid]] for jid in self.arm_joint_ids],
                             dtype=np.float32)
        try:
            sol_q, info = self.ik_solver.ik(target_mat, current_arm_motor_q=current_q)
            sol_q = np.asarray(sol_q, dtype=np.float32)[:len(self.arm_joint_ids)]
            self.last_q = sol_q
            for i in range(len(self.arm_joint_ids)):
                self.data.ctrl[i] = sol_q[i]
        except Exception:
            # IK 偶发失败时保持上一组控制量
            if self.last_q is not None:
                for i in range(len(self.arm_joint_ids)):
                    self.data.ctrl[i] = self.last_q[i]
        self.data.ctrl[5] = self.gripper_cmd

    def _ee_pos(self):
        return self.data.body(self.ee_body_id).xpos.copy()

    def _cube_pos(self):
        return self.data.body(self.cube_body_id).xpos.copy()

    def _jaw_q(self):
        return float(self.data.qpos[self.model.jnt_qposadr[self.jaw_joint_id]])

    def _reached(self, target, tol=GRASP_TOL):
        return np.linalg.norm(self._ee_pos() - np.asarray(target)) < tol

    def _start_approach(self):
        self.t_approach = np.array([
            self.cube_pos[0] + self.GRIP_XY_OFFSET,
            self.cube_pos[1],
            self.cube_pos[2] + self.GRIP_OFFSET + self.APPROACH_HEIGHT,
        ], dtype=np.float32)
        self.ee_target_pos = self.t_approach.copy()
        self.state_machine.transition_to(TaskStateMachine.STATE_APPROACH)

    def _update_approach(self):
        if self._reached(self.t_approach) or self.state_machine.timed_out():
            self._start_descend()

    def _start_descend(self):
        self.t_grasp = np.array([
            self.cube_pos[0] + self.GRIP_XY_OFFSET,
            self.cube_pos[1],
            self.cube_pos[2] + self.GRIP_OFFSET,
        ], dtype=np.float32)
        self.descend_z = self._ee_pos()[2]
        self.ee_target_pos = self.t_grasp.copy()
        self.state_machine.transition_to(TaskStateMachine.STATE_DESCEND)

    def _update_descend(self):
        dt = self.control_dt
        # 目标高度逐步下降
        self.descend_z = max(self.t_grasp[2], self.descend_z - self.DESCEND_SPEED * dt)
        self.ee_target_pos[2] = self.descend_z
        # 接触即停 (指尖碰到方块侧面)
        self.contact_detector.update()
        if self.descend_z <= self.t_grasp[2] + 0.001 or self.contact_detector.has_contact \
                or self.state_machine.timed_out():
            self._start_grasp()

    def _start_grasp(self):
        self.ee_target_pos = self.t_grasp.copy()
        self.gripper_target = self.gripper_close_val
        self.state_machine.transition_to(TaskStateMachine.STATE_GRASP)

    def _update_grasp(self):
        if self._jaw_q() <= self.gripper_close_val + 0.05 or self.state_machine.timed_out():
            self._start_lift()

    def _start_lift(self):
        self.t_lift = self.t_grasp + np.array([0, 0, self.LIFT_HEIGHT], dtype=np.float32)
        self.ee_target_pos = self.t_lift.copy()
        self.state_machine.transition_to(TaskStateMachine.STATE_LIFT)

    def _update_lift(self):
        if self._reached(self.t_lift) or self.state_machine.timed_out():
            # 抓起校验: 方块必须被抬起
            cube = self._cube_pos()
            if cube[2] > self.cube_start_pos[2] + 0.04 and \
               np.linalg.norm(cube[:2] - self.t_lift[:2]) < 0.06:
                self._start_move()
            else:
                self._fail(f"方块未被抓起 (cube z={cube[2]:.3f}, 期望 > {self.cube_start_pos[2] + 0.04:.3f})")

    def _start_move(self):
        self.t_move = np.array([
            self.place_pos[0] + self.PLACE_XY_OFFSET,
            self.place_pos[1],
            self.t_lift[2],
        ], dtype=np.float32)
        self.ee_target_pos = self.t_move.copy()
        self.state_machine.transition_to(TaskStateMachine.STATE_MOVE)

    def _update_move(self):
        if self._reached(self.t_move) or self.state_machine.timed_out():
            self._start_place_descend()

    def _start_place_descend(self):
        self.t_place = np.array([
            self.place_pos[0] + self.PLACE_XY_OFFSET,
            self.place_pos[1],
            self.place_pos[2] + self.GRIP_OFFSET,
        ], dtype=np.float32)
        self.descend_z = self._ee_pos()[2]
        self.ee_target_pos = self.t_place.copy()
        self.state_machine.transition_to(TaskStateMachine.STATE_PLACE_DESCEND)

    def _update_place_descend(self):
        dt = self.control_dt
        self.descend_z = max(self.t_place[2], self.descend_z - self.DESCEND_SPEED * dt)
        self.ee_target_pos[2] = self.descend_z
        if self.descend_z <= self.t_place[2] + 0.001 or self.state_machine.timed_out():
            self.ee_target_pos = self.t_place.copy()
            self._start_release()

    def _start_release(self):
        self.gripper_target = self.gripper_open_val
        self.state_machine.transition_to(TaskStateMachine.STATE_RELEASE)

    def _update_release(self):
        if self._jaw_q() >= self.gripper_open_val - 0.15 or self.state_machine.timed_out():
            self._start_settle()

    def _start_settle(self):
        self.ee_target_pos = self.t_place.copy()
        self.state_machine.transition_to(TaskStateMachine.STATE_SETTLE)

    def _update_settle(self):
        self.state_machine.progress += 1
        if self.state_machine.progress >= 30 or self.state_machine.timed_out():
            self._start_return()

    def _start_return(self):
        self.return_waypoint = self.t_place + np.array([0, 0, self.LIFT_HEIGHT], dtype=np.float32)
        self.return_at_home = False
        self.ee_target_pos = self.return_waypoint.copy()
        self.state_machine.transition_to(TaskStateMachine.STATE_RETURN)

    def _update_return(self):
        if getattr(self, "return_at_home", False):
            if self._reached(self.home_ee_pos, tol=0.01) or self.state_machine.timed_out():
                self._finish_check()
        else:
            self.ee_target_pos = self.return_waypoint.copy()
            if self._reached(self.return_waypoint, tol=0.01):
                self.return_at_home = True
                self.ee_target_pos = self.home_ee_pos.copy()

    def _fail(self, reason):
        self.task_ok = False
        self.fail_reason = reason
        self.state_machine.transition_to(TaskStateMachine.STATE_FAILED)
        print(f"[FAIL] {reason}")

    def _finish_check(self):
        cube = self._cube_pos()
        xy_err = np.linalg.norm(cube[:2] - self.place_pos[:2])
        z_err = abs(cube[2] - self.place_pos[2])
        contact = self.contact_detector.update()
        if xy_err < 0.015 and z_err < 0.015 and not contact and self._jaw_q() >= 0.5:
            self.task_ok = True
            self.fail_reason = None
            print(f"[OK] 任务成功: 方块 xy 误差 {xy_err * 1000:.1f}mm, z 误差 {z_err * 1000:.1f}mm")
        else:
            self._fail(f"放置校验失败 (xy 误差 {xy_err * 1000:.1f}mm > 15mm 或 "
                       f"z 误差 {z_err * 1000:.1f}mm > 15mm 或仍有夹爪接触)")
            return
        self.state_machine.transition_to(TaskStateMachine.STATE_IDLE)

    def _update_task(self):
        if self.state_machine.state == TaskStateMachine.STATE_IDLE:
            return
        sm = self.state_machine
        if sm.state == TaskStateMachine.STATE_APPROACH:
            self._update_approach()
        elif sm.state == TaskStateMachine.STATE_DESCEND:
            self._update_descend()
        elif sm.state == TaskStateMachine.STATE_GRASP:
            self._update_grasp()
        elif sm.state == TaskStateMachine.STATE_LIFT:
            self._update_lift()
        elif sm.state == TaskStateMachine.STATE_MOVE:
            self._update_move()
        elif sm.state == TaskStateMachine.STATE_PLACE_DESCEND:
            self._update_place_descend()
        elif sm.state == TaskStateMachine.STATE_RELEASE:
            self._update_release()
        elif sm.state == TaskStateMachine.STATE_SETTLE:
            self._update_settle()
        elif sm.state == TaskStateMachine.STATE_RETURN:
            self._update_return()

    def _save_episode(self):
        cube_end = self._cube_pos()
        metadata = {
            "episode_length": self.recorder.step_count,
            "success": True,
            "cube_start": self.cube_start_pos.tolist(),
            "cube_end": cube_end.tolist(),
            "place_target": self.place_pos.tolist(),
            "source_box_center": self.SOURCE_BOX_CENTER.tolist(),
            "target_box_center": self.TARGET_BOX_CENTER.tolist(),
            "box_half_size": self.SOURCE_BOX_HALF,
        }
        episode_dir = self.recorder.save_episode(metadata=metadata)
        print(f"[Save] episode {self.recorder.episode_num - 1} "
              f"({metadata['episode_length']} steps) -> {episode_dir}")

    def _record_step(self, elapsed):
        obs = self._get_observation_dict()
        images = self._capture_images()
        step_data = {
            "timestamp": elapsed,
            "observation": obs,
            "images": images,
            "gripper_target": float(self.gripper_target),
            "ee_target": self.ee_target_pos.tolist() if self.ee_target_pos is not None else None,
        }
        self.recorder.add_step(step_data)

    def run_one_episode(self, record=True, verbose=True):
        self.recording = record
        self.recorder.discard_episode()
        self.task_ok = False
        self.fail_reason = None
        self.stats["attempts"] += 1

        # 重置并设置本集目标
        self._reset_to_home()
        self.cube_default_pos = self._sample_cube_pos() if self.randomize else self.CUBE_DEFAULT_POS.copy()
        self._reset_cube(self.cube_default_pos)
        mujoco.mj_forward(self.model, self.data)
        self.cube_start_pos = self._cube_pos().copy()
        self.last_q = None

        if verbose:
            print(f"\n{'=' * 60}\nEpisode {self.recorder.episode_num} (attempt {self.stats['attempts']}): "
                  f"cube at {np.round(self.cube_start_pos, 3)}\n{'=' * 60}")

        self.ee_target_pos = self.home_ee_pos.copy()
        self.gripper_target = self.gripper_open_val
        # 平滑器从当前位置起步
        self._ee_smoother.reset(self.home_ee_pos)
        self.state_machine.transition_to(TaskStateMachine.STATE_IDLE)
        self._start_approach()
        self.start_time = time.time()

        while self.state_machine.state not in (TaskStateMachine.STATE_IDLE, TaskStateMachine.STATE_FAILED):
            elapsed = time.time() - self.start_time

            self._update_task()
            self._apply_ik_control()
            for _ in range(self.n_sim_steps):
                mujoco.mj_step(self.model, self.data)

            if self.recording:
                self._record_step(elapsed)

            if self.handle:
                self.handle.sync()

            if verbose and self.recorder.step_count % 30 == 0:
                print(f"\r  {self.state_machine.state:14s} progress={self.state_machine.progress:5.2f} "
                      f"ee={np.round(self._ee_pos(), 3)} cube={np.round(self._cube_pos(), 3)}", end="")
                sys.stdout.flush()
            print() if self.state_machine.state in (TaskStateMachine.STATE_FAILED, TaskStateMachine.STATE_IDLE) else None

            time.sleep(self.control_dt)

        if verbose:
            print()

        if self.task_ok:
            self.stats["successes"] += 1
            if self.recording and self.recorder.step_count > 0:
                self._save_episode()
        else:
            self.stats["failures"] += 1
            self.recorder.discard_episode()  # 失败不保存
        return self.task_ok

    def run_collection(self, episodes=50, record=True):
        print("\n" + "=" * 60)
        print("SO-100 Advanced Automated Box-to-Box Pickup and Place - Data Collection")
        print("=" * 60)
        print(f"任务: 从源框 {np.round(self.SOURCE_BOX_CENTER[:2], 2)} "
              f"-> 目标框 {np.round(self.TARGET_BOX_CENTER[:2], 2)} (方块初始位置源框内随机)")
        print(f"目标: 成功保存 {episodes} 集 (失败自动重试)")
        print(f"输出: {self.output_dir}\n")

        saved = 0
        t0 = time.time()
        while saved < episodes:
            if self.run_one_episode(record=record):
                saved += 1
                print(f"\n>>> 已保存 {saved}/{episodes} 集 "
                      f"(累计 {self.stats['attempts']} 次尝试, 用时 {time.time() - t0:.0f}s)")
            if saved == 0 and self.stats["attempts"] >= episodes * 3:
                print(f"[Abort] {self.stats['attempts']} 次尝试无一成功, 提前退出。请检查参数。")
                break

        print(f"\n{'=' * 60}\n采集完成: 保存 {saved} 集, 尝试 {self.stats['attempts']} 次, "
              f"总用时 {time.time() - t0:.0f}s\n{'=' * 60}")
        return saved

    def run_loop(self):
        try:
            from pynput import keyboard
        except Exception as e:
            print(f"[Warn] pynput 不可用 ({e}), 交互模式不可用")
            return

        print("\n" + "=" * 60)
        print("SO-100 Advanced Pickup and Place (Interactive)")
        print("=" * 60)
        print("Controls:  T: start task | R: reset cube | Q: quit\n" + "=" * 60 + "\n")

        self.keyboard = keyboard.Listener(on_press=self._on_key_press, on_release=self._on_key_release)
        self.keyboard.start()
        try:
            while self.keyboard.running:
                self._apply_ik_control()
                for _ in range(self.n_sim_steps):
                    mujoco.mj_step(self.model, self.data)
                if self.handle:
                    self.handle.sync()
                time.sleep(self.control_dt)
        except KeyboardInterrupt:
            print("\nInterrupted")
        finally:
            self.keyboard.stop()
            self.close()

    def _on_key_press(self, key):
        try:
            if hasattr(key, 'char'):
                if key.char == 't':
                    self.run_one_episode(record=True)
                elif key.char == 'r':
                    self._reset_cube()
                elif key.char == 'q':
                    self.keyboard.running = False
        except Exception:
            pass

    def _on_key_release(self, key):
        pass

    def close(self):
        if self.handle:
            self.handle.close()
        self._renderer.close()


def main():
    parser = argparse.ArgumentParser(description="SO-100 自动抓取放置数据采集")
    parser.add_argument("--episodes", type=int, default=100, help="要采集的 episode 数 (默认 100)")
    parser.add_argument("--no-gui", action="store_true", help="无头模式: 不打开查看器窗口")
    parser.add_argument("--no-record", action="store_true", help="只跑任务不保存数据")
    parser.add_argument("--no-randomize", action="store_true",
                        help="禁用每集在源框内随机化方块位置 (默认开启)")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--output-dir", type=str, default="data/automated_pick_place_dataset",
                        help="数据输出目录")
    parser.add_argument("--interactive", action="store_true", help="交互模式 (需键盘)")
    args = parser.parse_args()

    gui = not args.no_gui and os.environ.get("DISPLAY") is not None
    collector = AutomatedPickAndPlaceAdvanced(
        scene_path="model/trs_so_arm100/scene_with_cube_and_cameras.xml",
        arm_model_path="model/trs_so_arm100/so_arm100.xml",
        fps=30,
        image_size=(480, 640),
        gui=gui,
        randomize=not args.no_randomize,
        seed=args.seed,
        output_dir=args.output_dir,
    )

    try:
        if args.interactive:
            collector.run_loop()
        else:
            collector.run_collection(episodes=args.episodes, record=not args.no_record)
    finally:
        collector.close()
        print("Done!")


if __name__ == "__main__":
    main()
