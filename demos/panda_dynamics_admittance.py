"""Panda 机械臂末端 6 维导纳控制示例。"""

from collections import deque
from collections.abc import Sequence

import matplotlib.pyplot as plt
import mujoco
import numpy as np
import pinocchio as pin
from scipy.spatial.transform import Rotation

from src import mujoco_viewer

ARM_DOF = 7
CARTESIAN_DOF = 6
DEFAULT_TIMESTEP_S = 0.001
DEFAULT_FORCE_START_TIME_S = 0.5
DEFAULT_EXTERNAL_WRENCH_WORLD = (0.0, 0.0, 10.0, 0.0, 0.0, 0.0)
DEFAULT_VIRTUAL_INERTIA = (10.0, 10.0, 10.0, 1.0, 1.0, 1.0)
DEFAULT_VIRTUAL_DAMPING = (20.0, 20.0, 20.0, 4.0, 4.0, 4.0)
DEFAULT_VIRTUAL_STIFFNESS = (50.0, 50.0, 50.0, 10.0, 10.0, 10.0)
DEFAULT_PRINT_INTERVAL_STEPS = 200
MAX_HISTORY_SAMPLES = 100_000


class PandaAdmittanceDemo(mujoco_viewer.CustomViewer):
    """把世界坐标系外力转换为末端位姿偏移并用位置执行器跟踪。"""

    def __init__(
        self,
        scene_xml: str,
        arm_xml: str,
        *,
        ee_body_name: str = "ee_center_body",
        external_wrench_world: Sequence[float] = DEFAULT_EXTERNAL_WRENCH_WORLD,
        virtual_inertia: Sequence[float] = DEFAULT_VIRTUAL_INERTIA,
        virtual_damping: Sequence[float] = DEFAULT_VIRTUAL_DAMPING,
        virtual_stiffness: Sequence[float] = DEFAULT_VIRTUAL_STIFFNESS,
        force_start_time_s: float = DEFAULT_FORCE_START_TIME_S,
        print_interval_steps: int = DEFAULT_PRINT_INTERVAL_STEPS,
    ) -> None:
        super().__init__(scene_xml, distance=3, azimuth=-45, elevation=-30)

        if self.model.nkey == 0:
            raise ValueError("scene model must define a home keyframe")
        if self.model.nq < ARM_DOF or self.model.nv < ARM_DOF:
            raise ValueError("scene model must contain at least 7 arm joints")
        if self.model.nu < ARM_DOF:
            raise ValueError("scene model must contain at least 7 arm actuators")
        if force_start_time_s < 0:
            raise ValueError("force_start_time_s must be non-negative")
        if print_interval_steps <= 0:
            raise ValueError("print_interval_steps must be positive")

        self.arm_xml = arm_xml
        self.ee_body_name = ee_body_name
        self.external_wrench_world = self._six_vector(
            external_wrench_world, "external_wrench_world"
        )
        self.virtual_inertia = self._six_vector(
            virtual_inertia, "virtual_inertia", strictly_positive=True
        )
        self.virtual_damping = self._six_vector(
            virtual_damping, "virtual_damping", non_negative=True
        )
        self.virtual_stiffness = self._six_vector(
            virtual_stiffness, "virtual_stiffness", non_negative=True
        )
        self.force_start_time_s = force_start_time_s
        self.print_interval_steps = print_interval_steps

        self.max_force_n = 20.0
        self.max_torque_nm = 5.0
        self.max_translation_offset_m = 0.15
        self.max_rotation_offset_rad = np.deg2rad(20.0)
        self.max_linear_speed_m_s = 0.25
        self.max_angular_speed_rad_s = 0.5
        self.ik_damping = 0.02
        self.ik_iterations_per_step = 2
        self.max_ik_step_rad = 0.02

        self.setTimestep(DEFAULT_TIMESTEP_S)
        self.data.qpos[:] = self.model.key_qpos[0]
        self.data.qvel[:] = 0.0
        self.data.ctrl[:ARM_DOF] = self.data.qpos[:ARM_DOF]
        if self.model.nu > ARM_DOF:
            self.data.ctrl[ARM_DOF:] = self.model.key_ctrl[0, ARM_DOF:]

        self.ee_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, self.ee_body_name
        )
        if self.ee_body_id < 0:
            raise ValueError(f"MuJoCo body not found: {self.ee_body_name}")

        arm_actuators_limited = self.model.actuator_ctrllimited[:ARM_DOF]
        if not np.all(arm_actuators_limited):
            raise ValueError("all arm actuators must define ctrlrange limits")
        self.arm_ctrl_range_rad = self.model.actuator_ctrlrange[:ARM_DOF].copy()

        # position 执行器满足 tau = kp * (ctrl - q) - kv * qvel。
        self.position_actuator_kp = self.model.actuator_gainprm[:ARM_DOF, 0].copy()
        if np.any(self.position_actuator_kp <= 0):
            raise ValueError("all arm position actuators must have positive kp")

        self.step_index = 0
        self.time_s_history = deque(maxlen=MAX_HISTORY_SAMPLES)
        self.wrench_world_history = deque(maxlen=MAX_HISTORY_SAMPLES)
        self.offset_history = deque(maxlen=MAX_HISTORY_SAMPLES)
        self.target_position_m_history = deque(maxlen=MAX_HISTORY_SAMPLES)
        self.actual_position_m_history = deque(maxlen=MAX_HISTORY_SAMPLES)
        self.tracking_error_m_history = deque(maxlen=MAX_HISTORY_SAMPLES)

    @staticmethod
    def _six_vector(
        value: Sequence[float],
        name: str,
        *,
        non_negative: bool = False,
        strictly_positive: bool = False,
    ) -> np.ndarray:
        """校验并复制一个 6 维笛卡尔参数向量。"""
        vector = np.asarray(value, dtype=float)
        if vector.shape != (CARTESIAN_DOF,):
            raise ValueError(f"{name} must contain {CARTESIAN_DOF} values")
        if not np.all(np.isfinite(vector)):
            raise ValueError(f"{name} must contain only finite values")
        if strictly_positive and np.any(vector <= 0):
            raise ValueError(f"{name} must contain positive values")
        if non_negative and np.any(vector < 0):
            raise ValueError(f"{name} must contain non-negative values")
        return vector.copy()

    @staticmethod
    def _clip_vector_norm(
        vector: np.ndarray, max_norm: float
    ) -> tuple[np.ndarray, bool]:
        """保持方向不变，将向量的二范数限制在给定上限内。"""
        norm = float(np.linalg.norm(vector))
        if norm <= max_norm or norm == 0.0:
            return vector, False
        return vector * (max_norm / norm), True

    def runBefore(self) -> None:
        """构建运动学模型并记录不受力时的末端平衡位姿。"""
        robot = pin.RobotWrapper.BuildFromMJCF(self.arm_xml)
        self.pin_model = robot.model
        self.pin_data = self.pin_model.createData()

        if self.pin_model.nq != self.model.nq or self.pin_model.nv != self.model.nv:
            raise ValueError(
                "MuJoCo and Pinocchio model dimensions do not match: "
                f"MuJoCo (nq={self.model.nq}, nv={self.model.nv}), "
                f"Pinocchio (nq={self.pin_model.nq}, nv={self.pin_model.nv})"
            )

        self.pin_ee_frame_id = self.pin_model.getFrameId(self.ee_body_name)
        if self.pin_ee_frame_id >= self.pin_model.nframes:
            raise ValueError(f"Pinocchio frame not found: {self.ee_body_name}")

        mujoco.mj_forward(self.model, self.data)
        ee_body = self.data.body(self.ee_body_id)
        self.equilibrium_position_world_m = ee_body.xpos.copy()
        self.equilibrium_rotation_world = ee_body.xmat.reshape(3, 3).copy()

        self.admittance_offset = np.zeros(CARTESIAN_DOF)
        self.admittance_velocity = np.zeros(CARTESIAN_DOF)
        self.target_q = self.data.qpos.copy()

        print(
            "Admittance controller ready: "
            f"wrench={np.round(self.external_wrench_world, 2)} [N, N·m], "
            f"starts at t={self.force_start_time_s:.2f} s"
        )

    def _active_wrench_world(self) -> tuple[np.ndarray, bool]:
        """返回限幅后的教学用合成外力；启动延时之前返回零。"""
        if self.data.time < self.force_start_time_s:
            return np.zeros(CARTESIAN_DOF), False

        force_n, force_limited = self._clip_vector_norm(
            self.external_wrench_world[:3], self.max_force_n
        )
        torque_nm, torque_limited = self._clip_vector_norm(
            self.external_wrench_world[3:], self.max_torque_nm
        )
        return np.concatenate((force_n, torque_nm)), force_limited or torque_limited

    def _integrate_admittance(self, wrench_world: np.ndarray) -> None:
        """用半隐式 Euler 积分 6 维质量-阻尼-弹簧导纳模型。"""
        acceleration = (
            wrench_world
            - self.virtual_damping * self.admittance_velocity
            - self.virtual_stiffness * self.admittance_offset
        ) / self.virtual_inertia

        dt_s = self.model.opt.timestep
        self.admittance_velocity += acceleration * dt_s
        linear_velocity, linear_speed_limited = self._clip_vector_norm(
            self.admittance_velocity[:3], self.max_linear_speed_m_s
        )
        angular_velocity, angular_speed_limited = self._clip_vector_norm(
            self.admittance_velocity[3:], self.max_angular_speed_rad_s
        )
        self.admittance_velocity[:] = np.concatenate(
            (linear_velocity, angular_velocity)
        )

        self.admittance_offset += self.admittance_velocity * dt_s
        translation, translation_limited = self._clip_vector_norm(
            self.admittance_offset[:3], self.max_translation_offset_m
        )
        rotation, rotation_limited = self._clip_vector_norm(
            self.admittance_offset[3:], self.max_rotation_offset_rad
        )
        self.admittance_offset[:] = np.concatenate((translation, rotation))

        # 到达位移边界后清掉对应速度，避免积分器继续向边界外累积能量。
        if translation_limited:
            self.admittance_velocity[:3] = 0.0
        if rotation_limited:
            self.admittance_velocity[3:] = 0.0

        if not np.all(np.isfinite(self.admittance_offset)):
            raise FloatingPointError("admittance model generated a non-finite state")
        self.motion_limited = (
            linear_speed_limited
            or angular_speed_limited
            or translation_limited
            or rotation_limited
        )

    def _target_pose_world(self) -> tuple[np.ndarray, np.ndarray]:
        """将 6 维偏移转换为世界坐标系目标位置和旋转矩阵。"""
        target_position_world_m = (
            self.equilibrium_position_world_m + self.admittance_offset[:3]
        )
        rotation_offset_world = Rotation.from_rotvec(
            self.admittance_offset[3:]
        ).as_matrix()
        target_rotation_world = (
            rotation_offset_world @ self.equilibrium_rotation_world
        )
        return target_position_world_m, target_rotation_world

    def _update_ik_target(
        self,
        target_position_world_m: np.ndarray,
        target_rotation_world: np.ndarray,
    ) -> float:
        """执行少量阻尼最小二乘 IK 迭代并更新关节目标。"""
        pose_error = np.zeros(CARTESIAN_DOF)
        for _ in range(self.ik_iterations_per_step):
            pin.framesForwardKinematics(self.pin_model, self.pin_data, self.target_q)
            current_pose_world = self.pin_data.oMf[self.pin_ee_frame_id]
            pose_error[:3] = (
                target_position_world_m - current_pose_world.translation
            )
            pose_error[3:] = Rotation.from_matrix(
                target_rotation_world @ current_pose_world.rotation.T
            ).as_rotvec()

            jacobian = pin.computeFrameJacobian(
                self.pin_model,
                self.pin_data,
                self.target_q,
                self.pin_ee_frame_id,
                pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
            )[:, :ARM_DOF]
            regularized = (
                jacobian @ jacobian.T
                + self.ik_damping**2 * np.eye(CARTESIAN_DOF)
            )
            delta_q = jacobian.T @ np.linalg.solve(regularized, pose_error)
            delta_q, _ = self._clip_vector_norm(delta_q, self.max_ik_step_rad)

            self.target_q[:ARM_DOF] += delta_q
            self.target_q[:ARM_DOF] = np.clip(
                self.target_q[:ARM_DOF],
                self.pin_model.lowerPositionLimit[:ARM_DOF],
                self.pin_model.upperPositionLimit[:ARM_DOF],
            )

        if not np.all(np.isfinite(self.target_q[:ARM_DOF])):
            raise FloatingPointError("IK generated a non-finite joint target")
        return float(np.linalg.norm(pose_error))

    def _apply_position_control(self) -> bool:
        """加入重力前馈，并写入受 ctrlrange 限制的关节目标。"""
        gravity_tau_nm = pin.computeGeneralizedGravity(
            self.pin_model, self.pin_data, self.target_q
        )[:ARM_DOF]
        gravity_offset_rad = gravity_tau_nm / self.position_actuator_kp
        requested_ctrl_rad = self.target_q[:ARM_DOF] + gravity_offset_rad
        applied_ctrl_rad = np.clip(
            requested_ctrl_rad,
            self.arm_ctrl_range_rad[:, 0],
            self.arm_ctrl_range_rad[:, 1],
        )
        self.data.ctrl[:ARM_DOF] = applied_ctrl_rad
        return not np.array_equal(requested_ctrl_rad, applied_ctrl_rad)

    def runFunc(self) -> None:
        """推进一次导纳模型、IK 和关节位置控制。"""
        wrench_world, wrench_limited = self._active_wrench_world()
        self._integrate_admittance(wrench_world)
        target_position_world_m, target_rotation_world = self._target_pose_world()
        ik_error = self._update_ik_target(
            target_position_world_m, target_rotation_world
        )
        actuator_limited = self._apply_position_control()

        actual_position_world_m = self.data.body(self.ee_body_id).xpos.copy()
        tracking_error_m = float(
            np.linalg.norm(target_position_world_m - actual_position_world_m)
        )
        self.time_s_history.append(float(self.data.time))
        self.wrench_world_history.append(wrench_world.copy())
        self.offset_history.append(self.admittance_offset.copy())
        self.target_position_m_history.append(target_position_world_m.copy())
        self.actual_position_m_history.append(actual_position_world_m)
        self.tracking_error_m_history.append(tracking_error_m)

        if self.step_index % self.print_interval_steps == 0:
            limited = wrench_limited or self.motion_limited or actuator_limited
            limit_note = " [LIMITED]" if limited else ""
            print(
                f"t={self.data.time:6.3f} s | "
                f"F={np.round(wrench_world[:3], 2)} N | "
                f"offset={np.round(1000 * self.admittance_offset[:3], 1)} mm | "
                f"tracking error={1000 * tracking_error_m:5.1f} mm | "
                f"IK error={ik_error:.3e}{limit_note}"
            )
        self.step_index += 1

    def plot_results(self) -> None:
        """绘制目标/实际位置、导纳偏移和输入 wrench。"""
        if not self.time_s_history:
            print("No admittance samples were recorded; skip plotting.")
            return

        time_s = np.asarray(self.time_s_history)
        wrench_world = np.asarray(self.wrench_world_history)
        offset = np.asarray(self.offset_history)
        target_position_m = np.asarray(self.target_position_m_history)
        actual_position_m = np.asarray(self.actual_position_m_history)
        axis_names = ("x", "y", "z")

        pose_figure, pose_axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
        for axis_index, axis in enumerate(pose_axes):
            axis.plot(
                time_s,
                target_position_m[:, axis_index],
                label="target",
                linewidth=1.4,
            )
            axis.plot(
                time_s,
                actual_position_m[:, axis_index],
                label="actual",
                linewidth=1.1,
            )
            axis.set_ylabel(f"{axis_names[axis_index]} (m)")
            axis.grid(True, alpha=0.3)
            axis.legend()
        pose_axes[-1].set_xlabel("simulation time (s)")
        pose_figure.suptitle("End-effector target and actual position")
        pose_figure.tight_layout()

        offset_figure, offset_axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
        for axis_index, axis_name in enumerate(axis_names):
            offset_axes[0].plot(
                time_s, 1000 * offset[:, axis_index], label=axis_name
            )
            offset_axes[1].plot(
                time_s,
                np.rad2deg(offset[:, axis_index + 3]),
                label=f"r{axis_name}",
            )
        offset_axes[0].set_ylabel("translation (mm)")
        offset_axes[1].set_ylabel("rotation vector (deg)")
        offset_axes[1].set_xlabel("simulation time (s)")
        for axis in offset_axes:
            axis.grid(True, alpha=0.3)
            axis.legend(ncols=3)
        offset_figure.suptitle("Virtual admittance displacement")
        offset_figure.tight_layout()

        wrench_figure, wrench_axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
        for axis_index, axis_name in enumerate(axis_names):
            wrench_axes[0].plot(
                time_s, wrench_world[:, axis_index], label=f"F{axis_name}"
            )
            wrench_axes[1].plot(
                time_s,
                wrench_world[:, axis_index + 3],
                label=f"T{axis_name}",
            )
        wrench_axes[0].set_ylabel("force (N)")
        wrench_axes[1].set_ylabel("torque (N·m)")
        wrench_axes[1].set_xlabel("simulation time (s)")
        for axis in wrench_axes:
            axis.grid(True, alpha=0.3)
            axis.legend(ncols=3)
        wrench_figure.suptitle("Applied synthetic wrench in world frame")
        wrench_figure.tight_layout()
        plt.show()


def main() -> None:
    """运行 Panda 末端导纳示例，并在关闭 viewer 后显示曲线。"""
    demo = PandaAdmittanceDemo(
        scene_xml="model/franka_emika_panda/scene_pos.xml",
        arm_xml="model/franka_emika_panda/panda_pos.xml",
    )
    demo.run_loop()
    demo.plot_results()


if __name__ == "__main__":
    main()
