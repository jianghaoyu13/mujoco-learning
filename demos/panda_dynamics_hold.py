"""Panda 机械臂逆动力学补偿与关节阻抗保持示例。"""

from collections import deque
from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pinocchio as pin

from src import mujoco_viewer

ARM_DOF = 7
DEFAULT_STIFFNESS_NM_RAD = 100.0
DEFAULT_DAMPING_NM_S_RAD = 20.0
DEFAULT_PRINT_INTERVAL_STEPS = 100
MAX_HISTORY_SAMPLES = 100_000


class PandaHoldDemo(mujoco_viewer.CustomViewer):
    """用逆动力学补偿和关节阻抗控制将 Panda 保持在启动姿态。"""

    def __init__(
        self,
        scene_xml: str,
        arm_xml: str,
        *,
        stiffness_nm_rad: float | Sequence[float] = DEFAULT_STIFFNESS_NM_RAD,
        damping_nm_s_rad: float | Sequence[float] = DEFAULT_DAMPING_NM_S_RAD,
        print_interval_steps: int = DEFAULT_PRINT_INTERVAL_STEPS,
    ) -> None:
        super().__init__(scene_xml, distance=3, azimuth=-45, elevation=-30)

        if print_interval_steps <= 0:
            raise ValueError("print_interval_steps must be positive")
        if self.model.nkey == 0:
            raise ValueError("scene model must define a home keyframe")
        if self.model.nq < ARM_DOF or self.model.nv < ARM_DOF:
            raise ValueError("scene model must contain at least 7 arm joints")
        if self.model.nu < ARM_DOF:
            raise ValueError("scene model must contain at least 7 arm actuators")

        self.arm_xml = arm_xml
        self.stiffness_nm_rad = self._arm_gain_vector(
            stiffness_nm_rad, "stiffness_nm_rad"
        )
        self.damping_nm_s_rad = self._arm_gain_vector(
            damping_nm_s_rad, "damping_nm_s_rad"
        )
        self.print_interval_steps = print_interval_steps

        # 使用完整 keyframe 初始化，确保机械臂和夹爪状态一致。
        self.data.qpos[:] = self.model.key_qpos[0]
        self.data.qvel[:] = 0.0
        self.target_arm_q_rad = self.data.qpos[:ARM_DOF].copy()
        if self.model.nu > ARM_DOF:
            self.data.ctrl[ARM_DOF:] = self.model.key_ctrl[0, ARM_DOF:]

        arm_actuators_limited = self.model.actuator_ctrllimited[:ARM_DOF]
        if not np.all(arm_actuators_limited):
            raise ValueError("all arm actuators must define ctrlrange limits")
        self.arm_ctrl_range_nm = self.model.actuator_ctrlrange[:ARM_DOF].copy()
        self.arm_joint_names = [self.model.joint(i).name for i in range(ARM_DOF)]

        self.step_index = 0
        self.time_s_history = deque(maxlen=MAX_HISTORY_SAMPLES)
        self.dynamics_tau_nm_history = deque(maxlen=MAX_HISTORY_SAMPLES)
        self.impedance_tau_nm_history = deque(maxlen=MAX_HISTORY_SAMPLES)
        self.applied_tau_nm_history = deque(maxlen=MAX_HISTORY_SAMPLES)
        self.position_error_rad_history = deque(maxlen=MAX_HISTORY_SAMPLES)

    @staticmethod
    def _arm_gain_vector(value: float | Sequence[float], name: str) -> np.ndarray:
        """将一个统一增益或 7 个逐关节增益转换为安全的向量。"""
        gain = np.asarray(value, dtype=float)
        if gain.ndim == 0:
            gain = np.full(ARM_DOF, float(gain))
        if gain.shape != (ARM_DOF,):
            raise ValueError(f"{name} must be a scalar or contain {ARM_DOF} values")
        if not np.all(np.isfinite(gain)) or np.any(gain < 0):
            raise ValueError(f"{name} must contain finite, non-negative values")
        return gain

    def runBefore(self) -> None:
        """创建 Pinocchio 模型，并校验它与 MuJoCo 的状态维度。"""
        self.pin_model = pin.RobotWrapper.BuildFromMJCF(self.arm_xml).model
        self.pin_data = self.pin_model.createData()

        if self.pin_model.nq != self.model.nq or self.pin_model.nv != self.model.nv:
            raise ValueError(
                "MuJoCo and Pinocchio model dimensions do not match: "
                f"MuJoCo (nq={self.model.nq}, nv={self.model.nv}), "
                f"Pinocchio (nq={self.pin_model.nq}, nv={self.pin_model.nv})"
            )

        self.zero_qacc = np.zeros(self.model.nv)
        print(
            "Hold controller ready: "
            f"Kp={np.round(self.stiffness_nm_rad, 2)} N·m/rad, "
            f"Kd={np.round(self.damping_nm_s_rad, 2)} N·m·s/rad"
        )

    def _clip_arm_torque(self, requested_tau_nm: np.ndarray) -> tuple[np.ndarray, bool]:
        """按 MJCF actuator 的 ctrlrange 限制 7 轴控制力矩。"""
        lower_nm = self.arm_ctrl_range_nm[:, 0]
        upper_nm = self.arm_ctrl_range_nm[:, 1]
        applied_tau_nm = np.clip(requested_tau_nm, lower_nm, upper_nm)
        saturated = not np.array_equal(applied_tau_nm, requested_tau_nm)
        return applied_tau_nm, saturated

    def runFunc(self) -> None:
        """计算并施加一次动力学补偿与关节阻抗保持力矩。"""
        q = self.data.qpos.copy()
        qvel = self.data.qvel.copy()
        arm_q_rad = q[:ARM_DOF]
        arm_qvel_rad_s = qvel[:ARM_DOF]

        # qacc=0：只补偿维持当前运动状态所需的重力、科氏力等项。
        # 不把差分加速度反馈给控制器，以免放大仿真噪声。
        dynamics_tau_nm = np.asarray(
            pin.rnea(self.pin_model, self.pin_data, q, qvel, self.zero_qacc)
        ).reshape(-1)

        position_error_rad = self.target_arm_q_rad - arm_q_rad
        spring_tau_nm = self.stiffness_nm_rad * position_error_rad
        damping_tau_nm = -self.damping_nm_s_rad * arm_qvel_rad_s
        impedance_tau_nm = spring_tau_nm + damping_tau_nm
        requested_tau_nm = dynamics_tau_nm[:ARM_DOF] + impedance_tau_nm

        if not np.all(np.isfinite(requested_tau_nm)):
            self.data.ctrl[:ARM_DOF] = 0.0
            raise FloatingPointError("controller generated a non-finite torque command")

        applied_tau_nm, saturated = self._clip_arm_torque(requested_tau_nm)
        self.data.ctrl[:ARM_DOF] = applied_tau_nm

        self.time_s_history.append(float(self.data.time))
        self.dynamics_tau_nm_history.append(dynamics_tau_nm[:ARM_DOF].copy())
        self.impedance_tau_nm_history.append(impedance_tau_nm.copy())
        self.applied_tau_nm_history.append(applied_tau_nm.copy())
        self.position_error_rad_history.append(position_error_rad.copy())

        if self.step_index % self.print_interval_steps == 0:
            saturation_note = " [SATURATED]" if saturated else ""
            max_error_deg = np.rad2deg(np.max(np.abs(position_error_rad)))
            print(
                f"t={self.data.time:7.3f} s | "
                f"max |q error|={max_error_deg:6.2f} deg | "
                f"applied τ={np.round(applied_tau_nm, 2)} N·m{saturation_note}"
            )
        self.step_index += 1

    def plot_results(self) -> None:
        """绘制力矩分量和 7 个关节的位置误差。"""
        if not self.time_s_history:
            print("No controller samples were recorded; skip plotting.")
            return

        time_s = np.asarray(self.time_s_history)
        dynamics_tau_nm = np.asarray(self.dynamics_tau_nm_history)
        impedance_tau_nm = np.asarray(self.impedance_tau_nm_history)
        applied_tau_nm = np.asarray(self.applied_tau_nm_history)
        position_error_deg = np.rad2deg(np.asarray(self.position_error_rad_history))

        torque_figure, torque_axes = plt.subplots(
            ARM_DOF, 1, figsize=(11, 14), sharex=True
        )
        torque_figure.suptitle(
            "Panda hold-control torque components", fontsize=14, fontweight="bold"
        )

        for joint_index, axis in enumerate(torque_axes):
            axis.plot(
                time_s,
                dynamics_tau_nm[:, joint_index],
                label="dynamics compensation",
                color="tab:blue",
                linewidth=1.2,
            )
            axis.plot(
                time_s,
                impedance_tau_nm[:, joint_index],
                label="joint impedance",
                color="tab:red",
                linewidth=1.2,
            )
            axis.plot(
                time_s,
                applied_tau_nm[:, joint_index],
                label="applied total",
                color="tab:green",
                linewidth=1.0,
                alpha=0.8,
            )
            axis.set_ylabel(f"{self.arm_joint_names[joint_index]}\nτ (N·m)")
            axis.grid(True, alpha=0.3)
            axis.legend(loc="upper right", ncols=3, fontsize=8)

        torque_axes[-1].set_xlabel("simulation time (s)")
        torque_figure.tight_layout(rect=(0, 0, 1, 0.98))

        error_figure, error_axis = plt.subplots(figsize=(11, 5))
        for joint_index, joint_name in enumerate(self.arm_joint_names):
            error_axis.plot(
                time_s,
                position_error_deg[:, joint_index],
                label=joint_name,
                linewidth=1.2,
            )
        error_axis.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
        error_axis.set_title("Joint position error")
        error_axis.set_xlabel("simulation time (s)")
        error_axis.set_ylabel("target - actual (deg)")
        error_axis.grid(True, alpha=0.3)
        error_axis.legend(ncols=4)
        error_figure.tight_layout()
        plt.show()


def main() -> None:
    """运行 Panda 关节阻抗保持示例，并在 viewer 关闭后显示曲线。"""
    demo = PandaHoldDemo(
        scene_xml="model/franka_emika_panda/scene_tau.xml",
        arm_xml="model/franka_emika_panda/panda_tau.xml",
        stiffness_nm_rad=DEFAULT_STIFFNESS_NM_RAD,
        damping_nm_s_rad=DEFAULT_DAMPING_NM_S_RAD,
    )
    demo.run_loop()
    demo.plot_results()


if __name__ == "__main__":
    main()
