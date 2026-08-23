"""Panda 机械臂逆动力学补偿与关节阻尼拖动示例。"""

from collections import deque

import matplotlib.pyplot as plt
import numpy as np
import pinocchio as pin

from src import mujoco_viewer

ARM_DOF = 7
DEFAULT_DAMPING_NM_S_RAD = 30.0
DEFAULT_ACCELERATION_FILTER_ALPHA = 0.1
DEFAULT_PRINT_INTERVAL_STEPS = 100
MAX_HISTORY_SAMPLES = 100_000


class PandaDragDemo(mujoco_viewer.CustomViewer):
    """在 MuJoCo 中演示 Panda 的动力学补偿与阻尼拖动。"""

    def __init__(
        self,
        scene_xml: str,
        arm_xml: str,
        *,
        damping_nm_s_rad: float = DEFAULT_DAMPING_NM_S_RAD,
        compensate_inertia: bool = False,
        acceleration_filter_alpha: float = DEFAULT_ACCELERATION_FILTER_ALPHA,
        print_interval_steps: int = DEFAULT_PRINT_INTERVAL_STEPS,
    ) -> None:
        super().__init__(scene_xml, distance=3, azimuth=-45, elevation=-30)

        if damping_nm_s_rad < 0:
            raise ValueError("damping_nm_s_rad must be non-negative")
        if not 0 < acceleration_filter_alpha <= 1:
            raise ValueError("acceleration_filter_alpha must be in (0, 1]")
        if print_interval_steps <= 0:
            raise ValueError("print_interval_steps must be positive")
        if self.model.nkey == 0:
            raise ValueError("scene model must define a home keyframe")
        if self.model.nq < ARM_DOF or self.model.nv < ARM_DOF:
            raise ValueError("scene model must contain at least 7 arm joints")
        if self.model.nu < ARM_DOF:
            raise ValueError("scene model must contain at least 7 arm actuators")

        self.arm_xml = arm_xml
        self.damping_nm_s_rad = damping_nm_s_rad
        self.compensate_inertia = compensate_inertia
        self.acceleration_filter_alpha = acceleration_filter_alpha
        self.print_interval_steps = print_interval_steps

        # 使用完整 keyframe 初始化，确保 MuJoCo 中的机械臂和夹爪状态一致。
        self.data.qpos[:] = self.model.key_qpos[0]
        self.data.qvel[:] = 0.0
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
        self.damping_tau_nm_history = deque(maxlen=MAX_HISTORY_SAMPLES)
        self.applied_tau_nm_history = deque(maxlen=MAX_HISTORY_SAMPLES)

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

        self.last_qvel = self.data.qvel.copy()
        self.zero_qacc = np.zeros(self.model.nv)
        self.filtered_qacc = np.zeros(self.model.nv)

        compensation = (
            "gravity/Coriolis + filtered inertia"
            if self.compensate_inertia
            else "gravity/Coriolis"
        )
        print(
            f"Drag controller ready: {compensation}, "
            f"damping={self.damping_nm_s_rad:.1f} N·m·s/rad"
        )

    def _estimate_qacc(self, qvel: np.ndarray) -> np.ndarray:
        """用速度差分估计加速度，并用一阶低通滤波抑制数值噪声。"""
        raw_qacc = (qvel - self.last_qvel) / self.model.opt.timestep
        self.last_qvel = qvel.copy()

        alpha = self.acceleration_filter_alpha
        self.filtered_qacc = alpha * raw_qacc + (1.0 - alpha) * self.filtered_qacc
        return self.filtered_qacc.copy()

    def _clip_arm_torque(self, requested_tau_nm: np.ndarray) -> tuple[np.ndarray, bool]:
        """按 MJCF actuator 的 ctrlrange 限制 7 轴控制力矩。"""
        lower_nm = self.arm_ctrl_range_nm[:, 0]
        upper_nm = self.arm_ctrl_range_nm[:, 1]
        applied_tau_nm = np.clip(requested_tau_nm, lower_nm, upper_nm)
        saturated = not np.array_equal(applied_tau_nm, requested_tau_nm)
        return applied_tau_nm, saturated

    def runFunc(self) -> None:
        """计算并施加一次逆动力学补偿与关节阻尼力矩。"""
        q = self.data.qpos.copy()
        qvel = self.data.qvel.copy()

        # 默认不补偿惯性，避免把差分加速度的噪声反馈到力矩控制中。
        desired_qacc = (
            self._estimate_qacc(qvel) if self.compensate_inertia else self.zero_qacc
        )
        dynamics_tau_nm = np.asarray(
            pin.rnea(self.pin_model, self.pin_data, q, qvel, desired_qacc)
        ).reshape(-1)

        # 重力补偿 + 科氏力/离心力补偿 + 速度阻尼，关闭惯性补偿。
        arm_qvel_rad_s = qvel[:ARM_DOF]
        damping_tau_nm = -self.damping_nm_s_rad * arm_qvel_rad_s
        requested_tau_nm = dynamics_tau_nm[:ARM_DOF] + damping_tau_nm
        
        # 完全不控制
        # requested_tau_nm = np.zeros(ARM_DOF)
        
        # 只有重力补偿
        # gravity_tau_nm = pin.computeGeneralizedGravity(
        #     self.pin_model,
        #     self.pin_data,
        #     q,
        # )
        # damping_tau_nm = np.zeros(ARM_DOF)
        # requested_tau_nm = gravity_tau_nm[:ARM_DOF]
        
        if not np.all(np.isfinite(requested_tau_nm)):
            self.data.ctrl[:ARM_DOF] = 0.0
            raise FloatingPointError("controller generated a non-finite torque command")

        applied_tau_nm, saturated = self._clip_arm_torque(requested_tau_nm)
        self.data.ctrl[:ARM_DOF] = applied_tau_nm

        self.time_s_history.append(float(self.data.time))
        self.dynamics_tau_nm_history.append(dynamics_tau_nm[:ARM_DOF].copy())
        self.damping_tau_nm_history.append(damping_tau_nm.copy())
        self.applied_tau_nm_history.append(applied_tau_nm.copy())

        if self.step_index % self.print_interval_steps == 0:
            saturation_note = " [SATURATED]" if saturated else ""
            print(
                f"t={self.data.time:7.3f} s | "
                f"applied τ={np.round(applied_tau_nm, 2)} N·m{saturation_note}"
            )
        self.step_index += 1

    def plot_torques(self) -> None:
        """绘制最近一段历史中的动力学、阻尼和实际施加力矩。"""
        if not self.time_s_history:
            print("No torque samples were recorded; skip plotting.")
            return

        time_s = np.asarray(self.time_s_history)
        dynamics_tau_nm = np.asarray(self.dynamics_tau_nm_history)
        damping_tau_nm = np.asarray(self.damping_tau_nm_history)
        applied_tau_nm = np.asarray(self.applied_tau_nm_history)

        fig, axes = plt.subplots(ARM_DOF, 1, figsize=(11, 14), sharex=True)
        fig.suptitle(
            "Panda drag-control torque components", fontsize=14, fontweight="bold"
        )

        for joint_index, axis in enumerate(axes):
            axis.plot(
                time_s,
                dynamics_tau_nm[:, joint_index],
                label="dynamics compensation",
                color="tab:blue",
                linewidth=1.2,
            )
            axis.plot(
                time_s,
                damping_tau_nm[:, joint_index],
                label="damping",
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

        axes[-1].set_xlabel("simulation time (s)")
        fig.tight_layout(rect=(0, 0, 1, 0.98))
        plt.show()


def main() -> None:
    """运行 Panda 仿真拖动示例，并在 viewer 关闭后显示力矩曲线。"""
    demo = PandaDragDemo(
        scene_xml="model/franka_emika_panda/scene_tau.xml",
        arm_xml="model/franka_emika_panda/panda_tau.xml",
        damping_nm_s_rad=DEFAULT_DAMPING_NM_S_RAD,
        compensate_inertia=False,
    )
    demo.run_loop()
    demo.plot_torques()


if __name__ == "__main__":
    main()
