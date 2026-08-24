import numpy as np


class RateLimiter:
    """标量限速器: 以每秒至多 ``max_rate`` 的速度逼近目标值。

    用于限制单个控制量 (如夹爪关节指令) 的变化速率, 防止高速撞击物体。
    """

    def __init__(self, value: float, max_rate: float):
        self.value = float(value)
        self.max_rate = float(max_rate)

    def reset(self, value: float):
        self.value = float(value)

    def step(self, target: float, dt: float) -> float:
        """向 target 推进一步, 返回限速后的值。"""
        d = float(target) - self.value
        max_d = self.max_rate * dt
        self.value += float(np.clip(d, -max_d, max_d))
        return self.value


class SecondOrderTargetSmoother:
    """笛卡尔目标平滑器: 速度限制 + 二阶临界阻尼平滑。

    把阶跃目标 (home->approach、放置点->waypoint->home 等) 变成慢速圆滑
    的 S 曲线轨迹, 消除位置伺服追阶跃时的过冲/回摆 (末端可见晃动)。
    零稳态误差。

    平滑器为二阶系统::

        alpha = 1 - exp(-2 * w * dt),  w = 2 * pi * freq
        x_{k+1} = x_k + 2*alpha*(vt_k - x_k) - alpha^2 * (x_k - x_{k-1})

    其中 ``vt_k`` 为经过矢量速度限制后的目标。

    参数
    ----
    dim : int
        目标维度 (如笛卡尔位置取 3)。
    freq : float
        二阶平滑固有频率 (Hz)。越小越平滑、过渡时间越长。
    vel_limit : float | None
        矢量速度上限 (m/s), None 表示不限速。
    initial : np.ndarray | None
        初始位置/目标, 缺省为零向量。
    """

    def __init__(self, dim: int, freq: float = 0.6, vel_limit: float | None = 0.15,
                 initial: np.ndarray | None = None):
        self.dim = int(dim)
        self.freq = float(freq)
        self.vel_limit = None if vel_limit is None else float(vel_limit)
        if initial is None:
            initial = np.zeros(self.dim)
        self.reset(initial)

    def reset(self, x):
        """把平滑器状态全部重置到 x (位置、上一步位置、目标)。"""
        x = np.asarray(x, dtype=np.float64)
        if x.shape != (self.dim,):
            raise ValueError(f"expected shape ({self.dim},), got {x.shape}")
        self.smoothed = x.copy()
        self.smoothed_prev = x.copy()
        self.vel_target = x.copy()

    @property
    def x(self) -> np.ndarray:
        """当前平滑后的位置。"""
        return self.smoothed

    def step(self, target, dt: float) -> np.ndarray:
        """向 target 推进一步平滑, 返回平滑后的位置。"""
        target = np.asarray(target, dtype=np.float64)
        # 1) 矢量速度限制: 目标逼近速度不超过 vel_limit
        d = target - self.vel_target
        n = float(np.linalg.norm(d))
        if self.vel_limit is not None and n > self.vel_limit * dt:
            self.vel_target = self.vel_target + d * (self.vel_limit * dt / n)
        else:
            self.vel_target = target.copy()
        # 2) 二阶临界阻尼平滑
        w = 2.0 * np.pi * self.freq
        alpha = 1.0 - np.exp(-2.0 * w * dt)
        cur = self.smoothed
        self.smoothed_prev = cur
        self.smoothed = (cur + 2.0 * alpha * (self.vel_target - cur)
                         - alpha * alpha * (cur - self.smoothed_prev))
        return self.smoothed
