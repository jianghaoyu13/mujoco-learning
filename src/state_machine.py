import time


class StateMachine:
    """通用有限状态机: 每个状态可配置超时时间。

    子类通常声明状态名常量 (STATE_*) 和 ``TIMEOUTS`` 字典, 然后调用::

        super().__init__(initial_state=self.STATE_IDLE, timeouts=self.TIMEOUTS)

    参数
    ----
    initial_state : str
        初始状态。
    timeouts : dict | None
        各状态超时 (秒); 不在表中的状态永不超时。
    logger : callable
        状态切换时的回调 (默认打印 "[State] 旧 -> 新")。
    """

    def __init__(self, initial_state: str = "idle", timeouts: dict | None = None,
                 logger=print):
        self._timeouts = dict(timeouts or {})
        self._logger = logger
        self.state = initial_state
        self.progress = 0.0
        self.start_time = time.time()

    def transition_to(self, new_state: str) -> bool:
        """切换到 new_state 并重置 progress 与计时; 返回是否真的发生了切换。"""
        if self.state != new_state:
            self._logger(f"[State] {self.state} -> {new_state}")
            self.state = new_state
            self.progress = 0.0
            self.start_time = time.time()
            return True
        return False

    def timed_out(self) -> bool:
        """当前状态是否已超过其超时时间。"""
        limit = self._timeouts.get(self.state)
        if limit is None:
            return False
        return time.time() - self.start_time >= limit
