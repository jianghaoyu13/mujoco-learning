import mujoco


class ContactDetector:
    """MuJoCo 接触检测器: 监测两组 body 之间的接触力。

    - 每侧支持多个 body (如夹爪的 Fixed_Jaw + Moving_Jaw)。
    - 消抖: 力超过阈值后需连续 ``debounce_steps`` 步才判定接触,
      避免瞬态接触误报。

    参数
    ----
    model / data : MjModel / MjData
    bodies_a, bodies_b : sequence of str
        两侧 body 名称列表。
    force_threshold : float
        接触力阈值 (N)。
    debounce_steps : int
        确认接触所需的连续步数。
    """

    def __init__(self, model, data, bodies_a, bodies_b,
                 force_threshold: float = 0.01, debounce_steps: int = 3):
        self.model = model
        self.data = data
        self.bodies_a = self._name_set(model, bodies_a)
        self.bodies_b = self._name_set(model, bodies_b)
        self.force_threshold = float(force_threshold)
        self.debounce_steps = max(1, int(debounce_steps))
        self.has_contact = False
        self._contact_steps = 0

    @staticmethod
    def _name_set(model, names):
        ids = set()
        for name in names:
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            if bid < 0:
                raise ValueError(f"body '{name}' not found in model")
            ids.add(bid)
        return ids

    def contact_force(self) -> float:
        """两组 body 之间的最大接触力 (N), 无接触时为 0。

        MuJoCo 3.10+: 接触力在 data.efc_force[contact.efc_address]。
        """
        force = 0.0
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            b1 = self.model.geom_bodyid[contact.geom1]
            b2 = self.model.geom_bodyid[contact.geom2]
            if (b1 in self.bodies_a and b2 in self.bodies_b) or \
               (b2 in self.bodies_a and b1 in self.bodies_b):
                f = abs(self.data.efc_force[contact.efc_address]) \
                    if 0 <= contact.efc_address < self.data.nefc else 0.0
                force = max(force, f)
        return force

    def update(self) -> bool:
        """推进一步检测, 返回消抖后的接触状态。"""
        if self.contact_force() > self.force_threshold:
            self._contact_steps += 1
            self.has_contact = self._contact_steps >= self.debounce_steps
        else:
            self._contact_steps = 0
            self.has_contact = False
        return self.has_contact
