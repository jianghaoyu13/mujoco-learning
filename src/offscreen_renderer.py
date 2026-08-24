import glfw
import mujoco
import numpy as np


class OffscreenCameraRenderer:
    """MuJoCo 离屏渲染器: 把场景中的固定相机渲染成 RGB 数组。

    用法::

        renderer = OffscreenCameraRenderer(model, image_size=(480, 640))
        if renderer.enabled:
            img = renderer.render(data, cam_id)   # (H, W, 3) uint8
        renderer.close()

    初始化失败 (如无可用 OpenGL 上下文) 时 ``enabled`` 为 False,
    ``render`` 返回 None, 调用方可降级为只记录状态数据。
    """

    def __init__(self, model, image_size=(480, 640), maxgeom: int = 10000):
        self.model = model
        self.image_size = (int(image_size[0]), int(image_size[1]))  # (H, W)
        self.enabled = False
        self._window = None
        try:
            glfw.init()
            glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
            w, h = self.image_size[1], self.image_size[0]
            self._window = glfw.create_window(w, h, 'Offscreen', None, None)
            glfw.make_context_current(self._window)
            self._scene = mujoco.MjvScene(model, maxgeom=maxgeom)
            self._context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150.value)
            mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, self._context)
            self.enabled = True
        except Exception as e:
            print(f"[Warn] 离屏渲染初始化失败 ({e}), 相机图像不可用")

    def render(self, data, camera_id: int):
        """渲染一个固定相机, 返回 (H, W, 3) uint8 图像; 不可用时返回 None。"""
        if camera_id < 0 or not self.enabled:
            return None
        h, w = self.image_size
        camera_view = mujoco.MjvCamera()
        camera_view.type = mujoco.mjtCamera.mjCAMERA_FIXED
        camera_view.fixedcamid = camera_id
        viewport = mujoco.MjrRect(0, 0, w, h)
        mujoco.mjv_updateScene(self.model, data, mujoco.MjvOption(),
                               mujoco.MjvPerturb(), camera_view,
                               mujoco.mjtCatBit.mjCAT_ALL, self._scene)
        mujoco.mjr_render(viewport, self._scene, self._context)
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        mujoco.mjr_readPixels(rgb, None, viewport, self._context)
        return np.flipud(rgb)

    def close(self):
        if self._window is not None:
            try:
                glfw.destroy_window(self._window)
                glfw.terminate()
            except Exception:
                pass
            self._window = None
            self.enabled = False
