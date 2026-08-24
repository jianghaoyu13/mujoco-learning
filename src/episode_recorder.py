import json
from pathlib import Path

import cv2


def find_next_episode_num(output_dir: Path) -> int:
    """在输出目录中找已有最大 episode 编号并续号; 无已有目录时返回 0。"""
    output_dir = Path(output_dir)
    if not output_dir.exists():
        return 0
    nums = [int(p.name.split("_")[-1]) for p in output_dir.glob("episode_*")
            if p.name.split("_")[-1].isdigit()]
    return max(nums) + 1 if nums else 0


class EpisodeRecorder:
    """Episode 数据采集与保存: 逐步累积, 成功后整集保存, 失败时丢弃。

    每集目录结构::

        output_dir/
          episode_0000/
            steps.json         # 逐步数据 (images 存为相对路径)
            metadata.json      # 集级元数据 (base_metadata + 保存时传入的 metadata)
            images/            # 每步相机图像
              wrist_000123.png ...

    用法::

        rec = EpisodeRecorder("data/x", base_metadata={...})
        rec.add_step({"timestamp": 0.0, "observation": obs,
                      "images": {"wrist": img}, ...})
        if success:
            rec.save_episode(metadata={"episode_length": rec.step_count})
        else:
            rec.discard_episode()

    step 中的 ``timestamp`` 为秒 (用于图像文件名),
    ``images`` 为 {相机名: (H, W, 3) uint8 数组或 None}。
    """

    def __init__(self, output_dir, base_metadata: dict | None = None):
        self.output_dir = Path(output_dir)
        self.base_metadata = dict(base_metadata or {})
        self.episode_num = find_next_episode_num(self.output_dir)
        self._steps: list[dict] = []

    @property
    def step_count(self) -> int:
        return len(self._steps)

    def add_step(self, step: dict):
        """累积一步数据 (step 需含 timestamp 字段)。"""
        self._steps.append(step)

    def discard_episode(self):
        """丢弃当前集 (失败或新一集开始时调用)。"""
        self._steps = []

    def save_episode(self, metadata: dict | None = None) -> Path:
        """保存当前集, 返回集目录; 保存后自动续号并清空本集数据。"""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        episode_dir = self.output_dir / f"episode_{self.episode_num:04d}"
        episode_dir.mkdir(exist_ok=True)

        steps_data = []
        for step in self._steps:
            step_copy = dict(step)
            images_data = {}
            for cam_name, img in (step_copy.get("images") or {}).items():
                if img is not None:
                    img_path = f"images/{cam_name}_{int(step['timestamp'] * 1000):06d}.png"
                    img_file = episode_dir / img_path
                    img_file.parent.mkdir(exist_ok=True)
                    cv2.imwrite(str(img_file), img)
                    images_data[cam_name] = img_path
            step_copy["images"] = images_data
            steps_data.append(step_copy)

        with open(episode_dir / "steps.json", "w") as f:
            json.dump(steps_data, f, indent=2)

        meta = dict(self.base_metadata)
        meta.update(metadata or {})
        with open(episode_dir / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)

        self.episode_num += 1
        self._steps = []
        return episode_dir
