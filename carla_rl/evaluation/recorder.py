"""RGB-camera video recorder for evaluation episodes.

Attach after env.reset() (the env destroys all sensor.camera.rgb actors during
reset), detach before the next reset, then save. In synchronous mode the
camera delivers one frame per world tick, so fps should be 1/dt.

Frames are streamed straight into the ffmpeg writer from the sensor callback
(single CARLA callback thread, sequential appends) — buffering a 1000-step
episode in memory costs ~1.5 GB and risked OOM next to torch.

A lock guards the writer: the CARLA camera callback runs on its own thread and
an in-flight append must not race save()'s close() (that raised
'generator already executing' and corrupted the recording).
"""

import threading


class VideoRecorder:
    def __init__(self, width=960, height=544, fov=90, fps=10):
        # height divisible by 16 keeps ffmpeg from resizing (macro block size)
        # 高畫質錄製:env 變數 CARLA_VIDEO_W / CARLA_VIDEO_H 覆寫(預設 960x544)。
        # 例:720p → 1280x720;1080p → 1920x1088(高度需可被 16 整除)。注意高解析度會
        # 增加 GPU 負載,在高車流密度下更易觸發伺服器當機。
        import os
        self.width = int(os.environ.get('CARLA_VIDEO_W', width))
        self.height = int(os.environ.get('CARLA_VIDEO_H', height))
        self.fov = fov
        self.fps = fps
        self.camera = None
        self.writer = None
        self.frame_count = 0
        self._lock = threading.Lock()

    def attach(self, world, ego, path):
        import carla
        import imageio.v2 as imageio

        bp = world.get_blueprint_library().find('sensor.camera.rgb')
        bp.set_attribute('image_size_x', str(self.width))
        bp.set_attribute('image_size_y', str(self.height))
        bp.set_attribute('fov', str(self.fov))
        transform = carla.Transform(
            carla.Location(x=-6.5, z=3.5), carla.Rotation(pitch=-12)
        )
        self.writer = imageio.get_writer(
            str(path), fps=self.fps, codec='libx264', quality=7
        )
        self.frame_count = 0
        self.camera = world.spawn_actor(bp, transform, attach_to=ego)
        self.camera.listen(self._on_image)

    def _on_image(self, image):
        import numpy as np

        arr = np.frombuffer(image.raw_data, dtype=np.uint8)
        arr = arr.reshape((image.height, image.width, 4))[:, :, [2, 1, 0]]  # BGRA -> RGB
        with self._lock:
            if self.writer is None:
                return
            try:
                self.writer.append_data(arr)
                self.frame_count += 1
            except Exception:
                pass

    def detach(self):
        if self.camera is not None:
            try:
                self.camera.stop()
                self.camera.destroy()
            except Exception:
                pass
            self.camera = None

    def save(self):
        """Finalize the video; returns frame count written."""
        with self._lock:
            writer, self.writer = self.writer, None
        if writer is not None:
            writer.close()  # outside the lock so a slow close can't block callbacks
        return self.frame_count
