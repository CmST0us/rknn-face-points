# RK3588 RKNN 人脸姿态与表情

RK3588 / ROCK 5B 的开箱即用 ARM64 包：人脸检测、478 点人脸网格、头姿、52 个 ARKit 风格表情系数和 `tongueOut`。既可处理单图，也可作为 GStreamer `rknnfacemesh` 滤镜接入 V4L2、文件或其他视频节点。

## 一键运行

单图 CLI：

```sh
./run.sh [图片路径]
```

结果为 `output/face_result.ppm`，终端输出置信度、yaw/pitch/roll、表情系数和 NPU 耗时。

GStreamer 单图管线：

```sh
./run-gstreamer.sh [图片路径]
```

结果为 `output/gstreamer-face.jpg`。

MPP 文件解码、打点和编码：

```sh
./run-video.sh [H.264 MP4/MOV 路径]
```

结果为 `output/face-output.mp4`（不保留音轨）。视频主链路为：

```text
MPP H.264 decode → NV12 DMABUF → rknnfacemesh → 同一 DMABUF → MPP H.264 encode
```

没有全帧 RGB 转换、`videoconvert` 或帧拷贝。

V4L2 实时输入：

```sh
./run-v4l2.sh /dev/video0
```

结果为 `output/face-output.mkv`。ROCK 5B 的 `/dev/video0` 是 HDMI RX 时，需要先接入有效 HDMI 信号。

GNOME Wayland 实时显示：

```sh
./run-display.sh /dev/video0
# 或播放文件：./run-display.sh assets/test.mov
# 或 HLS/m3u8 直播：./run-display.sh 'https://server/live/stream.m3u8'
```

默认连接 `/run/user/1000/wayland-0`；其他会话可设置 `WAYLAND_RUNTIME_DIR` 和 `WAYLAND_DISPLAY`。V4L2 输入强制直接采集 NV12 DMABUF；HLS 支持 fMP4/m4s 与 MPEG-TS 分片，只选择 H.264 视频轨并按 PTS 播放，音频不会触发全局缓冲。MPP 解码到 RKNN 打点保持 DMABUF；`waylandsink` 使用 `/dev/dri/card0` 生成 compositor 可导入的显示缓冲，避免非标准 stride 花屏，其他设备可设置 `DRM_DEVICE`。

## GStreamer 节点

`rknnfacemesh` 接收并输出 NV12，运行时强制底层内存为 DMABUF；普通内存会直接报错，避免链路静默退化成拷贝模式。MPP 编码器的 pad template 不声明 `memory:DMABuf`，因此 caps 保持普通 `video/x-raw`，实际 FD 仍原样透传。

`draw=true` 默认直接在 NV12 DMABUF 上绘制人脸框和 478 点；`inference-interval=2` 默认以 30Hz 推理并在相邻帧复用结果，使 60fps 视频保持实时。需要逐帧新结果时设为 `1`：

```sh
docker compose run --rm --no-deps gstreamer-face -e \
  v4l2src device=/dev/video0 io-mode=dmabuf ! video/x-raw,format=NV12 ! \
  rknnfacemesh model-dir=/app/models draw=true inference-interval=1 ! \
  mpph264enc ! h264parse ! fakesink
```

每帧附带 `GstRknnFaceMeta`，包含 bbox、478 个 xyz 点、yaw/pitch/roll、52 个 blendshape、`tongue_out` 和 NPU 耗时。应用侧结构与访问函数见 `include/gst_rknn_face_meta.h`。

## 零拷贝与性能

MPP 解码输出、插件输入/输出和 MPP 编码输入始终是同一个 NV12 DMABUF。插件只映射模型实际采样的像素，预处理直接写入启动时创建并复用的 RKNN DMA tensor；框和点原地写回 NV12。`DMA_BUF_IOCTL_SYNC` 用于 CPU 与 MPP 的缓存一致性。

在 ROCK 5B / RK3588 上使用仓库测试视频（2688×1512、60fps、169 帧）：

- MPP 解码 + 打点：2.497 秒，67.7fps。
- MPP 解码 + 打点 + MPP H.264 编码：2.523 秒，67.0fps。
- `inference-interval=1` 的逐帧推理约 33fps。

指定 Bilibili HLS（1600×1280、30fps）经 MPP 解码 + DMABUF + RKNN + `fakesink` 连续验收 45 秒：1290 帧、累计平均 31.1fps、0 drops。HLS 解码按分片突发输出，因此推理前使用背压队列，不能使用两帧 `leaky` 队列。

旧 RGA/ORC-NEON `videoconvert` 补丁与构建脚本仍保留在 `rockchip/` 和 `scripts/build-rga-gstvideo.sh`，用于必须转换格式的外部管线；零拷贝主链路不再调用它们。

Rockchip MPP 在容器内导出解码帧需要访问板端媒体设备与 DMA heaps，因此 `gstreamer-face` 服务使用 `privileged: true`；RKNN 单图服务仍只映射 NPU。Rockchip MPP 插件和运行库从目标设备只读挂载，保持与板端驱动匹配。

## 构建

```sh
./scripts/build-native.sh          # CLI
./scripts/build-gstreamer.sh       # rknnfacemesh 插件
./scripts/build-rga-gstvideo.sh    # RGA + ORC/NEON libgstvideo
docker compose build
```

仓库已包含目标板预编译二进制、RKNN Runtime、模型、RGA/ORC 版 `libgstvideo` 和补丁，正常运行无需重新编译。

## ADB 部署

```sh
adb push face-rknn-package /opt/
adb shell 'cd /opt/face-rknn-package && ./run-gstreamer.sh'
adb pull /opt/face-rknn-package/output/gstreamer-face.jpg .
```

## 目录

```text
bin/          ARM64 CLI 与 GStreamer 插件
src/          共用推理核心、CLI、GStreamer 滤镜源码
include/      RKNN API 与 GstRknnFaceMeta 头文件
models/       三个 RKNN 模型
rockchip/     librga、RGA+ORC libgstvideo、头文件与补丁
scripts/      可复现构建脚本
```
