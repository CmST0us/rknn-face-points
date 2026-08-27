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

V4L2 实时输入：

```sh
./run-v4l2.sh /dev/video0
```

结果为 `output/face-output.mkv`。ROCK 5B 的 `/dev/video0` 是 HDMI RX 时，需要先接入有效 HDMI 信号。

## GStreamer 节点

`rknnfacemesh` 接收并输出 `video/x-raw,format=RGB`；`draw=true` 默认在输出帧上绘制人脸框和 478 点，设为 `false` 可仅输出元数据：

```sh
docker compose run --rm --no-deps gstreamer-face -e \
  v4l2src device=/dev/video0 ! videoconvert ! video/x-raw,format=RGB ! \
  rknnfacemesh model-dir=/app/models draw=true ! videoconvert ! fakesink
```

每帧附带 `GstRknnFaceMeta`，包含 bbox、478 个 xyz 点、yaw/pitch/roll、52 个 blendshape、`tongue_out` 和 NPU 耗时。应用侧结构与访问函数见 `include/gst_rknn_face_meta.h`。

## RGA 与 NEON

Compose 默认设置：

```sh
GST_VIDEO_CONVERT_USE_RGA=1
GST_VIDEO_CONVERT_RGA_DMA_HEAP=/dev/dma_heap/cma
```

镜像内的 GStreamer 1.24.2 `libgstvideo` 同时链接 Rockchip `librga` 和 ORC：支持格式走 RGA；RGA 不支持或失败时回退到 ORC 的 AArch64 NEON 路径。CMA DMA buffer 修复了 RK3588 旧补丁使用 4GB 以上虚拟地址时的 `RGA_BLIT EINVAL`。

容器仅映射所需设备：NPU `/dev/dri/card1`、`/dev/rga`、CMA DMA heap 和 V4L2 设备，不使用 `--privileged`。

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
scripts/      三个可复现构建脚本
```
