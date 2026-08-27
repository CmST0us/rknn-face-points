# RK3588 RKNN 人脸姿态与表情 Demo

离线可构建的 ARM64 Docker 包，包含人脸检测、478 点人脸网格、头姿估计、表情系数和 `tongueOut`。

## 环境

- RK3588 / Radxa ROCK 5B
- RKNPU 驱动 0.9.8
- Docker Engine 与 Docker Compose v2
- NPU DRM 节点 `/dev/dri/card1`

## 直接运行

```sh
chmod +x run.sh
./run.sh
```

指定图片：

```sh
./run.sh /absolute/path/face.jpg
```

结果写入 `output/face_result.ppm`，终端输出检测置信度、478 点、yaw/pitch/roll、表情系数和 NPU 耗时。

也可以直接使用 Compose：

```sh
mkdir -p output
docker compose run --rm --build face-rknn
```

## 目录

```text
bin/                 预编译 ARM64 Demo
src/                 C++ 源码
include/             RKNN API 头文件
third_party/         stb_image
lib/                 RKNN Runtime 2.3.2
runtime-libs/        scratch 镜像所需 ARM64 系统动态库
models/              三个 RKNN 模型
assets/test.jpg      测试图片
scripts/             可选的板端原生编译脚本
```

Dockerfile 使用 `FROM scratch`，构建过程不访问网络。宿主机只映射 `/dev/dri/card1`，不需要 `--privileged`。

## 重新编译 C++

设备装有 `g++` 时：

```sh
./scripts/build-native.sh
```

## ADB 部署与取回结果

```sh
adb push face-rknn-demo /opt/
adb shell 'cd /opt/face-rknn-demo && ./run.sh'
adb pull /opt/face-rknn-demo/output/face_result.ppm .
```

当前版本只处理单张图片并选择置信度最高的人脸；实时 HDMI/V4L2 取流尚未接入。
