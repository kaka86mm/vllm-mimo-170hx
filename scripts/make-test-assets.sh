#!/bin/bash
# 用镜像内置 ffmpeg 生成多模态测试素材（无需在宿主机安装任何东西）
# 产出: /tmp/mimo-test-video.mp4 (3s 带正弦音轨)、mimo-test-audio.wav (440Hz)、mimo-test-img.png
set -e
cd "$(dirname "$0")"
[ -f env.sh ] && . ./env.sh
IMAGE=${IMAGE:-docker.m.daocloud.io/lazymio/vllm-backport:v0.13.0-sm80}

docker run --rm --entrypoint ffmpeg -v /tmp:/tmp "$IMAGE" -y \
  -f lavfi -i "testsrc=duration=3:size=320x240:rate=15" \
  -f lavfi -i "sine=frequency=440:duration=3" \
  -c:v libx264 -pix_fmt yuv420p -c:a aac /tmp/mimo-test-video.mp4 2>&1 | tail -1

docker run --rm --entrypoint ffmpeg -v /tmp:/tmp "$IMAGE" -y \
  -i /tmp/mimo-test-video.mp4 -vn -acodec pcm_s16le -ar 16000 /tmp/mimo-test-audio.wav 2>&1 | tail -1

docker run --rm --entrypoint ffmpeg -v /tmp:/tmp "$IMAGE" -y \
  -f lavfi -i "color=c=red:size=320x240:duration=1:rate=1" /tmp/mimo-test-img.png 2>&1 | tail -1

ls -la /tmp/mimo-test-video.mp4 /tmp/mimo-test-audio.wav /tmp/mimo-test-img.png
echo "素材就绪。验证: python3 omni-test2.py && python3 video-diag.py"
