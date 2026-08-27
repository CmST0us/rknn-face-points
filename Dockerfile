FROM scratch
WORKDIR /app
COPY runtime-libs/ /lib/aarch64-linux-gnu/
COPY runtime-libs/ld-linux-aarch64.so.1 /lib/ld-linux-aarch64.so.1
COPY lib/librknnrt.so /opt/face-rknn/lib/librknnrt.so
COPY bin/face_tracking_demo .
COPY models/ models/
ENTRYPOINT ["/app/face_tracking_demo"]
