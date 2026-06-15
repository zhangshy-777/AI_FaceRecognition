# 模型文件

本项目使用 OpenCV Zoo 中的两个本地 ONNX 模型：

1. YuNet 人脸检测模型：
   `face_detection_yunet_2023mar.onnx`
2. SFace 人脸识别模型：
   `face_recognition_sface_2021dec.onnx`

下载后将文件放在本目录，最终结构如下：

```text
model/
├── face_detection_yunet_2023mar.onnx
└── face_recognition_sface_2021dec.onnx
```

模型来源：

- https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet
- https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface

模型必须下载到本地后再运行，代码不会调用任何云端人脸识别 API。

