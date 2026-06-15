# 人脸识别核心算法

本目录实现团队作业中的算法和后端代码部分。测试评估脚本与前端界面由其他组员在统一接口上继续开发。

## 算法流程

```text
注册图片 -> YuNet 人脸检测与五点定位 -> SFace 对齐与特征提取
        -> 同一身份特征取平均并归一化 -> 保存身份特征库

输入图片 -> 检测全部人脸 -> 提取特征 -> 余弦相似度匹配
        -> 超过阈值返回身份，否则返回 unknown
```

所有模型均在本地通过 OpenCV DNN/ONNX 推理，不调用云端 API，也不使用测试集进行训练。

## 文件说明

```text
config.py            路径和算法参数
face_engine.py       检测、对齐和 embedding 提取
face_database.py     身份库保存、加载和相似度匹配
build_database.py    使用注册集建立身份库
recognize.py         统一识别接口和命令行入口
model/               本地 ONNX 模型
output/              生成的身份特征库
dataset/             注册集、测试集和身份表
```

`app.py` 预留给负责前端的组员。测试组员可另建 `evaluate.py`，直接调用 `FaceRecognizer`。

## 环境安装

建议使用 Python 3.10：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

按 [model/README.md](model/README.md) 下载两个 ONNX 模型。

## 建立身份库

```powershell
python build_database.py
```

程序读取 `dataset/registered/p01` 到 `p20`。注册图中检测到多人时选择面积最大的人脸，同一身份的多张 embedding 取平均后进行 L2 归一化。

输出文件：

```text
output/face_database.npz
```

## 识别图片

```powershell
python recognize.py --image dataset/test/images/p01_t01.jpg
```

输出示例：

```json
[
  {
    "identity_id": "p01",
    "name": "成龙",
    "bbox": [124, 248, 437, 498],
    "similarity": 0.8123,
    "detection_score": 0.9971
  }
]
```

`bbox` 格式与作业标注一致，均为 `[x, y, width, height]`。

## 提供给其他组员的接口

```python
from recognize import FaceRecognizer

recognizer = FaceRecognizer()
results = recognizer.recognize_file("input.jpg")
```

如果前端已经得到 OpenCV BGR 图像：

```python
results = recognizer.recognize_image(image)
rendered = recognizer.draw_results(image, results)
```

## 参数说明

参数集中在 `config.py`：

- `detection_score_threshold`：YuNet 检测置信度阈值，当前为 `0.50`，用于兼顾侧脸和多人图召回率。
- `minimum_face_size`：过滤过小的人脸。
- `recognition_threshold`：已知身份与 `unknown` 的余弦相似度阈值。

当前识别阈值 `0.45` 是初始值。最终阈值应由测试负责人使用独立验证数据确定，不能使用正式测试集反复调参。


## 实现约束

- `identities.csv` 当前为 GBK 编码，建库脚本已自动兼容 UTF-8、GB18030 和 GBK。
- Windows 中文路径通过 `numpy.fromfile + cv2.imdecode` 读取。
- 注册图未检测到有效人脸时不会静默生成错误特征库。
- 模型或身份库缺失时会给出明确错误信息。
