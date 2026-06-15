# AI FaceRecognition 人脸识别系统

本项目是一个基于本地深度学习模型的人脸检测与识别系统，支持注册身份库构建、单张图片识别、多人脸识别、前端可视化展示和双数据集评测。

系统使用 YuNet 完成人脸检测与五点定位，使用 SFace 完成人脸对齐和特征提取，通过特征平均构建身份库，并使用余弦相似度完成人脸身份匹配。前端部分基于 Gradio 实现，支持图片上传、识别结果展示、人脸框绘制、身份 ID / 姓名 / 相似度显示。

---

## 1. 项目功能

本项目主要实现以下功能：

1. 人脸检测：检测输入图片中的一张或多张人脸。
2. 人脸识别：将检测到的人脸与已注册身份库进行相似度匹配。
3. 身份库构建：根据注册集图片生成本地人脸特征数据库。
4. 可视化展示：在前端页面中显示上传图片、识别结果图片和识别信息。
5. 中文姓名显示：使用 PIL 绘制中文标签，避免 OpenCV 中文乱码问题。
6. 测试评估：支持自采集 20 类数据集和 CelebA 100 类数据集的识别评测。

---

## 2. 算法流程

```text
注册图片
    ↓
YuNet 人脸检测与五点定位
    ↓
SFace 人脸对齐与特征提取
    ↓
同一身份特征取平均并归一化
    ↓
保存身份特征库

输入图片
    ↓
检测全部人脸
    ↓
提取人脸特征
    ↓
与身份库进行余弦相似度匹配
    ↓
超过阈值返回身份，否则返回 unknown
```

---

## 3. 项目目录结构

```text
AI_FaceRecognition/
│
├── app.py                  # Gradio 前端界面
├── build_database.py       # 构建人脸身份库
├── recognize.py            # 统一识别接口与命令行识别入口
├── face_engine.py          # 人脸检测、对齐、特征提取
├── face_database.py        # 身份库保存、加载与相似度匹配
├── config.py               # 路径与算法参数配置
├── evaluate.py             # 测试评估脚本
├── requirements.txt        # Python 依赖
├── README.md               # 项目说明文档
│
├── model/                  # 本地 ONNX 模型文件
│   ├── face_detection_yunet.onnx
│   └── face_recognition_sface.onnx
│
├── dataset/                # 数据集目录
│   ├── registered/         # 注册集图片
│   ├── test/               # 测试集图片与标注
│   └── identities.csv      # 身份编号、姓名、类别信息
│
└── output/                 # 输出目录
    ├── face_database.npz
    ├── celeba_database.npz
    └── evaluation_report.json
```

如果 `model/` 或 `dataset/` 文件夹未上传至 GitHub，需要根据教师或小组要求单独提供模型与数据集压缩包。

---

## 4. 环境配置

建议使用 Python 3.10。

### 4.1 创建虚拟环境

```cmd
python -m venv .venv
```

激活虚拟环境：

```cmd
.venv\Scripts\activate
```

### 4.2 安装依赖

```cmd
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

如果需要运行 Gradio 前端，还需要安装：

```cmd
python -m pip install gradio==4.44.1 gradio-client==1.3.0 huggingface_hub==0.25.2 pillow
```

如果遇到 Gradio 依赖版本兼容问题，可使用以下命令固定版本：

```cmd
python -m pip install --force-reinstall pydantic==2.10.6 fastapi==0.115.6 starlette==0.41.3 gradio==4.44.1 gradio-client==1.3.0 huggingface_hub==0.25.2 pillow
```

---

## 5. 模型文件准备

本项目使用本地 ONNX 模型进行推理，不调用云端 API。

请将以下模型文件放入 `model/` 文件夹：

```text
face_detection_yunet.onnx
face_recognition_sface.onnx
```

如果模型文件较大，建议不要直接上传 GitHub，可通过网盘或课程平台单独提供，并在提交说明中注明模型放置路径。

---

## 6. 构建身份库

在项目根目录运行：

```cmd
python build_database.py
```

程序会读取注册集图片，并生成身份特征库：

```text
output/face_database.npz
```

如果修改了注册图片、身份信息或建库逻辑，需要重新运行：

```cmd
python build_database.py
```

然后重新启动前端：

```cmd
python app.py
```

---

## 7. 命令行识别图片

可以使用以下命令识别单张图片：

```cmd
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

其中：

* `identity_id`：身份编号
* `name`：身份姓名
* `bbox`：人脸框，格式为 `[x, y, width, height]`
* `similarity`：与身份库中最相似身份的余弦相似度
* `detection_score`：人脸检测置信度

---

## 8. 启动前端界面

前端使用 Gradio 实现。

运行：

```cmd
python app.py
```

终端出现以下信息说明启动成功：

```text
Running on local URL: http://127.0.0.1:7860
```

然后在浏览器打开：

```text
http://127.0.0.1:7860
```

前端支持以下操作：

1. 上传待识别图片。
2. 点击“开始识别”。
3. 查看右侧识别结果图片。
4. 查看下方识别结果信息，包括身份 ID、姓名、相似度、检测置信度和人脸框坐标。

---

## 9. 局域网访问说明

如果需要让同一局域网内的其他设备访问前端页面，可以将 `app.py` 最后一段设置为：

```python
demo.launch(
    server_name="0.0.0.0",
    server_port=7860,
    inbrowser=True,
    show_api=False
)
```

然后在命令行查询本机 IPv4 地址：

```cmd
ipconfig
```

同一局域网内其他设备可访问：

```text
http://你的IPv4地址:7860
```

注意：如果无法访问，可能需要允许 Python 通过 Windows 防火墙。

---

## 10. 公网分享说明

如果需要生成公网链接，可在 `app.py` 中设置：

```python
share=True
```

示例：

```python
demo.launch(
    server_name="127.0.0.1",
    server_port=7860,
    inbrowser=True,
    show_api=False,
    share=True
)
```

运行后终端会生成类似：

```text
https://xxxxxx.gradio.live
```

该链接可发送给其他人临时访问。

注意：公网链接依赖 Gradio 的临时隧道服务，程序关闭后链接失效。

---

## 11. 测试评估

本项目支持自采集 20 类数据集和 CelebA 100 类数据集的测试评估。

运行全部评测：

```cmd
python evaluate.py --dataset all
```

只测试自采集数据：

```cmd
python evaluate.py --dataset self
```

只测试 CelebA 数据：

```cmd
python evaluate.py --dataset celeba
```

复用已经建立的数据库：

```cmd
python evaluate.py --dataset all --skip-build
```

评测结果会保存到：

```text
output/evaluation_report.json
```

当前数据集测试结果：

```text
自采集整体准确率：92.54%（124/134）
CelebA Top-1 准确率：98.67%（296/300）
```

---

## 12. 前端测试内容

前端测试主要覆盖以下情况：

| 测试类型      | 测试内容      | 观察重点            |
| --------- | --------- | --------------- |
| 单人图       | 上传单人测试图片  | 是否正确识别身份 ID 和姓名 |
| 多人图       | 上传多人合照    | 是否检测并识别多张人脸     |
| unknown 图 | 上传未注册人物图片 | 是否返回 unknown    |
| 模糊图       | 上传模糊或低清图片 | 系统是否稳定，有无报错     |
| 侧脸图       | 上传侧脸或遮挡图片 | 检测召回是否稳定        |
| 无人脸图      | 上传风景或物品图片 | 是否提示未检测到人脸      |
| 大尺寸图      | 上传较大尺寸图片  | 前端是否卡顿或报错       |

---

## 13. 常见问题

### 13.1 `ModuleNotFoundError: No module named 'cv2'`

说明 OpenCV 未安装，执行：

```cmd
python -m pip install opencv-contrib-python
```

### 13.2 `ModuleNotFoundError: No module named 'gradio'`

说明 Gradio 未安装，执行：

```cmd
python -m pip install gradio==4.44.1
```

### 13.3 图片上中文姓名显示为乱码

OpenCV 的 `cv2.putText()` 不支持中文。前端已改用 PIL 绘制中文标签，能够正常显示中文姓名。

### 13.4 修改身份库后前端没有变化

需要重新运行：

```cmd
python build_database.py
```

然后重启前端：

```cmd
python app.py
```

### 13.5 `Settings` 中缺少 `detector_max_input_size`

说明 `build_database.py` 调用了新的配置项，但 `config.py` 未同步更新。可在 `config.py` 的 `Settings` 类中添加：

```python
detector_max_input_size: int = 640
```

---

## 14. GitHub 上传注意事项

建议不要上传以下文件：

```text
.venv/
__pycache__/
output/
*.pyc
*.log
```

如果模型或数据集较大，也可以不上传：

```text
model/
dataset/
```

但需要在 README 或提交说明中注明模型和数据集的获取方式。

推荐 `.gitignore` 内容：

```gitignore
.venv/
venv/
__pycache__/
*.pyc
*.pyo
.vscode/
.idea/
.DS_Store
Thumbs.db
*.log
output/
```

---

## 15. 小组分工说明

本项目小组分工如下：

* 成员 1：数据收集与标注
* 成员 2：算法与后端代码开发
* 成员 3：前端界面、可视化展示与系统测试
* 成员 4：文档整合、报告撰写与课堂演示

本 README 主要说明系统运行、前端使用、身份库构建和测试评估流程，方便教师、小组成员或其他用户快速复现项目。
