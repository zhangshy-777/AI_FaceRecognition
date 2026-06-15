# 双数据集评测说明

`evaluate.py` 使用项目现有的 YuNet、SFace 和特征平均算法，分别评测：

- 自采集20类数据：使用 `dataset/test/annotations.jsonl` 中的人脸框和身份标注。
- CelebA 100类数据：使用 `register/` 建库，使用 `test/` 进行闭集 Top-1 识别。

两套数据会分别建立数据库，不会互相混合：

```text
output/face_database.npz
output/celeba_database.npz
```

## 运行全部评测

在项目根目录执行：

```powershell
.\.venv\Scripts\python.exe evaluate.py --dataset all
```

该命令会重新建立两套数据库并完成全部评测。

只测试自采集数据：

```powershell
.\.venv\Scripts\python.exe evaluate.py --dataset self
```

只测试 CelebA：

```powershell
.\.venv\Scripts\python.exe evaluate.py --dataset celeba
```

复用已经建立的数据库：

```powershell
.\.venv\Scripts\python.exe evaluate.py --dataset all --skip-build
```

## 输出指标

自采集数据输出：

- `Overall accuracy`：正确检测并正确判断身份的人脸数 / 标注人脸总数。
- `Detection recall`：成功匹配到检测框的人脸数 / 标注人脸总数。
- `Known-face accuracy`：已注册人物识别准确率。
- `Unknown recall`：未注册人物被正确拒识为 `unknown` 的比例。
- `Strict image acc.`：一张图片中所有人脸都正确且没有多余检测时才算正确。

CelebA 输出：

- `Top-1 accuracy`：300张测试图中，最高相似度身份与真实身份一致的比例。
- `Detection recall`：300张测试图中成功检测到人脸的比例。

CelebA 是100类闭集测试，因此直接取相似度最高的身份计算 Top-1，不使用
`0.45` 开放集拒识阈值。

完整结果和失败样例保存在：

```text
output/evaluation_report.json
```

当前数据运行结果：

```text
自采集整体准确率：92.54%（124/134）
CelebA Top-1准确率：98.67%（296/300）
```
