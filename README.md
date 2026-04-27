# 人物照片转绘画骨架/姿势图（支持本地网页）

你可以把照片中的人物转换为**绘画用骨架/姿势图**，支持：
- 单张图片
- 批量图片
- 本地网页交互操作

## 1) 安装依赖

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2) 本地网页运行（推荐）

```bash
python app.py
```

启动后打开浏览器：
- `http://127.0.0.1:7860`
- 或 `http://localhost:7860`

网页中提供：
- 单张上传 -> 实时生成骨架图
- 批量上传 -> 批量生成并下载结果

## 3) 命令行运行（可选）

### 单张

```bash
python pose_sketch.py \
  --input examples/person.jpg \
  --output outputs/person_pose.png
```

### 批量

```bash
python pose_sketch.py \
  --batch \
  --input examples/ \
  --output outputs/
```

## 常用参数

- `--model`：模型名或模型文件路径（默认 `yolov8n-pose.pt`）
- `--conf`：人体检测阈值（默认 `0.25`）
- `--kpt-conf`：关键点置信度阈值（默认 `0.4`）
- `--line-thickness`：骨架线条粗细（默认 `3`）
- `--point-radius`：关键点圆点大小（默认 `4`）

## 说明

- 首次运行会自动下载 `yolov8n-pose.pt`。
- 输出图为白底、黑色骨架线、红色关键点。
- 没检测到人时输出纯白画布（保持原图尺寸）。
