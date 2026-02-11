# sd-scripts-anima Quick Start / 快速开始

This fork focuses on **Anima (Cosmos-Predict2) LoRA/LoKr training** with a native entrypoint: `anima_train_network.py`.

本仓库聚焦 **Anima（Cosmos-Predict2）LoRA/LoKr 训练**，使用原生入口：`anima_train_network.py`。

## 中文：最快启动训练

### 1. 你需要准备

- Windows 10/11 或 Linux/WSL2
- NVIDIA GPU（建议 >= 12GB 显存）
- Python `3.10.x`
- Git

### 2. 依赖安装（Windows）

```powershell
git clone https://github.com/FHfanshu/sd-scripts-anima.git
cd sd-scripts-anima

python -m venv venv
.\venv\Scripts\activate

pip install --upgrade pip setuptools wheel
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
pip install --upgrade -r requirements.txt

accelerate config
```

`accelerate config` 推荐回答：

```text
This machine
No distributed training
NO
NO
NO
all
bf16 (or fp16)
```

如果 TensorBoard 报 `No module named 'pkg_resources'`：

```powershell
pip install "setuptools<81"
```

更完整安装说明见：[`docs/anima_dependency_guide.md`](docs/anima_dependency_guide.md)

### 3. 准备训练数据

每张图配同名 `.txt` 标注，例如：

```text
train_images/
  0001.png
  0001.txt
  0002.jpg
  0002.txt
```

### 4. 使用最小示例配置（已精简常用参数）

示例文件：

- [`configs/examples/anima_quickstart_train_args.toml`](configs/examples/anima_quickstart_train_args.toml)
- [`configs/examples/anima_quickstart_dataset.toml`](configs/examples/anima_quickstart_dataset.toml)

先把以下路径改成你本机路径：

- `anima_transformer`
- `vae`
- `qwen`
- `t5_tokenizer_dir`
- `image_dir`

### 5. 启动训练

```powershell
accelerate launch anima_train_network.py --config_file configs/examples/anima_quickstart_train_args.toml --dataset_config configs/examples/anima_quickstart_dataset.toml
```

### 6. 查看日志（默认 TensorBoard）

```powershell
tensorboard --logdir output/anima_quickstart/logs
```

默认日志目录规则：`<output_dir>/logs/<output_name>_YYYYMMDD_HHMMSS_ffffff`

### 7. 保留的常用参数（示例中已覆盖）

- 模型路径：`anima_transformer`、`vae`、`qwen`、`t5_tokenizer_dir`
- 训练核心：`max_train_epochs`、`train_batch_size`、`gradient_accumulation_steps`、`learning_rate`
- 网络：`network_module`、`network_dim`、`network_alpha`
- 性能：`mixed_precision`、`xformers`、`cache_latents`
- 输出：`output_dir`、`output_name`、`save_every_n_epochs`

不常用参数（监控阈值、高级 scheduler、大量 network_args 等）已从 quickstart 示例移除。

## English: Fastest Training Start

### 1. Requirements

- Windows 10/11 or Linux/WSL2
- NVIDIA GPU (recommended >= 12GB VRAM)
- Python `3.10.x`
- Git

### 2. Dependency Setup (Windows)

```powershell
git clone https://github.com/FHfanshu/sd-scripts-anima.git
cd sd-scripts-anima

python -m venv venv
.\venv\Scripts\activate

pip install --upgrade pip setuptools wheel
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
pip install --upgrade -r requirements.txt

accelerate config
```

Recommended `accelerate config` answers:

```text
This machine
No distributed training
NO
NO
NO
all
bf16 (or fp16)
```

If TensorBoard fails with `No module named 'pkg_resources'`:

```powershell
pip install "setuptools<81"
```

Full install guide: [`docs/anima_dependency_guide.md`](docs/anima_dependency_guide.md)

### 3. Dataset Layout

Use one caption file per image:

```text
train_images/
  0001.png
  0001.txt
  0002.jpg
  0002.txt
```

### 4. Use Minimal Example Configs (common args only)

- [`configs/examples/anima_quickstart_train_args.toml`](configs/examples/anima_quickstart_train_args.toml)
- [`configs/examples/anima_quickstart_dataset.toml`](configs/examples/anima_quickstart_dataset.toml)

Update these paths first:

- `anima_transformer`
- `vae`
- `qwen`
- `t5_tokenizer_dir`
- `image_dir`

### 5. Launch Training

```powershell
accelerate launch anima_train_network.py --config_file configs/examples/anima_quickstart_train_args.toml --dataset_config configs/examples/anima_quickstart_dataset.toml
```

### 6. TensorBoard (enabled by default)

```powershell
tensorboard --logdir output/anima_quickstart/logs
```

Default run directory format: `<output_dir>/logs/<output_name>_YYYYMMDD_HHMMSS_ffffff`

### 7. Common Parameters Kept in the Example

- Model paths: `anima_transformer`, `vae`, `qwen`, `t5_tokenizer_dir`
- Training core: `max_train_epochs`, `train_batch_size`, `gradient_accumulation_steps`, `learning_rate`
- Network: `network_module`, `network_dim`, `network_alpha`
- Performance: `mixed_precision`, `xformers`, `cache_latents`
- Output: `output_dir`, `output_name`, `save_every_n_epochs`

Less-common options (monitor thresholds, advanced scheduler knobs, large `network_args`, etc.) are intentionally removed from quickstart examples.

## More Docs

- Main Anima doc: [`docs/anima_train_network.md`](docs/anima_train_network.md)
- Dependency guide: [`docs/anima_dependency_guide.md`](docs/anima_dependency_guide.md)
- Root-style config migration tool: `tools/convert_anima_root_to_kohya.py`

