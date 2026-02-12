# sd-scripts-anima Quick Start / 快速开始

This fork focuses on **Anima (Cosmos-Predict2) LoRA/LoKr training** with a native entrypoint: `anima_train_network.py`.

本仓库聚焦 **Anima（Cosmos-Predict2）LoRA/LoKr 训练**，使用原生入口：`anima_train_network.py`。

## Quickstart Files / 快速入口文件

- Single-file template (with `[dataset]` section):
  `configs/examples/anima_quickstart_single.toml`
- Converter:
  `tools/convert_anima_root_to_kohya.py`
- Training entrypoint:
  `anima_train_network.py`

## 中文：最快启动训练

### 1. 你需要准备

- Windows 10/11 或 Linux/WSL2
- NVIDIA GPU（建议 >= 16GB 显存）
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

每张图配同名 `.txt` 标注，推荐使用 Kohya-style 子目录（例如 `10_subject`）：

```text
train/
  10_subject/
    0001.png
    0001.txt
    0002.jpg
    0002.txt
```

`dataset.data_dir` 建议指向 `.../train/10_subject`。重复次数由 `dataset.repeats` 控制。

### 4. 使用单文件 TOML 示例（含 `[dataset]` 分类）

示例文件：

- [`configs/examples/anima_quickstart_single.toml`](configs/examples/anima_quickstart_single.toml)

该文件按分区组织：

- `[model]`
- `[dataset]`
- `[training]`
- `[lora]`
- `[optimizer]`
- `[output]`

先把这些路径改成你本机路径：

- `model.transformer_path`
- `model.vae_path`
- `model.text_encoder_path`
- `model.t5_tokenizer_dir`
- `dataset.data_dir`

### 5. 单文件直接启动训练（自动转换，无需手动）

```powershell
accelerate launch anima_train_network.py --config_file configs/examples/anima_quickstart_single.toml
```

说明：当 `--config_file` 指向 root-style 单文件（含 `[model]` 与 `[training]`）时，入口会自动转换为 Kohya 训练参数和数据集配置，无需手动运行转换脚本。
该过程在内存中完成，不会写出 `train_args.toml` / `dataset.toml` 中间文件。
若同时传入 `--dataset_config`，则以 CLI 的 `--dataset_config` 为准，并忽略单文件中的 `[dataset]` 段。

### 6. 一键启动训练 + TensorBoard（默认自动拉起）

```powershell
accelerate launch anima_train_network.py --config_file configs/examples/anima_quickstart_single.toml
```

默认行为：

- 当 `log_with = "tensorboard"`（或 `all`）时，会在训练启动前自动拉起 TensorBoard Web 服务
- 控制台会打印访问地址，例如：`TensorBoard: http://127.0.0.1:6006`
- 若 `6006` 被占用，会自动尝试后续可用端口并打印最终端口

可选参数：

- `--tensorboard_host 127.0.0.1`
- `--tensorboard_port 6006`
- `--tensorboard_logdir <dir>`（默认使用 `logging_dir`）
- `--no-auto_start_tensorboard`（关闭自动拉起）

### 7. 单独启动 TensorBoard（需要另一个终端）

```powershell
.\venv\Scripts\python.exe -m tensorboard.main --logdir output/anima_quickstart/logs --host 127.0.0.1 --port 6006
```

浏览器访问：`http://127.0.0.1:6006`

默认日志目录规则：`./logs/<output_name>_YYYYMMDD_HHMMSS_ffffff`

### 8. 保留的常用参数（示例中已覆盖）

- 模型路径：`anima_transformer`、`vae`、`qwen`、`t5_tokenizer_dir`
- 训练核心：`max_train_epochs`、`train_batch_size`、`gradient_accumulation_steps`、`learning_rate`
- 网络：`network_module`、`network_dim`、`network_alpha`
- 性能：`mixed_precision`、`xformers`、`cache_latents`
- 输出：`output_dir`、`output_name`、`save_every_n_epochs`

不常用参数（监控阈值、高级 scheduler、大量 network_args 等）已从 quickstart 示例移除。

## English: Fastest Training Start

### 1. Requirements

- Windows 10/11 or Linux/WSL2
- NVIDIA GPU (recommended >= 16GB VRAM)
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

Use one caption file per image, and prefer a Kohya-style subfolder layout (for example `10_subject`):

```text
train/
  10_subject/
    0001.png
    0001.txt
    0002.jpg
    0002.txt
```

Set `dataset.data_dir` to `.../train/10_subject`. Repeat count is controlled by `dataset.repeats`.

### 4. Use Single-File TOML Example (with `[dataset]` section)

- [`configs/examples/anima_quickstart_single.toml`](configs/examples/anima_quickstart_single.toml)

This file is organized by sections:

- `[model]`
- `[dataset]`
- `[training]`
- `[lora]`
- `[optimizer]`
- `[output]`

Update these paths first:

- `model.transformer_path`
- `model.vae_path`
- `model.text_encoder_path`
- `model.t5_tokenizer_dir`
- `dataset.data_dir`

### 5. Launch Training from Single TOML (auto-convert, no manual step)

```powershell
accelerate launch anima_train_network.py --config_file configs/examples/anima_quickstart_single.toml
```

Note: when `--config_file` points to a root-style single TOML (with `[model]` and `[training]`), the entrypoint auto-converts it to Kohya train args + dataset config.
The conversion is performed in memory only (no intermediate `train_args.toml` / `dataset.toml` files are written).
If both inline `[dataset]` and CLI `--dataset_config` are provided, CLI `--dataset_config` takes precedence.

### 6. One-Command Start: Training + TensorBoard (auto-start by default)

```powershell
accelerate launch anima_train_network.py --config_file configs/examples/anima_quickstart_single.toml
```

Default behavior:

- If `log_with = "tensorboard"` (or `all`), TensorBoard web service is auto-started before training
- The URL is printed in console, for example: `TensorBoard: http://127.0.0.1:6006`
- If `6006` is busy, the entrypoint automatically picks the next available port and prints it

Optional flags:

- `--tensorboard_host 127.0.0.1`
- `--tensorboard_port 6006`
- `--tensorboard_logdir <dir>` (defaults to `logging_dir`)
- `--no-auto_start_tensorboard` (disable auto-start)

### 7. Start TensorBoard Separately (in another terminal)

```powershell
.\venv\Scripts\python.exe -m tensorboard.main --logdir output/anima_quickstart/logs --host 127.0.0.1 --port 6006
```

Open in browser: `http://127.0.0.1:6006`

Default run directory format: `./logs/<output_name>_YYYYMMDD_HHMMSS_ffffff`

### 8. Common Parameters Kept in the Example

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

