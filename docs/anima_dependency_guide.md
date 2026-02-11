# Anima Dependency Installation Guide / Anima 依赖安装向导

## 中文

### 1. 推荐环境

- Python: `3.10.x`
- PyTorch: `2.6.0`
- CUDA: `12.4`（示例）
- 系统: Windows 10/11 或 Linux/WSL2

### 2. Windows 安装步骤

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

`accelerate config` 推荐：

```text
This machine
No distributed training
NO
NO
NO
all
bf16 (or fp16)
```

### 3. Linux / WSL2

```bash
git clone https://github.com/FHfanshu/sd-scripts-anima.git
cd sd-scripts-anima

python -m venv venv
source venv/bin/activate

pip install --upgrade pip setuptools wheel
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
pip install --upgrade -r requirements.txt

accelerate config
```

### 4. 可选项

安装 xformers（可选）：

```bash
pip install xformers --index-url https://download.pytorch.org/whl/cu124
```

### 5. 常见问题

- TensorBoard 报错 `No module named 'pkg_resources'`

```bash
pip install "setuptools<81"
```

- `ValueError: fp16 mixed precision requires a GPU`
  - 重新执行 `accelerate config`，GPU 选择填 `0`（而不是 `all`）。

- CUDA 版本不同
  - 替换 PyTorch 安装 URL，例如 `cu121`、`cu124`。

## English

### 1. Recommended Environment

- Python: `3.10.x`
- PyTorch: `2.6.0`
- CUDA: `12.4` (example)
- OS: Windows 10/11 or Linux/WSL2

### 2. Windows Setup

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

### 3. Linux / WSL2

```bash
git clone https://github.com/FHfanshu/sd-scripts-anima.git
cd sd-scripts-anima

python -m venv venv
source venv/bin/activate

pip install --upgrade pip setuptools wheel
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
pip install --upgrade -r requirements.txt

accelerate config
```

### 4. Optional

Install xformers (optional):

```bash
pip install xformers --index-url https://download.pytorch.org/whl/cu124
```

### 5. Common Issues

- TensorBoard error `No module named 'pkg_resources'`

```bash
pip install "setuptools<81"
```

- `ValueError: fp16 mixed precision requires a GPU`
  - Re-run `accelerate config` and set GPU list to `0` instead of `all`.

- Different CUDA version
  - Change the PyTorch wheel URL, e.g. `cu121` or `cu124`.
