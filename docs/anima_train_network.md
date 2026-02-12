# Anima LoRA/LoKr Training (`anima_train_network.py`)

`anima_train_network.py` is the native Anima training entrypoint in this repo.

`anima_train_network.py` 是本仓库内的 Anima 原生训练入口。

## 中文

## 1. 配置模式

- 主推荐：单文件 root-style TOML（`[model] [dataset] [training] [lora] [optimizer] [output]`）
- 兼容模式：`--config_file <train_args.toml>` + `--dataset_config <dataset.toml>`
- 旧的 root-style `--config` 参数已移除

如需从旧配置迁移：

```bash
python tools/convert_anima_root_to_kohya.py --input old_root.toml --output_dir converted
```

## 2. 最小启动命令

```bash
accelerate launch anima_train_network.py --config_file configs/examples/anima_quickstart_single.toml
```

当 `--config_file` 指向 root-style 单文件时，入口会在内存中自动适配为 Kohya 参数（不生成中间 `train_args.toml` / `dataset.toml` 文件）。

若同时提供 `--dataset_config`，则优先使用 `--dataset_config`，并忽略单文件中的 `[dataset]` 段。

## 3. 必填参数（最小）

- `anima_transformer`（或 `pretrained_model_name_or_path`）
- `vae`
- `qwen`
- `t5_tokenizer_dir`
- `network_module`（`networks.lora_anima` 或 `networks.lokr_anima`）
- `output_dir` / `output_name`

## 4. 推荐常用参数（保留高频）

- `max_train_epochs`
- `train_batch_size`
- `gradient_accumulation_steps`
- `learning_rate`
- `lr_scheduler = "constant"`
- `lr_warmup_steps = 0`
- `mixed_precision = "bf16"`（或 `fp16`）
- `xformers = true`
- `cache_latents = true`

## 5. LoKr 快速规则

- 使用 `network_module = "networks.lokr_anima"`
- 若 `network_dim >= 100000`，会自动触发 full-matrix sentinel 语义
- 若只想快速跑通，可改为较小 `network_dim`（例如 `16`）

## 6. 默认行为（无需额外参数）

- 默认启用 TensorBoard 日志
- 当 `log_with=tensorboard`（或 `all`）时，训练启动前自动拉起 TensorBoard Web 服务
- 控制台会打印 TensorBoard 访问地址（若默认端口占用会自动顺延到可用端口）
- 日志根目录默认 `./logs`
- 日志子目录默认 `<output_name>_YYYYMMDD_HHMMSS_ffffff`
- `train_norm` 默认开启
- `t5_tokenizer_dir` 缺文件时默认自动下载补齐（HF 优先，失败后 ModelScope 回退）

可选参数：

- `--tensorboard_host`（默认 `127.0.0.1`）
- `--tensorboard_port`（默认 `6006`）
- `--tensorboard_logdir`（默认复用 `logging_dir`）
- `--no-auto_start_tensorboard`（关闭自动拉起）

## 7. 续训建议

- 完整续训用 `--resume <state_dir>`
- 只加载网络权重用 `--network_weights <file>`（不是完整断点恢复）

## English

## 1. Config Mode

- Recommended: single root-style TOML (`[model] [dataset] [training] [lora] [optimizer] [output]`)
- Compatible mode: `--config_file <train_args.toml>` + `--dataset_config <dataset.toml>`
- Legacy root-style `--config` flag is removed

Migration from old config:

```bash
python tools/convert_anima_root_to_kohya.py --input old_root.toml --output_dir converted
```

## 2. Minimal Launch Command

```bash
accelerate launch anima_train_network.py --config_file configs/examples/anima_quickstart_single.toml
```

When `--config_file` points to a root-style single TOML, the entrypoint adapts it in memory to Kohya-compatible args (no intermediate `train_args.toml` / `dataset.toml` files are written).

If both inline `[dataset]` and CLI `--dataset_config` are provided, CLI `--dataset_config` takes precedence.

## 3. Required Parameters (minimal)

- `anima_transformer` (or `pretrained_model_name_or_path`)
- `vae`
- `qwen`
- `t5_tokenizer_dir`
- `network_module` (`networks.lora_anima` or `networks.lokr_anima`)
- `output_dir` / `output_name`

## 4. Recommended Common Parameters

- `max_train_epochs`
- `train_batch_size`
- `gradient_accumulation_steps`
- `learning_rate`
- `lr_scheduler = "constant"`
- `lr_warmup_steps = 0`
- `mixed_precision = "bf16"` (or `fp16`)
- `xformers = true`
- `cache_latents = true`

## 5. LoKr Quick Rule

- Use `network_module = "networks.lokr_anima"`
- `network_dim >= 100000` automatically triggers full-matrix sentinel behavior
- For quick smoke runs, use a smaller `network_dim` (for example `16`)

## 6. Defaults (no extra flags needed)

- TensorBoard logging is enabled by default
- When `log_with=tensorboard` (or `all`), TensorBoard web service is auto-started before training
- TensorBoard URL is printed in console (if default port is busy, the next available port is used)
- Default log root: `./logs`
- Default run dir: `<output_name>_YYYYMMDD_HHMMSS_ffffff`
- `train_norm` is enabled by default
- Missing tokenizer files under `t5_tokenizer_dir` are auto-downloaded by default (HF first, ModelScope fallback)

Optional flags:

- `--tensorboard_host` (default `127.0.0.1`)
- `--tensorboard_port` (default `6006`)
- `--tensorboard_logdir` (defaults to `logging_dir`)
- `--no-auto_start_tensorboard` (disable auto-start)

## 7. Resume Recommendation

- Full resume: `--resume <state_dir>`
- Weight init only: `--network_weights <file>` (not full optimizer/scheduler resume)

