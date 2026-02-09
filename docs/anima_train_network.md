# Anima LoRA / LoKr Training (`sd-scripts-anima`)

`anima_train_network.py` 是 `sd-scripts-anima` 内的原生 `NetworkTrainer` 训练入口，不依赖上级目录仓库。

## 1. 配置策略（重要）

从当前版本开始，Anima 训练入口只支持 Kohya 原生参数体系：

- 训练参数：CLI 直接传参，或 `--config_file <train_args.toml>`
- 数据集参数：`--dataset_config <dataset.toml>`
- 训练语义：固定 LLMAdapter 条件链路（`Qwen hidden + T5 ids -> preprocess_text_embeds`）

旧的 root-style Anima 配置（`[model] / [dataset] / [training] ...`）不再由 `anima_train_network.py` 直接加载。  
`--config` 参数已移除。

如果你手上是旧 root-style 配置，请先转换：

```bash
python tools/convert_anima_root_to_kohya.py \
  --input /path/to/old_anima_root.toml \
  --output_dir /path/to/converted
```

将生成：

- `train_args.toml`（用于 `--config_file`）
- `dataset.toml`（用于 `--dataset_config`）

## 2. 快速启动（Kohya 原生配置）

```bash
accelerate launch anima_train_network.py \
  --config_file /path/to/train_args.toml \
  --dataset_config /path/to/dataset.toml
```

项目内提供了一个 10-step smoke 配置模板：

- `configs/smoke/anima_10steps_train_args.toml`
- `configs/smoke/anima_dataset.toml`

示例：

```bash
accelerate launch anima_train_network.py \
  --config_file configs/smoke/anima_10steps_train_args.toml \
  --dataset_config configs/smoke/anima_dataset.toml
```

你也可以不使用 `--config_file`，直接传参数：

```bash
accelerate launch anima_train_network.py \
  --network_module networks.lokr_anima \
  --anima_transformer /path/to/anima_transformer.safetensors \
  --vae /path/to/anima_vae.safetensors \
  --qwen /path/to/qwen_model_dir \
  --t5_tokenizer_dir /path/to/t5_tokenizer_dir \
  --dataset_config /path/to/dataset.toml \
  --output_dir /path/to/output \
  --output_name anima_lokr \
  --max_train_steps 1000
```

## 3. 必填参数

- `--anima_transformer`（或 `--pretrained_model_name_or_path`）
- `--vae`
- `--qwen`
- `--t5_tokenizer_dir`
- `--network_module`：`networks.lora_anima` 或 `networks.lokr_anima`
- `--auto_download_t5_tokenizer`（默认 `true`，启动时缺文件自动下载）
- `--t5_tokenizer_repo_id`（自动下载源，支持 Hugging Face repo id / Hugging Face URL / ModelScope URL）
- `--t5_tokenizer_subfolder`（默认 `tokenizer`，用于 Cosmos tokenizer 子目录）
- `--t5_tokenizer_modelscope_fallback`（默认 `true`，HF 下载失败时自动回退 ModelScope）
- `--t5_tokenizer_modelscope_repo_id`（默认 `nv-community/Cosmos-Predict2-2B-Text2Image`）
- `--t5_tokenizer_modelscope_revision`（默认 `master`）
- `--t5_tokenizer_modelscope_subfolder`（默认 `tokenizer`）
- `--train_norm`（默认 `true`，训练 LayerNorm / RMSNorm 的 weight/bias；可用 `--no-train_norm` 关闭）

## 4. 文本条件训练语义

训练链路固定为：

1. `Qwen` 编码得到 hidden states
2. `AnimaTransformer.preprocess_text_embeds(qwen_hidden, t5_ids)`
3. 将融合后的条件送入 Anima Transformer 主前向

即 `T5 tokenizer -> text_ids` 是必经链路。
当前入口不提供 `dit_only` 之类的无 T5 训练模式。

## 5. 导出格式与续训建议

- Anima LoRA / LoKr 导出默认使用 ComfyUI 可识别键（`diffusion_model.*`）。
- 训练范围默认覆盖 `attn + mlp + llm_adapter + norm`。
- LoKr 采用 Kohya/LyCORIS 的 full-matrix sentinel 语义：当 `network_dim >= 100000` 时，自动强制 `lokr_full_matrix=true`（不改写原始 dim）。
- 若需要严格断点续训，建议使用 `--resume` 状态目录（包含 optimizer/scheduler/step）。
- `--network_weights` 单文件热启动属于权重初始化，不等价于完整断点恢复。

## 6. ScheduleFree 约束

当 `optimizer_type=RAdamScheduleFree`（或别名 `radam_schedulefree`）时：

- `lr_scheduler` 必须是 `constant`
- `lr_warmup_steps` 必须是 `0`
- `lr_warmup_ratio` 必须是 `0`

否则会在参数校验阶段直接报错退出。

## 7. 常见问题

- `T5 tokenizer directory is required`
  - 必须提供 `--t5_tokenizer_dir`。若目录缺 `config.json`、`spiece.model`、`tokenizer.json`，默认会在训练启动时自动下载补齐。
  - 自动下载默认先尝试 `--t5_tokenizer_repo_id`（支持 HF repo id / HF URL），失败后自动回退到 ModelScope（可用 `--no-t5_tokenizer_modelscope_fallback` 关闭）。
  - 如果你不希望联网下载，可设置 `--no-auto_download_t5_tokenizer`，此时缺文件会直接报错。

- `Unsupported network module`
  - 仅支持 `networks.lora_anima` / `networks.lokr_anima`。

- `image too large, but cropping and bucketing are disabled`
  - 在 `dataset.toml` 开启 bucket（`enable_bucket=true`）或启用裁剪策略（`random_crop` / `face_crop_aug_range`）。
