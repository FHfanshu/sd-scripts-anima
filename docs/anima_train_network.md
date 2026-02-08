# Anima LoRA / LoKr Training (`sd-scripts-anima`)

`anima_train_network.py` 是 `sd-scripts-anima` 内的原生 `NetworkTrainer` 训练入口，不依赖上级目录仓库。

## 1. 快速启动

LoRA:

```bash
accelerate launch anima_train_network.py \
  --network_module networks.lora_anima \
  --anima_transformer /path/to/anima_transformer.safetensors \
  --vae /path/to/anima_vae.safetensors \
  --qwen /path/to/qwen_model_dir \
  --t5_tokenizer_dir /path/to/t5_tokenizer_dir \
  --dataset_config /path/to/dataset.toml \
  --output_dir /path/to/output \
  --output_name anima_lora \
  --max_train_steps 1000
```

LoKr:

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

## 2. 必填参数

- `--anima_transformer`（或 `--pretrained_model_name_or_path`）
- `--vae`
- `--qwen`
- `--t5_tokenizer_dir`（必填，训练全模式强制）
- `--network_module`：`networks.lora_anima` 或 `networks.lokr_anima`

## 3. 文本条件训练语义

当前链路固定为：

1. `Qwen` 编码得到 hidden states
2. `AnimaTransformer.preprocess_text_embeds(qwen_hidden, t5_ids)`
3. 将融合后的条件送入 Anima Transformer 主前向

即 `T5 tokenizer -> text_ids` 是必经链路。

## 4. 配置文件适配

可选传入 `--config <toml>` 使用根仓风格 Anima TOML 映射为 `sd-scripts` 参数：

- 模型路径（transformer / vae / qwen / t5）
- 数据与训练参数（batch、lr、scheduler、warmup 等）
- LoRA / LoKr 参数与输出参数

CLI 显式参数优先级高于 `--config` 映射值。

## 5. ScheduleFree 约束

当 `optimizer_type=RAdamScheduleFree`（或别名 `radam_schedulefree`）时：

- `lr_scheduler` 必须是 `constant`
- `lr_warmup_steps` 必须是 `0`
- `lr_warmup_ratio` 必须是 `0`

否则会在参数校验阶段直接报错退出。

## 6. 常见问题

- `T5 tokenizer directory is required`
  - 必须提供 `--t5_tokenizer_dir`，且目录内包含 `config.json`、`spiece.model`、`tokenizer.json`。

- `Unsupported network module`
  - 仅支持 `networks.lora_anima` / `networks.lokr_anima`。

- `optimizer=RAdamScheduleFree requires lr_scheduler=constant`
  - 调整 scheduler/warmup 为受支持组合后再启动训练。
