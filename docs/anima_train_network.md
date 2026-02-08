# Anima LoRA / LoKr Training (`sd-scripts-anima`)

> `anima_train_network.py` is a bridge entry that forwards arguments to root `anima_train.py`.

## 1. Quick Start

```bash
accelerate launch anima_train_network.py \
  --config ../config/save/anima_lora_config.example.toml \
  --network_module networks.lora_anima
```

LoKr:

```bash
accelerate launch anima_train_network.py \
  --config ../config/save/anima_lora_config.example.toml \
  --network_module networks.lokr_anima
```

## 2. Required Inputs

- `--config`: root trainer TOML config path
- model paths in TOML:
  - `[model].transformer_path`
  - `[model].vae_path` (optional but recommended)
  - `[model].text_encoder_path`

## 3. CLI Behavior

- `--network_module networks.lora_anima` => forwards `--network-type lora`
- `--network_module networks.lokr_anima` => forwards `--network-type lokr`
- all unknown arguments are passed through to root `anima_train.py`

Useful:

```bash
python anima_train_network.py --config <cfg> --network_module networks.lora_anima --print-cmd
```

## 4. Common Errors

- `Could not locate anima_train.py`  
  Set `--anima-train-py` or env `ANIMA_TRAIN_PY`.

- `optimizer=radam_schedulefree requires lr_scheduler=constant`  
  Set:
  - `[optimizer].type = "radam_schedulefree"`
  - `[training].lr_scheduler = "constant"`
  - `[training].lr_warmup_steps = 0`
  - `[training].lr_warmup_ratio = 0.0`

- TensorBoard port conflict  
  root trainer auto-increments from `tensorboard_port` (default `6006`).

