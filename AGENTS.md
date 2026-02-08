# AGENTS 工作记录

## 2026-02-08

- 阅读并遵循 `.ai/claude.prompt.md` 与 `.ai/context/01-overview.md`。
- 新增 Anima 训练桥接入口：`anima_train_network.py`（将参数转发到根仓 `anima_train.py`）。
- 新增 Anima 适配器网络模块：
  - `networks/lora_anima.py`
  - `networks/lokr_anima.py`
  - `networks/_anima_adapter_common.py`（内置 LoRA/LoKr 注入实现，无外部 LyCORIS 依赖）。
- 新增 Anima 模型/策略模块：
  - `library/anima_models.py`
  - `library/strategy_anima.py`
- 新增文档：`docs/anima_train_network.md`，并在 `README.md`、`README-ja.md` 增加入口链接。
- 新增测试：
  - `tests/test_anima_network_modules.py`
  - `tests/test_anima_train_network.py`
- 执行测试：`4 passed`（针对新增 Anima 模块与桥接脚本）。
- 新增并启用本地后端/运行时模块，推进 `sd-scripts-anima` 独立化：
  - `library/anima_backend/modelling/*`（vendor Anima transformer / cosmos_predict2 / wan vae）
  - `library/anima_runtime/model_loading.py`
  - `library/anima_runtime/flow_training.py`
  - `library/anima_runtime/config_adapter.py`
- 将 `anima_train_network.py` 从桥接入口改为原生 `NetworkTrainer` 实现：
  - 移除桥接参数（`--anima-train-py` / `--print-cmd` / `--dry-run`）
  - 增加本地加载与参数校验（含 `RAdamScheduleFree` 约束）
  - 训练语义对齐为 `Qwen -> preprocess_text_embeds(qwen_hidden, t5_ids) -> Anima Transformer`
- 强制 T5 链路：`t5_tokenizer_dir` 必填，训练 batch 中 `t5_ids` 缺失直接报错。
- 切断外部依赖路径：
  - `library/anima_models.py` 仅调用本地 `library/anima_runtime/*`
  - 移除父目录探测、`sys.path` 注入与 `models.*` 虚拟包依赖
- 更新文档：
  - 重写 `docs/anima_train_network.md` 为独立原生训练说明
  - 更新 `README.md` / `README-ja.md` 标注 Anima 入口不依赖上级仓库
- 更新/新增测试：
  - 更新 `tests/test_anima_train_network.py`
  - 新增 `tests/test_anima_independence.py`
  - 新增 `tests/test_anima_process_batch.py`
