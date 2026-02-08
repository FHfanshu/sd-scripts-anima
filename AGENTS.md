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

