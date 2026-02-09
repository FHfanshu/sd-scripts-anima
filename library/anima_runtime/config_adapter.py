from __future__ import annotations

OPTIMIZER_ALIASES = {
    "radam_schedulefree": "RAdamScheduleFree",
    "adamw_schedulefree": "AdamWScheduleFree",
    "sgd_schedulefree": "SGDScheduleFree",
}


def coerce_optimizer_type(value: str) -> str:
    normalized = str(value or "").strip()
    if not normalized:
        return normalized
    return OPTIMIZER_ALIASES.get(normalized.lower(), normalized)


def normalize_optimizer_aliases(args):
    if getattr(args, "optimizer_type", None):
        args.optimizer_type = coerce_optimizer_type(args.optimizer_type)
    return args

