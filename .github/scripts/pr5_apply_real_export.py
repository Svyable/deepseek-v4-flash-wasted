#!/usr/bin/env python3
"""Apply the narrow classifier corrections proven by the real 0731 headers.

This helper is intentionally temporary PR machinery. It lets the GitHub runner
validate the exact patch against all 48 real headers before committing that
patch back to the branch. It is idempotent: once the source is fixed, rerunning
it makes no changes.
"""

from pathlib import Path


PATH = Path("tools/inventory.py")

REPLACEMENTS = (
    (
        '    (re.compile(r"\\.mlp\\.gate\\.|e_score_correction_bias|\\brouter\\b"),\n'
        '     "router"),',
        '    (re.compile(r"(?:^|\\.)ffn\\.gate\\.|\\.mlp\\.gate\\.|"\n'
        '                r"e_score_correction_bias|\\brouter\\b"),\n'
        '     "router"),',
    ),
    (
        '    (re.compile(r"lm_head|output\\.weight$"), "lm_head"),',
        '    (re.compile(r"lm_head|output\\.weight$|(?:^|\\.)head\\.weight$"),\n'
        '     "lm_head"),',
    ),
    (
        'LAYER_RE = re.compile(r"\\.layers\\.(\\d+)\\.")',
        'LAYER_RE = re.compile(r"(?:^|\\.)layers\\.(\\d+)\\.")',
    ),
)


def main():
    text = PATH.read_text(encoding="utf-8")
    changed = False
    for old, new in REPLACEMENTS:
        if old in text:
            text = text.replace(old, new, 1)
            changed = True
        elif new not in text:
            raise SystemExit(f"expected inventory source fragment not found:\n{old}")

    PATH.write_text(text, encoding="utf-8")

    # Pin the exact real-export names that caused the first Gate A run to fail.
    import importlib.util

    spec = importlib.util.spec_from_file_location("inventory_pr5", PATH)
    inventory = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(inventory)

    cases = {
        "head.weight": ("lm_head", None, None),
        "layers.0.ffn.gate.weight": ("router", 0, None),
        "layers.3.ffn.gate.bias": ("router", 3, None),
        "layers.0.ffn.gate.tid2eid": ("router_hash", 0, None),
        "layers.0.ffn.experts.0.w1.weight": ("routed_expert", 0, 0),
        "layers.42.ffn.experts.255.w2.scale": ("routed_expert", 42, 255),
    }
    for name, expected in cases.items():
        row = inventory.classify(name, 43)
        got = (row["subsystem"], row["layer"], row["expert"])
        if got != expected:
            raise SystemExit(f"{name}: classified as {got}, expected {expected}")

    print("real 0731 export classifier patch: " + ("applied" if changed else "already present"))


if __name__ == "__main__":
    main()
