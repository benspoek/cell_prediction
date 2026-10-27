#!/usr/bin/env python3
"""Supply fresh RWTH GPU quota to existing budget gates without editing config.

Keeping this launcher outside src/ preserves scientific computation fingerprints.
CPP_QUOTA_SOURCE=config explicitly selects the existing manual fallback.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile


class QuotaError(ValueError):
    """The available GPU-hour allowance could not be established reliably."""


def _number(value):
    if isinstance(value, bool):
        raise QuotaError("GPU-hour allowance must be a finite nonnegative number")
    if isinstance(value, str):
        if not re.fullmatch(r"(?:[0-9]+|[0-9]{1,3}(?:,[0-9]{3})+)(?:\.[0-9]+)?", value.strip()):
            raise QuotaError(f"Malformed GPU-hour allowance: {value!r}")
        value = value.replace(",", "")
    try:
        number = float(value)
    except (ValueError, TypeError) as error:
        raise QuotaError("GPU-hour allowance must be a finite nonnegative number") from error
    if not math.isfinite(number) or number < 0:
        raise QuotaError("GPU-hour allowance must be a finite nonnegative number")
    return number


def parse_quota(output: str, account: str) -> float:
    """Read the absolute ML consumable GPU hours, excluding CPU hours and percent."""
    text = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", output)
    accounts = re.findall(r"^\s*Account\s*:\s*(\S+)\s*$", text, flags=re.I | re.M)
    if accounts != [account]:
        raise QuotaError(f"Quota response account does not uniquely match {account!r}: {accounts}")
    in_ml = False
    ml_blocks = 0
    values = []
    for line in text.splitlines():
        partition = re.fullmatch(r"\s*Partitiontype\s*:\s*([A-Za-z]+)\s*", line, flags=re.I)
        heading = re.fullmatch(r"\s*=+\s*(HPC|ML)\s+partitions\s*\([^)]*\)\s*=+\s*", line, flags=re.I)
        if partition or heading:
            in_ml = (partition or heading).group(1).upper() == "ML"
            ml_blocks += int(in_ml)
            if heading and in_ml and not re.search(r"gpu[- ]?h", line, flags=re.I):
                raise QuotaError("ML quota header does not declare GPU-hour units")
            continue
        row = re.fullmatch(r"\s*Consumable\s+gpu[- ]?(?:h|hours?)\s*:\s*(.*?)\s*", line, flags=re.I)
        if in_ml and row:
            values.append(_number(row.group(1)))
    if ml_blocks != 1 or len(values) != 1:
        raise QuotaError("Expected one ML section with one absolute 'Consumable gpu-h' value")
    return values[0]


def query_quota(account: str) -> dict:
    if not isinstance(account, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", account):
        raise QuotaError("A valid cluster.account is required for automatic quota lookup")
    command = ["r_wlm_usage", "-q", "-p", account]
    environment = dict(os.environ, LC_ALL="C", LANG="C")
    try:
        response = subprocess.run(command, capture_output=True, text=True, check=True,
                                  timeout=30, env=environment)
    except (OSError, subprocess.SubprocessError) as error:
        raise QuotaError(f"Could not query r_wlm_usage for {account}: {error}. "
                         "No configured or cached allowance was substituted.") from error
    hours = parse_quota(response.stdout, account)
    return {"source": "r_wlm_usage", "account": account, "remaining_gpu_hours": hours,
            "checked_at": datetime.now(timezone.utc).isoformat(), "command": command,
            "output_sha256": hashlib.sha256(response.stdout.encode()).hexdigest()}


def _invoke_cli(arguments: list[str]) -> int:
    from cell_property_prediction.cli import main
    return main(arguments)


def run_budget(root: Path, config_path: Path, command: str, arguments: list[str], source="cluster") -> int:
    import yaml
    from cell_property_prediction.config import compute_config, load_config

    root = Path(root).resolve()
    config_path = Path(config_path)
    if not config_path.is_absolute():
        config_path = root / config_path
    original = load_config(config_path)
    if source == "cluster":
        observation = query_quota(original["cluster"].get("account"))
    elif source == "config":
        observation = {"source": "config", "account": original["cluster"].get("account"),
                       "remaining_gpu_hours": _number(original["cluster"].get("remaining_gpu_hours")),
                       "checked_at": datetime.now(timezone.utc).isoformat()}
    else:
        raise QuotaError(f"Unknown quota source: {source}")
    if command == "quota":
        print(json.dumps({"remaining_gpu_hours": observation["remaining_gpu_hours"],
                          "quota": observation}, indent=2))
        return 0
    if command not in {"budget", "hvg-encoder-budget"}:
        raise QuotaError(f"Unsupported budget command: {command}")
    runtime = copy.deepcopy(original)
    runtime["cluster"]["remaining_gpu_hours"] = observation["remaining_gpu_hours"]
    if compute_config(runtime) != compute_config(original):
        raise QuotaError("Quota injection changed scientific configuration")
    output = io.StringIO()
    failure = None
    status = 0
    with tempfile.TemporaryDirectory(prefix="cellpred-quota-") as directory:
        path = Path(directory) / "runtime.yaml"
        path.write_text(yaml.safe_dump(runtime, sort_keys=False), encoding="utf-8")
        try:
            with contextlib.redirect_stdout(output):
                status = int(_invoke_cli(["--root", str(root), "--config", str(path), command, *arguments]) or 0)
        except (OSError, RuntimeError, ValueError) as error:
            failure = str(error)
            status = 2
        except SystemExit as error:
            status = error.code if isinstance(error.code, int) else 2
            failure = str(error) if status else None
    try:
        result = json.loads(output.getvalue()) if output.getvalue().strip() else {}
        if not isinstance(result, dict):
            raise ValueError("budget output must be a JSON object")
    except ValueError as error:
        raise QuotaError(f"Could not read budget result: {error}") from error
    result["quota"] = observation
    if failure:
        if failure.startswith("Set the actual remaining GPU hours"):
            failure = ("Measured GPU-hour estimate exceeds the available allowance; review the budget result "
                       "and pilot completeness before submission")
        result["error"] = failure
        print(failure, file=sys.stderr)
    print(json.dumps(result, indent=2, default=str))
    return status


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--config", type=Path, default=os.environ.get("CPP_CONFIG", "configs/benchmark.yaml"))
    parser.add_argument("--source", choices=("cluster", "config"), default=os.environ.get("CPP_QUOTA_SOURCE", "cluster"))
    parser.add_argument("command", choices=("quota", "budget", "hvg-encoder-budget"))
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    try:
        return run_budget(args.root, args.config, args.command, args.arguments, source=args.source)
    except (QuotaError, OSError, ValueError) as error:
        print(f"GPU quota check failed: {error} "
              "To explicitly use a manually verified YAML allowance, set CPP_QUOTA_SOURCE=config.", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
