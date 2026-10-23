"""Offline adapters for single-outcome, full-context tabular regression."""
from __future__ import annotations

import importlib
import importlib.metadata
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from .config import file_sha256, resolve


def _version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed-as-distribution"


def _resource(root: Path, value: str | Path, *, directory: bool = False) -> Path:
    path = resolve(root, value).resolve()
    if not (path.is_dir() if directory else path.is_file()):
        raise FileNotFoundError(f"Required local {'directory' if directory else 'file'}: {path}")
    return path


def _offline_environment(cache_dir: Path, offline: bool) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(cache_dir / "huggingface")
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["TABPFN_DISABLE_TELEMETRY"] = "1"
    if offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"


def _git_revision(repo: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def _arrays(X_train, y_train, X_query) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train = np.asarray(X_train, dtype=np.float32)
    query = np.asarray(X_query, dtype=np.float32)
    target = np.asarray(y_train, dtype=np.float64)
    if train.ndim != 2 or query.ndim != 2 or target.ndim != 1:
        raise ValueError("Expected two feature matrices and one one-dimensional target")
    if not len(train) or not len(query) or not train.shape[1]:
        raise ValueError("Empty context, query, or feature axis")
    if train.shape[1] != query.shape[1] or len(target) != len(train):
        raise ValueError("Feature axes and context/target lengths must align")
    if not np.isfinite(target).all() or np.isinf(train).any() or np.isinf(query).any():
        raise ValueError("Targets must be finite; feature infinities are forbidden (NaNs are supported)")
    return train, target, query


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


class Predictor:
    """One outcome per call. Constructor and calls do not download weights."""

    def __init__(self, name, config, *, device, cache_dir, offline, root):
        self.name, self.config = name, dict(config)
        self.device, self.root = str(device), Path(root).resolve()
        self.cache_dir = Path(cache_dir).resolve()
        self.seed = int(config.get("seed", 42))
        self.n_estimators = int(config.get("n_estimators", 8))
        if self.n_estimators < 1:
            raise ValueError("n_estimators must be positive")
        _offline_environment(self.cache_dir, offline)
        self.audit = {"predictor": name, "device": self.device, "offline": bool(offline),
                      "seed": self.seed, "n_estimators": self.n_estimators,
                      "numpy_version": _version("numpy"), "python": sys.version,
                      "extra_feature_reduction": None, "context_subsampling": False}

    def _start(self, X_train, y_train, X_query, run_id):
        train, target, query = _arrays(X_train, y_train, X_query)
        self._started = time.perf_counter()
        self.audit.update(run_id=str(run_id), context_rows=len(train), query_rows=len(query),
                          input_features=train.shape[1], torch_version=_version("torch"))
        return train, target, query

    def _finish(self, prediction, n_query):
        prediction = np.asarray(prediction, dtype=np.float64).reshape(-1)
        if len(prediction) != n_query or not np.isfinite(prediction).all():
            raise ValueError("Vendor output must be one finite prediction per query row")
        self.audit["elapsed_seconds"] = time.perf_counter() - self._started
        return prediction, _jsonable(self.audit)

    def predict(self, X_train, y_train, X_query, *, run_id):
        raise NotImplementedError


class TabPFNPredictor(Predictor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.checkpoint = _resource(self.root, self.config["checkpoint"])
        if "v3.5" not in self.checkpoint.name:
            raise ValueError("TabPFN benchmark requires the explicit v3.5 checkpoint")
        self.audit.update(checkpoint=str(self.checkpoint), checkpoint_sha256=file_sha256(self.checkpoint),
                          package_version=_version("tabpfn"), prediction_statistic="mean")

    def predict(self, X_train, y_train, X_query, *, run_id):
        train, target, query = self._start(X_train, y_train, X_query, run_id)
        from tabpfn import TabPFNRegressor

        regressor = TabPFNRegressor(
            model_path=str(self.checkpoint), device=self.device, n_estimators=self.n_estimators,
            auto_scale_n_estimators=False, random_state=self.seed,
            fit_mode="fit_preprocessors", ignore_pretraining_limits=False,
            n_preprocessing_jobs=int(self.config.get("n_preprocessing_jobs", 1)),
            show_progress_bar=False,
        )
        regressor.fit(train, target)
        prediction = regressor.predict(query, output_type="mean")
        self.audit["resolved_inference_config"] = _jsonable(getattr(regressor, "inference_config_", None))
        return self._finish(prediction, len(query))


class LimiXPredictorAdapter(Predictor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.repo = _resource(self.root, self.config["repo_dir"], directory=True)
        self.checkpoint = _resource(self.root, self.config["checkpoint"])
        self.config_path = _resource(self.root, self.config["inference_config"])
        self.inference_config = json.loads(self.config_path.read_text(encoding="utf-8"))
        pipelines = self.inference_config.get("pipelines")
        if not isinstance(pipelines, list) or len(pipelines) != self.n_estimators:
            raise ValueError("LimiX V2 config must have exactly n_estimators pipelines")
        if self.inference_config.get("adaptive_svd", {}).get("enabled", False):
            raise ValueError("Disable adaptive_svd: hardware-dependent feature fallback is forbidden")
        if any(item.get("retrieval_config", {}).get("use_retrieval", False) for item in pipelines):
            raise ValueError("The full-context benchmark requires no retrieval")
        self.audit.update(checkpoint=str(self.checkpoint), checkpoint_sha256=file_sha256(self.checkpoint),
                          vendor_commit=_git_revision(self.repo), inference_config=str(self.config_path),
                          inference_config_sha256=file_sha256(self.config_path),
                          native_preprocessing=self.inference_config,
                          inactive_subsample_ratio_fields=True, package_version=_version("limix"))
        self._predictor = None

    def predict(self, X_train, y_train, X_query, *, run_id):
        train, target, query = self._start(X_train, y_train, X_query, run_id)
        import torch

        if self._predictor is None:
            sys.path.insert(0, str(self.repo))
            module = importlib.import_module("inference.predictor")
            module_path = Path(module.__file__).resolve()
            if not module_path.is_relative_to(self.repo):
                raise ImportError(f"Wrong LimiX module loaded from {module_path}")
            self._predictor = module.LimiXPredictor(
                device=torch.device(self.device), model_path=str(self.checkpoint),
                inference_config=self.inference_config, seed=self.seed,
                mix_precision=bool(self.config.get("mixed_precision", True)),
                use_data_cache=False,
            )
        prediction = self._predictor.predict(train, target, query, task_type="Regression",
                                             unique_dataset_name=str(run_id))
        for key in ("cuda_pipeline_fallback_audit", "svd_runtime_retry_audit",
                    "polynomial_interaction_runtime_retry_audit"):
            records = getattr(self._predictor, key, [])
            self.audit[key] = _jsonable(records)
            if records:
                raise RuntimeError(f"LimiX changed/skipped a native pipeline: {key}={records!r}")
        self.audit["query_batch_diagnostics"] = _jsonable(
            getattr(self._predictor, "last_query_batch_diagnostics", None))
        self.audit["svd_adaptation_audit"] = _jsonable(
            getattr(self._predictor, "svd_adaptation_audit", None))
        return self._finish(prediction, len(query))


class KumoPredictor(Predictor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.config.get("size", "large") != "large":
            raise ValueError("This benchmark pins Kumo Tabular Large")
        if self.config.get("point_prediction", "median") != "median":
            raise ValueError("Kumo point prediction is preregistered as q500 (median)")
        checkpoint_value = self.config.get("checkpoint")
        if checkpoint_value is None:
            checkpoint_value = Path(self.config["checkpoint_dir"]) / "large" / "regressor.pt"
        self.checkpoint = _resource(self.root, checkpoint_value)
        self.audit.update(checkpoint=str(self.checkpoint), checkpoint_sha256=file_sha256(self.checkpoint),
                          package_version=_version("structured-data-models"), size="large",
                          prediction_statistic="q500", native_column_selection_per_estimator=500,
                          native_column_selection="Latin column shuffle then first 500",
                          no_multi_target=True)
        self._model = None

    def predict(self, X_train, y_train, X_query, *, run_id):
        train, target, query = self._start(X_train, y_train, X_query, run_id)
        import torch
        import sdm

        torch.manual_seed(self.seed)
        if self.device.startswith("cuda"):
            torch.cuda.manual_seed_all(self.seed)
        if self._model is None:
            self._model = sdm.models.KumoTabular(task="regression", size="large", pretrained=False,
                                                device="cpu")
            state = torch.load(self.checkpoint, map_location="cpu", weights_only=True)
            self._model.models[sdm.Task.regression].load_state_dict(state, strict=True, assign=True)
            self._model.to(self.device).eval()
        names = [f"x{index}" for index in range(train.shape[1])]

        def table(array, columns):
            return sdm.TableTensor(columns={sdm.Stype.numerical: columns},
                                   numerical=torch.as_tensor(array, dtype=torch.float32, device=self.device))

        with torch.inference_mode(), torch.amp.autocast(
            device_type=torch.device(self.device).type, dtype=torch.float16,
            enabled=self.device.startswith("cuda"),
        ):
            result = self._model(x_context=table(train, names), y_context=table(target[:, None], ["target"]),
                                 x_query=table(query, names), num_estimators=self.n_estimators)
        columns = list(result.columns[sdm.Stype.numerical])
        if len(columns) != 999 or "q500" not in columns:
            raise RuntimeError("Expected native 999-quantile Kumo regression output")
        median = result["q500"].numerical.detach().float().cpu().numpy()
        self.audit["effective_feature_cap"] = min(train.shape[1], 500)
        return self._finish(median, len(query))


def build_predictor(name: str, config: dict, *, device: str, cache_dir: str | Path,
                    offline: bool = True, root: Path = Path(".")) -> Predictor:
    """Build a selected local-checkpoint adapter, never substitute weights."""
    classes = {"tabpfn": TabPFNPredictor, "limix": LimiXPredictorAdapter, "kumo": KumoPredictor}
    if name not in classes:
        raise ValueError(f"Unsupported pretrained predictor {name!r}; choose {list(classes)}")
    return classes[name](name, config, device=device, cache_dir=cache_dir, offline=offline, root=root)
