"""モデルカタログとダウンロード処理（初回のみオンライン、以後は完全オフライン動作）。"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

APP_NAME = "koemoji"

ProgressCallback = Callable[[int, int], None]


@dataclass(frozen=True)
class ModelSpec:
    name: str
    repo_id: str
    download_size: str
    speed: str
    accuracy: str
    note: str
    supports_word_timestamps: bool = True

    @property
    def label(self) -> str:
        return f"{self.name}（{self.note}・{self.download_size}）"


MODEL_CATALOG: dict[str, ModelSpec] = {
    "turbo": ModelSpec(
        "turbo",
        "vokasug/faster-whisper-large-v3-turbo-int8",
        "約820MB",
        "標準",
        "高",
        "バランス・おすすめ",
    ),
    "kotoba": ModelSpec(
        "kotoba",
        "kotoba-tech/kotoba-whisper-v2.0-faster",
        "約1.5GB",
        "標準",
        "最高（日本語）",
        "日本語特化・話者分離不可",
        supports_word_timestamps=False,
    ),
    "small": ModelSpec("small", "Systran/faster-whisper-small", "約480MB", "高速", "良好", "軽量"),
    "medium": ModelSpec("medium", "Systran/faster-whisper-medium", "約1.5GB", "低速", "高", "高精度"),
    "large-v3": ModelSpec("large-v3", "Systran/faster-whisper-large-v3", "約3.1GB", "かなり低速", "最高", "最高精度"),
    "tiny": ModelSpec("tiny", "Systran/faster-whisper-tiny", "約75MB", "最速", "低", "動作確認用"),
    "base": ModelSpec("base", "Systran/faster-whisper-base", "約145MB", "高速", "やや低", "軽量"),
}

DEFAULT_MODEL = "turbo"


def default_model_dir() -> Path:
    """OS ごとの既定モデル保存先を返す。"""
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
    else:
        base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / APP_NAME / "models"


def get_spec(name: str) -> ModelSpec:
    try:
        return MODEL_CATALOG[name]
    except KeyError as exc:
        raise KeyError(
            f"未知のモデル: {name}（利用可能: {', '.join(MODEL_CATALOG)}）"
        ) from exc


def _snapshot(repo_id: str, cache_dir: Path, local_only: bool) -> str:
    from huggingface_hub import snapshot_download

    return snapshot_download(repo_id, cache_dir=str(cache_dir), local_files_only=local_only)


def is_downloaded(name: str, model_dir: Optional[Path] = None) -> bool:
    """モデルがローカルに揃っているかを返す（通信は行わない）。"""
    spec = get_spec(name)
    cache_dir = Path(model_dir) if model_dir else default_model_dir()
    try:
        _snapshot(spec.repo_id, cache_dir, local_only=True)
        return True
    except Exception:
        return False


def ensure_model(
    name: str,
    model_dir: Optional[Path] = None,
    progress_cb: Optional[ProgressCallback] = None,
) -> str:
    """モデルをローカルに用意してスナップショットパスを返す。

    未取得の場合だけダウンロードする。取得済みなら通信せず、そのまま
    ローカルパスを返すので、2 回目以降は完全オフラインで動作する。
    """
    from huggingface_hub import HfApi, hf_hub_download
    from tqdm.auto import tqdm

    spec = get_spec(name)
    cache_dir = Path(model_dir) if model_dir else default_model_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)

    # 取得済みなら即返す（オフラインでも動く）
    try:
        return _snapshot(spec.repo_id, cache_dir, local_only=True)
    except Exception:
        pass

    try:
        info = HfApi().model_info(spec.repo_id, files_metadata=True)
    except Exception as exc:
        raise RuntimeError(
            f"モデル「{name}」のダウンロード情報を取得できませんでした。"
            "初回のみインターネット接続が必要です。接続を確認してください。"
        ) from exc

    files = [s for s in info.siblings if not s.rfilename.startswith(".")]
    total = max(sum(s.size or 0 for s in files), 1)
    lock = threading.Lock()
    done_before = 0

    for sibling in files:
        fname = sibling.rfilename
        fsize = sibling.size or 0

        class _ProgressTqdm(tqdm):
            def update(self, n: int = 1):
                result = super().update(n)
                if progress_cb is not None:
                    with lock:
                        done = done_before + min(int(self.n), fsize or int(self.n))
                        progress_cb(min(done, total), total)
                return result

        hf_hub_download(spec.repo_id, fname, cache_dir=str(cache_dir), tqdm_class=_ProgressTqdm)
        done_before += fsize

    if progress_cb is not None:
        progress_cb(total, total)
    return _snapshot(spec.repo_id, cache_dir, local_only=True)


# ---------------------------------------------------------------------------
# 話者分離モデル（sherpa-onnx の声紋抽出 / CPU）
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DiarizationModelSpec:
    key: str
    url: str
    filename: str
    size: str
    note: str


DIARIZATION_CATALOG: dict[str, DiarizationModelSpec] = {
    "titanet": DiarizationModelSpec(
        "titanet",
        "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/nemo_en_titanet_small.onnx",
        "titanet.onnx",
        "約40MB",
        "声紋抽出（NVIDIA TitaNet Small・標準）",
    ),
    "3dspeaker": DiarizationModelSpec(
        "3dspeaker",
        "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx",
        "3dspeaker.onnx",
        "約40MB",
        "声紋抽出（3D-Speaker ERes2Net・アジア音声向け）",
    ),
}

DEFAULT_EMBEDDING = "titanet"


def diarization_model_dir(model_dir: Optional[Path] = None) -> Path:
    base = Path(model_dir) if model_dir else default_model_dir()
    return base / "diarization"


def diarization_model_path(key: str, model_dir: Optional[Path] = None) -> Path:
    spec = DIARIZATION_CATALOG[key]
    return diarization_model_dir(model_dir) / spec.filename


def _download_file(url: str, dest: Path, progress_cb: Optional[ProgressCallback] = None) -> None:
    import urllib.request

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": APP_NAME})
    try:
        response = urllib.request.urlopen(request, timeout=60)
    except Exception as exc:
        raise RuntimeError(
            f"モデルをダウンロードできませんでした（初回のみオンラインが必要です）: {exc}"
        ) from exc

    with response:
        total = int(response.headers.get("Content-Length") or 0)
        done = 0
        with open(tmp, "wb") as handle:
            while True:
                chunk = response.read(262144)
                if not chunk:
                    break
                handle.write(chunk)
                done += len(chunk)
                if progress_cb is not None:
                    progress_cb(done, total or done)
    tmp.replace(dest)


def ensure_diarization_model(
    key: str = DEFAULT_EMBEDDING,
    model_dir: Optional[Path] = None,
    progress_cb: Optional[ProgressCallback] = None,
) -> Path:
    if key not in DIARIZATION_CATALOG:
        raise KeyError(f"未知の声紋モデル: {key}（利用可能: {', '.join(DIARIZATION_CATALOG)}）")
    spec = DIARIZATION_CATALOG[key]
    dest = diarization_model_path(key, model_dir)
    if dest.is_file():
        return dest
    _download_file(spec.url, dest, progress_cb)
    return dest
