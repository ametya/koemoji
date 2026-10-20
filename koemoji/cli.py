"""コマンドラインインターフェース。"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional

from . import __version__
from .core import FORMATS, TranscribeCancelled, TranscribeOptions, transcribe_file
from .models import (
    DEFAULT_EMBEDDING,
    DEFAULT_MODEL,
    DIARIZATION_CATALOG,
    MODEL_CATALOG,
    default_model_dir,
    diarization_model_path,
    ensure_diarization_model,
    ensure_model,
    is_downloaded,
)


def _fix_stdio() -> None:
    """GUI 版 exe（コンソールなし）でも落ちないようにする。

    エンコーディングはあえて変更しない。Windows コンソールでは UTF-8、
    パイプ時は日本語ロケール（cp932 等）になり、それぞれ正しく表示される。
    変換不能な文字だけ置換して例外を防ぐ。
    """
    for name in ("stdout", "stderr"):
        stream = getattr(sys, name, None)
        if stream is None:
            setattr(sys, name, open(os.devnull, "w", encoding="utf-8"))
            continue
        try:
            stream.reconfigure(errors="replace")
        except Exception:
            pass


def _progress_printer(quiet: bool):
    def _cb(frac: float, message: str) -> None:
        if quiet:
            return
        frac = max(0.0, min(frac, 1.0))
        bar_len = 24
        filled = int(bar_len * frac)
        bar = "#" * filled + "-" * (bar_len - filled)
        sys.stdout.write(f"\r  [{bar}] {frac * 100:5.1f}%  {message}   ")
        sys.stdout.flush()

    return _cb


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="koemoji",
        description="CPUだけで動くオフライン日本語文字起こしツール（faster-whisper / 初回のみモデルDL）",
    )
    parser.add_argument("-V", "--version", action="version", version=f"koemoji {__version__}")
    sub = parser.add_subparsers(dest="command")

    p_tr = sub.add_parser(
        "transcribe",
        aliases=["tr"],
        help="音声・動画ファイルを文字起こしする",
    )
    p_tr.add_argument("inputs", nargs="+", help="入力ファイル（wav/mp3/m4a/mp4/mkv など）")
    p_tr.add_argument("-o", "--output-dir", default=None, help="出力先ディレクトリ（既定: 入力ファイルと同じ場所）")
    p_tr.add_argument(
        "-m", "--model", default=DEFAULT_MODEL, choices=sorted(MODEL_CATALOG),
        help=f"使用モデル（既定: {DEFAULT_MODEL}）",
    )
    p_tr.add_argument("-l", "--language", default="ja", help="言語コード。auto で自動判定（既定: ja）")
    p_tr.add_argument("--no-vad", action="store_true", help="無音区間スキップ（VAD）を無効化")
    p_tr.add_argument("--beam-size", type=int, default=5, help="ビームサイズ（既定: 5）")
    p_tr.add_argument(
        "--compute-type", choices=["int8", "int8_float32", "float32"], default="int8",
        help="計算精度。精度重視なら float32（遅い）、バランスなら int8_float32（既定: int8）",
    )
    p_tr.add_argument(
        "--initial-prompt", default=None,
        help="認識を誘導するヒント文（例: 「株式会社ほげ、プロジェクトふが」）",
    )
    p_tr.add_argument("--formats", default="txt", help="出力形式。txt,srt,vtt,log をカンマ区切り（既定: txt）")
    p_tr.add_argument("--model-dir", default=None, help="モデル保存先ディレクトリ")
    p_tr.add_argument("--cpu-threads", type=int, default=None, help="使用CPUスレッド数（既定: 自動）")
    p_tr.add_argument(
        "--diarize", action="store_true",
        help="話者分離を行い「話者1: ...」形式の会話ログにする（会議・インタビューなど）",
    )
    p_tr.add_argument(
        "--speakers", type=int, default=0,
        help="話者数が決まっている場合に指定（0=自動・既定）",
    )
    p_tr.add_argument(
        "--diarize-embedding", choices=["titanet", "3dspeaker"], default=DEFAULT_EMBEDDING,
        help=f"声紋モデル（既定: {DEFAULT_EMBEDDING}）",
    )
    p_tr.add_argument(
        "--diarize-threshold", type=float, default=0.5,
        help="話者の分かれやすさ（0.3〜0.8 目安。小さいほど人数が減り、大きいほど増える）",
    )
    p_tr.add_argument("-q", "--quiet", action="store_true", help="進捗表示を抑制")

    sub.add_parser("gui", help="GUI を起動する")

    p_m = sub.add_parser("models", help="モデルの一覧表示・ダウンロード")
    p_m.add_argument(
        "-d", "--download", metavar="NAME", action="append", default=None,
        help="指定の音声認識モデルを事前ダウンロード（複数指定可）",
    )
    p_m.add_argument(
        "--download-diarization", action="store_true",
        help="話者分離用の声紋モデルを事前ダウンロード",
    )
    p_m.add_argument(
        "--embedding", choices=["titanet", "3dspeaker"], default=DEFAULT_EMBEDDING,
        help=f"--download-diarization で使う声紋モデル（既定: {DEFAULT_EMBEDDING}）",
    )
    p_m.add_argument("--model-dir", default=None, help="モデル保存先ディレクトリ")
    return parser


def cmd_transcribe(args: argparse.Namespace) -> int:
    formats = [f.strip().lower() for f in args.formats.split(",") if f.strip()]
    unknown = [f for f in formats if f not in FORMATS]
    if unknown:
        print(f"エラー: 未対応の出力形式です: {', '.join(unknown)}（対応: {', '.join(FORMATS)}）")
        return 2

    model_dir = Path(args.model_dir) if args.model_dir else None
    print(f"モデル: {args.model} / 言語: {args.language} / 出力形式: {', '.join(formats)}")
    exit_code = 0

    for raw in args.inputs:
        path = Path(raw)
        if not path.is_file():
            print(f"エラー: ファイルが見つかりません: {path}")
            exit_code = 1
            continue

        print(f"== {path.name}")
        options = TranscribeOptions(
            model=args.model,
            language=args.language,
            vad=not args.no_vad,
            beam_size=args.beam_size,
            compute_type=args.compute_type,
            initial_prompt=args.initial_prompt,
            model_dir=model_dir,
            cpu_threads=args.cpu_threads,
            diarize=args.diarize,
            num_speakers=args.speakers if args.speakers > 0 else -1,
            diarize_threshold=args.diarize_threshold,
            diarize_embedding=args.diarize_embedding,
        )
        try:
            result = transcribe_file(path, options, on_progress=_progress_printer(args.quiet))
        except TranscribeCancelled:
            print("\n  キャンセルしました")
            exit_code = 130
            break
        except KeyboardInterrupt:
            print("\n  キャンセルしました")
            return 130
        except Exception as exc:
            print(f"\n  エラー: {exc}")
            exit_code = 1
            continue

        if not args.quiet:
            print()

        outdir = Path(args.output_dir) if args.output_dir else path.parent
        outdir.mkdir(parents=True, exist_ok=True)
        for fmt in formats:
            out_path = outdir / f"{path.stem}.{fmt}"
            out_path.write_text(FORMATS[fmt](result.segments), encoding="utf-8")
            print(f"  書き出し: {out_path}")

        speed = result.duration / result.elapsed if result.elapsed > 0 else 0.0
        detail = f"  完了: {len(result.segments)}セグメント"
        if args.diarize:
            detail += f" / 話者 {result.num_speakers}人を検出"
        detail += (
            f" / 音声 {result.duration:.0f}秒 を {result.elapsed:.1f}秒で処理（約{speed:.1f}倍速・CPU）"
        )
        print(detail)

    return exit_code


def _progress_bytes(printer) -> "callable":
    def _cb(done: int, total: int) -> None:
        frac = done / total if total else 0.0
        printer(frac, f"{done / 1048576:.0f}MB / {total / 1048576:.0f}MB")

    return _cb


def cmd_models(args: argparse.Namespace) -> int:
    model_dir = Path(args.model_dir) if args.model_dir else default_model_dir()

    if args.download:
        for name in args.download:
            print(f"モデル「{name}」を準備しています...")
            try:
                ensure_model(name, model_dir, progress_cb=_progress_bytes(_progress_printer(False)))
            except Exception as exc:
                print(f"\n  エラー: {exc}")
                return 1
            print(f"\n  準備完了: {name}")

    if args.download_diarization:
        print(f"話者分離モデル（声紋: {args.embedding}）を準備しています...")
        try:
            ensure_diarization_model(
                args.embedding, model_dir, progress_cb=_progress_bytes(_progress_printer(False))
            )
        except Exception as exc:
            print(f"\n  エラー: {exc}")
            return 1
        print("\n  準備完了（話者分離モデル）")

    if args.download or args.download_diarization:
        return 0

    print(f"モデル保存先: {model_dir}")
    print()
    print("  【音声認識モデル】")
    print("  名前        状態    ダウンロード  速度        精度    備考")
    print("  " + "-" * 64)
    for name, spec in MODEL_CATALOG.items():
        state = "取得済" if is_downloaded(name, model_dir) else "未取得"
        print(
            f"  {name:<10}  {state}  {spec.download_size:<12}  {spec.speed:<10}  {spec.accuracy:<6}  {spec.note}"
        )
    print()
    print("  【話者分離モデル（--diarize 使用時に初回DL）】")
    print("  名前         状態    サイズ    備考")
    print("  " + "-" * 68)
    for key, spec in DIARIZATION_CATALOG.items():
        state = "取得済" if diarization_model_path(key, model_dir).is_file() else "未取得"
        print(f"  {key:<12}  {state}  {spec.size:<8}  {spec.note}")
    print()
    print("  例: koemoji models --download small --download-diarization")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    _fix_stdio()
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command in (None, "gui"):
        try:
            from .gui import run_gui

            return run_gui()
        except Exception as exc:
            print(f"GUI を起動できませんでした: {exc}", file=sys.stderr)
            parser.print_help()
            return 1

    if args.command in ("transcribe", "tr"):
        try:
            return cmd_transcribe(args)
        except KeyboardInterrupt:
            print("\nキャンセルしました")
            return 130

    if args.command == "models":
        return cmd_models(args)

    parser.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
