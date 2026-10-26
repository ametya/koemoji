"""tkinter 製 GUI（追加依存なしで動く）。"""

from __future__ import annotations

import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, ttk

from . import __version__
from .core import (
    FORMATS,
    TranscribeCancelled,
    TranscribeOptions,
    TranscribeResult,
    transcribe_file,
)
from .glossary import load_glossary
from .models import DEFAULT_MODEL, MODEL_CATALOG, get_spec

APP_TITLE = f"声文字 (koemoji) v{__version__}"

AUDIO_FILETYPES = [
    (
        "音声・動画ファイル（Teams/Zoom録画など）",
        "*.wav *.mp3 *.m4a *.aac *.flac *.ogg *.oga *.opus *.wma *.amr *.aiff *.mp4 *.mkv *.mov *.avi *.webm *.wmv *.mpg *.mpeg *.ts *.mts *.m2ts *.3gp *.flv",
    ),
    ("すべてのファイル", "*.*"),
]


class App:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.messages: "queue.Queue[tuple]" = queue.Queue()
        self.cancel_event = threading.Event()
        self.worker: threading.Thread | None = None
        self.result: TranscribeResult | None = None

        root.title(APP_TITLE)
        root.minsize(760, 560)

        self._build_widgets()
        root.after(100, self._poll)

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_widgets(self) -> None:
        outer = ttk.Frame(self.root, padding=8)
        outer.pack(fill="both", expand=True)

        file_row = ttk.Frame(outer)
        file_row.pack(fill="x", pady=4)
        ttk.Label(file_row, text="入力ファイル:").pack(side="left")
        self.file_var = tk.StringVar()
        ttk.Entry(file_row, textvariable=self.file_var).pack(
            side="left", fill="x", expand=True, padx=6
        )
        ttk.Button(file_row, text="参照...", command=self._choose_file).pack(side="left")

        opts_row = ttk.Frame(outer)
        opts_row.pack(fill="x", pady=4)
        ttk.Label(opts_row, text="モデル:").pack(side="left")
        self.model_keys = list(MODEL_CATALOG.keys())
        labels = [MODEL_CATALOG[key].label for key in self.model_keys]
        self.model_var = tk.StringVar(value=MODEL_CATALOG[DEFAULT_MODEL].label)
        combo = ttk.Combobox(
            opts_row, textvariable=self.model_var, values=labels, state="readonly", width=40
        )
        combo.pack(side="left", padx=6)
        self.vad_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            opts_row, text="無音区間をスキップ (VAD・高速化)", variable=self.vad_var
        ).pack(side="left", padx=8)

        diarize_row = ttk.Frame(outer)
        diarize_row.pack(fill="x", pady=4)
        self.diarize_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            diarize_row,
            text="話者分離（会議・インタビューなど複数人の会話ログを作る）",
            variable=self.diarize_var,
        ).pack(side="left")
        ttk.Label(diarize_row, text="話者数:").pack(side="left", padx=(16, 2))
        self.speakers_var = tk.StringVar(value="自動")
        ttk.Combobox(
            diarize_row,
            textvariable=self.speakers_var,
            values=["自動", "2", "3", "4", "5", "6", "7", "8"],
            state="readonly",
            width=5,
        ).pack(side="left")

        prompt_row = ttk.Frame(outer)
        prompt_row.pack(fill="x", pady=4)
        ttk.Label(prompt_row, text="ヒント文（任意・固有名詞など）:").pack(side="left")
        self.prompt_var = tk.StringVar()
        ttk.Entry(prompt_row, textvariable=self.prompt_var).pack(
            side="left", fill="x", expand=True, padx=6
        )

        glossary_row = ttk.Frame(outer)
        glossary_row.pack(fill="x", pady=4)
        ttk.Label(glossary_row, text="用語集（任意・固有名詞/誤認識の置換）:").pack(side="left")
        self.glossary_var = tk.StringVar()
        ttk.Entry(glossary_row, textvariable=self.glossary_var).pack(
            side="left", fill="x", expand=True, padx=6
        )
        ttk.Button(glossary_row, text="参照...", command=self._choose_glossary).pack(side="left")

        self.glossary_prompt_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            outer, text="用語集をヒント文としても使う（狭帯域の音声ではオフ推奨）",
            variable=self.glossary_prompt_var,
        ).pack(anchor="w", padx=2)

        btn_row = ttk.Frame(outer)
        btn_row.pack(fill="x", pady=8)
        self.start_btn = ttk.Button(btn_row, text="文字起こし開始", command=self._start)
        self.start_btn.pack(side="left")
        self.cancel_btn = ttk.Button(
            btn_row, text="キャンセル", command=self._cancel, state="disabled"
        )
        self.cancel_btn.pack(side="left", padx=6)

        self.save_btn = ttk.Button(btn_row, text="保存...", command=self._save, state="disabled")
        self.save_btn.pack(side="right")
        self.copy_btn = ttk.Button(btn_row, text="コピー", command=self._copy, state="disabled")
        self.copy_btn.pack(side="right", padx=6)

        prog_row = ttk.Frame(outer)
        prog_row.pack(fill="x", pady=4)
        self.progress = ttk.Progressbar(prog_row, mode="determinate", maximum=1000)
        self.progress.pack(fill="x", expand=True)

        self.status_var = tk.StringVar(
            value="ファイルを選択して「文字起こし開始」を押してください（初回はモデルのダウンロードがあります）"
        )
        ttk.Label(outer, textvariable=self.status_var, anchor="w").pack(fill="x", pady=2)

        self.text = scrolledtext.ScrolledText(outer, wrap="word", height=18)
        self.text.pack(fill="both", expand=True, pady=4)

    # ------------------------------------------------------------------
    # 操作
    # ------------------------------------------------------------------

    def _choose_file(self) -> None:
        path = filedialog.askopenfilename(title="文字起こしするファイルを選択", filetypes=AUDIO_FILETYPES)
        if path:
            self.file_var.set(path)

    def _choose_glossary(self) -> None:
        path = filedialog.askopenfilename(
            title="用語集ファイルを選択",
            filetypes=[
                ("用語集 (*.txt *.tsv)", "*.txt *.tsv"),
                ("すべてのファイル", "*.*"),
            ],
        )
        if path:
            self.glossary_var.set(path)

    def _selected_model(self) -> str:
        label = self.model_var.get()
        for key in self.model_keys:
            if MODEL_CATALOG[key].label == label:
                return key
        return DEFAULT_MODEL

    def _start(self) -> None:
        if self.worker is not None and self.worker.is_alive():
            return
        raw = self.file_var.get().strip().strip('"')
        if not raw:
            messagebox.showwarning(APP_TITLE, "入力ファイルを選択してください。")
            return
        path = Path(raw)
        if not path.is_file():
            messagebox.showerror(APP_TITLE, f"ファイルが見つかりません:\n{path}")
            return

        speakers_raw = self.speakers_var.get()
        num_speakers = int(speakers_raw) if speakers_raw.isdigit() else -1
        model_key = self._selected_model()

        if self.diarize_var.get() and not get_spec(model_key).supports_word_timestamps:
            messagebox.showerror(
                APP_TITLE,
                f"モデル「{model_key}」は話者分離に対応していません。\n"
                "話者分離を使う場合は turbo / small / medium / large-v3 を選んでください。",
            )
            return

        glossary = None
        glossary_raw = self.glossary_var.get().strip().strip('"')
        if glossary_raw:
            try:
                glossary = load_glossary(Path(glossary_raw))
            except Exception as exc:
                messagebox.showerror(APP_TITLE, f"用語集を読み込めません:\n{exc}")
                return

        options = TranscribeOptions(
            model=model_key,
            language="ja",
            vad=self.vad_var.get(),
            initial_prompt=self.prompt_var.get().strip() or None,
            diarize=self.diarize_var.get(),
            num_speakers=num_speakers,
            glossary=glossary,
            glossary_prompt=self.glossary_prompt_var.get(),
        )

        self.cancel_event.clear()
        self.result = None
        self.text.delete("1.0", "end")
        self.progress["value"] = 0
        self._set_busy(True)
        self.status_var.set("準備中...")

        def _run() -> None:
            try:
                result = transcribe_file(
                    path,
                    options,
                    on_progress=lambda frac, msg: self.messages.put(("progress", frac, msg)),
                    cancel_event=self.cancel_event,
                )
                self.messages.put(("done", result))
            except TranscribeCancelled:
                self.messages.put(("cancelled", None))
            except Exception as exc:  # noqa: BLE001
                self.messages.put(("error", exc))

        self.worker = threading.Thread(target=_run, daemon=True)
        self.worker.start()

    def _cancel(self) -> None:
        self.cancel_event.set()
        self.status_var.set("キャンセル中...（現在の区間の処理が終わり次第停止します）")

    def _copy(self) -> None:
        content = self.text.get("1.0", "end-1c")
        if not content.strip():
            return
        self.root.clipboard_clear()
        self.root.clipboard_append(content)
        self.status_var.set("クリップボードにコピーしました")

    def _save(self) -> None:
        content = self.text.get("1.0", "end-1c")
        if not content.strip():
            return
        initial = "transcript"
        if self.result is not None:
            initial = self.result.source.stem
        path = filedialog.asksaveasfilename(
            title="保存先を選択",
            defaultextension=".txt",
            initialfile=initial,
            filetypes=[
                ("テキスト (*.txt)", "*.txt"),
                ("会話ログ (*.log)", "*.log"),
                ("SRT 字幕 (*.srt)", "*.srt"),
                ("WebVTT 字幕 (*.vtt)", "*.vtt"),
            ],
        )
        if not path:
            return
        out = Path(path)
        suffix = out.suffix.lower().lstrip(".")
        if suffix in ("srt", "vtt", "log") and self.result is not None:
            out.write_text(FORMATS[suffix](self.result.segments), encoding="utf-8")
        else:
            out.write_text(content.rstrip() + "\n", encoding="utf-8-sig")
        self.status_var.set(f"保存しました: {path}")

    # ------------------------------------------------------------------
    # ワーカーとのやり取り
    # ------------------------------------------------------------------

    def _set_busy(self, busy: bool) -> None:
        self.start_btn.config(state="disabled" if busy else "normal")
        self.cancel_btn.config(state="normal" if busy else "disabled")
        self.copy_btn.config(state="disabled" if busy else "normal")
        self.save_btn.config(state="disabled" if busy else "normal")

    def _poll(self) -> None:
        try:
            while True:
                kind, *payload = self.messages.get_nowait()
                if kind == "progress":
                    frac, message = payload
                    self.progress["value"] = int(frac * 1000)
                    self.status_var.set(message)
                elif kind == "done":
                    self._on_done(payload[0])
                elif kind == "cancelled":
                    self._set_busy(False)
                    self.status_var.set("キャンセルしました")
                elif kind == "error":
                    self._set_busy(False)
                    self.status_var.set("エラーが発生しました")
                    messagebox.showerror(APP_TITLE, str(payload[0]))
        except queue.Empty:
            pass
        self.root.after(100, self._poll)

    def _on_done(self, result: TranscribeResult) -> None:
        self.result = result
        self._set_busy(False)
        self.text.delete("1.0", "end")
        self.text.insert("1.0", result.text)
        self.progress["value"] = 1000
        speed = result.duration / result.elapsed if result.elapsed > 0 else 0.0
        message = (
            f"完了: 音声 {result.duration:.0f}秒 を {result.elapsed:.1f}秒 で処理"
            f"（約{speed:.1f}倍速・CPU・{len(result.segments)}セグメント"
        )
        if result.num_speakers > 0:
            message += f"・話者 {result.num_speakers}人"
        message += "）"
        self.status_var.set(message)


def run_gui() -> int:
    try:
        import ctypes

        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass

    root = tk.Tk()
    App(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(run_gui())
