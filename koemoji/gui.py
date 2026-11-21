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
from .models import (
    DEFAULT_MODEL,
    DEFAULT_SUMMARY_MODEL,
    MODEL_CATALOG,
    SUMMARY_CATALOG,
    get_spec,
)
from .summary import SummaryCancelled, SummaryOptions, SummaryResult, summarize_result

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
        self.summary: SummaryResult | None = None

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

        summary_row = ttk.LabelFrame(outer, text="議事録（任意・CPUのみ）", padding=6)
        summary_row.pack(fill="x", pady=6)
        self.summary_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            summary_row,
            text="文字起こし後に議事録を作る",
            variable=self.summary_var,
        ).pack(side="left")
        self.summary_ai_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            summary_row,
            text="ローカルAIで文章を整える（初回のみモデルDL）",
            variable=self.summary_ai_var,
        ).pack(side="left", padx=(12, 4))
        self.summary_model_keys = list(SUMMARY_CATALOG.keys())
        summary_labels = [SUMMARY_CATALOG[key].label for key in self.summary_model_keys]
        self.summary_model_var = tk.StringVar(
            value=SUMMARY_CATALOG[DEFAULT_SUMMARY_MODEL].label
        )
        ttk.Combobox(
            summary_row,
            textvariable=self.summary_model_var,
            values=summary_labels,
            state="readonly",
            width=26,
        ).pack(side="left", padx=4)

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

        self.notebook = ttk.Notebook(outer)
        self.notebook.pack(fill="both", expand=True, pady=4)
        transcript_tab = ttk.Frame(self.notebook)
        summary_tab = ttk.Frame(self.notebook)
        self.notebook.add(transcript_tab, text="文字起こし")
        self.notebook.add(summary_tab, text="議事録")

        self.text = scrolledtext.ScrolledText(transcript_tab, wrap="word", height=18)
        self.text.pack(fill="both", expand=True)
        self.summary_text = scrolledtext.ScrolledText(summary_tab, wrap="word", height=18)
        self.summary_text.pack(fill="both", expand=True)

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

    def _selected_summary_model(self) -> str:
        label = self.summary_model_var.get()
        for key in self.summary_model_keys:
            if SUMMARY_CATALOG[key].label == label:
                return key
        return DEFAULT_SUMMARY_MODEL

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

        want_summary = self.summary_var.get() or self.summary_ai_var.get()
        summary_options = SummaryOptions(
            use_llm="always" if self.summary_ai_var.get() else "never",
            model=self._selected_summary_model(),
        )

        self.cancel_event.clear()
        self.result = None
        self.summary = None
        self.text.delete("1.0", "end")
        self.summary_text.delete("1.0", "end")
        self.progress["value"] = 0
        self._set_busy(True)
        self.status_var.set("準備中...")

        def _progress(frac: float, msg: str) -> None:
            self.messages.put(("progress", frac, msg))

        def _run() -> None:
            try:
                result = transcribe_file(
                    path,
                    options,
                    on_progress=_progress,
                    cancel_event=self.cancel_event,
                )
            except TranscribeCancelled:
                self.messages.put(("cancelled", None))
                return
            except Exception as exc:  # noqa: BLE001
                self.messages.put(("error", exc))
                return

            summary: SummaryResult | None = None
            if want_summary:
                self.messages.put(("progress", 1.0, "議事録を整理しています..."))
                try:
                    summary = summarize_result(
                        result,
                        summary_options,
                        on_progress=_progress,
                        cancel_event=self.cancel_event,
                    )
                except SummaryCancelled:
                    self.messages.put(("notice", "議事録の作成をキャンセルしました"))
                except Exception as exc:  # noqa: BLE001
                    self.messages.put(("notice", f"議事録の作成に失敗しました: {exc}"))
            self.messages.put(("done", result, summary))

        self.worker = threading.Thread(target=_run, daemon=True)
        self.worker.start()

    def _cancel(self) -> None:
        self.cancel_event.set()
        self.status_var.set("キャンセル中...（現在の区間の処理が終わり次第停止します）")

    def _active_text(self) -> str:
        widget = self.summary_text if self.notebook.index(self.notebook.select()) == 1 else self.text
        return widget.get("1.0", "end-1c")

    def _copy(self) -> None:
        content = self._active_text()
        if not content.strip():
            return
        self.root.clipboard_clear()
        self.root.clipboard_append(content)
        self.status_var.set("クリップボードにコピーしました")

    def _save(self) -> None:
        content = self._active_text()
        if not content.strip():
            return
        on_summary = self.notebook.index(self.notebook.select()) == 1
        initial = "summary" if on_summary else "transcript"
        if self.result is not None:
            initial = self.result.source.stem + (".summary" if on_summary else "")
        if on_summary:
            default_ext = ".md"
            filetypes = [("Markdown (*.md)", "*.md"), ("テキスト (*.txt)", "*.txt")]
        else:
            default_ext = ".txt"
            filetypes = [
                ("テキスト (*.txt)", "*.txt"),
                ("会話ログ (*.log)", "*.log"),
                ("SRT 字幕 (*.srt)", "*.srt"),
                ("WebVTT 字幕 (*.vtt)", "*.vtt"),
            ]
        path = filedialog.asksaveasfilename(
            title="保存先を選択",
            defaultextension=default_ext,
            initialfile=initial,
            filetypes=filetypes,
        )
        if not path:
            return
        out = Path(path)
        suffix = out.suffix.lower().lstrip(".")
        if not on_summary and suffix in ("srt", "vtt", "log") and self.result is not None:
            out.write_text(FORMATS[suffix](self.result.segments), encoding="utf-8")
        elif on_summary:
            out.write_text(content.rstrip() + "\n", encoding="utf-8")
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
                    self._on_done(*payload)
                elif kind == "notice":
                    self.status_var.set(str(payload[0]))
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

    def _on_done(self, result: TranscribeResult, summary: SummaryResult | None = None) -> None:
        self.result = result
        self.summary = summary
        self._set_busy(False)
        self.text.delete("1.0", "end")
        self.text.insert("1.0", result.text)
        if summary is not None:
            self.summary_text.delete("1.0", "end")
            self.summary_text.insert("1.0", summary.to_markdown())
            self.notebook.select(1)
        else:
            self.summary_text.delete("1.0", "end")
            self.summary_text.insert(
                "1.0",
                "議事録は作成されていません。\n"
                "「議事録を作る」にチェックを入れて文字起こしを実行してください。",
            )
        self.progress["value"] = 1000
        speed = result.duration / result.elapsed if result.elapsed > 0 else 0.0
        message = (
            f"完了: 音声 {result.duration:.0f}秒 を {result.elapsed:.1f}秒 で処理"
            f"（約{speed:.1f}倍速・CPU・{len(result.segments)}セグメント"
        )
        if result.num_speakers > 0:
            message += f"・話者 {result.num_speakers}人"
        message += "）"
        if summary is not None:
            message += f" / 議事録: {summary.method}"
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
