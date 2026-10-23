# 声文字 (koemoji)

**CPUだけで動く、オフライン日本語文字起こしツール（Windows GUI + CLI）**

GIGAZINE で紹介された [Embedding Gemma Transcribe](https://gigazine.net/news/20261014-embetting-gemma-transcribe/)（EmbeddingGemma 2 を音声認識に転用し、15分の音声を約3秒で文字起こし）の「**オフライン・ローカル完結**」という思想を参考に、日本語向けに実用重視で作り直したツールです。

**Teams / Zoom の会議録画も文字起こしできます。話者分離（誰が話したか）にも対応。**

## できること

- 音声ファイル（wav / mp3 / m4a / flac など）と**動画ファイル（Teams・Zoom録画の mp4 / mkv / mov など）**の文字起こし
- **話者分離**: 複数人の会話を「誰が話したか」付きの会話ログに
- 字幕ファイル（SRT / WebVTT）の出力
- 完全ローカル処理（音声は外部に送信されません・CPUのみ・GPU不要）

会話ログの出力例（`--diarize` 使用時）:

```
[00:00:00] 話者1: こんにちは。今日は会議を始めます。
[00:00:05] 話者2: おはようございます。まず議事録を確認しましょう。
[00:00:12] 話者1: 新しい機能の進捗はどうですか。
[00:00:17] 話者2: 来週リリースできる見込みです。テストも順調です。
```

## 特徴

- **CPU 前提**（GPU 不要・int8 量子化でメモリも省める）
- **高精度モデルを標準搭載**（Whisper large-v3-turbo。日本語特化の kotoba-whisper も選択可）
- **完全ローカル処理**。音声がサーバーに送られることはありません
- **初回だけモデルをダウンロード**（文字起こし `turbo` 約820MB / 話者分離 約40MB）。以後は**オフラインで動作**
- **日本語に最適化**（言語固定 `ja`、VAD で無音区間を自動スキップ）
- **幅広い入力形式**: wav / mp3 / m4a / aac / flac / ogg / opus / wma / amr / aiff、動画は mp4 / mkv / mov / avi / webm / wmv / mpg / ts / mts / 3gp など
- **話者分離**（会議・インタビュー・鼎談など、複数人の会話を会話ログ化）
- **GUI（tkinter）と CLI の両方**。GUI は exe をダブルクリックするだけ
- **txt / 会話ログ / SRT / VTT 字幕**出力
- **固有名詞ヒント機能**（`--initial-prompt`）で人名・社名の認識率を改善
- **用語集**（`--glossary`）で固有名詞をヒントに追加し、よくある誤認識を自動修正
- PyInstaller で **配布用 zip を自動作成**、GitHub Actions で**リリース自動作成**

## Embedding Gemma Transcribe との違い

| | Embedding Gemma Transcribe | 声文字 (koemoji) |
|---|---|---|
| エンジン | EmbeddingGemma 2 + 英単語辞書（22.9万語） | faster-whisper (Whisper / CTranslate2 int8) + sherpa-onnx（話者分離） |
| 日本語 | 非対応（英語の空耳になる） | **対応（標準）** |
| 話者分離 | なし | **あり（会話ログ形式）** |
| 動作環境 | Apple Silicon Mac のみ | **Windows / macOS / Linux（CPUのみでOK）** |
| 処理 | 完全ローカル | 完全ローカル（初回のみモデルDL、以後オフライン可） |

> 埋め込みモデルを音声認識に転用する手法は日本語ではまだ研究段階のため、本ツールは Whisper 系の多言語モデルを CPU 向けに量子化（int8）して動かす構成にしています。話者分離も同様に CPU 向けの軽量な声紋モデル（sherpa-onnx）を使用しています。

## インストール

### 方法1: 配布版 exe（Python 不要・かんたん）

1. [Releases](https://github.com/ametya/koemoji/releases) から `koemoji-windows-x64.zip` をダウンロード
2. 好きな場所に展開（例: `C:\koemoji`）
3. `koemoji-gui.exe` をダブルクリック → GUI が起動
4. CLI で使いたい場合は `koemoji.exe` を PowerShell / コマンドプロンプトから実行

> 初回起動時にモデル（既定: 文字起こし用 turbo・約820MB）をダウンロードします。2回目以降はネットなしで動きます。
> 署名なし exe のため、Windows SmartScreen の警告（「詳細情報」→「実行」）が出ることがあります。

### 方法2: pip でインストール（Python 3.10 以上）

```powershell
pip install "git+https://github.com/ametya/koemoji.git"
# またはクローンして
git clone https://github.com/ametya/koemoji.git
cd koemoji
pip install .
```

## 使い方

### GUI

```powershell
koemoji            # 引数なしで GUI が起動
koemoji-gui.exe    # 配布版の場合
```

1. 音声・動画ファイルを選択（Teams録画の mp4 もそのまま選べます）
2. モデルを選択（迷ったら既定の `turbo` のまま。軽さ重視なら `small`）
3. 会議なら「**話者分離**」にチェック（話者数は通常「自動」。うまく分かれない場合だけ人数を指定）
4. 必要なら「ヒント文」や「**用語集**」を指定（固有名詞の認識改善・誤認識の自動修正）
5. 「文字起こし開始」→ 完了後にコピー / 保存（txt・会話ログ・SRT・VTT）

### CLI

```powershell
# 基本（txt 出力、日本語固定）
koemoji transcribe 会議録音.m4a

# Teams の会議録画を話者分離付きで（会話ログ + 字幕）
koemoji transcribe 会議録画.mp4 -m medium --diarize --formats log,srt

# 参加者が3人と分かっている場合
koemoji transcribe インタビュー.mp3 --diarize --speakers 3

# 固有名詞ヒント付き・高精度モデル
koemoji transcribe 講演.mp4 -m medium --initial-prompt "株式会社ほげ、開発部のふが"

# 用語集で固有名詞を補正（ヒント文と誤認識の置換）
koemoji transcribe 会議録画.mp4 --diarize --glossary 用語集.txt

# モデル一覧と事前ダウンロード（配布先でオフライン運用したい場合など）
koemoji models
koemoji models --download turbo --download-diarization
```

| オプション | 説明 | 既定 |
|---|---|---|
| `-m, --model` | 使用モデル | `turbo` |
| `-l, --language` | 言語コード（`auto` で自動判定） | `ja` |
| `-o, --output-dir` | 出力先ディレクトリ | 入力と同じ場所 |
| `--formats` | `txt,log,srt,vtt` をカンマ区切り（`txt` は話者分離時、自動で会話ログ形式） | `txt` |
| `--diarize` | 話者分離を行う（会話ログ形式に） | オフ |
| `--speakers` | 話者数が決まっている場合に指定 | `0`（自動） |
| `--diarize-embedding` | 声紋モデル（`titanet` / `3dspeaker`） | `titanet` |
| `--diarize-threshold` | 声紋の距離しきい値を手動指定（通常は自動。指定時は小さいほど話者が増える） | 自動 |
| `--no-vad` | 無音スキップを無効化 | VAD 有効 |
| `--beam-size` | ビームサイズ（大きいほど丁寧・遅い） | `5` |
| `--compute-type` | 計算精度（`int8` / `int8_float32` / `float32`）。精度重視なら `int8_float32` か `float32` | `int8` |
| `--initial-prompt` | 固有名詞などのヒント文 | なし |
| `--glossary` | 用語集ファイル（ヒント追加＋誤認識の置換） | なし |
| `--cpu-threads` | 使用スレッド数 | 自動（全コア） |
| `-q, --quiet` | 進捗表示を抑制 | - |

## モデル選択の目安（CPU・int8）

### 文字起こしモデル

| モデル | DLサイズ | 速度目安 | 精度 | 用途 |
|---|---|---|---|---|
| **turbo** | 約820MB | 標準（2〜4倍速程度） | 高 | **おすすめ（既定）** |
| kotoba | 約1.5GB | 標準（2〜3倍速程度） | 最高（日本語特化） | 日本語の精度重視（**話者分離は不可**） |
| small | 約480MB | 高速（5〜10倍速程度） | 良好 | 速度重視 |
| medium | 約1.5GB | 低速（1〜2倍速程度） | 高 | 精度重視（話者分離可） |
| large-v3 | 約3.1GB | かなり低速（1倍速未満） | 最高 | 最高精度重視（話者分離可） |
| tiny / base | 約75MB / 145MB | 最速 / 高速 | 低 | 動作確認・軽量運用 |

※ 速度は8コア前後のCPUでの目安。話者分離を有効にすると処理時間が少し増えます（上記 turbo の実測は約2.3倍速）。
※ `kotoba`（kotoba-whisper v2.0）は日本語特化で文字精度が高い一方、単語タイムスタンプに未対応のため `--diarize` と併用できません（指定すると分かりやすいエラーを表示します）。

### 精度をもう一段上げたいとき

1. `--compute-type int8_float32`（または `float32`）を付ける（速度は少し落ちます）
2. `--beam-size 8` のようにビーム幅を広げる
3. `--initial-prompt "株式会社ほげ、山田さん、プロジェクトふが"` で固有名詞をヒントとして与える
4. `--glossary 用語集.txt` で固有名詞を継続的に与え、決まった誤認識を置換する
5. 日本語特化の `-m kotoba` を使う（話者分離を使わない場合）
6. 録音品質を上げる（マイクに近づく・ノイズを減らす）のが最も効果的です

### 用語集で固有名詞をそろえる

用語集は1行1項目のテキストファイルです（UTF-8 / Shift_JIS、BOM 可）。`#` で始まる行はコメントになります。区切りがない行は「正しい用語」として Whisper のヒント文に追加され、`=>`・`->`・`→`・タブの区切りがある行は「誤認識=>正しい表記」の置換として使われます。置換の「正しい表記」もヒント文に追加されます。

```
# ヒント文に追加する用語
株式会社サンプル
プロジェクトあおぞら

# よくある誤認識を自動修正
サンプル社=>株式会社サンプル
あおぞらプロジェクト=>プロジェクトあおぞら
```

置換は文字列の完全一致で、長いものから一度だけ適用されます。意図しない箇所まで置換され得るため、初回は出力を確認してください。用語集は GUI の「用語集」欄からも指定できます。

### 話者分離モデル（初回DL）

| モデル | DLサイズ | 特徴 |
|---|---|---|
| **titanet** | 約40MB | NVIDIA TitaNet Small（標準） |
| 3dspeaker | 約40MB | 3D-Speaker ERes2Net（アジア音声向け） |

> 話者分離は音声の声紋から自動推定するため、100%正確ではありません。短い相づち、かぶせ話、声質の近い話者は苦手です。通常は 3 秒ごとの声紋から話者数を自動推定します（長い会議でも、途中で話者が混ざりにくい方式です）。**うまく分かれない場合は `--speakers N` で人数を指定**してください。`--diarize-threshold` は距離しきい値を手動で固定したい上級者向けで、小さいほど人数が増えます。

## 対応ファイル形式

音声と動画の区別なく、ffmpeg（PyAV）が読める形式をそのまま入力できます。

- **音声**: wav / mp3 / m4a / aac / flac / ogg / opus / wma / amr / aiff
- **動画**: mp4 / mkv / mov / avi / webm / wmv / mpg / mpeg / ts / mts / m2ts / 3gp / flv
- **Teams / Zoom / Google Meet のローカル録画**（mp4 等）はそのまま指定OK

## 完全オフラインで使う

- 一度モデルをダウンロードすれば、以後の文字起こし・話者分離は通信しません
- 別のPCへは `%LOCALAPPDATA%\koemoji\models` フォルダをコピーすれば、そのままオフライン運用できます
- `koemoji models --download turbo --download-diarization` で事前ダウンロードしておくと安心です

## ビルド（配布用 zip の作成）

```powershell
powershell -ExecutionPolicy Bypass -File scripts\build_win.ps1
# → dist\koemoji-windows-x64.zip ができます
```

手動でビルドする場合:

```powershell
py -3.12 -m venv .venv-build
.\.venv-build\Scripts\python.exe -m pip install ".[build]"
.\.venv-build\Scripts\python.exe -m PyInstaller packaging\koemoji.spec --noconfirm --clean --distpath dist --workpath build
```

### GitHub Releases で自動配布

タグを push するだけで、GitHub Actions が Windows 用 zip をビルドして Release に添付します。

```powershell
git tag v0.1.0
git push origin v0.1.0
```

## テスト

```powershell
pip install ".[dev]"
pytest
```

## ライセンス

[MIT License](LICENSE)

## 謝辞・参考

- [GoogleのEmbeddingGemma 2を音声認識モデルに変換した「Embedding Gemma Transcribe」（GIGAZINE）](https://gigazine.net/news/20261014-embetting-gemma-transcribe/)
- [faster-whisper (SYSTRAN)](https://github.com/SYSTRAN/faster-whisper) / [CTranslate2](https://github.com/OpenNMT/CTranslate2) / [OpenAI Whisper](https://github.com/openai/whisper)
- 話者分離: [sherpa-onnx (k2-fsa)](https://github.com/k2-fsa/sherpa-onnx)（NVIDIA TitaNet / 3D-Speaker 声紋モデル）
- 文字起こしモデル配布: [Systran/faster-whisper-* (Hugging Face)](https://huggingface.co/Systran)
- [Whisper large-v3-turbo (OpenAI)](https://huggingface.co/openai/whisper-large-v3-turbo) / CTranslate2 変換: [deepdml/faster-whisper-large-v3-turbo-ct2](https://huggingface.co/deepdml/faster-whisper-large-v3-turbo-ct2)
- 日本語特化モデル: [Kotoba-Whisper (Kotoba Technologies)](https://huggingface.co/kotoba-tech/kotoba-whisper-v2.0-faster)
