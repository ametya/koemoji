# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller 用 spec。

使い方（リポジトリのルートで実行）:
    pyinstaller packaging/koemoji.spec --noconfirm --clean --distpath dist --workpath build

ビルド成果物:
    dist/koemoji/koemoji.exe       … CLI（コンソールあり）
    dist/koemoji/koemoji-gui.exe   … GUI（ダブルクリック起動用・コンソールなし）
"""

import os

from PyInstaller.utils.hooks import collect_all

SPECPATH = os.path.dirname(os.path.abspath(SPEC))
PROJECT_ROOT = os.path.abspath(os.path.join(SPECPATH, os.pardir))

datas = []
binaries = []
hiddenimports = [
    "koemoji",
    "koemoji.cli",
    "koemoji.core",
    "koemoji.gui",
    "koemoji.models",
]

for package in (
    "ctranslate2",
    "faster_whisper",
    "huggingface_hub",
    "tokenizers",
    "onnxruntime",
    "av",
    "tqdm",
    "sherpa_onnx",
):
    try:
        pkg_datas, pkg_binaries, pkg_hidden = collect_all(package)
    except Exception:
        continue
    datas += pkg_datas
    binaries += pkg_binaries
    hiddenimports += pkg_hidden

a = Analysis(
    [os.path.join(PROJECT_ROOT, "run.py")],
    pathex=[PROJECT_ROOT],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=[
        "torch",
        "tensorflow",
        "matplotlib",
        "pandas",
        "scipy",
        "IPython",
        "notebook",
        "pytest",
    ],
    noarchive=False,
)

# PyInstaller は PATH 上の古い MSVC ランタイム（msvcp140/vcruntime140）を
# 同梱することがあり、新しい ctranslate2 がアクセス違反でクラッシュする。
# System32 の新しいランタイムに置き換える（無ければ同梱しない）。
_VC_RUNTIME_NAMES = {
    "msvcp140.dll",
    "msvcp140_1.dll",
    "msvcp140_2.dll",
    "vcruntime140.dll",
    "vcruntime140_1.dll",
}
_system32 = os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32")
_replacements = {}
for _name, _src, _typ in a.binaries:
    _base = os.path.basename(_name).lower()
    if _base in _VC_RUNTIME_NAMES:
        _sys_src = os.path.join(_system32, os.path.basename(_name))
        _replacements[_name] = _sys_src if os.path.isfile(_sys_src) else None

if _replacements:
    _new_binaries = []
    for _entry in a.binaries:
        if _entry[0] in _replacements:
            _sys_src = _replacements[_entry[0]]
            if _sys_src is not None:
                _new_binaries.append((_entry[0], _sys_src, _entry[2]))
        else:
            _new_binaries.append(_entry)
    a.binaries = _new_binaries

pyz = PYZ(a.pure)

exe_cli = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="koemoji",
    debug=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
)

exe_gui = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="koemoji-gui",
    debug=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
)

coll = COLLECT(
    exe_cli,
    exe_gui,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="koemoji",
)
