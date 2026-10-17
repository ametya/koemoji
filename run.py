"""PyInstaller 用のエントリーポイント。"""

import sys

from koemoji.cli import main

if __name__ == "__main__":
    sys.exit(main())
