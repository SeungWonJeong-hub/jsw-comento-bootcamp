# -*- mode: python ; coding: utf-8 -*-
"""실행 파일 묶음 설정.

주의할 점
---------
streamlit 은 자기 버전을 importlib.metadata 로 읽어서 metadata 를 같이
넣어야 하고, 화면에 쓰는 정적 파일이 패키지 안에 있어 collect_all 이
필요합니다. rasterio 와 pyproj 는 좌표 변환에 쓰는 proj.db 같은 자료
파일이 없으면 조용히 실패합니다.
"""
from PyInstaller.utils.hooks import collect_all, copy_metadata

datas, binaries, hiddenimports = [], [], []
for pkg in ("streamlit", "ultralytics", "rasterio", "pyproj", "torch"):
    d, b, h = collect_all(pkg)
    datas += d; binaries += b; hiddenimports += h

for pkg in ("streamlit", "ultralytics", "torch", "numpy", "opencv-python"):
    try:
        datas += copy_metadata(pkg)
    except Exception:
        pass

# 앱 본체와 가중치는 실행 파일 옆이 아니라 안에 넣습니다
datas += [("app_ship.py", "."), ("ship_core.py", "."), ("point_net.py", "."),
          ("weights", "weights")]

hiddenimports += ["streamlit.web.cli", "streamlit.runtime.scriptrunner.magic_funcs",
                  "pyproj._compat", "rasterio._shim", "rasterio.sample",
                  "rasterio.vrt", "rasterio._features"]

a = Analysis(["launch.py"], pathex=["."], binaries=binaries, datas=datas,
             hiddenimports=hiddenimports, hookspath=[], runtime_hooks=[],
             excludes=["matplotlib", "tkinter", "PyQt5", "PySide2", "notebook",
                       "IPython", "pytest"],
             noarchive=False)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="위성선박탐지",
          console=True, disable_windowed_traceback=False, upx=False)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False,
               name="위성선박탐지")
