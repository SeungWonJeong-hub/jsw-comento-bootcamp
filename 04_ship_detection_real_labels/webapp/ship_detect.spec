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

# 용량을 줄이려고 앱이 실제로 쓰는 것만 모읍니다. rasterio·pyproj 는 GeoTIFF
# 업로드용이었는데 지금 앱엔 업로드가 없어 뺐습니다(-100 MB).
datas, binaries, hiddenimports = [], [], []
for pkg in ("streamlit", "ultralytics", "torch"):
    d, b, h = collect_all(pkg)
    datas += d; binaries += b; hiddenimports += h

for pkg in ("streamlit", "ultralytics", "torch", "numpy", "opencv-python"):
    try:
        datas += copy_metadata(pkg)
    except Exception:
        pass

# 앱 본체와 가중치는 실행 파일 옆이 아니라 안에 넣습니다
# 항만 5곳 test 영상·라벨·manifest 도 같이 넣습니다 — 실행 파일만으로 돌아가게
# 이 spec 은 webapp/ 안에 있고, 자료·가중치·그림은 한 단계 위(과제 폴더)에 있습니다.
datas += [("app_ship.py", "."), ("ship_core.py", "."), ("../data/hrsc", "data/hrsc"),
          ("../outputs/port_metrics.json", "outputs"),
          ("../weights", "weights")]

hiddenimports += ["streamlit.web.cli", "streamlit.runtime.scriptrunner.magic_funcs",
                  "pyproj._compat", "rasterio._shim", "rasterio.sample",
                  "rasterio.vrt", "rasterio._features"]

a = Analysis(["launch.py"], pathex=["."], binaries=binaries, datas=datas,
             hiddenimports=hiddenimports, hookspath=[], runtime_hooks=[],
             # matplotlib 은 빼면 안 됩니다 — ultralytics 가 OBB 추론 경로에서
             # import 합니다. 빼고 빌드한 exe 가 화면을 열자마자 죽었습니다.
             # ultralytics 가 끌고 오지만 이 앱이 안 쓰는 큰 패키지들 (약 -420 MB).
             # 빼면 안 되는 것 — matplotlib(ultralytics OBB 추론), pyarrow(streamlit 표).
             # 둘 다 뺐다가 exe 가 화면을 열자마자 죽었습니다.
             excludes=["tkinter", "PyQt5", "PySide2", "notebook", "IPython", "pytest",
                       "polars", "_polars_runtime_32", "llvmlite", "numba",
                       "pyogrio", "transformers", "onnxruntime", "rasterio", "pyproj",
                       "pandas.tests", "scipy.tests", "sklearn", "tensorboard"],
             noarchive=False)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="위성선박탐지",
          console=True, disable_windowed_traceback=False, upx=False)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False,
               name="위성선박탐지")
