# -*- coding: utf-8 -*-
"""실행 파일 진입점 — 웹앱을 띄우고 브라우저를 엽니다.

streamlit 은 명령줄로 스크립트를 받는 구조라, 실행 파일 안에서도
같은 방식으로 부릅니다. 묶인 상태에서는 app_ship.py 가 임시 폴더에
풀리므로 sys._MEIPASS 를 먼저 봅니다.
"""
import os
import sys
import socket
import threading
import webbrowser


def base_dir():
    """묶였으면 풀린 자리, 아니면 이 파일이 있는 자리."""
    return getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))


def free_port(start=8502, tries=20):
    """이미 쓰는 포트를 피합니다 — 두 번 실행해도 되게."""
    for p in range(start, start + tries):
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", p)) != 0:
                return p
    return start


def safe_console():
    """윈도 콘솔 기본 인코딩(cp949)은 em-dash 같은 글자에서 죽습니다.

    묶인 실행 파일에서 실제로 UnicodeEncodeError 로 종료했습니다.
    출력만 죽는 것이 프로그램 전체를 죽이지 않게 합니다.
    """
    for st in (sys.stdout, sys.stderr):
        try:
            st.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def main():
    safe_console()
    here = base_dir()
    app = os.path.join(here, "app_ship.py")
    if not os.path.exists(app):
        print("app_ship.py 를 찾지 못했습니다: %s" % app)
        input("엔터를 누르면 닫습니다...")
        return 1

    port = free_port()
    url = "http://localhost:%d" % port
    print("위성 선박탐지  %s" % url)
    print("이 창을 닫으면 프로그램이 멈춥니다.")
    threading.Timer(3.0, lambda: webbrowser.open(url)).start()

    sys.argv = ["streamlit", "run", app,
                "--server.port", str(port),
                "--server.headless", "true",
                "--browser.gatherUsageStats", "false",
                "--global.developmentMode", "false"]
    from streamlit.web import cli
    return cli.main()


if __name__ == "__main__":
    sys.exit(main() or 0)
