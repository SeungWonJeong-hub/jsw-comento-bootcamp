# -*- coding: utf-8 -*-
"""브라우저 없이 Streamlit 세션을 열어 화면 코드를 실제로 실행시킵니다.

    py webapp/tests/drive_session.py --port 8503

exe 에서 빠진 패키지 때문에 화면이 열리자마자 죽는 문제는 health 검사로는
안 잡힙니다(서버는 살아 있고 스크립트만 죽음). 그래서 실제 세션을 열고
스크립트 완료 신호와 예외 메시지를 봅니다.
"""
import sys
import time
import argparse

import websocket
from streamlit.proto import BackMsg_pb2, ForwardMsg_pb2


def run(port, timeout):
    ws = websocket.create_connection("ws://localhost:%d/_stcore/stream" % port,
                                     timeout=timeout, subprotocols=["streamlit"])
    bm = BackMsg_pb2.BackMsg()
    bm.rerun_script.query_string = ""
    ws.send(bm.SerializeToString(), opcode=websocket.ABNF.OPCODE_BINARY)
    t0 = time.time()
    n_msgs, errors, finished, deltas = 0, [], False, 0
    while time.time() - t0 < timeout:
        try:
            raw = ws.recv()
        except websocket.WebSocketTimeoutException:
            break
        if not isinstance(raw, (bytes, bytearray)):
            continue
        fm = ForwardMsg_pb2.ForwardMsg()
        fm.ParseFromString(raw)
        n_msgs += 1
        kind = fm.WhichOneof("type")
        if kind == "delta":
            deltas += 1
            el = fm.delta.new_element if fm.delta.HasField("new_element") else None
            if el is not None and el.WhichOneof("type") == "exception":
                errors.append("%s: %s" % (el.exception.type, el.exception.message[:200]))
        elif kind == "script_finished":
            finished = True
            break
    ws.close()
    return n_msgs, deltas, finished, errors


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8502)
    ap.add_argument("--timeout", type=float, default=120)
    a = ap.parse_args()
    n, d, fin, errs = run(a.port, a.timeout)
    print("messages %d · elements %d · finished %s · exceptions %d" % (n, d, fin, len(errs)))
    for e in errs:
        print("  !!", e)
    sys.exit(1 if (errs or not fin) else 0)
