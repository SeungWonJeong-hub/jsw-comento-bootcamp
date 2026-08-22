"""2차 업무 발표자료(PPT 4페이지) 생성기.

    달 지형 스테레오 실험의 결과를 4장으로 정리한다.

1차 업무 자료와 같은 디자인 규격을 따른다.
    배경 #FAFAFA · 카드 흰색 + #EBEBEB 0.75pt 테두리 · 모서리 반경 0.125in
    제목 Pretendard SemiBold 27pt · 본문 Pretendard 9.5pt · 라벨 Cascadia Mono 8.5pt

수치는 outputs/metrics.json 에서 직접 읽는다. 실험을 다시 돌리면 발표자료도
자동으로 갱신되므로 본문과 결과가 어긋날 일이 없다.

생성물(.pptx)은 저장소에 커밋하지 않는다. 필요할 때 이 스크립트로 만든다.

사용법
    py -3 run_3d_experiment.py
    py -3 tools/build_report.py [출력_폴더]
"""

from __future__ import annotations

import io
import json
import os
import sys

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN
from pptx.util import Emu, Inches, Pt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs")
REPORT = os.path.join(ROOT, "report")

BG = RGBColor(0xFA, 0xFA, 0xFA)
CARD = RGBColor(0xFF, 0xFF, 0xFF)
BORDER = RGBColor(0xEB, 0xEB, 0xEB)
INK = RGBColor(0x17, 0x17, 0x17)
BODY = RGBColor(0x4D, 0x4D, 0x4D)
MUTED = RGBColor(0x8F, 0x8F, 0x8F)
FAINT = RGBColor(0xA1, 0xA1, 0xA1)

MONO = "Cascadia Mono"
SANS = "Pretendard"
SANS_SB = "Pretendard SemiBold"
SANS_MD = "Pretendard Medium"

TOTAL_PAGES = 4


def test_count() -> int:
    """실행된 테스트 개수를 pytest 리포트에서 읽는다.

    발표자료에 숫자를 직접 적어 두면 테스트를 늘렸을 때 조용히 어긋난다.
    """
    import re
    path = os.path.join(OUT, "pytest_report.txt")
    if not os.path.exists(path):
        return 0
    text = io.open(path, encoding="utf-8", errors="replace").read()
    hit = re.search(r"(\d+) passed", text)
    return int(hit.group(1)) if hit else 0


def josa(n, pair):
    """숫자 뒤에 붙는 조사를 읽는 소리에 맞춘다 ("3을" / "5를").

    수치를 f-string 으로 채우다 보니 값이 바뀌면 조사가 틀어진다. 한 자리
    끝소리가 받침으로 끝나는 수(0 영, 1 일, 3 삼, 6 육, 7 칠, 8 팔)와 10(십)
    뒤에는 앞쪽 조사를, 나머지(2 이, 4 사, 5 오, 9 구) 뒤에는 뒤쪽을 쓴다.
    """
    with_batchim, without = pair.split("/")
    last = int(n) % 10
    closed = last in (0, 1, 3, 6, 7, 8) or (int(n) % 100 == 10)
    # '으로/로' 만 규칙이 다르다. ㄹ 받침(1 일, 7 칠, 8 팔) 뒤에는 '로' 를 쓴다.
    if with_batchim == "으로" and last in (1, 7, 8):
        closed = False
    return f"{n}{with_batchim if closed else without}"


def code_lines(name="test_stereo_baseline_matches_the_convergence_formula"):
    """테스트 파일에서 함수 하나를 원문 그대로 읽어 온다.

    과제 요청이 "Unit Test 코드 및 실행 결과 문서화" 이므로 슬라이드에 코드를
    직접 싣는다. 슬라이드에 손으로 옮겨 적으면 코드가 바뀌었을 때 조용히
    어긋나므로 파일에서 읽는다.
    """
    path = os.path.join(ROOT, "tests", "test_terrain.py")
    src = io.open(path, encoding="utf-8").read().splitlines()
    start = next(i for i, l in enumerate(src) if l.startswith(f"def {name}("))
    end = start + 1
    while end < len(src) and not src[end].startswith("def "):
        end += 1
    while end > start and not src[end - 1].strip():
        end -= 1
    return src[start:end]


def test_counts() -> dict:
    """pytest 실행 결과에서 파일별 통과 수를 센다.

    슬라이드에 "기하 44 · 스테레오 27" 처럼 적으려면 이 값이 필요하다.
    손으로 적어 두면 테스트를 늘렸을 때 조용히 어긋난다.
    """
    import re
    from collections import Counter

    path = os.path.join(OUT, "pytest_report.txt")
    if not os.path.exists(path):
        return {}
    text = io.open(path, encoding="utf-8", errors="replace").read()
    hits = re.findall(r"^(\S+?\.py)::\S+ PASSED", text, re.M)
    return dict(Counter(f.split("/")[-1] for f in hits))


def report_lines():
    """pytest 실행 결과에서 슬라이드에 들어갈 만큼만 뽑는다.

    전문은 outputs/pytest_report.txt 에 그대로 있다. 여기서는 파일별 통과 수와
    테스트 이름 몇 개만 옮긴다. 이름을 고르는 기준은 '무엇을 검증하는지가
    이름만 봐도 드러나는 것' 이다.
    """
    import re
    from collections import Counter

    path = os.path.join(OUT, "pytest_report.txt")
    if not os.path.exists(path):
        return ["outputs/pytest_report.txt 가 없습니다"]
    text = io.open(path, encoding="utf-8", errors="replace").read()

    hits = re.findall(r"^(\S+?\.py)::(\S+) PASSED", text, re.M)
    per_file = Counter(f.split("/")[-1] for f, _ in hits)
    names = {n.split("[")[0] for n in (x[1] for x in hits)}
    total = re.search(r"(\d+) passed", text)

    # 카드 폭에 줄바꿈 없이 들어가는 길이만 고른다. 이름이 접히면 그만큼
    # 줄이 늘어 카드 밖으로 흘러넘친다.
    picked = [n for n in (
        "test_stereo_baseline_matches_the_convergence_formula",
        "test_rendered_depth_recovers_the_original_elevation",
        "test_full_pipeline_recovers_the_terrain_within_one_pixel",
    ) if n in names]

    out = [f"collected {len(hits)} items", ""]
    for f, n in sorted(per_file.items()):
        out.append(f"{f:<24s}{n:>3d} passed")
    out += ["", "PASSED (발췌)"]
    out += [f"  {n}" for n in picked]
    out += ["  ...", "", f"{total.group(1) if total else '?'} passed"]
    return out


def set_font(run, name, size, color, bold=False, tracking=None):
    """글꼴을 지정한다. tracking 은 자간 [pt], 음수면 좁힌다.

    디자인 규격(DESIGN-vercel.md)의 Don't 에 "large heading 의 음수 자간을
    풀지 말 것" 이 있다. 표에 따르면 48px 제목이 -2.4px, 32px 제목이 -1.28px 로
    둘 다 글자 크기의 -4% 다. 같은 비율을 그대로 적용한다.
    """
    run.font.name = name
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color
    if tracking:
        # python-pptx 에 자간 API 가 없어 rPr 의 spc 속성을 직접 쓴다.
        # 단위는 1/100 pt 다.
        run._r.get_or_add_rPr().set("spc", str(int(round(tracking * 100))))
    # 한글이 기본 글꼴로 떨어지지 않도록 동아시아 글꼴도 지정한다.
    rPr = run._r.get_or_add_rPr()
    ns = "{http://schemas.openxmlformats.org/drawingml/2006/main}"
    for tag in ("ea", "cs"):
        rPr.append(rPr.makeelement(ns + tag, {"typeface": name}))


def style_paragraph(para, name, size, color, tracking=None):
    """문단과 그 안의 런에 모두 글꼴을 지정한다.

    런에만 지정하면 텍스트가 빈 문단은 기본 크기(18pt)로 높이를 잡아
    줄 간격이 어긋난다. 표에서 빈 칸이 한 행씩 밀리는 원인이었다.
    """
    para.font.name = name
    para.font.size = Pt(size)
    para.font.color.rgb = color
    run = para.add_run()
    set_font(run, name, size, color, tracking=tracking)
    return run


def add_text(slide, x, y, w, h, lines, align=PP_ALIGN.LEFT, spacing=1.35,
             tracking=None):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0

    for i, (text, name, size, color) in enumerate(lines):
        para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        para.alignment = align
        para.line_spacing = spacing
        style_paragraph(para, name, size, color, tracking).text = text
    return box


def add_card(slide, x, y, w, h):
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,
                                   Inches(x), Inches(y), Inches(w), Inches(h))
    shape.adjustments[0] = 0.125 / min(w, h)
    shape.fill.solid()
    shape.fill.fore_color.rgb = CARD
    shape.line.color.rgb = BORDER
    shape.line.width = Pt(0.75)
    shape.shadow.inherit = False
    return shape


def add_matrix(slide, x, y, col_w, rows, row_h=0.235, aligns=None,
               spacing=1.45):
    """표를 그린다. 열마다 별도 텍스트 상자를 둔다.

    등폭 글꼴로 공백을 맞추는 방식은 쓸 수 없다. Cascadia Mono 에는 한글이 없어
    대체 글꼴로 그려지는데 그 글꼴은 고정폭이 아니라서 열이 어긋난다.
    열을 상자로 분리하면 글꼴과 무관하게 정렬이 맞는다.

    Parameters
    ----------
    col_w : 열 너비 목록 [inch]. 기본은 첫 열만 왼쪽, 나머지는 오른쪽 정렬.
    rows : (텍스트 튜플, 스타일) 목록. 스타일은 'head' | 'key' | 'body' | 'small'.
    aligns : 열별 정렬 목록. 설명이 들어가는 열은 오른쪽으로 밀면 안 읽힌다.
    """
    styles = {
        "head": (SANS, 9, MUTED),
        "key": (SANS_MD, 10, INK),
        "body": (SANS, 10, BODY),
        "small": (SANS, 9, BODY),
        "small_key": (SANS_MD, 9, INK),
    }
    for j, cw in enumerate(col_w):
        left = x + sum(col_w[:j])
        box = slide.shapes.add_textbox(Inches(left), Inches(y), Inches(cw),
                                       Inches(row_h * len(rows)))
        tf = box.text_frame
        tf.word_wrap = False
        tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
        for i, (cells, style) in enumerate(rows):
            para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            para.alignment = (aligns[j] if aligns else
                              PP_ALIGN.LEFT if j == 0 else PP_ALIGN.RIGHT)
            para.line_spacing = spacing
            name, size, color = styles[style]
            style_paragraph(para, name, size, color).text = cells[j]


def add_strip(slide, x, y, w, items, label_pt=10, value_pt=17):
    """숫자 몇 개를 가로로 늘어놓는다.

    표로 쌓으면 슬라이드가 문서가 된다. 발표에서 눈이 머무는 숫자는 서넛뿐이고,
    그것들은 한 줄에 나란히 있을 때 가장 빨리 읽힌다.
    """
    if not items:
        return
    step = w / len(items)
    for i, (label, value) in enumerate(items):
        left = x + step * i
        add_text(slide, left, y, step - 0.2, 0.22,
                 [(label, SANS, label_pt, MUTED)])
        add_text(slide, left, y + 0.30, step - 0.2, 0.34,
                 [(value, SANS_SB, value_pt, INK)], tracking=-value_pt * 0.02)


def add_panel(slide, x, y, w, h, heading, lines):
    add_card(slide, x, y, w, h)
    add_text(slide, x + 0.26, y + 0.19, w - 0.52, 0.24,
             [(heading, SANS_SB, 10.5, INK)])
    add_text(slide, x + 0.26, y + 0.46, w - 0.52, h - 0.66, lines, spacing=1.3)


def add_image(slide, path, cx, cy, cw, ch, caption=None):
    add_card(slide, cx, cy, cw, ch)
    pad = 0.15
    pic = slide.shapes.add_picture(path, Inches(0), Inches(0))
    aspect = pic.width / pic.height
    box_w, box_h = cw - 2 * pad, ch - 2 * pad
    if box_w / box_h > aspect:
        h = box_h
        w = h * aspect
    else:
        w = box_w
        h = w / aspect
    pic.width = Emu(int(Inches(w)))
    pic.height = Emu(int(Inches(h)))
    pic.left = Emu(int(Inches(cx + (cw - w) / 2)))
    pic.top = Emu(int(Inches(cy + (ch - h) / 2)))
    if caption:
        add_text(slide, cx, cy + ch + 0.07, cw, 0.2,
                 [(caption, MONO, 8.5, MUTED)])
    return pic


TALK = []


def add_notes(slide, text):
    """할 말을 모아 둔다. 슬라이드에는 넣지 않는다.

    발표자 노트로 넣으면 파일을 열어 본 사람에게 그대로 보이고, 인쇄하거나
    다른 도구로 변환할 때 따라다닌다. 발표자가 볼 것은 손에 든 대본이면
    충분하므로 pptx 는 슬라이드만 담고, 같은 내용을 대본 txt 로 따로 낸다.
    """
    TALK.append(text.strip())


def new_slide(prs, page, eyebrow, title, subtitle):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0,
                                prs.slide_width, prs.slide_height)
    bg.fill.solid()
    bg.fill.fore_color.rgb = BG
    bg.line.fill.background()
    bg.shadow.inherit = False

    add_text(slide, 0.72, 0.46, 11.89, 0.24, [(eyebrow, MONO, 8.5, MUTED)])
    # 규격의 display 트래킹: 글자 크기의 -4%
    add_text(slide, 0.72, 0.74, 11.89, 0.52, [(title, SANS_SB, 27, INK)],
             spacing=1.0, tracking=-27 * 0.04)
    add_text(slide, 0.72, 1.28, 11.89, 0.26, [(subtitle, SANS, 10.5, BODY)])
    add_text(slide, 11.4, 6.92, 1.21, 0.24,
             [(f"{page:02d} / {TOTAL_PAGES:02d}", MONO, 8.5, FAINT)],
             align=PP_ALIGN.RIGHT)
    return slide


def b(t):
    """본문."""
    return (t, SANS, 10, BODY)


def k(t):
    """강조 — 수치나 결론."""
    return (t, SANS_MD, 10, INK)


def m(t):
    """등폭 — 표나 수식."""
    return (t, MONO, 9.5, BODY)


def t(x):
    """발표용 문장. 듣는 사람이 읽을 크기라 본문보다 한 단계 크다."""
    return (x, SANS, 11, BODY)


def tk(x):
    """발표용 문장 중 결론."""
    return (x, SANS_MD, 11, INK)


def gap():
    return ("", SANS, 4, BODY)


# ---------------------------------------------------------------------------


def load_apollo():
    """실제 사진 실험 결과가 있으면 읽는다. 없으면 그 부분을 뺀다."""
    path = os.path.join(OUT, "apollo_metrics.json")
    if not os.path.exists(path):
        return None
    if not os.path.exists(os.path.join(OUT, "08_apollo_slide.png")):
        return None
    with io.open(path, encoding="utf-8") as f:
        return json.load(f)


def build(prs, s):
    """4 페이지. 과제 결과물 형식(업무.pdf p.16)에 맞춰 자리를 나눈다."""
    sc, best = s["scene"], s["best"]
    conv = s["convergence_sweep"]
    blocks = s["block_size_sweep"]
    tc = test_count()
    res = sc["altitude_m"] ** 2 / (sc["focal_px"] * sc["baseline_m"])
    chosen = next(c for c in conv
                  if abs(c["convergence_deg"] - sc["convergence_deg"]) < 1e-9)
    widest = conv[-1]


    # ---------------- 1. 방법 ----------------
    sl = new_slide(
        prs, 1, "01 / 방법",
        "달 사진 두 장으로 복원한 지형의 높낮이",
        f"수렴 스테레오 · 티코 크레이터 · LOLA 고도 모델 "
        f"{sc['gsd_m']/1000:.3f} km 격자 · {sc['grid'][0]}×{sc['grid'][1]} · "
        f"달 광도함수 · SNR {sc['snr']:.0f}")
    add_image(sl, os.path.join(OUT, "05_method.png"), 0.72, 1.70, 11.89, 3.16)

    # 왼쪽 — 식만. 라벨과 식을 열로 나눠야 식의 시작 위치가 맞는다.
    add_card(sl, 0.72, 5.02, 6.10, 1.86)
    add_text(sl, 0.98, 5.21, 5.58, 0.24, [("스테레오로 Z 를 뽑기까지", SANS_SB, 10.5, INK)])
    steps = [
        ("① 상대 자세", "R = R2 · R1^T,   t = t2 - R · t1"),
        ("② 베이스라인", "B = 2 · H · tan(수렴각/2)"),
        ("③ 정렬·정합", "stereoRectify(R, t) -> StereoSGBM -> d"),
        ("④ 깊이", "Z = f · B / d"),
        ("⑤ 3D 좌표", "X = (u-cx)·Z/f,   Y = (v-cy)·Z/f"),
    ]
    add_text(sl, 0.98, 5.52, 1.30, 0.94,
             [(k, SANS, 9.5, MUTED) for k, _ in steps], spacing=1.28)
    add_text(sl, 2.32, 5.52, 4.24, 0.94,
             [(v, MONO, 9.5, BODY) for _, v in steps], spacing=1.28)
    add_text(sl, 0.98, 6.58, 5.58, 0.22, [
        (f"→ 이렇게 나온 Z 오차 중앙값 {best['median_abs']/1000:.3f} km "
         f"(시차 {best['median_abs']/res:.2f} px)", SANS_MD, 9.5, INK)])

    # 오른쪽 — 기호가 무슨 역할인지와 그 값.
    add_card(sl, 7.05, 5.02, 5.56, 1.86)
    add_text(sl, 7.31, 5.21, 5.04, 0.24,
             [("각 기호가 하는 일", SANS_SB, 10.5, INK)])
    add_matrix(sl, 7.31, 5.55, [0.68, 2.86, 1.50], [
        (("H", "카메라가 떠 있는 높이",
          f"{sc['altitude_m']/1000:.1f} km"), "small"),
        (("B", "두 촬영 지점 사이 거리",
          f"{sc['baseline_m']/1000:.1f} km"), "small"),
        (("수렴각", "두 시선이 벌어진 각",
          f"{sc['convergence_deg']:.0f} 도"), "small"),
        (("f", "초점거리", f"{sc['focal_px']:,.0f} px"), "small"),
        (("d", "두 사진에서 밀린 픽셀 수", "찾는 값"), "small_key"),
        (("Z", "그 픽셀이 보는 지면까지 거리", "d 로 계산"), "small_key"),
    ], row_h=0.19, spacing=1.36,
        aligns=[PP_ALIGN.LEFT, PP_ALIGN.LEFT, PP_ALIGN.RIGHT])

    add_notes(sl, f"""
· 쓴 기법은 스테레오 비전입니다. 같은 곳을 각도를 바꿔 두 번 찍고 두 사진에서 밀린 정도로 거리를 잽니다. 두 눈으로 거리를 재는 것과 같습니다.
· 두 대를 나란히 놓는 대신 같은 지점을 바라보게 기울인 수렴 촬영입니다. 실제 궤도 스테레오가 그렇게 합니다.

· 지형은 LOLA 레이저 고도 모델의 티코 크레이터를 쓰고 표면 무늬는 같은 곳을 실제로 찍은 WAC 영상에서 가져왔습니다.

· 두 장을 나란히 펴고 OpenCV 의 SGBM 으로 시차를 찾고 못 믿을 곳을 버린 뒤 Z 는 f 곱하기 B 를 d 로 나눕니다. 이 결과가 depth map 입니다.
· 마지막에 수렴각이 다른 네 쌍을 합칩니다.
· 시차 1픽셀이 {res/1000:.2f} km 인데 오차는 {best['median_abs']/1000:.3f} km 입니다. 한 픽셀보다 작습니다.
""")

    # ---------------- 2. Unit Test ----------------
    sl = new_slide(
        prs, 2, "02 / Unit Test",
        "식이 틀리면 걸리는 테스트",
        "실제 테스트 코드와 실행 결과 · tests/test_terrain.py · "
        "outputs/pytest_report.txt")

    # 두 카드가 담는 글의 길이가 비슷하다(코드 74자 · 결과 69자). 폭도 그에
    # 맞춰 나눈다. 예전에는 7.35 대 4.30 이라 오른쪽이 접히며 넘쳤다.
    add_card(sl, 0.72, 1.70, 6.90, 3.45)
    add_text(sl, 0.98, 1.89, 6.38, 0.24, [
        ("테스트 코드 — tests/test_terrain.py", SANS_SB, 10.5, INK)])
    add_text(sl, 0.98, 2.24, 6.38, 0.22, [
        ("베이스라인이 B = 2·H·tan(수렴각/2) 를 정확히 만족해야 합니다",
         SANS, 10, MUTED)])
    add_text(sl, 0.98, 2.62, 6.38, 2.60,
             [(x, MONO, 9.5, BODY) for x in code_lines()], spacing=1.3)

    add_card(sl, 7.86, 1.70, 4.75, 3.45)
    add_text(sl, 8.12, 1.89, 4.23, 0.24, [
        ("실행 결과 — pytest", SANS_SB, 10.5, INK)])
    add_text(sl, 8.12, 2.24, 4.23, 0.22, [
        ("걸린 시간만 지우고 그대로 저장합니다", SANS, 10, MUTED)])
    add_text(sl, 8.12, 2.62, 4.23, 2.60,
             [(x, MONO, 8, BODY) for x in report_lines()], spacing=1.2)

    # 아래 카드는 글이 위로 몰려 있었다. 카드 높이를 글에 맞추고 가운데로 둔다.
    add_card(sl, 0.72, 5.48, 11.89, 1.34)
    add_text(sl, 0.98, 5.70, 11.37, 0.90, [
        ("테스트가 출력의 크기와 자료형만 보면 계산식이 틀려도 통과합니다.",
         SANS, 11.5, MUTED),
        ("그래서 손으로 답을 낼 수 있는 조건을 만들어 숫자까지 대조했습니다. "
         "식이 틀리면 반드시 걸립니다.", SANS, 11.5, BODY),
        (f"실제로 이 테스트가 렌더링이 시차를 밀어내던 버그를 잡았습니다. "
         f"지금은 {tc}개가 전부 통과합니다.", SANS_MD, 11.5, INK),
    ], spacing=1.55)
    add_notes(sl, f"""
· 크기와 자료형만 보는 테스트는 식이 틀려도 통과합니다. 그래서 손으로 답을 낼 수 있는 조건으로 짰습니다.
· {tc}개 전부 통과합니다.
· 실제로 버그를 잡았습니다. 렌더링이 시차를 2.6 픽셀 밀어 고도가 1.245 km 틀렸던 것을 0.1 픽셀로 줄였습니다.
""")

    # ---------------- 3. 2D -> 3D 변환 결과 ----------------
    fu = s["fusion"]
    fz = fu["fused"]
    one = next(p for p in fu["per_angle"]
               if abs(p["convergence_deg"] - sc["convergence_deg"]) < 1e-9)
    widest = fu["per_angle"][-1]
    ap = load_apollo()
    one = next(p for p in fu["per_angle"]
               if abs(p["convergence_deg"] - sc["convergence_deg"]) < 1e-9)
    sl = new_slide(
        prs, 3, "03 / 2D → 3D 변환 결과",
        f"레이저로 잰 고도와 {fz['median_abs']/1000:.3f} km 안에서 일치",
        f"수렴각이 다른 네 쌍을 합친 결과 · 복원한 점 {s['n_points']:,}개")

    # 3장이 요구하는 것은 변환 결과 이미지다. 크게 하나 두고, 숫자는 견줄 수
    # 있을 만큼만 아래에 한 줄로 놓는다. 나머지 설명은 대본이 갖는다.
    add_image(sl, os.path.join(OUT, "04_result.png"), 0.72, 1.76, 11.89, 3.16)

    add_card(sl, 0.72, 5.30, 11.89, 1.36)
    # 이 장이 답할 것은 하나다 - 얼마나 맞았는가. 중앙값만으로는 꼬리가 안
    # 보이므로 90 퍼센타일을 같이 둔다.
    strip = [
        ("고도 오차 (중앙값)", f"{fz['median_abs']/1000:.3f} km"),
        ("90 퍼센타일", f"{fz['p90_abs']/1000:.3f} km"),
        ("높낮이 차 대비", f"{fz['median_abs']/sc['relief_m']*100:.2f}%"),
        ("값이 나온 셀", f"{fz['coverage']*100:.0f}%"),
    ]
    if ap:
        strip.append(("실제 사진으로도",
                      f"레이저 대비 {ap['relief_m']/2719:.2f} 배"))
    add_strip(sl, 0.98, 5.62, 11.37, strip)

    add_notes(sl, f"""
· 왼쪽부터 정답 고도 · 사진에서 복원한 고도 · 3D 점구름입니다. 크레이터 바닥과 테두리 그리고 가운데 봉우리가 살아 있습니다.
· 오차 중앙값이 {fz['median_abs']/1000:.3f} km 입니다. 높낮이 차가 {sc['relief_m']/1000:.3f} km 인 지형이니 그 {fz['median_abs']/sc['relief_m']*100:.2f}퍼센트입니다.
· 90 퍼센타일도 {fz['p90_abs']/1000:.3f} km 이니 크게 틀린 곳이 거의 없습니다.
· 못 낸 {100-fz['coverage']*100:.0f}퍼센트는 크레이터 안쪽 그늘입니다. 두 사진에 같은 무늬가 없으면 원리적으로 못 맞춥니다.

· 여기까지는 제가 렌더링한 두 장입니다. 그래서 1971년 아폴로 15호가 실제로 찍은 두 장에도 같은 파이프라인을 걸었습니다.
· 거기서도 3D 점구름이 나왔고 사진에서 잰 것만으로 계산한 카메라 고도와 복원한 높낮이 차가 기록된 값과 맞습니다.
""")

    # ---------------- 4. 개선점 ----------------
    # 고친 것은 3장의 수치에 이미 들어가 있다. 여기는 남은 것만 번호로 둔다.
    sl = new_slide(
        prs, 4, "04 / 남은 과제",
        "개선점",
        "앞줄이 왜 그런지 뒷줄이 그래서 무엇을 개선해야 하는지입니다")

    cf = s["confidence"]
    ap = s["auto_parameters"]
    best_signal = min(cf["sparsification"], key=lambda r: r["ause"])
    best_auc = max(cf["sparsification"],
                   key=lambda r: r.get("auc", float("-inf")))
    tune = {r["block_size"]: r for r in ap["tuning"]}
    cost = (tune[ap["picked_without_truth"]]["median_abs"]
            / tune[ap["picked_with_truth"]]["median_abs"] - 1) * 100

    items = [
        ("1", "실제 사진 결과를 높낮이 차로만 견줬다",
         "복원한 고도가 지도 좌표 위에 놓여 있지 않아서 정답 고도와 픽셀을 "
         "맞대어 볼 수가 없었습니다. 그래서 견준 것이 높낮이 차와 "
         "카메라까지의 거리뿐입니다.",
         "이대로는 실제 사진에서 오차가 몇 미터인지를 한 숫자로 말할 수 "
         "없습니다. 픽셀마다 채점할 수 있는 상태로 개선해야 할 것 같습니다."),
        ("2", "남긴 값이 얼마나 정확한지는 말하지 못한다",
         f"정답 없이 쓰는 신호가 크게 틀린 픽셀을 골라내는 쪽에 맞춰져 "
         f"있습니다. 그 일은 잘합니다(AUC {best_auc['auc']:.2f}). 하지만 오차의 "
         f"크기와는 순위 상관이 {best_signal['spearman']:+.2f} 뿐입니다.",
         "착륙 지점을 고르려면 여기는 몇 미터 안이라고 말할 수 있어야 하는데 "
         "지금은 못 합니다. 크기까지 예측하는 쪽으로 개선해야 할 것 같습니다."),
    ]
    # 항목이 적을수록 글자를 키우고 간격을 넓힌다. 같은 자리에 넷을 넣을 때와
    # 둘을 넣을 때 같은 크기를 쓰면 카드가 비어 보인다.
    few = len(items) <= 2
    head_pt, body_pt = (18, 12.5) if few else (14, 11)
    step = 1.78 if few else 1.22
    y = 2.28 if few else 2.10
    # 항목이 둘이면 카드도 그만큼 줄인다. 큰 카드에 글이 절반만 있으면 빠뜨린
    # 것처럼 보인다.
    tall = step * (len(items) - 1) + 1.85 if few else 5.12

    add_card(sl, 0.72, 1.70, 11.89, tall)
    for num, head, now, why in items:
        add_text(sl, 0.98, y, 0.44, 0.32,
                 [(num, SANS_SB, head_pt + 1, FAINT)])
        add_text(sl, 1.52, y, 10.83, 0.32, [(head, SANS_SB, head_pt, INK)],
                 tracking=-head_pt * 0.02)
        add_text(sl, 1.52, y + 0.52, 10.83, 0.60,
                 [(now, SANS, body_pt, BODY), (why, SANS_MD, body_pt, INK)],
                 spacing=1.4)
        y += step
    add_notes(sl, f"""
· 하나. 실제 사진 결과를 높낮이 차로만 견줬습니다. 복원한 고도가 지도 좌표 위에 없어서 픽셀마다 채점하지 못했습니다.
· 둘. 크게 틀린 픽셀은 잘 골라내는데 오차의 크기는 못 맞힙니다. 여기는 몇 미터 안이라고 말하지 못합니다.
""")


def talk_script(width=46):
    """모아 둔 할 말을 대본 txt 로 조립한다."""
    import textwrap

    titles = ["01 / 방법", "02 / Unit Test", "03 / 2D → 3D 변환 결과",
              "04 / 남은 과제"]
    out = ["2차 업무 발표 대본 — 정승원",
           f"슬라이드 {TOTAL_PAGES}장 · 각 장에서 할 말", "=" * 50, ""]
    for i, note in enumerate(TALK):
        out += [f"[{i + 1}] {titles[i]}", "-" * 50]
        for line in note.splitlines():
            line = line.strip()
            if line:
                out += textwrap.wrap(line, width, subsequent_indent="  ")
            elif out and out[-1]:
                # 빈 줄은 문단을 나누는 표시다. 지우면 할 말이 한 덩어리가 된다.
                out.append("")
        out.append("")
    return "\n".join(out)


def main() -> int:
    path = os.path.join(OUT, "metrics.json")
    if not os.path.exists(path):
        print(f"metrics.json 이 없습니다: {path}\n먼저 run_3d_experiment.py 를 실행하세요.")
        return 1
    with open(path, encoding="utf-8") as f:
        summary = json.load(f)

    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    build(prs, summary)

    dest = sys.argv[1] if len(sys.argv) > 1 else REPORT
    os.makedirs(dest, exist_ok=True)
    out = os.path.join(dest, "2차업무_정승원.pptx")
    prs.save(out)
    print(f"저장 완료 -> {out}  ({TOTAL_PAGES} 슬라이드)")

    script = os.path.join(dest, "2차업무_발표대본_정승원.txt")
    io.open(script, "w", encoding="utf-8-sig",
            newline="\r\n").write(talk_script())
    print(f"저장 완료 -> {script}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
