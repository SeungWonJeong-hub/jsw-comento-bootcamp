"""2차 업무 발표자료(PPT 4페이지) 생성기.

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


def code_lines(name="test_relative_pose_maps_view_i_to_view_j"):
    """테스트 파일에서 함수 하나를 원문 그대로 읽어 온다.

    과제 요청이 "Unit Test 코드 및 실행 결과 문서화" 이므로 슬라이드에 코드를
    직접 싣는다. 슬라이드에 손으로 옮겨 적으면 코드가 바뀌었을 때 조용히
    어긋나므로 파일에서 읽는다.
    """
    path = os.path.join(ROOT, "tests", "test_stereo.py")
    src = io.open(path, encoding="utf-8").read().splitlines()
    start = next(i for i, l in enumerate(src) if l.startswith(f"def {name}("))
    end = start + 1
    while end < len(src) and not src[end].startswith("def "):
        end += 1
    while end > start and not src[end - 1].strip():
        end -= 1
    return src[start:end]


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

    picked = [n for n in (
        "test_disparity_to_depth_matches_formula",
        "test_relative_pose_maps_view_i_to_view_j",
        "test_brightness_depth_confuses_albedo_with_distance",
        "test_vertical_reconstruct_returns_unrotated_maps",
        "test_fit_window_keeps_geometry_and_contains_the_source",
        "test_reference_depth_lands_in_the_same_frame_as_reconstruct",
        "test_unlit_surfaces_break_matching",
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


def add_matrix(slide, x, y, col_w, rows, row_h=0.235):
    """표를 그린다. 열마다 별도 텍스트 상자를 둔다.

    등폭 글꼴로 공백을 맞추는 방식은 쓸 수 없다. Cascadia Mono 에는 한글이 없어
    대체 글꼴로 그려지는데 그 글꼴은 고정폭이 아니라서 열이 어긋난다.
    열을 상자로 분리하면 글꼴과 무관하게 정렬이 맞는다.

    Parameters
    ----------
    col_w : 열 너비 목록 [inch]. 첫 열은 왼쪽 정렬, 나머지는 오른쪽 정렬.
    rows : (텍스트 튜플, 스타일) 목록. 스타일은 'head' | 'key' | 'body'.
    """
    styles = {
        "head": (SANS, 9, MUTED),
        "key": (SANS_MD, 10, INK),
        "body": (SANS, 10, BODY),
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
            para.alignment = PP_ALIGN.LEFT if j == 0 else PP_ALIGN.RIGHT
            para.line_spacing = 1.45
            name, size, color = styles[style]
            style_paragraph(para, name, size, color).text = cells[j]


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


def build(prs, s):
    """4 페이지. 과제 결과물 형식(업무.pdf p.16)에 맞춰 자리를 나눈다.

        1. 방법          - 카메라가 고정인데 어떻게 깊이를 재는가
        2. Unit Test     - 코드 및 실행 결과 문서화 (p.16 요구)
        3. 2D -> 3D 변환 - 변환 결과 이미지 첨부 (p.16 요구)
        4. 개선점        - 요청내용 3번 "개선점을 도출"

    쓰는 방침
        한 장에 카드 두 개, 카드 하나에 문장 세 줄까지만 둔다. 발표에서 듣는
        사람이 읽을 수 있는 양이 그 정도다. 자세한 수치와 근거는 README 와
        metrics.json 에 있고 슬라이드는 그중 결론만 옮긴다.

        전문 용어는 처음 나올 때 한 번만 풀어 준다. '정합' 대신 '같은 지점
        찾기', 'blockSize' 대신 '오려서 비교하는 네모의 크기' 로 쓴다.

        다만 풀어 쓴 말이 다른 말과 겹치면 원래 용어를 쓴다. '시차' 를 '밀린
        거리' 로 바꿔 봤더니 재려는 대상인 깊이(거리)와 헷갈렸다. 지금은
        '시차 d' 를 쓰고 '옆으로 밀린 픽셀 수' 라고 한 번만 덧붙인다.

        그림은 한 장에 하나만 넣고, 4 장은 표만 둔다.
    """
    syn = s["synthetic_validation"]
    ss = syn["stereo"]
    best = s["best_pair"]
    tc = test_count()
    # 대조군은 스테레오가 값을 낸 화소에서만 채점한 'common' 을 쓴다. 실루엣
    # 전체에서 채점한 'full' 과 나란히 놓으면 두 방법이 서로 다른 화소에서
    # 채점되기 때문이다.
    se = syn["example_code"]["common"]
    bex = s["best_pair_example_code"]["common"]
    narrow = s["disparity_range_ablation"][0]
    cov = s["surface_coverage"]
    cov_one = cov["stereo_single_pair"]
    cov_fuse = cov["multiview_stereo_fusion"]
    fuse0 = cov_fuse["stages"][0]
    blocks = s["block_size_ablation"]
    blk_old = blocks[0]
    spread = cov_fuse["view_spread"]
    posesens = s["pose_error_sensitivity"]
    inc = cov_fuse["incremental"]
    top2 = sorted(inc, key=lambda r: -r["gain"])[:2]
    top2_share = sum(r["gain"] for r in top2) / inc[-1]["cumulative_coverage"]
    blk_new = next(x for x in blocks
                   if x["best_pair"]["valid_ratio"] == best["valid_ratio"])
    gap_ratio = bex["median_abs"] / best["median_abs"]

    # ---------------- 1. 방법 ----------------
    # 스테레오가 무엇인지는 설명하지 않는다. 이 장에서 남겨야 할 것은
    # "이 데이터에 어떻게 적용했고, 어떤 식을 거쳐, 무슨 값이 나왔는가" 다.
    sl = new_slide(
        prs, 1, "01 / 방법",
        "카메라는 그대로인데, 위성이 돌아서 깊이를 쟀습니다",
        "SPE3R 위성 근접 영상 1,000장 · 256×256 · 카메라 병진은 항상 (0, 0, Z)")
    add_image(sl, os.path.join(OUT, "00_concept.png"), 0.72, 1.70, 11.89, 3.16)

    add_card(sl, 0.72, 5.02, 6.60, 1.86)
    add_text(sl, 0.98, 5.21, 6.08, 0.24,
             [("Z 를 뽑기까지", SANS_SB, 10.5, INK)])
    add_text(sl, 0.98, 5.48, 6.08, 0.20, [
        ("시차 d = 같은 지점이 두 사진에서 옆으로 밀린 픽셀 수",
         SANS, 9.5, MUTED)])
    # 첨자는 i, j 가 아니라 1, 2 를 쓴다. i 와 j 는 생김새가 비슷해 어느 쪽이
    # 어느 사진인지 한눈에 안 들어온다. 상대 자세에는 첨자를 아예 붙이지
    # 않는다 - 뒤에서 R, t 로만 쓰이므로 이름이 짧을수록 식이 읽힌다.
    # 전치는 유니코드 위첨자 대신 ^T 로 쓴다. 등폭 글꼴에 없으면 다른 글꼴로
    # 대체돼 줄이 어긋나기 때문이다.
    add_text(sl, 0.98, 5.72, 6.08, 1.10, [
        (x, MONO, 9.5, BODY) for x in (
            "R1, t1 / R2, t2 = 사진 1·2 의 자세 (데이터셋이 준 값)",
            "① 상대 자세   R = R2 · R1^T,    t = t2 − R · t1",
            "② 베이스라인  B = t 중 화면과 나란한 성분",
            "③ 정렬·정합   stereoRectify(R, t) → StereoSGBM → 시차 d",
            "④ 깊이        Z = f · B / d",
            "⑤ 3D 좌표     X = (u−cx)·Z/f,    Y = (v−cy)·Z/f",
        )], spacing=1.32)

    add_card(sl, 7.45, 5.02, 5.16, 1.86)
    add_text(sl, 7.71, 5.21, 4.64, 0.24,
             [("식에 들어간 값과 나온 값 (최적 쌍)", SANS_SB, 10.5, INK)])
    add_matrix(sl, 7.71, 5.57, [2.80, 1.84], [
        (("초점거리 f", f"{best['focal_px']:,.0f} px"), "body"),
        (("베이스라인 B", f"{best['baseline_m']:.3f} m"), "body"),
        (("시차 d (기대)", f"{best['expected_disparity_px']:.1f} px"), "body"),
        (("d 1픽셀이 바꾸는 깊이",
          f"{best['depth_resolution_m_per_px']*100:.1f} cm"), "body"),
        # '거리 오차' 라고 부르지 않는다. Z 는 광축 방향 깊이이고, 카메라에서
        # 점까지의 실제 거리는 sqrt(X^2+Y^2+Z^2) 라 화면 가장자리에서 갈린다.
        # 우리가 재는 것은 Z 이므로 Z 라고 부르는 편이 짧고 정확하다.
        (("→ Z 오차 (중앙값)", f"{best['median_abs']*100:.2f} cm"), "key"),
    ])

    add_notes(sl, f"""
· 이 데이터는 카메라가 제자리에 고정돼 있습니다. 위치 값 천 개를 확인해도 옆으로 1 mm 도 안 움직입니다. 그대로는 스테레오를 쓸 수 없습니다.
· 대신 위성이 매 장면 무작위로 돌아갑니다. 위성을 가만히 놓고 보면 카메라가 그 주위를 돈 것과 같습니다. 그림 ①②가 그 이야기입니다.
· 그래서 두 사진의 자세에서 상대 자세를 계산했습니다. 회전 R 은 사진 2의 회전에 사진 1의 회전을 전치해 곱한 것이고, 이동 t 는 사진 2의 위치에서 R 과 사진 1의 위치를 곱한 값을 뺀 것입니다.
· 그 t 중에서 화면과 나란한 성분이 베이스라인입니다. 회전 {best['rotation_deg']:.1f}도짜리 쌍에서 {best['baseline_m']:.3f} m 가 나왔습니다.
· 이것을 stereoRectify 에 넣으면 두 사진이 평행하게 펴지고, StereoSGBM 이 같은 지점이 옆으로 몇 픽셀 밀렸는지를 찾습니다. 그 픽셀 수가 시차 d 이고, 이 쌍에서는 {best['expected_disparity_px']:.0f} 픽셀입니다.
· 깊이는 Z = f 곱하기 B 나누기 d 입니다. 초점거리 {best['focal_px']:,.0f} 픽셀, 베이스라인 {best['baseline_m']:.3f} m 를 넣습니다.
· 여기서 1픽셀이 바꾸는 깊이가 {best['depth_resolution_m_per_px']*100:.1f} cm 입니다. 이보다 미세한 구조는 원리적으로 구분되지 않습니다.
· 마지막으로 화소마다 X 는 u 빼기 cx 에 Z 나누기 f 를, Y 는 v 빼기 cy 에 Z 나누기 f 를 곱해 3D 좌표로 폅니다.
· 결과는 Z 오차 중앙값 {best['median_abs']*100:.2f} cm 입니다. 위성까지 {best['distance_m']:.1f} m 인데 그 0.13퍼센트입니다. 5 cm 안에 드는 비율은 3장에서 보여 드리겠습니다.
""")

    # ---------------- 2. Unit Test ----------------
    # 과제 요청은 "Unit Test 코드 및 실행 결과 문서화" 다. 설계 설명만 적으면
    # 요구를 반만 채우므로, 실제 테스트 코드 한 조각과 실제 실행 결과를 그대로
    # 싣는다. 둘 다 파일에서 읽어 오므로 코드가 바뀌면 슬라이드도 따라간다.
    sl = new_slide(
        prs, 2, "02 / Unit Test",
        "크기와 자료형만 확인하면, 식이 틀려도 통과합니다",
        "실제 테스트 코드와 실행 결과 · tests/ · outputs/pytest_report.txt")

    add_card(sl, 0.72, 1.70, 7.35, 3.30)
    add_text(sl, 0.98, 1.89, 6.83, 0.24, [
        ("테스트 코드 — tests/test_stereo.py", SANS_SB, 10.5, INK)])
    add_text(sl, 0.98, 2.24, 6.83, 0.22, [
        ("1장에서 말한 '돌아간 각도가 두 번째 눈이 된다' 를 식으로 확인합니다",
         SANS, 10, MUTED)])
    add_text(sl, 0.98, 2.58, 6.83, 2.30,
             [(x, MONO, 9.5, BODY) for x in code_lines()], spacing=1.3)

    add_card(sl, 8.31, 1.70, 4.30, 3.30)
    add_text(sl, 8.57, 1.89, 3.78, 0.24, [
        ("실행 결과 — pytest", SANS_SB, 10.5, INK)])
    add_text(sl, 8.57, 2.24, 3.78, 0.22, [
        ("걸린 시간만 지우고 그대로 저장합니다", SANS, 10, MUTED)])
    add_text(sl, 8.57, 2.58, 3.78, 2.20,
             [(x, MONO, 8.5, BODY) for x in report_lines()], spacing=1.2)

    add_panel(sl, 0.72, 5.18, 11.89, 1.64, "왜 이렇게 짰나", [
        t("공과 상자로 위성을 세우면 정답 거리를 식으로 구할 수 있습니다. "
          "정답에 오차가 없으니, 틀린 만큼이 그대로 알고리즘의 오차입니다."),
        t("그래서 '그림이 그럴듯한가' 가 아니라 '숫자가 맞는가' 로 확인합니다. "
          f"정답 깊이 차이 {syn['gt_span_m']:.2f} m 를 저희는 {ss['span_m']:.2f} m 로 "
          f"되살렸고, 과제 예시 코드는 {se['span_m']:.3f} m 로 납작해졌습니다."),
        tk("이렇게까지 한 이유는, 이번에 찾은 버그 4건이 전부 테스트가 없던 "
           "자리에서 나왔기 때문입니다. 고친 뒤에는 일부러 되돌려 그 테스트가 "
           "진짜 실패하는지까지 확인했습니다."),
    ])
    add_notes(sl, f"""
· 크기와 자료형만 보는 테스트는 계산식이 틀려도 통과합니다.
· 그래서 정답을 손으로 풀 수 있는 장면을 만들어 숫자까지 맞춰 봤습니다.
· 왼쪽이 실제 코드입니다. 1장에서 말한 "각도 차이가 두 번째 눈이 된다" 를 소수점 12자리까지 확인합니다.
· 오른쪽이 실행 결과입니다. {tc}개 전부 통과, outputs 폴더에 그대로 저장돼 있습니다.
· 정답 깊이 차이 {syn['gt_span_m']:.2f} m 를 저희는 {ss['span_m']:.2f} m 로, 과제 예시 코드는 {se['span_m']:.3f} m 로 복원했습니다.
· 이렇게까지 한 이유 — 이번에 찾은 버그 4건이 전부 테스트가 없던 자리에서 나왔습니다.
""")

    # ---------------- 3. 2D -> 3D 변환 결과 ----------------
    sl = new_slide(
        prs, 3, "03 / 2D → 3D 변환 결과",
        "앞면은 정확하고, 뒷면이 비어 있습니다",
        f"왼쪽부터 정답 · 사진 2장으로 만든 결과 · "
        f"사진 {cov_fuse['pairs_used']}쌍을 합친 결과")
    add_image(sl, os.path.join(OUT, "04_pointclouds_slide.png"),
              0.72, 1.70, 11.89, 3.16)
    add_card(sl, 0.72, 5.02, 4.30, 1.86)
    add_text(sl, 0.98, 5.21, 3.78, 0.24, [("결과", SANS_SB, 10.5, INK)])
    add_matrix(sl, 0.98, 5.60, [2.34, 1.44], [
        (("Z 오차 (중앙값)", f"{best['median_abs']*100:.2f} cm"), "key"),
        (("5 cm 안에 든 비율", f"{best['within_5cm']*100:.1f}%"), "key"),
        (("과제 예시 코드보다", f"{gap_ratio:.1f}배 정확"), "key"),
        (("겉면을 덮은 비율 · 2장", f"{cov_one['surface_coverage']*100:.1f}%"), "body"),
        ((f"겉면을 덮은 비율 · {cov_fuse['pairs_used']}쌍",
          f"{fuse0['surface_coverage']*100:.1f}%"), "body"),
    ])
    # 카드 제목은 "무엇을 보라" 가 아니라 "여기서 하려는 말" 이어야 한다.
    # 각 줄도 결론이 되는 말을 앞에 두고 근거를 뒤에 붙인다.
    add_panel(sl, 5.26, 5.02, 7.35, 1.86,
              "깊이는 정확합니다 · 남은 문제는 덮는 범위입니다", [
        tk(f"Z 는 정확합니다 — 100번 중 {best['within_5cm']*100:.0f}번이 "
           f"5 cm 안에 들어옵니다."),
        tk("뒷면은 원리적으로 안 보입니다 — 알고리즘이 부족해서가 아니라 "
           "한 방향에서는 가려지기 때문입니다."),
        tk(f"넓히면 정확도를 내줍니다 — {cov_fuse['pairs_used']}쌍을 합쳐 겉면 "
           f"{cov_one['surface_coverage']*100:.0f} → "
           f"{fuse0['surface_coverage']*100:.0f}%, 정확한 점 "
           f"{cov_one['precision']*100:.0f} → {fuse0['precision']*100:.0f}%."),
    ])
    add_notes(sl, f"""
· 왼쪽부터 정답, 사진 2장으로 만든 결과, 사진 {cov_fuse['pairs_used']}쌍을 합친 결과입니다.
· 거리는 정확합니다. 오차 {best['median_abs']*100:.2f} cm, 100번 중 {best['within_5cm']*100:.0f}번이 5 cm 안입니다. 과제 예시 코드보다 {gap_ratio:.1f}배 정확합니다.
· 대신 가운데를 보시면 한쪽 면만 있습니다. 겉면 전체로는 {cov_one['surface_coverage']*100:.0f}% 입니다.
· 알고리즘이 부족해서가 아니라 한 방향에서는 뒷면이 가려서 안 보이기 때문입니다.
· 그래서 사진을 더 모아 {cov_fuse['pairs_used']}쌍을 합쳐 {fuse0['surface_coverage']*100:.0f}% 까지 올렸습니다.
· 대신 정확한 점의 비율이 {cov_one['precision']*100:.0f} 에서 {fuse0['precision']*100:.0f}% 로 떨어집니다. 둘을 동시에 얻을 수는 없었습니다.
""")

    # ---------------- 4. 개선점 ----------------
    # 이미 고친 것은 3 장까지의 수치에 반영돼 있다. 여기 또 적으면 "무엇이
    # 남았는가" 가 흐려지므로 남은 것만 둔다. 항목마다 지금 무엇이 문제인지를
    # 측정값으로 먼저 대고, 그래서 무엇을 할 것인지를 붙인다. 왜 그런지는
    # 슬라이드에 적지 않고 대본에서 말한다.
    sl = new_slide(
        prs, 4, "04 / 개선점",
        "실제로 쓰려면 남은 것 네 가지",
        "각 항목의 앞줄이 지금 잰 값입니다 · outputs/metrics.json 에 그대로 남습니다")

    p025 = next(r for r in posesens if abs(r["degrees"] - 0.25) < 1e-9)
    items = [
        ("1", "자세가 조금만 틀려도 무너집니다",
         f"자세에 0.25도 오차를 주면 5 cm 안에 드는 비율이 "
         f"{best['within_5cm']*100:.1f}% 에서 {p025['within_5cm']*100:.1f}% 로 "
         f"떨어집니다.",
         "→ 자세를 그대로 받지 말고, 영상의 대응점으로 상대 자세를 다시 맞춰야 "
         "합니다."),
        ("2", "보는 방향이 한쪽에 몰려 있습니다",
         f"시점 간 최대각은 {spread['max_angle_deg']:.0f}도인데 구면 "
         f"{spread['sphere_cells_total']}칸 중 {spread['sphere_cells_filled']}칸에만 "
         f"있고, 상위 2쌍이 겉면의 {top2_share*100:.0f}% 를 만듭니다.",
         "→ 조건에 맞는 사진을 전부 쓰지 말고, 빈 방향을 채우는 쪽으로 골라야 "
         "합니다."),
        ("3", "넓게 덮으면 정확한 점의 비율이 떨어집니다",
         f"{cov_fuse['pairs_used']}쌍을 합치면 겉면은 "
         f"{cov_one['surface_coverage']*100:.0f} → {fuse0['surface_coverage']*100:.0f}% "
         f"로 늘지만 정확한 점의 비율이 {cov_one['precision']*100:.0f} → "
         f"{fuse0['precision']*100:.0f}% 로 떨어집니다.",
         "→ 값을 버리거나 남기거나로 나누지 말고, 점마다 신뢰도를 함께 내야 "
         "합니다."),
        ("4", "화면의 대부분이 빈 우주입니다",
         f"위성은 화면의 일부만 차지하는데 전체를 그대로 봅니다. 깊이 분해능이 "
         f"{best['depth_resolution_m_per_px']*100:.1f} cm/픽셀에서 막힙니다.",
         "→ 위성이 있는 부분만 잘라 같은 계산량으로 더 조밀하게 봐야 합니다."),
    ]

    add_card(sl, 0.72, 1.70, 11.89, 5.12)
    y = 2.14
    for num, head, now, todo in items:
        add_text(sl, 0.98, y, 0.40, 0.28, [(num, SANS_SB, 15, FAINT)])
        add_text(sl, 1.42, y, 10.93, 0.28, [(head, SANS_SB, 14, INK)],
                 tracking=-14 * 0.02)
        add_text(sl, 1.42, y + 0.40, 10.93, 0.46,
                 [(now, SANS, 11, BODY), (todo, SANS_MD, 11, INK)], spacing=1.3)
        y += 1.24

    add_notes(sl, f"""
· 실제로 쓰려면 남은 것 네 가지입니다. 앞줄이 지금 잰 값이고 뒷줄이 할 일입니다.
· 1번이 가장 큽니다. 지금은 위성의 자세를 데이터셋이 준 정답으로 받아 씁니다. 실제 상대항법에서는 그 자세도 추정해야 합니다.
· 그래서 자세를 일부러 조금씩 틀리게 넣어 봤습니다. 0.25도만 틀려도 5 cm 안에 드는 비율이 {best['within_5cm']*100:.0f}에서 {p025['within_5cm']*100:.0f}퍼센트로 무너집니다.
· 이유는 베이스라인이 짧아서입니다. 0.25도를 돌리면 5.6 m 거리에서 타겟이 2.5 cm 옮겨 앉는데, 베이스라인이 {best['baseline_m']*100:.0f} cm 뿐이라 그 7퍼센트가 그대로 깊이 배율 오차가 됩니다.
· 그래서 자세를 그대로 믿지 말고, 두 사진의 대응점으로 상대 자세를 다시 맞추는 단계가 필요합니다.
· 2번은 3장에서 사진 {cov_fuse['pairs_used']}쌍을 합쳐 {fuse0['surface_coverage']*100:.0f}퍼센트를 덮었다고 말씀드린 것의 원인입니다. 시점 간 최대각은 {spread['max_angle_deg']:.0f}도로 넓어 보이지만 구면 {spread['sphere_cells_total']}칸 중 {spread['sphere_cells_filled']}칸에만 있습니다.
· 조건에 맞는 사진을 전부 쓰다 보니 그렇습니다. 실제로 상위 두 쌍이 겉면의 {top2_share*100:.0f}퍼센트를 만들고 나머지는 이미 덮은 데를 또 덮습니다. 빈 방향을 채우는 쪽으로 고르면 같은 장수로 더 넓게 덮을 수 있습니다.
· 3번은 넓게 덮는 것과 정확한 것이 맞바뀌는 문제입니다. 지금은 자신 없는 값을 그냥 버립니다. 그래서 "모른다" 와 "틀렸다" 가 구분되지 않습니다. 점마다 신뢰도를 함께 내면 쓰는 쪽에서 골라 쓸 수 있습니다.
· 4번은 정렬한 화면의 대부분이 빈 우주라는 것입니다. 위성이 있는 부분만 잘라 쓰면 같은 계산량으로 더 조밀하게 볼 수 있습니다. 지금 깊이 분해능이 {best['depth_resolution_m_per_px']*100:.1f} 센티미터 퍼 픽셀에서 막혀 있습니다.
· 이 밖에 재 봤지만 쓰지 않은 것도 있습니다. 시차를 찾는 범위를 좁히면 빈 곳은 줄지만 5 cm 안이 {narrow['current']['within_5cm']*100:.1f}에서 {narrow['narrowed']['within_5cm']*100:.1f}퍼센트로 떨어져서, 잰 값만 남겼습니다.
· 한계도 말씀드리면 위성은 한 종류만 실험했습니다.
""")


def talk_script(width=46):
    """모아 둔 할 말을 대본 txt 로 조립한다."""
    import textwrap

    titles = ["01 / 방법", "02 / Unit Test", "03 / 2D → 3D 변환 결과",
              "04 / 개선점"]
    out = ["2차 업무 발표 대본 — 정승원",
           f"슬라이드 {TOTAL_PAGES}장 · 각 장에서 할 말", "=" * 50, ""]
    for i, note in enumerate(TALK):
        out += [f"[{i + 1}] {titles[i]}", "-" * 50]
        for line in note.splitlines():
            line = line.strip()
            if line:
                out += textwrap.wrap(line, width, subsequent_indent="  ")
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
