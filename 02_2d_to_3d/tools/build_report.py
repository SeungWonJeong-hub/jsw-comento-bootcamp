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


def add_notes(slide, text):
    """발표자 노트. PowerPoint 발표자 보기에서 보인다."""
    tf = slide.notes_slide.notes_text_frame
    tf.text = text.strip()


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

        전문 용어는 풀어 쓴다. '시차' 대신 '두 사진에서 밀린 거리', '정합'
        대신 '같은 지점 찾기', 'blockSize' 대신 '오려서 비교하는 네모의
        크기' 로 쓴다. 듣는 사람이 용어를 해석하는 동안 다음 문장을 놓친다.

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
    blk_new = next(x for x in blocks
                   if x["best_pair"]["valid_ratio"] == best["valid_ratio"])
    gap_ratio = bex["median_abs"] / best["median_abs"]

    # ---------------- 1. 방법 ----------------
    sl = new_slide(
        prs, 1, "01 / 방법",
        "카메라는 그대로인데, 위성이 돌아서 깊이를 쟀습니다",
        "SPE3R 위성 근접 영상 1,000장 · 256×256 · Stanford SLAB")
    add_image(sl, os.path.join(OUT, "00_concept.png"), 0.72, 1.70, 11.89, 3.16)
    add_panel(sl, 0.72, 5.02, 5.86, 1.86, "왜 사진이 두 장 필요한가", [
        t("사진 한 장에는 멀고 가까움이 남지 않습니다."),
        t("한 방향으로 늘어선 점이 전부 같은 자리에 찍히기 때문입니다."),
        t("그래서 떨어진 두 곳에서 찍습니다. 가까운 것일수록 많이 "
          "밀리고, 그 밀린 정도가 곧 거리입니다."),
    ])
    add_panel(sl, 6.96, 5.02, 5.65, 1.86, "그런데 이 데이터는 카메라가 고정입니다", [
        t("옆으로 1 mm 도 움직이지 않아, 그대로면 두 번째 눈이 없습니다."),
        tk("대신 위성이 매 장면 무작위로 돌아갑니다."),
        t(f"위성을 가만히 놓고 보면 카메라가 돈 셈이라, 옆에서 "
          f"{best['baseline_m']*100:.0f} cm 떨어져 찍은 효과가 나옵니다."),
    ])
    add_notes(sl, f"""
사진 한 장으로는 멀고 가까움을 알 수 없습니다. 한 방향으로 늘어선 점들이 전부 같은
자리에 찍히기 때문입니다. 사람도 눈이 둘이라 거리를 느끼는 것과 같습니다. 그래서
서로 다른 위치에서 찍은 사진이 두 장 필요합니다. 가까운 것일수록 두 사진 사이에서
많이 밀리는데, 그 밀린 정도로 거리를 계산합니다.

그런데 이 데이터셋은 카메라가 제자리에 고정돼 있습니다. 위치 값 천 개를 확인해 보니
옆으로 1밀리미터도 움직이지 않습니다. 교과서대로면 여기서는 이 방법을 쓸 수 없습니다.

핵심은 카메라가 움직였느냐가 아니라, 카메라와 위성 사이의 관계가 달라졌느냐입니다.
이 데이터셋은 위성이 매 장면 무작위로 돌아갑니다. 위성을 가만히 놓고 보면 카메라가
그 주위를 돈 것과 같습니다. 두 장의 각도 차이를 계산하면 옆에서
{best['baseline_m']*100:.0f}센티미터 떨어져 찍은 것과 같은 효과가 나옵니다.
회전 {best['rotation_deg']:.1f}도짜리 사진 쌍에서 나온 값입니다.

그 다음은 교과서 그대로입니다. 두 사진을 나란히 펴고, 같은 지점이 옆으로 얼마나
밀렸는지 찾고, 그 값으로 거리를 계산합니다. 1픽셀 차이가
{best['depth_resolution_m_per_px']*100:.1f}센티미터에 해당합니다.
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
과제 예시의 테스트는 결과의 크기와 자료형만 확인합니다. 그것만으로는 계산식이 틀려도
통과합니다. 그래서 정답을 손으로 풀 수 있는 조건을 만들어 숫자까지 맞춰 봤습니다.

왼쪽이 실제 테스트 코드 한 조각입니다. 앞 장에서 카메라는 고정인데 위성이 돌아서
두 번째 눈이 생긴다고 말씀드렸는데, 그 말이 진짜인지 확인하는 테스트입니다. 위성에
찍힌 점 200개를 두 장면의 좌표로 각각 옮겨 놓고, 저희가 계산한 각도 차이로 한쪽을
다른 쪽으로 옮겼을 때 정확히 겹치는지를 봅니다. 소수점 열두 자리까지 맞아야
통과합니다.

오른쪽이 실행 결과입니다. 테스트 {tc}개가 전부 통과합니다. 이 파일은 outputs 폴더에
그대로 저장돼 있습니다. 걸린 시간만 지웠는데, 실행할 때마다 달라지는 값이라 결과
파일이 매번 바뀌는 것을 막기 위해서입니다.

아래가 왜 이렇게 짰는지입니다. 공과 상자로 위성 모양을 세우면 거리를 식으로 정확히
구할 수 있습니다. 정답에 오차가 전혀 없으니 틀린 만큼이 곧 알고리즘의 오차입니다.
정답 깊이 차이 {syn['gt_span_m']:.2f}미터를 저희는 {ss['span_m']:.2f}미터로 되살렸고,
과제 예시 코드는 {se['span_m']:.3f}미터, 사실상 납작한 판 하나였습니다.

테스트는 크게 세 가지를 봅니다. 식이 맞는지, 틀린 방법이 왜 틀리는지, 그리고 한 번 난
버그가 다시 나지 않는지입니다. 마지막이 중요했습니다. 이번에 버그를 네 건 찾았는데
전부 테스트가 없던 자리에서 나왔습니다. 고친 뒤에는 일부러 다시 망가뜨려서 그 테스트가
진짜로 실패하는지까지 확인했습니다.
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
        (("거리 오차 (중앙값)", f"{best['median_abs']*100:.2f} cm"), "key"),
        (("5 cm 안에 든 비율", f"{best['within_5cm']*100:.1f}%"), "key"),
        (("과제 예시 코드보다", f"{gap_ratio:.1f}배 정확"), "key"),
        (("겉면을 덮은 비율 · 2장", f"{cov_one['surface_coverage']*100:.1f}%"), "body"),
        ((f"겉면을 덮은 비율 · {cov_fuse['pairs_used']}쌍",
          f"{fuse0['surface_coverage']*100:.1f}%"), "body"),
    ])
    add_panel(sl, 5.26, 5.02, 7.35, 1.86, "이렇게 읽어 주십시오", [
        t(f"거리는 정확합니다. 100번 중 {best['within_5cm']*100:.0f}번이 "
          f"5 cm 안에 들어옵니다."),
        tk("대신 가운데 그림을 보시면 한쪽 면만 있습니다. 알고리즘이 부족해서가 "
           "아니라, 한 방향에서는 뒷면이 가려서 안 보이기 때문입니다."),
        t(f"그래서 사진을 더 모아 {cov_fuse['pairs_used']}쌍을 합쳤습니다. 오른쪽처럼 "
          f"반대편이 채워져 {fuse0['surface_coverage']*100:.0f}% 가 되지만, "
          f"대신 정확한 점의 비율이 떨어집니다."),
    ])
    add_notes(sl, f"""
2D 사진에서 3D 로 바꾼 결과입니다. 세 장을 왼쪽부터 봐 주십시오.

첫 번째가 정답 모양입니다. 두 번째가 사진 두 장으로 만든 결과, 점
{cov_one['n_points']:,}개입니다. 거리는 정확합니다. 오차 중앙값이
{best['median_abs']*100:.2f}센티미터이고, 100번 중 {best['within_5cm']*100:.0f}번은
5센티미터 안에 들어옵니다. 앞 장에서 보신 과제 예시 코드보다
{gap_ratio:.1f}배 정확합니다.

그런데 정답과 비교해 보시면 한쪽 면만 있습니다. 위성 겉면 전체로 따지면
{cov_one['surface_coverage']*100:.0f}퍼센트입니다. 알고리즘이 부족해서가 아니라, 한
방향에서 보면 뒷면이 가려서 안 보이기 때문입니다. 아무리 잘 만들어도 절반을 넘길 수
없습니다.

그래서 사진을 더 모았습니다. 조건을 풀어 {cov_fuse['candidates']}쌍까지 후보를 늘리면
{cov_fuse['pairs_used']}쌍이 복원되고, 다 합치면 세 번째 그림이 됩니다. 반대편이 채워져
{fuse0['surface_coverage']*100:.0f}퍼센트가 됩니다.

다만 공짜는 아닙니다. 정확한 점의 비율이 {cov_one['precision']*100:.0f}에서
{fuse0['precision']*100:.0f}퍼센트로 떨어집니다. 사진이 늘면 서로 어긋난 점도 같이 늘기
때문입니다. 넓게 덮는 것과 정확한 것을 동시에 얻을 수는 없었고, 어느 쪽을 쓸지는
용도가 정할 문제라고 봅니다.
""")

    # ---------------- 4. 개선점 ----------------
    # 이 장은 그림을 넣지 않는다. 쌍별 산점도는 한 눈에 읽히지 않아 발표에서
    # 설명이 그림을 따라가는 모양이 된다. 보여야 하는 것은 "설정 하나만 바꿔
    # 가며 잰 표" 이므로 표를 크게 놓는 편이 낫다.
    sl = new_slide(
        prs, 4, "04 / 개선점",
        "가짜 영상에서 고른 설정을, 실제 영상에서 다시 골랐습니다",
        "재 본 것과 아직 안 해 본 것을 나눠서 적었습니다")

    add_panel(sl, 0.72, 1.70, 7.35, 2.85, "무엇이 문제였나", [
        t("두 사진에서 같은 지점을 찾을 때, 점 하나만 보면 어디가 어딘지"),
        t("알 수 없어서 주변을 네모나게 오려 통째로 비교합니다."),
        gap(),
        t("그 네모의 크기를 3으로 두고 있었는데, 근거가 2장에서 보여 드린"),
        t("가짜(합성) 영상이었습니다. 무늬가 많아 작은 네모로도 잘 찾아지고,"),
        t("작을수록 경계가 선명해서 3이 제일 좋았습니다."),
        gap(),
        tk("그런데 실제 위성은 금속이라 무늬가 거의 없습니다. 작은 네모로는"),
        tk("아예 못 찾습니다. 실제 영상으로 3부터 17까지 다시 재서 11로 바꿨습니다."),
    ])
    add_card(sl, 8.31, 1.70, 4.30, 2.85)
    add_text(sl, 8.57, 1.89, 3.78, 0.24,
             [("바꾼 뒤 (같은 사진, 같은 기준)", SANS_SB, 10.5, INK)])
    add_matrix(sl, 8.57, 2.32, [1.70, 1.04, 1.04], [
        (("", "전", "후"), "head"),
        (("거리 오차", f"{blk_old['best_pair']['median_abs']*100:.2f} cm",
          f"{blk_new['best_pair']['median_abs']*100:.2f} cm"), "key"),
        (("5 cm 안", f"{blk_old['best_pair']['within_5cm']*100:.1f}%",
          f"{blk_new['best_pair']['within_5cm']*100:.1f}%"), "key"),
        (("값이 나온 픽셀", f"{blk_old['best_pair']['valid_ratio']*100:.1f}%",
          f"{blk_new['best_pair']['valid_ratio']*100:.1f}%"), "key"),
        (("쓸 수 있는 쌍", f"{blk_old['pairs_reconstructed']}쌍",
          f"{blk_new['pairs_reconstructed']}쌍"), "body"),
    ])
    add_text(sl, 8.57, 3.72, 3.78, 0.60, [
        ("하나를 내주고 얻은 것이 아니라", SANS, 11, BODY),
        ("전부 같이 좋아졌습니다.", SANS_MD, 11, INK),
    ])

    add_panel(sl, 0.72, 4.72, 5.86, 2.10, "해 봤지만 쓰지 않은 것", [
        t("밀린 거리를 찾는 범위도 좁혀 봤습니다. 빈 곳은 줄어드는데,"),
        t(f"5 cm 안에 드는 비율이 {narrow['current']['within_5cm']*100:.1f}% 에서 "
          f"{narrow['narrowed']['within_5cm']*100:.1f}% 로 떨어졌습니다."),
        gap(),
        tk("이번엔 하나를 얻고 하나를 내주는 쪽이라 그대로 두고,"),
        tk("잰 값만 남겨 뒀습니다."),
    ])
    add_panel(sl, 6.75, 4.72, 5.86, 2.10, "아직 안 해 본 것", [
        tk("① 사진을 고를 때 보는 방향이 골고루 퍼지게 고르기"),
        t("　지금은 한쪽에 몰려 있어, 같은 장수로 더 넓게 덮을 수 있습니다."),
        t("② 화면에서 위성이 차지하는 부분만 잘라 더 크게 보기"),
        t("③ 자신 없는 값을 버리는 대신 '자신 없음' 으로 표시해 남기기"),
    ])
    add_notes(sl, f"""
개선점입니다. 재 본 것과 아직 안 해 본 것을 나눠 적었습니다.

가장 크게 바꾼 것부터 말씀드리겠습니다. 두 사진에서 같은 지점을 찾을 때, 점 하나만
보면 어디가 어딘지 알 수 없어서 그 주변을 네모나게 오려서 통째로 비교합니다. 그
네모의 크기를 3으로 두고 있었습니다.

문제는 그 3을 고른 근거가 가짜 영상이었다는 점입니다. 앞 장에서 보여 드린, 정답을
손으로 풀 수 있는 합성 장면입니다. 그 영상은 무늬가 많아서 작은 네모로도 잘 찾아지고,
작을수록 물체 경계가 선명하게 나옵니다. 그래서 3이 제일 좋았습니다.

그런데 실제 위성 영상은 정반대였습니다. 금속 표면이라 무늬가 거의 없어서, 작은
네모로는 어디가 같은 지점인지 아예 못 찾습니다. 가짜 영상에서 고른 값을 실제 영상에
그대로 쓰고 있었던 겁니다.

그래서 사진 스무 쌍을 그대로 두고 네모 크기만 3부터 17까지 바꿔 가며 다시 재서 11을
골랐습니다. 오른쪽 표가 그 결과입니다. 거리 오차가
{blk_old['best_pair']['median_abs']*100:.2f}에서 {blk_new['best_pair']['median_abs']*100:.2f}센티미터,
5센티미터 안에 드는 비율이 {blk_old['best_pair']['within_5cm']*100:.0f}에서
{blk_new['best_pair']['within_5cm']*100:.0f}퍼센트, 값이 나온 픽셀이
{blk_old['best_pair']['valid_ratio']*100:.0f}에서 {blk_new['best_pair']['valid_ratio']*100:.0f}퍼센트가
됐습니다. 쓸 수 있는 사진 쌍도 {blk_old['pairs_reconstructed']}쌍에서
{blk_new['pairs_reconstructed']}쌍으로 늘었습니다. 보통은 하나를 얻으면 하나를 내주는데
이건 전부 같이 좋아졌습니다.

더 키우면 값이 나온 픽셀만 늘고 정확도는 다시 나빠집니다. 그래서 11은 끝값이 아니라
가운데에서 고른 값입니다.

여기서 배운 건 11이라는 숫자가 아니라 순서라고 생각합니다. 가짜 데이터로 고른 설정을
실제 데이터에 그냥 옮기면 안 됩니다. 두 데이터에서 걸리는 지점이 다르기 때문입니다.
가짜 데이터는 식이 맞는지 확인하는 데 쓰고, 설정은 실제로 쓸 데이터에서 골라야
합니다.

왼쪽 아래는 해 봤지만 쓰지 않은 것입니다. 찾는 범위를 좁히면 빈 곳은 줄어드는데,
5센티미터 안에 드는 비율이 {narrow['current']['within_5cm']*100:.1f}에서
{narrow['narrowed']['within_5cm']*100:.1f}퍼센트로 떨어집니다. 넓게 덮는 것과 정확한 것을
맞바꾸는 셈이라 그대로 뒀고 잰 값만 남겨 뒀습니다.

오른쪽 아래는 아직 안 해 본 것입니다. 첫 번째를 가장 유력하게 봅니다. 지금은 조건에
맞는 사진을 전부 쓰는데, 그러다 보니 보는 방향이 한쪽에 몰려 있습니다. 방향이
골고루 퍼지게 고르면 같은 장수로 더 넓게 덮을 수 있을 것으로 봅니다.

마지막으로 한계도 말씀드리면, 위성의 자세는 데이터셋이 준 정답을 그대로 썼고, 위성은
한 종류만 실험했습니다.
""")


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
    return 0


if __name__ == "__main__":
    sys.exit(main())
