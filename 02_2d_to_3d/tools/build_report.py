"""2차 업무 발표자료(PPT 3페이지) 생성기.

1차 업무 자료와 같은 디자인 규격을 따른다.
    배경 #FAFAFA · 카드 흰색 + #EBEBEB 0.75pt 테두리 · 모서리 반경 0.125in
    제목 Pretendard SemiBold 27pt · 본문 Pretendard 9.5pt · 라벨 Cascadia Mono 8.5pt

수치는 outputs/metrics.json 에서 직접 읽는다. 실험을 다시 돌리면 발표자료도
자동으로 갱신되므로 본문과 결과가 어긋날 일이 없다.

사용법
    py -3 run_3d_experiment.py
    py -3 tools/build_report.py
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


def gap():
    return ("", SANS, 4, BODY)


# ---------------------------------------------------------------------------


def build(prs, s):
    """4 페이지. 과제 결과물 형식(업무.pdf p.16)에 맞춰 자리를 나눈다.

        1. 방법          - 베이스라인이 없는 데이터에서 왜 스테레오가 되는가
        2. Unit Test     - 코드 및 실행 결과 문서화 (p.16 요구)
        3. 2D -> 3D 변환 - 변환 결과 이미지 첨부 (p.16 요구)
        4. 개선점        - 요청내용 3번 "개선점을 도출"

    1 번을 뺄까 했지만 남긴다. 카메라 병진이 항상 (0, 0, Z) 인 데이터에서 왜
    삼각측량이 성립하는지를 먼저 깔지 않으면 나머지 세 장이 읽히지 않는다.
    """
    syn = s["synthetic_validation"]
    ss = syn["stereo"]
    sur = s["spe3r_pair_survey"]
    best = s["best_pair"]
    ch = s["best_pair_chamfer"]
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
    cov_both = cov["stereo_plus_carving"]
    carve = s["silhouette_carving"]
    cdm = s["carved_depth_map"]
    gap_ratio = bex["median_abs"] / best["median_abs"]
    ch_ratio = ch["target_to_pred"] / ch["pred_to_target"]

    # ---------------- 1. 방법 ----------------
    sl = new_slide(
        prs, 1, "01 / 방법",
        "카메라가 고정이어도 타겟이 돌면 스테레오가 된다",
        "SPE3R aqua · 위성 근접 영상 1,000장 · 256×256 · Stanford SLAB · CC BY-NC-SA 4.0")
    add_image(sl, os.path.join(OUT, "00_concept.png"), 0.72, 1.70, 11.89, 3.16)
    add_panel(sl, 0.72, 5.02, 5.86, 1.86, "왜 시점이 두 개 필요한가", [
        b("영상 한 장은 깊이를 잃습니다. 한 시선 위의 모든 점이"),
        b("같은 화소에 맺히기 때문입니다."),
        gap(),
        k("SPE3R은 카메라가 제자리에 고정돼 있습니다."),
        k("대신 타겟이 매 프레임 무작위로 회전합니다."),
    ])
    add_panel(sl, 6.96, 5.02, 5.65, 1.86, "파이프라인 5단계", [
        b("① 쌍 선별 — 회전 8° 이내, 옆으로 움직인 쌍만"),
        b("② 정렬 — cv2.stereoRectify 로 에피폴라선을 가로로"),
        b("③ 시차 — cv2.StereoSGBM 으로 가로 이동량 d 탐색"),
        k("④ 깊이 맵 — Z = f · B / d"),
        k("⑤ 포인트 클라우드 — X = (u−cx)·Z/f,  Y = (v−cy)·Z/f"),
    ])
    add_notes(sl, f"""
영상 한 장으로는 깊이를 알 수 없습니다. 한 시선 위의 점이 전부 같은 화소에 맺히기
때문입니다. 그래서 시점이 두 개 필요합니다.

그런데 이 데이터셋은 카메라가 제자리에 고정돼 있습니다. 포즈 라벨 1,000개를 재보니
병진이 항상 0, 0, Z 입니다. 옆으로 1밀리미터도 움직이지 않습니다. 교과서대로면
여기서 삼각측량은 불가능합니다.

삼각측량에 필요한 것은 카메라의 절대 운동이 아니라 카메라와 타겟 사이의 상대
운동입니다. 이 데이터셋은 타겟이 매 프레임 무작위로 회전하므로, 타겟을 고정으로 놓고
보면 카메라가 궤도를 돈 것과 같습니다. 두 뷰의 상대 자세를 계산하면 베이스라인이
드러납니다. 회전 {best['rotation_deg']:.2f}도, 거리 {best['distance_m']:.2f}미터인 쌍에서 유효 베이스라인
{best['baseline_m']:.3f}미터를 얻었습니다.

이것을 cv2.stereoRectify 에 넣으면 평행 정렬된 쌍이 되어 표준 SGBM 을 그대로 쓸 수
있습니다. 시차 1픽셀이 깊이 {best['depth_resolution_m_per_px']*100:.1f}센티미터에 해당합니다.
""")

    # ---------------- 2. Unit Test ----------------
    sl = new_slide(
        prs, 2, "02 / Unit Test",
        "출력 크기와 자료형만 보면 수식이 틀려도 통과합니다",
        f"pytest {tc}개 · 손으로 풀 수 있는 조건을 만들어 수치까지 대조 · "
        "실행 결과는 outputs/pytest_report.txt 에 저장")
    add_image(sl, os.path.join(OUT, "01_synthetic_validation_slide.png"),
              0.72, 1.70, 11.89, 3.16)
    add_panel(sl, 0.72, 5.02, 5.86, 1.86, "정답 깊이에 오차가 없는 조건을 만든다", [
        b("구와 직육면체로 위성을 세우면 광선과 도형의 교차를"),
        b("손으로 풀 수 있습니다. 정답에 렌더링 오차가 0 이므로"),
        b("남는 오차는 전부 정합 알고리즘에서 온 것입니다."),
        gap(),
        k(f"정답 깊이 폭 {syn['gt_span_m']:.3f} m → 스테레오 {ss['span_m']:.3f} m "
          f"/ 과제 예시 {se['span_m']:.3f} m"),
    ])
    add_panel(sl, 6.96, 5.02, 5.65, 1.86, f"{tc}개를 다섯 갈래로", [
        b("해석해 — Z = f·B/d 를 1e-12 까지 대조"),
        b("불변식 — B와 d를 함께 2배 하면 Z 는 그대로"),
        b("실패 특성화 — 같은 거리라도 반사율이 다르면 4배 다른 깊이"),
        b("경계 조건 — 시차 0(무한원점), None 입력, 크기 불일치"),
        k("회귀 — 실제로 찾은 버그를 다시 나지 않게 고정"),
    ])
    add_notes(sl, f"""
과제 예시의 테스트는 출력 크기와 자료형만 확인합니다. 그것만으로는 수식이 틀려도
통과합니다. 그래서 손으로 풀 수 있는 조건을 만들어 수치까지 대조했습니다.

구와 직육면체로 위성을 세우면 광선과 도형의 교차를 해석적으로 풀 수 있습니다.
정답 깊이에 렌더링 오차가 전혀 없으니, 남는 오차는 전부 정합 알고리즘에서 온
것입니다.

그림 세 장을 같은 색 범위로 놓았습니다. 왼쪽이 정답 깊이인데 태양전지판 양 끝이
{syn['gt_span_m']:.2f}미터 차이 납니다. 가운데 스테레오가 그 차이를 {ss['span_m']:.2f}미터로 되살립니다.
오른쪽 과제 예시 코드는 {se['span_m']:.3f}미터, 사실상 평면 하나입니다.

테스트는 네 갈래로 시작했습니다. 해석해 대조, 불변식, 실패 특성화, 경계 조건입니다.
여기에 회귀 갈래를 하나 더 붙였습니다. 파이프라인을 검토하면서 실제로 버그를 여러 건
찾았고 전부 테스트가 없던 영역에서 나왔기 때문입니다. 고친 뒤에는 수정을 일부러
되돌려 해당 테스트가 실제로 실패하는지까지 확인했습니다.
""")

    # ---------------- 3. 2D -> 3D 변환 결과 ----------------
    sl = new_slide(
        prs, 3, "03 / 2D → 3D 변환 결과",
        f"타겟 표면의 {cov_both['surface_coverage']*100:.0f}% 를 복원했습니다",
        "두 경로를 같은 지표로 비교 · 정답 메시 표면 40만 점 중 복원점이 "
        "2 cm 안에 있는 비율")
    add_image(sl, os.path.join(OUT, "04_pointclouds.png"), 0.72, 1.70, 11.89, 3.16)
    add_card(sl, 0.72, 5.02, 5.86, 1.86)
    add_text(sl, 0.98, 5.21, 5.34, 0.24, [("표면 커버리지", SANS_SB, 10.5, INK)])
    add_matrix(sl, 0.98, 5.51, [2.10, 1.15, 1.30], [
        (("", "점 수", "표면 덮음"), "head"),
        (("A 스테레오 (2장)", f"{cov_one['n_points']:,}",
          f"{cov_one['surface_coverage']*100:.1f}%"), "body"),
        (("B 실루엣 카빙 (20뷰)", f"{carve['n_points']:,}",
          f"{carve['surface_coverage']*100:.1f}%"), "body"),
        (("A + B", f"{cov_both['n_points']:,}",
          f"{cov_both['surface_coverage']*100:.1f}%"), "key"),
    ])
    add_panel(sl, 6.96, 5.02, 5.65, 1.86, "같은 시점에서 두 깊이 맵을 비교하면", [
        k(f"A 오차 중앙값 {best['median_abs']*100:.1f} cm · 유효화소 "
          f"{best['valid_ratio']*100:.0f}%   (영상 2장)"),
        k(f"B 오차 중앙값 {cdm['median_abs']*100:.1f} cm · 유효화소 "
          f"{cdm['valid_ratio']*100:.0f}%   (마스크 20장)"),
        gap(),
        b("B 가 낫지만 입력이 열 배 많고 마스크는 정답입니다."),
        b("어느 쪽이 낫다가 아니라 필요한 입력이 다른 두 경로입니다."),
    ])
    add_notes(sl, f"""
2D 에서 3D 로의 변환 결과입니다. 네 장을 왼쪽부터 봐 주십시오.

첫 번째가 정답 메시입니다. 두 번째가 경로 A, 영상 두 장으로 만든 스테레오 복원
{cov_one['n_points']:,}점입니다. 정확합니다 — 깊이 오차 중앙값이 {best['median_abs']*100:.1f}센티미터입니다. 그런데
정답과 비교해 보시면 한쪽 면만 있습니다. 표면의 {cov_one['surface_coverage']*100:.0f}퍼센트입니다.

단일 시점이 뒷면을 못 보는 것은 알고리즘 문제가 아니라 원리 문제입니다. 그래서 시점을
늘려 봤습니다. 쌍 선별 조건을 풀어 후보를 {cov_fuse['candidates']}쌍까지 늘리고 전부 융합해도
{cov_fuse['stages'][0]['surface_coverage']*100:.0f}퍼센트에서 멈춥니다. {cov_fuse['candidates']}쌍 중 {cov_fuse['pairs_used']}쌍만 복원되기 때문입니다. 원인은 쌍 개수가
아니라 무늬 부족이었습니다. 앞 장에서 조명을 뒤집었을 때와 같은 결론입니다.

그래서 세 번째, 경로 B 를 함께 만들었습니다. 실루엣과 자세만으로 복셀을 깎는
visual hull 입니다. 무늬가 필요 없으니 스테레오에 최악인 조건이 오히려 유리합니다.
20뷰로 표면의 {carve['surface_coverage']*100:.0f}퍼센트를 덮습니다.

두 경로의 약점이 상보적입니다. A 는 정확하지만 보이는 면만, B 는 전방위지만 실루엣의
교집합이라 오목한 곳을 못 만듭니다. 합치면 {cov_both['surface_coverage']*100:.0f}퍼센트입니다.

전제가 다르다는 점은 짚어 두겠습니다. B 는 정답 마스크와 정답 자세를 둘 다 씁니다.
A 는 마스크 없이도 동작합니다. 그래서 이 표는 어느 쪽이 낫다는 뜻이 아니라, 덮는
범위와 필요한 입력이 다른 두 경로라는 뜻입니다.

맨 오른쪽은 과제 예시 코드입니다. X, Y 가 픽셀 인덱스이고 Z 가 0에서 255 밝기값이라
세 축의 단위가 서로 다릅니다. 납작한 판으로 나옵니다.
""")

    # ---------------- 4. 개선점 ----------------
    sl = new_slide(
        prs, 4, "04 / 개선점",
        f"스테레오를 전방위로 밀어봤지만 {cov_fuse['stages'][0]['surface_coverage']*100:.0f}% 에서 멈춥니다",
        "재 본 것만 적습니다 · 수치는 outputs/metrics.json 에 그대로 남습니다")
    add_image(sl, os.path.join(OUT, "02_pair_survey.png"), 0.72, 1.70, 11.89, 3.16)
    add_card(sl, 0.72, 5.02, 5.86, 1.86)
    add_text(sl, 0.98, 5.21, 5.34, 0.24,
             [("시차 탐색 범위를 좁히면 (최적 쌍)", SANS_SB, 10.5, INK)])
    add_matrix(sl, 0.98, 5.51, [1.55, 1.35, 1.35], [
        (("", "현재", "좁힘"), "head"),
        (("유효화소", f"{narrow['current']['valid_ratio']*100:.1f}%",
          f"{narrow['narrowed']['valid_ratio']*100:.1f}%"), "key"),
        (("5cm 이내", f"{narrow['current']['within_5cm']*100:.1f}%",
          f"{narrow['narrowed']['within_5cm']*100:.1f}%"), "body"),
    ])
    add_panel(sl, 6.96, 5.02, 5.65, 1.86, "쌍을 늘려도 벽이 있습니다", [
        b(f"후보를 {cov_fuse['candidates']}쌍으로 늘려도 {cov_fuse['pairs_used']}쌍만 복원됩니다."),
        b("일관성 필터를 걸면 정밀도는 오르지만 커버리지가 무너집니다."),
        gap(),
        k("원인은 쌍 개수가 아니라 무늬 부족입니다."),
        b("그래서 전방위는 실루엣 기반으로 갔습니다 (앞 장)."),
    ])
    add_notes(sl, f"""
개선점입니다. 재 본 것만 적었습니다.

가장 큰 약점은 유효화소 {best['valid_ratio']*100:.0f}퍼센트입니다. 나머지는 답을 내지 못한 것이지
맞힌 것이 아닙니다. 앞 장의 정확도 수치는 전부 이 안에서 잰 값입니다.

첫 번째 후보는 시차 탐색 범위입니다. 타겟의 경계 반지름을 알면 깊이가 어느 구간에
있는지 알고, 따라서 시차도 그렇습니다. 최적 쌍에서 물리적으로 가능한 폭은 24픽셀인데
지금은 144픽셀을 훑고 있습니다. 좁혀 보니 유효화소가 {narrow['current']['valid_ratio']*100:.0f}에서 {narrow['narrowed']['valid_ratio']*100:.0f}퍼센트로
올랐습니다.

그런데 공짜가 아닙니다. 다른 쌍에서는 커버리지만 오르고 정확도가 떨어집니다. 탐색
후보가 줄면 uniquenessRatio 검사를 통과하기 쉬워져서, 원래는 기각됐을 애매한 대응이
살아남기 때문입니다. 트레이드오프라 기본값으로 채택하지 않고 측정값만 남겼습니다.

두 번째는 다중 뷰 융합입니다. 지금은 한 쌍만 씁니다. 쓸 만한 쌍이 {sur['pairs_within_10cm']}개 있으니
여러 쌍의 깊이를 합치면 한 쌍에서 비는 부분을 다른 쌍이 채웁니다. 지금 구조에서 가장
싸게 얻을 수 있는 개선이라고 봅니다. 이건 아직 재 보지 않았습니다.

위 그림은 쌍별 결과입니다. 베이스라인이 크다고 정확한 것이 아니고, 커버리지와
정확도가 함께 가지도 않는다는 것이 보입니다.

한계도 함께 말씀드리면, 자세는 데이터셋이 준 정답을 그대로 썼고, 위성은 한 종만
실험했으며, 깊이 분해능 하한이 {best['depth_resolution_m_per_px']*100:.1f} 센티미터 퍼 픽셀입니다.
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

    os.makedirs(REPORT, exist_ok=True)
    out = os.path.join(REPORT, "2차업무_정승원.pptx")
    prs.save(out)
    print(f"저장 완료 -> {out}  ({TOTAL_PAGES} 슬라이드)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
