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

TOTAL_PAGES = 3


def set_font(run, name, size, color, bold=False):
    run.font.name = name
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color
    # 한글이 기본 글꼴로 떨어지지 않도록 동아시아 글꼴도 지정한다.
    rPr = run._r.get_or_add_rPr()
    ns = "{http://schemas.openxmlformats.org/drawingml/2006/main}"
    for tag in ("ea", "cs"):
        rPr.append(rPr.makeelement(ns + tag, {"typeface": name}))


def style_paragraph(para, name, size, color):
    """문단과 그 안의 런에 모두 글꼴을 지정한다.

    런에만 지정하면 텍스트가 빈 문단은 기본 크기(18pt)로 높이를 잡아
    줄 간격이 어긋난다. 표에서 빈 칸이 한 행씩 밀리는 원인이었다.
    """
    para.font.name = name
    para.font.size = Pt(size)
    para.font.color.rgb = color
    run = para.add_run()
    set_font(run, name, size, color)
    return run


def add_text(slide, x, y, w, h, lines, align=PP_ALIGN.LEFT, spacing=1.35):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0

    for i, (text, name, size, color) in enumerate(lines):
        para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        para.alignment = align
        para.line_spacing = spacing
        style_paragraph(para, name, size, color).text = text
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
    add_text(slide, 0.72, 0.74, 11.89, 0.52, [(title, SANS_SB, 27, INK)], spacing=1.0)
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
    syn = s["synthetic_validation"]
    ss, se = syn["stereo"], syn["example_code"]
    sur = s["spe3r_pair_survey"]
    best = s["best_pair"]
    bex = s["best_pair_example_code"]

    # ---------------- 1. 방법 ----------------
    sl = new_slide(
        prs, 1, "01 / 방법",
        "카메라가 고정이어도 타겟이 돌면 스테레오가 된다",
        "SPE3R aqua · 위성 근접 영상 1,000장 · 256×256 · Stanford SLAB · CC BY-NC-SA 4.0")
    add_image(sl, os.path.join(OUT, "00_concept.png"), 0.72, 1.78, 11.89, 2.75)
    add_panel(sl, 0.72, 4.78, 5.86, 2.07, "왜 시점이 두 개 필요한가", [
        b("영상 한 장은 깊이를 잃습니다. 한 시선 위의 모든 점이"),
        b("같은 화소에 맺히기 때문입니다."),
        gap(),
        k("SPE3R은 카메라가 제자리에 고정돼 있습니다."),
        k("대신 타겟이 매 프레임 무작위로 회전합니다."),
        b("이 회전이 카메라를 궤도에 올린 것과 같은 효과를 냅니다."),
    ])
    add_panel(sl, 6.96, 4.78, 5.65, 2.07, "파이프라인 5단계", [
        b("① 쌍 선별 — 회전 8° 이내, 옆으로 움직인 쌍만"),
        b("② 정렬 — cv2.stereoRectify 로 에피폴라선을 가로로"),
        b("③ 시차 — cv2.StereoSGBM 으로 가로 이동량 d 탐색"),
        k("④ 깊이 맵 — Z = f · B / d"),
        k("⑤ 포인트 클라우드 — X = (u−cx)·Z/f,  Y = (v−cy)·Z/f"),
    ])
    add_notes(sl, f"""
이번 과제는 위성 근접 영상에서 깊이 맵을 만들고 3D 포인트 클라우드로 바꾸는 것입니다.

영상 한 장으로는 깊이를 알 수 없습니다. 한 시선 위에 있는 점들이 전부 같은 화소에
맺히기 때문입니다. 그래서 시점이 두 개 필요합니다.

그런데 쓴 데이터셋은 카메라가 제자리에 고정돼 있습니다. 왼쪽 그림처럼 병진이 항상
(0, 0, Z)라서, 카메라 좌표계만 보면 베이스라인이 없습니다.

대신 타겟이 매 프레임 무작위로 회전합니다. 가운데 그림처럼 타겟을 기준으로 보면
카메라가 궤도를 돈 것과 같습니다. 두 뷰의 상대 자세를 계산하면 회전 {best['rotation_deg']:.2f}도짜리
쌍에서 베이스라인 {best['baseline_m']:.3f} 미터가 나옵니다.

이걸 평행 정렬하면 오른쪽처럼 표준 삼각측량 공식 Z = f 곱하기 B 나누기 d 를
그대로 쓸 수 있습니다. 시차 1픽셀이 깊이 {best['depth_resolution_m_per_px']*100:.1f} 센티미터에 해당합니다.
""")

    # ---------------- 2. 검증 ----------------
    sl = new_slide(
        prs, 2, "02 / 검증",
        f"정답 깊이 폭 {syn['gt_span_m']:.2f} m 를 {ss['span_m']:.2f} m 로 복원",
        "구와 직육면체로 위성을 세우고 광선 교차를 해석적으로 풀면 정답 깊이에 "
        "오차가 없다 · 남는 오차는 전부 정합에서 온다")
    add_image(sl, os.path.join(OUT, "01_synthetic_validation.png"),
              0.72, 1.78, 11.89, 2.75)
    add_panel(sl, 0.72, 4.78, 5.86, 2.07, "같은 영상, 두 가지 방법", [
        b("왼쪽·오른쪽 영상을 만들고 두 방법으로 깊이를 구했습니다."),
        b("과제 예시 코드에는 정답에 맞춘 최적 정렬까지 해 줬습니다."),
    ])
    add_matrix(sl, 0.98, 5.70, [1.55, 1.25, 1.30, 1.05], [
        (("", "깊이 폭", "오차 중앙값", "5cm 이내"), "head"),
        (("정답", f"{syn['gt_span_m']:.3f} m", "—", "—"), "body"),
        (("스테레오", f"{ss['span_m']:.3f} m", f"{ss['median_abs']*100:.1f} cm",
          f"{ss['within_5cm']*100:.1f}%"), "key"),
        (("과제 예시", f"{se['span_m']:.3f} m", f"{se['median_abs']*100:.1f} cm",
          f"{se['within_5cm']*100:.1f}%"), "body"),
    ])
    add_panel(sl, 6.96, 4.78, 5.65, 2.07, "Unit Test 65개로 수식을 검증", [
        b("출력 크기와 자료형만 보면 수식이 틀려도 통과합니다."),
        b("손으로 풀 수 있는 조건을 만들어 수치까지 확인했습니다."),
        gap(),
        k("해석해 — Z = f·B/d 를 1e-12 까지 대조"),
        k("불변식 — B와 d를 함께 2배 하면 Z 는 그대로"),
        k("실패 특성화 — 같은 거리라도 반사율이 다르면 4배 다른 깊이"),
        b("경계 조건 — 시차 0(무한원점), None 입력, 크기 불일치"),
    ])
    add_notes(sl, f"""
실제 데이터에 적용하기 전에, 정답을 아는 조건에서 파이프라인이 맞는지 먼저 확인했습니다.

구와 직육면체로 위성을 세우면 광선과 도형의 교차를 손으로 풀 수 있습니다. 그래서
정답 깊이에 렌더링 오차가 전혀 없고, 남는 오차는 전부 정합 알고리즘에서 온 것입니다.

오른쪽 세 장을 비교해 주십시오. 네 번째가 정답 깊이인데, 태양전지판 왼쪽 끝과
오른쪽 끝이 {syn['gt_span_m']:.2f} 미터 차이 납니다. 다섯 번째 스테레오 복원이 그 차이를
{ss['span_m']:.2f} 미터로 되살립니다. 마지막 과제 예시 코드는 {se['span_m']:.3f} 미터,
사실상 평면 하나입니다.

오차 중앙값은 {ss['median_abs']*100:.1f} 센티미터 대 {se['median_abs']*100:.1f} 센티미터,
5센티미터 안에 든 화소는 {ss['within_5cm']*100:.1f} 퍼센트 대 {se['within_5cm']*100:.1f} 퍼센트입니다.
예시 코드에는 정답에 맞춘 최적 정렬까지 해 준 결과입니다.

Unit Test 는 65개입니다. 출력 크기와 자료형만 보면 수식이 틀려도 통과하기 때문에,
손으로 풀 수 있는 조건을 만들어 수치까지 대조했습니다.
""")

    # ---------------- 3. 결과와 한계 ----------------
    sl = new_slide(
        prs, 3, "03 / 결과와 한계",
        f"실제 위성 영상에서 깊이 오차 중앙값 {best['median_abs']*100:.1f} cm",
        "기준 깊이 = 동봉 메시 40만 점을 z-buffer 로 투영 · "
        "대조군에는 정답에 맞춘 최적 정렬을 적용해 유리한 조건을 부여")
    add_image(sl, os.path.join(OUT, "03_spe3r_stereo.png"), 0.72, 1.78, 11.89, 2.55)
    add_card(sl, 0.72, 4.56, 6.62, 2.29)
    add_text(sl, 0.98, 4.75, 6.10, 0.24, [("결과", SANS_SB, 10.5, INK)])
    add_matrix(sl, 0.98, 5.05, [1.55, 1.40, 1.40, 1.15], [
        (("", "RMSE", "오차 중앙값", "5cm 이내"), "head"),
        (("스테레오", f"{best['rmse']:.4f} m", f"{best['median_abs']*100:.1f} cm",
          f"{best['within_5cm']*100:.1f}%"), "key"),
        (("과제 예시", f"{bex['rmse']:.4f} m", f"{bex['median_abs']*100:.1f} cm",
          f"{bex['within_5cm']*100:.1f}%"), "body"),
    ])
    add_text(sl, 0.98, 5.85, 6.10, 0.92, [
        k(f"포인트 클라우드 {s['best_pair_points']:,}점 · 메시 대비 Chamfer "
          f"{s['best_pair_chamfer_pred_to_gt']:.4f}  →"),
        gap(),
        b(f"한계 — 쓸 만한 쌍이 후보 20개 중 {sur['pairs_within_10cm']}개, "
          f"유효화소 {best['valid_ratio']*100:.0f}%, 단일 시점이라 뒷면은 복원 불가,"),
        b(f"정답 자세를 그대로 사용, 깊이 분해능 하한 "
          f"{best['depth_resolution_m_per_px']*100:.1f} cm/px (dZ = Z²/fB)"),
    ], spacing=1.3)
    add_image(sl, os.path.join(OUT, "04_pointclouds.png"), 7.58, 4.56, 5.03, 2.29)
    add_notes(sl, f"""
실제 SPE3R 위성 영상 결과입니다.

왼쪽 두 장이 정렬된 스테레오 쌍입니다. 배경에 지구가 보이고, 타겟이 {best['rotation_deg']:.2f}도
돌아간 두 시점입니다. 세 번째가 찾아낸 시차이고, 네 번째가 동봉된 메시로 만든 기준 깊이입니다.

네 번째와 다섯 번째를 비교해 주십시오. 기준 깊이는 위쪽 태양전지판이 노랗고 아래
본체로 갈수록 파래지는 그라데이션인데, 스테레오 복원이 그 구조를 그대로 잡아냅니다.
반면 맨 오른쪽 과제 예시 코드는 거의 균일한 초록입니다. 깊이 정보가 없다는 뜻입니다.

수치로는 오차 중앙값 {best['median_abs']*100:.1f} 센티미터, 5센티미터 이내가 {best['within_5cm']*100:.1f} 퍼센트입니다.
예시 코드는 {bex['median_abs']*100:.1f} 센티미터, {bex['within_5cm']*100:.1f} 퍼센트입니다.

이 깊이 맵을 역투영하면 오른쪽 아래 포인트 클라우드가 나옵니다. {s['best_pair_points']:,}점이고
정답 메시 대비 Chamfer 거리는 {s['best_pair_chamfer_pred_to_gt']:.4f} 입니다.

한계도 말씀드리겠습니다. 조건을 만족하는 쌍이 후보 20개 중 {sur['pairs_within_10cm']}개뿐이고,
유효 화소는 {best['valid_ratio']*100:.0f} 퍼센트입니다. 단일 시점이라 타겟 뒷면은 원리적으로
복원되지 않고, 자세는 데이터셋이 준 정답을 그대로 썼습니다. 실제 상대항법에서는
자세도 추정해야 하고 그 오차가 삼각측량에 미치는 영향은 이번에 측정하지 않았습니다.
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
