"""4차 업무 PPT 생성 — 코멘토 4차 업무 / 정승원

    py make_ppt.py --out 4차업무_정승원.pptx

2·3차 업무 PPT 서식을 그대로 따릅니다.
  13.333 x 7.5 in · 배경 #FAFAFA · 카드 #FFFFFF + #EBEBEB 0.75pt 테두리
  제목 Pretendard SemiBold 27pt · 본문 Pretendard · 수치 Cascadia Mono

구성
----
  01 데이터셋    어디서 · 항만 식별 · GSD 검증
  02 파이프라인  받기 → 라벨 → 학습 → 평가 → 웹앱
  03 결과        전체 · 항만별 실측 · 항만 5곳 탐지 예
"""
import os
import argparse

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE

W, H = 13.333, 7.5
INK, BODY, MUTE, FAINT = "171717", "4D4D4D", "8F8F8F", "A1A1A1"
HAIR, CANVAS, CARD = "EBEBEB", "FAFAFA", "FFFFFF"
ACC, WARN = "0070F3", "D4443C"
SANS, SEMI, MED, MONO = ("Pretendard", "Pretendard SemiBold",
                         "Pretendard Medium", "Cascadia Mono")
TOTAL = 3
HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "outputs")


def C(h):
    return RGBColor.from_string(h)


def txt(sl, l, t, w, h, s, font=SANS, size=10, color=BODY,
        align=PP_ALIGN.LEFT, line=1.25):
    tb = sl.shapes.add_textbox(Inches(l), Inches(t), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    for i, ln in enumerate(s.split("\n")):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.line_spacing = line
        p.space_after = Pt(0)
        r = p.add_run()
        r.text = ln
        r.font.name = font
        r.font.size = Pt(size)
        r.font.color.rgb = C(color)
    return tb


def card(sl, l, t, w, h, radius=0.045):
    sh = sl.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,
                             Inches(l), Inches(t), Inches(w), Inches(h))
    sh.adjustments[0] = radius
    sh.fill.solid()
    sh.fill.fore_color.rgb = C(CARD)
    sh.line.color.rgb = C(HAIR)
    sh.line.width = Pt(0.75)
    sh.shadow.inherit = False
    if sh.has_text_frame:
        sh.text_frame.text = ""
    return sh


def base(prs, eyebrow, title, sub, page):
    sl = prs.slides.add_slide(prs.slide_layouts[6])
    bg = sl.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(W), Inches(H))
    bg.fill.solid()
    bg.fill.fore_color.rgb = C(CANVAS)
    bg.line.fill.background()
    bg.shadow.inherit = False
    txt(sl, .72, .46, 11.89, .24, eyebrow, MONO, 8.5, MUTE)
    txt(sl, .72, .74, 11.89, .52, title, SEMI, 27, INK)
    txt(sl, .72, 1.28, 11.89, .26, sub, SANS, 10.5, BODY)
    txt(sl, 11.40, 6.92, 1.21, .24, "%02d / %02d" % (page, TOTAL),
        MONO, 8.5, FAINT, PP_ALIGN.RIGHT)
    return sl


def rows(sl, l, t, cols, data, widths, size=9.0):
    x = l
    for c, wd in zip(cols, widths):
        if c:
            txt(sl, x, t, wd, .2, c, MED, 8.5, MUTE)
        x += wd
    y = t + .26
    for r in data:
        x = l
        for ci, (v, wd) in enumerate(zip(r, widths)):
            col = BODY
            if isinstance(v, tuple):
                v, col = v
            txt(sl, x, y, wd, .2, str(v), SANS if ci == 0 else MONO, size, col)
            x += wd
        y += .245
    return y


def strip(sl, l, t, w, stats, gap=None):
    gap = gap or w / len(stats)
    x = l
    for lab, val in stats:
        col = INK
        if isinstance(val, tuple):
            val, col = val
        txt(sl, x, t, gap - .1, .22, lab, SANS, 9.0, MUTE)
        txt(sl, x, t + .30, gap - .1, .34, val, SEMI, 17, col)
        x += gap


def pic(sl, name, l, t, w=None, h=None):
    p = os.path.join(FIG, name)
    if not os.path.exists(p):
        txt(sl, l, t, w or 4, .3, "(그림 없음: %s)" % name, MONO, 9, WARN)
        return
    kw = {}
    if w:
        kw["width"] = Inches(w)
    if h:
        kw["height"] = Inches(h)
    sl.shapes.add_picture(p, Inches(l), Inches(t), **kw)


def step(sl, x, w, n, title, body):
    """파이프라인 한 칸. 번호 · 제목 · 설명."""
    txt(sl, x, 2.20, w, .20, n, MONO, 9.0, ACC)
    txt(sl, x, 2.46, w, .24, title, SEMI, 11.5, INK)
    txt(sl, x, 2.80, w, 1.2, body, SANS, 9.0, MUTE, line=1.45)


# ------------------------------------------------------------------ 슬라이드
def slide1(prs):
    sl = base(prs, "01 / 데이터셋", "실측 라벨이 있는 미국 해군기지 5곳을 확보했습니다",
              "HRSC2016 — Google Earth 약 0.45 m · 사람이 그린 회전상자 2,964척 · 라벨 1,070장 중 1,059장(99 %)이 미국 항만", 1)
    card(sl, .72, 1.80, 7.35, 4.85)
    pic(sl, "fig2_port_spotcheck.png", .90, 1.98, w=7.0)
    txt(sl, .90, 6.20, 7.0, .4, "노란 상자 = 정답. 샌디에이고 · 노퍽 · 메이포트 · 에버렛 · 뉴포트.",
        SANS, 8.5, MUTE)

    card(sl, 8.30, 1.80, 4.31, 1.5)
    txt(sl, 8.50, 1.95, 3.9, .22, "어디서", SEMI, 11, INK)
    txt(sl, 8.50, 2.25, 3.95, 1.0,
        "Kaggle guofeng/hrsc2016 (8 GB, 5분할 RAR).\n"
        "정답은 HRSC2016 XML 의 회전상자 그대로 — 우리가 만든 라벨은 없습니다.",
        SANS, 9.0, BODY, line=1.45)

    card(sl, 8.30, 3.45, 4.31, 1.6)
    txt(sl, 8.50, 3.60, 3.9, .22, "항만은 좌표로 식별", SEMI, 11, INK)
    txt(sl, 8.50, 3.90, 3.95, 1.1,
        "문서엔 'six famous harbors' 뿐. XML 좌표를 군집화하니 6곳.\n"
        "경도 부호가 빠져 있어 서경으로 보정 → 실제 기지와 2 km 안, 지형으로 확인.",
        SANS, 9.0, BODY, line=1.45)

    card(sl, 8.30, 5.20, 4.31, 1.45)
    txt(sl, 8.50, 5.35, 3.9, .22, "GSD 는 함급으로 검증", SEMI, 11, INK)
    txt(sl, 8.50, 5.65, 3.95, .95,
        "XML 의 1.07 m 는 명목값(배 길이 중앙 360 m). 타일 레벨·위도로 0.45 m —\n"
        "선박 p99 347 m ↔ Nimitz 급 333 m (+4 %).",
        SANS, 9.0, BODY, line=1.45)
    return sl


def slide2(prs):
    sl = base(prs, "02 / 파이프라인", "받기 → 라벨 → 학습 → 평가 → 웹앱",
              "전 과정이 스크립트로 재현됩니다. 학습만 Kaggle T4, 나머지는 CPU.", 2)
    card(sl, .72, 1.80, 11.89, 2.35)
    w = 11.89 / 5
    step(sl, .95, w - .2, "01", "받기 · 풀기",
         "Kaggle 에서 내려받고 Windows 기본 bsdtar 로 RAR 추출.\n영상 1,680장 · XML 1,681개.")
    step(sl, .95 + w, w - .2, "02", "항만 · GSD",
         "좌표 군집 → 항만 5곳. 타일 레벨·위도로 GSD 계산,\n함급 치수로 검증. manifest 생성.")
    step(sl, .95 + 2 * w, w - .2, "03", "회전상자 라벨",
         "XML mbox(중심·폭·높이·라디안) → YOLO OBB 8좌표 정규화.\n꼭짓점 순서 고정, 100장 육안 검증.")
    step(sl, .95 + 3 * w, w - .2, "04", "학습",
         "YOLO11m-OBB · imgsz 640 · 67 epoch · 시드 3.\n학습 중 val 끄고 last.pt — 학습량 동일.")
    step(sl, .95 + 4 * w, w - .2, "05", "평가 · 웹앱",
         "공식 test 451장. 항만별 P/R/F1/AP50 을 따로.\n웹앱에서 항만을 골라 test 영상을 돌아가며 탐지.")

    card(sl, .72, 4.35, 5.8, 2.3)
    txt(sl, .92, 4.50, 5.4, .22, "분할", SEMI, 11, INK)
    txt(sl, .92, 4.80, 5.4, 1.8,
        "HRSC2016 공식 train / val / test = 436 / 181 / 453.\n"
        "항만이 세 쪽에 섞이므로 항만 단위 재분할(학습 샌디에이고·노퍽 → 시험 에버렛·뉴포트)로\n"
        "누수 영향을 따로 확인 — 결론이 바뀌지 않았습니다.",
        SANS, 9.0, BODY, line=1.5)
    card(sl, 6.75, 4.35, 5.86, 2.3)
    txt(sl, 6.95, 4.50, 5.4, .22, "왜 회전상자인가", SEMI, 11, INK)
    txt(sl, 6.95, 4.80, 5.5, 1.8,
        "군함은 부두에 비스듬히 댑니다. 축정렬 상자는 이웃 배와 부두를 크게 포함해\n"
        "길이·폭을 못 재고 IoU 도 흐려집니다. 정답이 회전상자라 모델도 회전상자를 내고,\n"
        "길이·폭·방향이 나옵니다.",
        SANS, 9.0, BODY, line=1.5)
    return sl


def slide3(prs):
    sl = base(prs, "03 / 결과", "test 451장에서 F1 0.936 — 항만 5곳 모두 0.9 이상",
              "학습에 쓰지 않은 공식 test · conf 0.25 · IoU 0.5 · 학습 시드 3개 평균 (항만별은 seed 0 실측)", 3)
    card(sl, .72, 1.80, 11.89, 1.15)
    strip(sl, .95, 1.98, 11.5, [
        ("precision", "0.916"), ("recall", "0.957"), ("F1", ("0.936", ACC)),
        ("AP50", "0.970"), ("AP50-95", "0.770"), ("시드 편차 (F1)", "±0.004")])

    card(sl, .72, 3.15, 4.9, 3.5)
    txt(sl, .92, 3.30, 4.6, .22, "항만별 실측", SEMI, 11, INK)
    rows(sl, .92, 3.65, ["항만", "선박", "P", "R", "F1", "AP50"],
         [["샌디에이고", "683", "0.923", "0.952", "0.937", "0.966"],
          ["노퍽", "308", "0.946", "0.964", "0.955", "0.986"],
          ["메이포트", "160", "0.927", "0.950", "0.938", "0.940"],
          ["에버렛", "52", "0.839", "1.000", "0.912", "0.959"],
          ["뉴포트", "23", "0.952", "0.870", "0.909", "0.942"]],
         [1.2, .6, .7, .7, .7, .7])
    txt(sl, .92, 5.30, 4.6, 1.2,
        "초록 = 탐지, 노랑 = 정답. 웹앱에서 항만을 고르고\n◀ ▶ 으로 test 영상을 돌아가며 같은 것을 봅니다.",
        SANS, 9.0, MUTE, line=1.45)

    ports = [("fig1_san_diego.png", "샌디에이고 11/12"), ("fig1_norfolk.png", "노퍽 7/7"),
             ("fig1_mayport.png", "메이포트 6/5"), ("fig1_everett.png", "에버렛 4/5"),
             ("fig1_newport.png", "뉴포트 3/1")]
    x0, y0, cw, ch, gap = 5.85, 3.15, 2.18, 1.62, .12
    for k, (f, cap) in enumerate(ports):
        r, c = divmod(k, 3)
        x, y = x0 + c * (cw + gap), y0 + r * (ch + .22 + gap)
        card(sl, x, y, cw, ch + .22)
        pic(sl, f, x + .06, y + .06, w=cw - .12, h=ch - .12)
        txt(sl, x + .08, y + ch - .02, cw - .16, .2, cap + "  (정답/탐지)", MED, 8.0, INK)
    return sl


def build(out):
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(W), Inches(H)
    for f in (slide1, slide2, slide3):
        f(prs)
    prs.save(out)
    print("저장:", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "4차업무_정승원.pptx"))
    a = ap.parse_args()
    build(a.out)
