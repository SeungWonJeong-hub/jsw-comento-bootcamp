"""3차 업무 PPT 생성 — 코멘토 3차 업무 / 정승원

2차 업무 PPT 서식을 그대로 따른다.
  13.333 x 7.5 in · 배경 #FAFAFA · 카드 #FFFFFF + #EBEBEB 0.75pt 테두리
  제목 Pretendard SemiBold 27pt · 본문 Pretendard · 수치 Cascadia Mono

구성
----
  01 방법    핀란드 연안 데이터로 학습셋을 만들고 파인튜닝한 경로
  02 결과    대조군 넷의 기여도, 그리고 지표를 바꾸니 뒤집힌 결론
  03 확인    원본 위성사진과 탐지 결과를 나란히
  04 확장    한국으로 가져갈 때 무엇이 되고 무엇이 안 되는가 (4차)
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
TOTAL = 4


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
    """텍스트박스 격자로 만든 간단한 표. 셀이 (값, 색) 튜플이면 색을 적용."""
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
    """카드 안에 큰 수치를 나란히 놓는다."""
    gap = gap or w / len(stats)
    x = l
    for lab, val in stats:
        txt(sl, x, t, gap - .1, .22, lab, SANS, 9.0, MUTE)
        txt(sl, x, t + .30, gap - .1, .34, val, SEMI, 17, INK)
        x += gap


# ------------------------------------------------------------------ 슬라이드
def slide1(prs, fig):
    sl = base(prs, "01 / 방법",
              "10 m 위성사진에서 5 픽셀짜리 선박을 찾는다",
              "Sentinel-2 GSD 10 m · 핀란드 연안 7,010 타일 / 13,069 인스턴스 · "
              "DOTA 사전학습 YOLO11s-OBB 파인튜닝 · 지역 단위 분할", 1)

    card(sl, .72, 1.70, 6.42, 2.94, .04)
    txt(sl, .98, 1.90, 5.90, .24, "주석만 공개된 데이터를 학습셋으로 만든다",
        SEMI, 10.5, INK)
    txt(sl, .98, 2.22, 5.90, .22,
        "Zenodo 는 GeoPackage 주석만 준다. 사진은 따로 맞춰야 한다.",
        SANS, 9.5, MUTE)
    txt(sl, .98, 2.60, 5.90, 1.70,
        "1.  Zenodo 15019034 주석 (CC-BY-4.0, RSE 2025)\n"
        "2.  AWS 공개 COG 에서 Sentinel-2 L2A TCI 수신 — 인증 불필요\n"
        "3.  주석 좌표를 COG 격자에 정합해 320 x 320 타일로 절단\n"
        "4.  DOTA 사전학습 YOLO11s-OBB 파인튜닝 (T4, 100 epoch)\n"
        "5.  학습에 안 쓴 해역(34WFT)에서만 평가",
        SANS, 9.5, BODY, line=1.52)
    txt(sl, .98, 4.28, 5.90, .24,
        "→ 주석은 L1C 기준이지만 L2A 는 같은 촬영·같은 격자라 기하가 같다",
        MED, 9.5, INK)

    card(sl, 7.36, 1.70, 5.25, 2.94, .04)
    txt(sl, 7.62, 1.90, 4.73, .24, "지역으로 갈랐다 — 수치를 믿을 근거",
        SEMI, 10.5, INK)
    txt(sl, 7.62, 2.22, 4.73, .22,
        "같은 해역이 학습과 평가에 섞이면 성능이 부풀려진다", SANS, 9.5, MUTE)
    rows(sl, 7.62, 2.58, ["split", "타일", "인스턴스", "지역"],
         [["train", "6,261", "12,354", "34VEM 외"],
          ["val", "353", "349", "34VER"],
          ["test", "396", "366", "34WFT"]],
         [1.15, 1.02, 1.28, 1.28])
    txt(sl, 7.62, 3.52, 4.73, .84,
        "선박 크기는 20 m 부터 700 m 까지 넓다. 150 m 이상 대형선이 665척으로,\n"
        "\"레저용 소형선만 있을 것\" 이라는 예상은 측정으로 뒤집혔다.\n"
        "장변 중앙값은 52.3 m = 5.23 px 이다.",
        SANS, 9.0, MUTE, line=1.42)

    card(sl, .72, 4.86, 11.89, 1.90, .08)
    txt(sl, .98, 5.08, 11.37, .24, "최종 성능 — 못 본 해역에서",
        SEMI, 10.5, INK)
    strip(sl, .98, 5.42, 11.37,
          [("test mAP50", "0.9014"), ("test mAP50-95", "0.5544"),
           ("Precision", "0.834"), ("Recall", "0.877"),
           ("CPU 타일 한 장", "54 ms")], 2.28)
    txt(sl, .98, 6.32, 11.37, .24,
        "Tesla T4 로 학습, 추론은 CPU 로도 된다 — 항만 한 곳 전체 주사가 1.61 초",
        SANS, 9.5, MUTE)


def slide2(prs, fig):
    sl = base(prs, "02 / 결과",
              "무엇이 성능을 만들었고, 무엇이 지표를 속였나",
              "대조군 넷으로 기여도를 갈랐다 · 같은 모델을 점 기준으로 재니 결론이 뒤집혔다", 2)

    card(sl, .72, 1.70, 6.42, 3.06, .04)
    txt(sl, .98, 1.90, 5.90, .24, "대조군 넷 — 무엇이 얼마나 기여했나",
        SEMI, 10.5, INK)
    txt(sl, .98, 2.22, 5.90, .22,
        "한 번에 한 축만 바꿔 기여도를 귀속시킨다", SANS, 9.5, MUTE)
    rows(sl, .98, 2.58, ["실험", "val mAP50", "test mAP50", "Recall"],
         [["밑바닥 (사전학습 없음)", "0.8106", "0.8734", "0.831"],
          ["증강 off", ("0.7068", WARN), ("0.7678", WARN), ("0.735", WARN)],
          ["DOTA 사전학습 (n)", "0.8390", "0.8809", "0.828"],
          ["DOTA 사전학습 (s)", ("0.8628", ACC), ("0.9014", ACC), ("0.877", ACC)]],
         [2.30, 1.20, 1.24, 1.16])
    txt(sl, .98, 3.96, 5.90, .58,
        "증강  +13.2%p      사전학습  +2.8%p      모델 크기  +2.4%p",
        MONO, 10.0, INK, line=1.3)
    txt(sl, .98, 4.42, 5.90, .24,
        "→ 가장 크게 기여한 것은 구조가 아니라 증강이었다", MED, 9.5, INK)

    card(sl, 7.36, 1.70, 5.25, 3.06, .04)
    txt(sl, 7.62, 1.90, 4.73, .24, "지표를 바꾸니 결론이 뒤집혔다",
        SEMI, 10.5, INK)
    txt(sl, 7.62, 2.22, 4.73, .22,
        "같은 모델, 같은 평가셋. 재는 방법만 다르다", SANS, 9.5, MUTE)
    rows(sl, 7.62, 2.58, ["지표", "값"],
         [["IoU 기반 mAP50", "0.901"],
          ["IoU 기반 mAP50-95", ("0.459", WARN)],
          ["점 기반 AP", ("0.911", ACC)],
          ["점 기반 재현율", ("0.992", ACC)],
          ["중심 오차 (중앙값)", "0.46 px ≈ 5 m"]],
         [2.70, 2.03])
    txt(sl, 7.62, 4.06, 4.73, .58,
        "5 픽셀짜리 물체에 IoU 0.75 를 요구하는 것은\n"
        "부분 픽셀 정확도를 요구하는 것과 같다.",
        MED, 9.5, INK, line=1.4)

    card(sl, .72, 4.98, 11.89, 1.76, .085)
    txt(sl, .98, 5.18, 11.37, .24,
        "왜 IoU 가 낮았나 — 라벨 좌표를 전수로 찍어 확인했다", SEMI, 10.5, INK)
    txt(sl, .98, 5.52, 11.37, 1.02,
        "여덟 좌표로 배포되기에 회전 박스라고 믿었는데, 13,069개 전부가 축 정렬이었습니다. "
        "장변 각도가 0~10도에 6,343개, 80~90도에 6,726개, 그 사이는 0개입니다.\n"
        "종횡비도 물리가 아니었습니다. 0.958 에서 0.493 으로 크기에 따라 단조 감소하는데, "
        "실제 선박 비율은 크기와 무관하므로 이것은 배의 모양이 아니라 작은 배를 뭉툭하게 칠한 주석 습관입니다.\n"
        "그리고 폭은 어떤 종횡비를 가정해도 72~95% 가 2 px 미만입니다 — 규약을 정하는 문제가 아니라 잴 대상이 없습니다.",
        SANS, 9.0, MUTE, line=1.42)


def slide3(prs, fig):
    sl = base(prs, "03 / 확인",
              "숫자 대신 눈으로 — 원본과 탐지 결과",
              "학습에 안 쓴 해역(34WFT) · 선박이 몰린 0.72 km 구간을 잘라 9배 확대 · "
              "신뢰도 0.25", 3)

    for i, (lab, sub, png) in enumerate([
            ("원본 위성사진", "Sentinel-2 L2A TCI, 받은 그대로",
             "fig6_before_after_raw.png"),
            ("탐지 결과", "모델이 찾은 선박을 노란 박스로",
             "fig6_before_after_det.png")]):
        x = 2.36 + i * 4.50
        txt(sl, x, 1.62, 4.05, .22, lab, SEMI, 10.5, INK)
        txt(sl, x, 1.88, 4.05, .20, sub, SANS, 9.0, MUTE)
        card(sl, x - .08, 2.14, 4.21, 4.21, .03)
        p = fig(png)
        if os.path.exists(p):
            sl.shapes.add_picture(p, Inches(x), Inches(2.22), width=Inches(4.05))

    txt(sl, 2.36, 6.56, 8.55, .24,
        "네 구간 합계 — 정답 17척, 탐지 17척.  발트해는 물도 숲도 어두워 대비를 늘려 표시했다.",
        MED, 9.5, INK)


def slide4(prs, fig):
    sl = base(prs, "04 / 확장",
              "다음은 한국이다 — 무엇이 되고 무엇이 안 되는가",
              "핀란드 모델을 한국 항만 22곳에 그대로 적용해 미리 확인했다 · 4차 웹앱의 밑그림", 4)

    card(sl, .72, 1.70, 6.42, 3.10, .04)
    txt(sl, .98, 1.90, 5.90, .24, "이미 되는 것", SEMI, 10.5, INK)
    txt(sl, .98, 2.22, 5.90, .22,
        "항만 이름 → 무운 장면 검색 → 수신 → 겹쳐 탐지 → NMS", SANS, 9.5, MUTE)
    rows(sl, .98, 2.58, ["항만", "탐지", "판정"],
         [["부산 북항", "36척", "안정"],
          ["광양항", "95척", "안정"],
          ["울산항·앞바다", "145척", "안정"],
          ["인천항 (탁수)", "42척", "동작"],
          ["통영항", ("395척", WARN), ("양식장 오탐", WARN)]],
         [2.30, 1.30, 2.30])
    txt(sl, .98, 4.20, 5.90, .58,
        "임계값 0.05 + 물 게이트로 인천이 25 → 42척이 됐다.\n"
        "탁한 서해에서도 항로의 선박을 되찾는다.",
        MED, 9.5, INK, line=1.4)

    card(sl, 7.36, 1.70, 5.25, 3.10, .04)
    txt(sl, 7.62, 1.90, 4.73, .24, "안 되는 것 — 가두리 양식장", SEMI, 10.5, INK)
    txt(sl, 7.62, 2.22, 4.73, .22,
        "세 가지 후처리를 시험했고 전부 실패했다", SANS, 9.5, MUTE)
    rows(sl, 7.62, 2.58, ["방법", "결과"],
         [["임계값 0.05 → 0.60", ("572 → 130척", WARN)],
          ["물 게이트", ("양식장도 물 위", WARN)],
          ["밀집도 필터", ("21% vs 33%", WARN)]],
         [2.15, 2.58])
    txt(sl, 7.62, 3.70, 4.73, .96,
        "발트해에는 가두리가 없어 모델이 배운 적이 없습니다.\n"
        "후처리로는 못 고칩니다. 학습 데이터에 넣는 것이 정공법입니다.\n"
        "GFW 합성 라벨로 한 재학습은 mAP50 0.106 에 그쳤는데, 폭의 72.6% 가\n"
        "상수로 고정돼 있어 학습할 대상이 아니었습니다.",
        SANS, 9.0, MUTE, line=1.42)

    card(sl, .72, 5.02, 11.89, 1.72, .085)
    txt(sl, .98, 5.22, 11.37, .24, "4차에서 할 것", SEMI, 10.5, INK)
    txt(sl, .98, 5.56, 11.37, .82,
        "1.  양식장 하드 네거티브 — 통영·완도 해역 타일을 배경으로 라벨해 재학습\n"
        "2.  서해 탁도 — NIR(B08) 을 4번째 채널로. 탁한 물도 근적외에서는 어둡다 (NDWI 가 작동하는 원리)\n"
        "3.  웹앱 — CPU 만으로 항만 한 곳 1.61 초를 실측했다. ONNX Runtime 직접 호출로 1 초 아래를 노린다",
        SANS, 9.5, BODY, line=1.52)


def build(out, fig):
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(W), Inches(H)
    slide1(prs, fig)
    slide2(prs, fig)
    slide3(prs, fig)
    slide4(prs, fig)
    prs.save(out)
    print("저장:", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="outputs")
    ap.add_argument("--out", default="3차업무_정승원.pptx")
    a = ap.parse_args()
    build(a.out, lambda n: os.path.join(a.outdir, n))
