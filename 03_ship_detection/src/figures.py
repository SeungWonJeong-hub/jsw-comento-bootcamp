"""그림 생성 — 코멘토 3차 업무 / 정승원"""
import os, glob, json, math, argparse
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

INK, ACC, WARN, MUTE = '#1f2430', '#2b7fd4', '#d4443c', '#8f8f8f'
PORTS4 = ['busan_anchorage', 'tongyeong', 'yeosu', 'gwangyang']


def fig_samples(root, out, tile=320):
    """그림 1 — 학습 데이터가 어떻게 생겼나. 크기 분위별 표본."""
    recs = []
    for f in glob.glob(f'{root}/labels/train/*.txt'):
        if not os.path.getsize(f):
            continue
        stem = os.path.basename(f)[:-4]
        for ln in open(f):
            t = ln.split()
            if len(t) < 9:
                continue
            p = np.array([[float(t[i]) * tile, float(t[i + 1]) * tile]
                          for i in range(1, 9, 2)], np.float32)
            e = [math.dist(p[i], p[(i + 1) % 4]) for i in range(4)]
            recs.append((max(e[0], e[1]), stem, p))
    recs.sort(key=lambda r: r[0])
    picks = [recs[int(len(recs) * q)] for q in (0.05, 0.35, 0.65, 0.88, 0.97, 0.999)]

    fig, axes = plt.subplots(1, 6, figsize=(16.5, 3.2))
    for ax, (L, stem, p) in zip(axes, picks):
        img = cv2.imread(f'{root}/images/train/{stem}.png')[:, :, ::-1]
        cx, cy = p[:, 0].mean(), p[:, 1].mean()
        r = 40
        x0 = int(np.clip(cx - r, 0, tile - 2 * r))
        y0 = int(np.clip(cy - r, 0, tile - 2 * r))
        big = cv2.resize(img[y0:y0 + 2 * r, x0:x0 + 2 * r], None,
                         fx=6, fy=6, interpolation=cv2.INTER_NEAREST).copy()
        pts = np.array([[(v[0] - x0) * 6, (v[1] - y0) * 6] for v in p], np.int32)
        cv2.polylines(big, [pts], True, (255, 212, 0), 2)
        ax.imshow(big); ax.axis('off')
        ax.set_title(f'{L:.1f} px  ·  약 {L*10:.0f} m', fontsize=11)
    fig.suptitle('핀란드 연안 학습 데이터 — 선박 크기 분위별 (6배 확대)', fontsize=13)
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)
    print('저장:', out)


def fig_strata(j_test, j_val, out):
    """그림 2 — 크기별 층화. 작을수록 못 찾고, 찾아도 헐겁습니다."""
    a = json.load(open(j_test, encoding='utf-8'))['bins']
    b = json.load(open(j_val, encoding='utf-8'))['bins']
    keep = [i for i, r in enumerate(a) if r['n_gt'] > 0 and r['ap50'] == r['ap50']]
    x = np.arange(len(keep))
    lab = [f"{a[i]['bin']}px\n{a[i]['metres']}" for i in keep]

    fig, ax = plt.subplots(1, 2, figsize=(13.5, 4.4))
    w = 0.38
    ax[0].bar(x - w/2, [a[i]['recall'] for i in keep], w, label='test (34WFT)', color=ACC)
    ax[0].bar(x + w/2, [b[i]['recall'] for i in keep if i < len(b)], w,
              label='val (34VER)', color='#9ab8d8')
    ax[0].set_ylabel('Recall'); ax[0].set_ylim(0, 1.05)
    ax[0].set_title('작은 배일수록 놓칩니다', fontsize=12)
    ax[0].legend(fontsize=9)

    ax[1].plot(x, [a[i]['mean_iou'] for i in keep], 'o-', color=ACC, lw=2.2, ms=8,
               label='test')
    ax[1].plot(x, [b[i]['mean_iou'] for i in keep if i < len(b)], 's--', color=INK,
               lw=2, ms=7, label='val')
    ax[1].axhline(0.5, ls=':', c=WARN, lw=1.4)
    ax[1].text(0.05, 0.52, 'IoU 0.5 — 매칭 기준선', color=WARN, fontsize=9)
    ax[1].set_ylabel('평균 IoU'); ax[1].set_ylim(0.4, 0.85)
    ax[1].set_title('작은 배일수록 박스가 헐겁습니다', fontsize=12)
    ax[1].legend(fontsize=9)
    for a_ in ax:
        a_.set_xticks(x); a_.set_xticklabels(lab, fontsize=9)
        a_.grid(alpha=.25, ls=':')
    fig.suptitle('선박 크기별 층화 평가 — mAP 하나가 가리는 것', fontsize=13)
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)
    print('저장:', out)


def fig_operating(j_test, j_val, out):
    """그림 3 — 운용 임계값. 재학습 없이 오탐을 97% 줄입니다."""
    a = json.load(open(j_test, encoding='utf-8'))
    b = json.load(open(j_val, encoding='utf-8'))
    fig, ax = plt.subplots(1, 2, figsize=(13.5, 4.4))
    for d, nm, c, ls in [(a, 'test (34WFT)', ACC, '-'), (b, 'val (34VER)', INK, '--')]:
        cv = d['curve']
        t = [r['conf'] for r in cv]
        ax[0].plot(t, [r['n_fp'] for r in cv], ls, color=c, lw=2.2, marker='o', ms=5, label=nm)
        ax[1].plot([r['recall'] for r in cv], [r['precision'] for r in cv], ls,
                   color=c, lw=2.2, marker='o', ms=5, label=nm)
        bf = d['best_f1']
        ax[1].plot(bf['recall'], bf['precision'], '*', color=WARN, ms=18, zorder=5)
    ax[0].set_yscale('log'); ax[0].set_xlabel('신뢰도 임계값'); ax[0].set_ylabel('오탐 수 (로그)')
    ax[0].set_title('임계값만 올려도 오탐이 사라집니다', fontsize=12)
    ax[1].set_xlabel('Recall'); ax[1].set_ylabel('Precision')
    ax[1].set_title('PR 곡선 — 별표가 F1 최대점', fontsize=12)
    for a_ in ax:
        a_.grid(alpha=.25, ls=':'); a_.legend(fontsize=9)
    fig.suptitle('운용 임계값 결정 — 재학습 없이 되는 것', fontsize=13)
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)
    print('저장:', out)


def fig_korea(kdir, out, ports=PORTS4):
    """그림 4 — 한국 항만 4곳 적용 결과."""
    s = json.load(open(f'{kdir}/summary.json', encoding='utf-8'))
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 13.6))
    for ax, name in zip(axes.flat, ports):
        img = cv2.imread(f'{kdir}/{name}.jpg')
        if img is None:
            ax.axis('off'); continue
        ax.imshow(img[:, :, ::-1]); ax.axis('off')
        v = s.get(name, {})
        L = [d['length_m'] for d in v.get('detections', [])]
        ax.set_title(f"{v.get('label', name)}   {v.get('datetime','')[:10]}\n"
                     f"탐지 {v.get('n_detections',0)}척 · 물 위 {v.get('water_ratio',0):.0f}% · "
                     f"길이중앙 {np.median(L) if L else 0:.0f} m",
                     fontsize=12)
    fig.suptitle('한국 항만 적용 — 핀란드 데이터로 학습한 모델을 그대로 (신뢰도 0.5)',
                 fontsize=14, y=0.995)
    fig.subplots_adjust(hspace=0.16, wspace=0.06, top=0.93, bottom=0.01,
                        left=0.01, right=0.99)
    fig.savefig(out, dpi=110); plt.close(fig)
    print('저장:', out)


def fig_training(curve_dir, out):
    """그림 5 — 학습 곡선과 실험 비교."""
    import csv
    runs = [('yolo11s_dota', 'YOLO11s + DOTA', ACC, '-'),
            ('yolo11n_dota', 'YOLO11n + DOTA', INK, '-'),
            ('yolo11n_scratch', 'YOLO11n 밑바닥', MUTE, '--'),
            ('yolo11n_noaug', 'YOLO11n 증강 off', WARN, '--')]
    fig, ax = plt.subplots(1, 2, figsize=(13.5, 4.4))
    for k, lab, c, ls in runs:
        f = os.path.join(curve_dir, k + '.csv')
        if not os.path.exists(f):
            continue
        r = list(csv.DictReader(open(f)))
        ep = [int(x['epoch']) for x in r]
        ax[0].plot(ep, [float(x['metrics/mAP50(B)']) for x in r], ls, color=c, lw=2, label=lab)
        ax[1].plot(ep, [float(x['metrics/mAP50-95(B)']) for x in r], ls, color=c, lw=2, label=lab)
    ax[0].set_ylabel('mAP50'); ax[0].set_title('mAP50 — 증강을 끄면 크게 떨어집니다', fontsize=12)
    ax[1].set_ylabel('mAP50-95'); ax[1].set_title('mAP50-95 — 절반 수준에 머뭅니다', fontsize=12)
    for a_ in ax:
        a_.set_xlabel('epoch'); a_.grid(alpha=.25, ls=':'); a_.legend(fontsize=9)
    fig.suptitle('학습 곡선 — Tesla T4 · 100 epoch · imgsz 320', fontsize=13)
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)
    print('저장:', out)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='C:/Users/seung/datasets/S2Ships/yolo')
    ap.add_argument('--outdir', default='outputs')
    a = ap.parse_args()
    O = a.outdir
    os.makedirs(O, exist_ok=True)
    fig_samples(a.data, f'{O}/fig1_samples.png')
    if os.path.exists(f'{O}/size_strata_test.json'):
        fig_strata(f'{O}/size_strata_test.json', f'{O}/size_strata_val.json',
                   f'{O}/fig2_size_strata.png')
    if os.path.exists(f'{O}/operating_point_test.json'):
        fig_operating(f'{O}/operating_point_test.json', f'{O}/operating_point_val.json',
                      f'{O}/fig3_operating.png')
    if os.path.exists(f'{O}/korea/summary.json'):
        fig_korea(f'{O}/korea', f'{O}/fig4_korea.png')
    if os.path.isdir(f'{O}/curves'):
        fig_training(f'{O}/curves', f'{O}/fig5_training.png')
