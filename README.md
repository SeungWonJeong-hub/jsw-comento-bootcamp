# 코멘토 직무부트캠프 1차 업무 — Git 코드 관리 & 픽셀 단위 이미지 처리

Computer Vision POC를 위한 기초 실습입니다. Git 브랜치 전략으로 코드를 관리하고,
OpenCV로 픽셀 단위 색상 검출과 AI 학습용 전처리 파이프라인을 구현했습니다.

## 1. 구성

```
jsw-comento-bootcamp/
├── color_pixel_filter.py       # [요청내용 2] 특정 색상 픽셀 감지 및 필터링
├── image_preprocessing.py      # [추가 요청] 전처리 파이프라인 (기본 + 심화)
├── preprocessed_samples/       # [제출 항목] 전처리를 마친 이미지 5장 + 품질 리포트
├── outputs/                    # 색상 검출 결과 이미지
│   └── preview/                # 참고용: 단계별 비교 격자 + 학습 입력 텐서(.npy)
├── data/                       # 원본 캐시 (용량 문제로 .gitignore 처리)
├── requirements.txt
└── README.md
```

## 2. 실행 방법

```bash
pip install -r requirements.txt

# 전처리 파이프라인 (Hugging Face food101에서 40장 수집 → 5장 저장)
# 네트워크가 없으면 --offline 으로 data/ 캐시만 사용해 동작합니다.
python image_preprocessing.py --num-images 40 --num-samples 5
```

색상 검출에 쓰는 위성 이미지는 `data/`가 `.gitignore` 대상이라 저장소에 없습니다.
아래 명령으로 원본을 받으면 그대로 재현됩니다.

```bash
mkdir -p data
curl -o data/saltpond_aiguesmortes_landsat8.jpg \
  https://assets.science.nasa.gov/content/dam/science/esd/eo/images/imagerecords/144000/144777/graudoroi_oli_2018241_lrg.jpg

# 색상 픽셀 검출
python color_pixel_filter.py --image data/saltpond_aiguesmortes_landsat8.jpg --color red-orange
```

## 3. 픽셀 단위 색상 처리 (`color_pixel_filter.py`)

| 단계 | 사용 함수 | 목적 |
|---|---|---|
| 로드 | `cv2.imread()` | BGR uint8 배열로 읽습니다. 실패 시 `None`을 돌려주므로 명시적으로 검사합니다 |
| 통계 | `cv2.split()` | 채널별 평균·표준편차·최솟값·최댓값으로 픽셀 분포를 파악합니다 |
| 색공간 변환 | `cv2.cvtColor(BGR2HSV)` | 색상(H)과 밝기(V)를 분리해 조명 변화에 강인하게 만듭니다 |
| 색상 감지 | `cv2.inRange()` | HSV 범위 안의 픽셀만 255로 남긴 마스크를 만듭니다 |
| 이진화 | `cv2.threshold()` | 밝기 기준 필터링 결과를 비교군으로 함께 출력합니다 |
| 필터링 | `cv2.bitwise_and()` | 마스크 영역의 원본 픽셀만 추출합니다 |
| 후처리 | `cv2.morphologyEx()` | Open으로 점 노이즈를 제거하고 Close로 구멍을 메웁니다 |
| 검증 | `cv2.findContours()` | 검출 영역에 바운딩 박스를 그려 육안으로 확인합니다 |

**빨간색을 두 범위로 나누는 이유**: OpenCV의 Hue는 0~179이고 빨강은 0을 기준으로
양쪽 끝(0~10, 170~180)에 걸쳐 있어 한 구간만으로는 절반을 놓칩니다.
두 마스크를 합치는 방식은 과제 예제의 `mask = mask1 + mask2` 형태를 그대로 따랐습니다.
`red-orange`가 H 0~15 한 구간뿐인 것은 대상이 H 0 위쪽에만 분포해 반대쪽 끝(170~180)에
걸리는 화소가 없기 때문입니다.

### 대상 이미지

`data/saltpond_aiguesmortes_landsat8.jpg` — 프랑스 Aigues-Mortes 염전입니다.
염전 물이 붉게 보이는 것은 고염도 환경에서 번식하는 베타카로틴 함유 플랑크톤과
호염성 세균(halobacteria) 때문입니다.

| 항목 | 값 |
|---|---|
| 센서 / 촬영일 | Landsat 8 OLI / 2018-08-20 |
| 출처 | [NASA Earth Observatory](https://science.nasa.gov/earth/earth-observatory/salin-aigues-mortes-144777) (public domain) |
| 크레딧 | NASA Earth Observatory image by Lauren Dauphin, using Landsat data from the U.S. Geological Survey |
| 크기 | 2131×1421 (원본 그대로 사용) |

### 실행 결과와 한계

```
mean_bgr          : [88.63, 81.83, 69.08]
[red 검출]         12,324 px / 3,028,151 px (0.41%)
[red-orange 검출]  26,501 px / 3,028,151 px (0.88%)
```

과제 예제의 빨강 범위(H 0~10, S≥120)로는 염전이 **0.41%만 검출됩니다.**
염전 화소를 직접 측정하니 **H 7~11, S 96~135, V 76~223** 이었습니다. 색상(H)은 범위
안에 들어오지만, 채도(S)가 96~119인 화소가 하한 120에 걸려 잘려나간 것이 원인입니다.

측정값에 맞춰 별도 범위 `red-orange`(H 0~15, S≥100)를 추가하니 검출량이 **2.15배**로
늘었습니다. `red` 범위는 과제 예제값 그대로 유지했습니다.

전체 화면 대비 비율이 0.88%에 그치는 것은 염전이 장면에서 차지하는 면적 자체가
작기 때문입니다.

**한계**: 정답 마스크가 없으므로 이 수치가 얼마나 정확한지는 판정할 수 없습니다.
검출 영역이 염전과 일치한다는 판단은 결과 이미지의 육안 확인에 근거한 것입니다.
`red-orange` 범위 역시 이 한 장면에서 측정한 값이라, 다른 조명·계절·센서 조건에서도
통할지는 확인되지 않았습니다.

## 4. 전처리 파이프라인 (`image_preprocessing.py`)

### 4.1 데이터 수집
`datasets` 라이브러리의 `streaming=True`로 food101에서 필요한 장수만 수신합니다.
전체는 5GB 이상이라 전량 다운로드가 불필요합니다. food101은 클래스 순으로 정렬돼 있어
그대로 받으면 전부 같은 음식이 나오므로, `shuffle(seed=42)`로 클래스를 섞고
seed를 고정해 재현성을 확보했습니다.

### 4.2 처리 순서와 근거

| 순서 | 단계 | 구현 | 왜 이 순서인가 |
|---|---|---|---|
| 1 | 크기 조정 224×224 | `cv2.resize(INTER_AREA)` | 축소가 대부분이라 `INTER_AREA`의 에일리어싱이 가장 적습니다 |
| 2 | 노이즈 제거 | `cv2.GaussianBlur(5,5)` | 리사이즈 **뒤**에 두어야 커널이 항상 같은 화소 비율로 작동합니다 |
| 3 | 그레이스케일 | `cv2.cvtColor(BGR2GRAY)` | 3채널을 1채널로 줄여 연산량이 1/3이 됩니다 |
| 4 | 정규화 | `/255.0 → float32` | 입력을 0~1로 맞춰 학습 시 gradient 스케일을 안정화합니다 |

증강은 원본 손상을 피하기 위해 **리사이즈 결과(컬러)** 에서 분기합니다.

| 증강 | 구현 | 모사하는 실제 변화 |
|---|---|---|
| 좌우 반전 | `cv2.flip(img, 1)` | 촬영 방향 차이 |
| 회전 ±15° | `cv2.getRotationMatrix2D` + `warpAffine` | 카메라 기울기. `BORDER_REPLICATE`로 검은 테두리를 막습니다 |
| 색상 변화 | HSV의 S×1.4, V×1.25 | 조명·화이트밸런스 차이 |

### 4.3 이상치 필터링 (심화 문제)

전처리 **이전**에 원본을 검사해 학습에 해로운 이미지를 걸러냅니다.

**① 너무 어두운 이미지** — HSV의 V 채널 평균이 40 미만이면 제외합니다.
BGR 단순 평균 대신 V를 쓰는 이유는 색상·채도의 영향을 배제하고 밝기만 보기 위해서입니다.

**② 객체가 너무 작은 이미지** — 주요 객체 면적 비율이 0.10 미만이면 제외합니다.
Otsu 이진화로 전경과 배경을 자동 분리하고, 배경이 밝은 경우까지 처리하기 위해
전경 픽셀이 과반이면 마스크를 반전합니다. 모폴로지로 잡음을 정리한 뒤
가장 큰 윤곽선의 면적을 전체 픽셀 수로 나눠 비율을 구합니다.

검사 결과는 `preprocessed_samples/quality_report.json`에 전량 기록되므로
임계값을 바꿔가며 판정 근거를 추적할 수 있습니다.

## 5. 결과물

- `preprocessed_samples/*.png` — 전처리를 마친 이미지 5장입니다 (224×224 그레이스케일).
  정규화 결과는 0~1 float이라 저장할 때 0~255 uint8로 되돌립니다.
- `preprocessed_samples/quality_report.json` — 전수 검사 수치와 제외 사유입니다
- `outputs/` — 색상 검출 결과 7장입니다. 원본, 마스크 2종(`red` · `red-orange`),
  필터링 2종, 밝기 이진화, 검출 박스가 들어 있습니다. 사진 계열은 `.jpg`로,
  마스크·이진화는 값이 0/255로 유지돼야 하므로 무손실 `.png`로 저장합니다
- `outputs/preview/*_stages.png` — 단계별 결과와 증강 3종을 한 장의 격자로 비교합니다 (참고용)
- `outputs/preview/*_normalized.npy` — 실제 학습 입력이 되는 `float32 [0,1]` 텐서입니다

### 전처리 실행 결과 (40장 수집 기준)

```
[info] 원본 이미지 40장 확보
[info] 이상치 3장 제외 / 정상 37장
       - food101_006_label51: tiny_object(ratio=0.087<0.1)
       - food101_018_label6 : tiny_object(ratio=0.068<0.1)
       - food101_038_label74: tiny_object(ratio=0.059<0.1)
[save] food101_000_label87.png | shape=(224, 224) dtype=float32 range=[0.020, 0.992]
```

최종 출력 텐서는 `(224, 224)` / `float32` / 값 범위 `0.0~1.0` 으로, 모델 입력 규격을 만족합니다.

### 검사 지표 분포 (40장)

|  | 최소 | Q1 | 중앙값 | Q3 | 최대 | 임계값 |
|---|---|---|---|---|---|---|
| 밝기 (V 채널 평균) | 54.5 | 122.1 | 144.8 | 160.9 | 205.2 | 40 |
| 객체 면적비 | 0.059 | 0.184 | 0.244 | 0.325 | 0.515 | 0.10 |

**밝기 필터가 한 번도 발동하지 않은 이유**: 표본 40장의 V 채널 평균이 54.5~205.2
범위여서 임계값 40을 밑도는 이미지가 없었습니다. 임계값 40은 판독이 거의 불가능한
수준을 기준으로 잡은 값이라, 정상 데이터셋에서 발동하지 않는 것 자체는 정상 동작입니다.

다만 **최저값 54.5는 임계값과 14.5밖에 떨어져 있지 않습니다.** 표본이 조금만 달랐어도
걸렸을 거리이므로, 발동하지 않았다는 사실을 안전하다는 뜻으로 읽을 수 없습니다.
객체 면적비 쪽도 경계가 촘촘합니다. 제외된 최대값이 0.087이고 통과한 최소값이 0.107로,
임계값 0.10을 사이에 두고 0.02 폭 안에 두 그룹이 붙어 있습니다.

임계값 40과 면적비 0.10은 모두 데이터 분포를 보고 정한 값이 아니라 감으로 잡은
값이라 근거가 약합니다.

## 6. Git 워크플로

```
main ──────────────────────────●  (PR merge)
  └── feature/image-processing ┘
```

1. `git init` → `git remote add origin <URL>`
2. `git checkout -b feature/image-processing`
3. 기능 단위로 `git add` / `git commit`
4. `git push origin feature/image-processing`
5. GitHub에서 PR 생성 → 코드 리뷰 → `main` 병합

`main`에 직접 커밋하지 않고 feature 브랜치를 거치는 이유는, 리뷰 전 코드가
기준 브랜치를 오염시키지 않게 하고 PR 단위로 변경 이력을 추적하기 위해서입니다.

## 7. 환경

Python 3.11.9 / OpenCV 5.0.0 / NumPy 2.4.6 / Windows 11
