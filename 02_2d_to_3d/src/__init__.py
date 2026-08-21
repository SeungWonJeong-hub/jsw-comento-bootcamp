"""위성 상대항법용 2D -> 3D 변환 파이프라인 (2차 업무).

모듈 구성
    camera     : 핀홀 카메라 모델, 쿼터니언/자세 변환, 투영과 역투영
    scene      : 해석적 합성 장면 (Unit Test 픽스처, 정답 오차 0)
    spe3r      : SPE3R 공개 데이터셋 로더
    stereo     : 스테레오 삼각측량 -> 깊이 맵 -> 포인트 클라우드 (본 과제의 주 경로)
    depth      : 밝기 기반 깊이(과제 예시 대조군), 스케일/오프셋 정렬
    baseline   : 과제 예시 코드 원문 재현
    pointcloud : 포인트 클라우드 변환, PLY 입출력, 메시 샘플링
    metrics    : Chamfer 거리, 깊이 오차 지표
"""

__all__ = ["camera", "scene", "spe3r", "stereo", "depth", "baseline",
           "pointcloud", "metrics"]
