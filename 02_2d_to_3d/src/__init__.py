"""달 지형 2D -> 3D 변환 파이프라인 (2차 업무).

모듈 구성
    camera     : 핀홀 카메라, 자세, 투영·역투영
    terrain    : 고도 모델 로드·렌더링·촬영 기하
    stereo     : 상대 자세, 정렬, 시차, 깊이 맵
    pointcloud : 포인트 클라우드 변환, PLY 입출력
    metrics    : 깊이 오차, Chamfer, F-score
"""

__all__ = ["camera", "terrain", "stereo", "pointcloud", "metrics"]
