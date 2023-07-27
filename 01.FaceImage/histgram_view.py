import cv2
from matplotlib import pyplot as plt

# 불러올 이미지 경로
image_file = "../image/my_face.jpg"

# cv2.IMREAD_COLOR : RGB 값만 불러서 이미지 출력
rgb = cv2.imread(image_file, cv2.IMREAD_COLOR)

# cv2.IMREAD_GRAYSCALE : 흑백사진으로 수정해서 이미지 읽기
gray = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)

# OpenCV 색상을 일반적으로 부르는 RGB로 순서가 아닌, BGR 순서로 사용함
color = ("b", "g", "r")

# 색상이 존재하는 원본이미지 히스토그램 보기
# BGR 순서대로 그래프 그리기 위해 반복문 사용함
for i, col in enumerate(color):
    # calcHist 파라미터 설명
    # 첫번째 파라미터(images) : 분석할 이미지 파일
    # 두번째 파라미터(Channel) : 컬러이미지(BGR)이면, 배열 값 3개로 정의
    # 세번째 파라미터(Mask) : 분석할 영역의 형태인 mask
    # 네번째 파라미터(histSize) : 히스토그램의 hist 크기, 예 : 64면 256/64 = 4 => 픽셀 4개를 1개의 픽셀로 합쳐서 연산함
    # 다섯번째 파라미터(범위) : 컬러이미지(BGR)이면 0- 256까지 배열
    hist = cv2.calcHist([rgb], [i], None, [256], [0, 256])
    plt.figure(1)
    plt.plot(hist, color=col)

# 원본 이미지 히스토그램 출력
plt.show()

# 흑백사진 히스토그램 보기
hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
plt.figure(2)
plt.plot(hist)
plt.show()

# 히스토그램 평활화, 흑백사진만 가능함
gray = cv2.equalizeHist(gray)

# 히스토그램 평활화된 히스토그램 보기
hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
plt.figure(3)
plt.plot(hist)
plt.show()

