# 설치한 OpenCV 패키지 불러오기
import cv2

# 분석하기 위한 이미지 불러오기
image = cv2.imread("../image/my_face.jpg", cv2.IMREAD_COLOR)

# 흑백사진으로 변경
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 변환한 흑백사진을 검정색과 흰색을 변경하기(검정색 -> 흰색 / 흰색 -> 검정색)
invert = cv2.bitwise_not(gray)

# 가우시안 필터링 적용
# 가우시안 커널(KSize): 가우시안 커널 크기
# sigmaX: x방향 sigma로 가우시안 커널 크기(KSize)는 sigmaX 값에 따라 자동 계산
# 설정값 : 1부터 4까지 사용 / 0은 sigmaX 사용하지 않는 것으로 가우시안 커널(KSize) 값을 설정함
# 사용 이유 : 밝은 곳은 더 밝게,  어두운 곳은 더 어둡게 변환
blur = cv2.GaussianBlur(invert, (0, 0), 10)

# blur 블러 처리한 흑백사진을 검정색과 흰색을 변경하기(검정색 -> 흰색 / 흰색 -> 검정색)
invertedBlur = cv2.bitwise_not(blur)

# 스케치 효과를 주기 위해 흑백 사진과 블러 흑백 이미지를 나눈 뒤, 255를 곱함
sketch = cv2.divide(gray, invertedBlur, scale=255)

# 스케치 이미지 보여주기
cv2.imshow("sketch", sketch)

# 입력받는 것 대기하기, 작성안하면, 결과창이 바로 닫힘
cv2.waitKey(0)
