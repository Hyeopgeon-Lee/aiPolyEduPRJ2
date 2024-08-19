# 설치한 OpenCV 패키지 불러오기
import cv2

# 웹캠으로부터 영상 가져오기(0 : 컴퓨터에 기본 설치된 웹캠 장치 / cv2.CAP_DSHOW : 화면에 웹캠영상 바로 보여주기)
cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# 동영상은 시작부터 종료될때까지 프레임을 지속적으로 받아야 하기 때문에 while문을 계속 반복함
while True:

    # 웹캠의 영상은 이미지파일과 유사한 많은 프레임들이 빠르게 움직이면서 보여줌
    # 웹캠의 영상으로부터 프레임 1개마다 읽어오기 위해 사용
    ret, webcam_image = cam.read()  # 프레임 1개마다 읽기

    # 웹캠 영상으로부터 프레임(이미지)를 잘 받았으면 실행함
    if ret is True:

        # 흑백사진으로 변경
        gray = cv2.cvtColor(webcam_image, cv2.COLOR_BGR2GRAY)

        # 변환한 흑백사진을 검정색과 흰색을 변경하기(검정색 -> 흰색 / 흰색 -> 검정색)
        invert = cv2.bitwise_not(gray)

        # 가우시안 필터링 적용
        # 가우시안 커널(KSize): 가우시안 커널 크기
        # sigmaX: x방향 sigma로 가우시안 커널 크기(KSize)는 sigmaX 값에 따라 자동 계산
        # 사용 이유 : 밝은 곳은 더 밝게,  어두운 곳은 더 어둡게 변환
        blur = cv2.GaussianBlur(invert, (0, 0), 10)

        # blur 블러 처리한 흑백사진을 검정색과 흰색을 변경하기(검정색 -> 흰색 / 흰색 -> 검정색)
        invertedBlur = cv2.bitwise_not(blur)

        # 스케치 효과를 주기 위해 흑백 사진과 블러 흑백 이미지를 나눈 뒤, 255를 곱함
        sketch = cv2.divide(gray, invertedBlur, scale=255)

        # 스케치 이미지 보여주기
        cv2.imshow("sketch", sketch)

    # 입력받는 것 대기하기, 작성안하면, 결과창이 바로 닫힘
    if cv2.waitKey(1) > 0:
        break

