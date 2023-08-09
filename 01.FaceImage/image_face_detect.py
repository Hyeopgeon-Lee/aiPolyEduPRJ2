# 설치한 OpenCV 패키지 불러오기
import cv2

# 분석하기 위한 이미지 불러오기
image = cv2.imread("../image/emoticon.png", cv2.IMREAD_UNCHANGED)

# 흑백사진으로 변경
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 변환한 흑백사진으로부터 히스토그램 평활화
gray = cv2.equalizeHist(gray)

if image is None: raise Exception("이미지 읽기 실패")

# 학습된 얼굴 정면검출기 사용하기(OpenCV에 존재하는 파일)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml")

# 얼굴 검출 수행(정확도 높이는 방법의 아래 파라미터를 조절함)
# 얼굴 검출은 히스토그램 평황화한 이미지 사용
# scaleFactor : 1.5
# minNeighbors : 인근 유사 픽셀 발견 비율이 5번 이상
# flags : 0 => 더이상 사용하지 않는 인자값
# 분석할 이미지의 최소 크기 : 가로 100, 세로 100
faces = face_cascade.detectMultiScale(gray, 1.1, 5, 0, (100, 100))

# 인식된 얼굴의 수
facesCnt = len(faces)

# 인식된 얼굴의 수 출력
print(len(faces))

# 얼굴이 검출되었다면,
if facesCnt > 0:

    # 사진에 여러 명 얼굴이 있다면, 반복하여 사각형 그리기
    for face in faces:
        cv2.rectangle(image, face, (255, 0, 0), 4)  # 얼굴 검출 사각형 그리기

else:
    print("얼굴 미검출")

# 얼굴 영역이 그려진 사진 보여주기
cv2.imshow("MyFace", image)

# 입력받는 것 대기하기, 작성안하면, 결과창이 바로 닫힘
cv2.waitKey(0)

