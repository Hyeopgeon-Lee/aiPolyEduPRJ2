# 설치한 OpenCV 패키지 불러오기
import cv2

# 분석하기 위한 이미지 불러오기
image = cv2.imread("../image/izone.jpg", cv2.IMREAD_COLOR)

# 흑백사진으로 변경
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 변환한 흑백사진으로부터 히스토그램 평활화
gray = cv2.equalizeHist(gray)

# 얼굴을 변경할 이모티콘 이미지
emoticon_image = cv2.imread("../image/emoticon.png", cv2.IMREAD_COLOR)

if image is None: raise Exception("이미지 읽기 실패")

# 학습된 얼굴 정면검출기 사용하기
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml")

# 얼굴 검출 수행(정확도 높이는 방법의 아래 파라미터를 조절함)
# 얼굴 검출은 히스토그램 평황화한 이미지 사용
# scaleFactor : 1.1
# minNeighbors : 인근 유사 픽셀 발견 비율이 5번 이상
# flags : 0 => 더이상 사용하지 않는 인자값
# 분석할 이미지의 최소 크기 : 가로 100, 세로 100
faces = face_cascade.detectMultiScale(gray, 1.1, 5, 0, (100, 100))

# 인식된 얼굴의 수
facesCnt = len(faces)

# 얼굴이 검출되었다면,
if facesCnt > 0:

    # 검출된 얼굴의 수만큼 반복하여 실행함
    for face in faces:
        # 얼굴 위치 값을 가져오기
        x, y, w, h = face

        # 얼굴 이미지
        face_image = image[y:y + h, x:x + w]

        # 이모티콘 이미지를 얼굴 영역 크기의 사이즈로 변경하며, 얼굴 영역 크기에 맞추고,
        # 얼굴 영역 이미지를 이모티콘 이미지로 변경
        emoticon_image_resize = cv2.resize(emoticon_image, (w, h), interpolation=cv2.INTER_AREA)

        gray_mask = cv2.cvtColor(emoticon_image_resize, cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(gray_mask, 240, 255, cv2.THRESH_BINARY_INV)
        mask_inv = cv2.bitwise_not(mask)
        masked_face = cv2.bitwise_and(emoticon_image_resize, emoticon_image_resize, mask=mask)
        masked_frame = cv2.bitwise_and(face_image, face_image, mask=mask_inv)

        image[y:y + h, x:x + w] = cv2.add(masked_face, masked_frame)

    # 이모티콘 처리된 이미지 파일 생성하기
    cv2.imwrite("../result/emoticon_png.jpg", image)

    # 모자이크 처리된 이미지 보여주기
    cv2.imshow("emoticon_image", cv2.imread("../result/emoticon_png.jpg", cv2.IMREAD_COLOR))

else:
    print("얼굴 미검출")

# 입력받는 것 대기하기, 작성안하면, 결과창이 바로 닫힘
cv2.waitKey(0)
