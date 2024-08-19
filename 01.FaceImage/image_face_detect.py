import cv2

# 분석할 이미지 불러오기
image = cv2.imread("../image/my_face.jpg", cv2.IMREAD_COLOR)

# 이미지가 정상적으로 불러와졌는지 확인
if image is None:
    raise Exception("이미지를 불러올 수 없습니다.")

# 이미지를 그레이스케일로 변환 및 히스토그램 평활화 적용
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.equalizeHist(gray)

# 얼굴 검출기를 로드
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml")

# 얼굴 검출 수행
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

# 검출된 얼굴의 수 출력
print(f"검출된 얼굴 수: {len(faces)}")

# 얼굴이 검출되었을 때 사각형 그리기
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 4)

# 결과 이미지 출력
cv2.imshow("Detected Faces", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
