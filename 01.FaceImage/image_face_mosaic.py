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
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=2, minSize=(100, 100))

# 검출된 얼굴의 수 출력
print(f"검출된 얼굴 수: {len(faces)}")

# 얼굴이 검출되었을 때 모자이크 처리
if len(faces) > 0:
    mosaic_rate = 30  # 모자이크 비율 설정

    for (x, y, w, h) in faces:
        # 얼굴 영역 가져오기
        face_image = image[y:y+h, x:x+w]

        # 얼굴 영역을 모자이크 비율에 맞게 축소
        small_face = cv2.resize(face_image, (w // mosaic_rate, h // mosaic_rate))

        # 축소된 이미지를 다시 원래 크기로 확대
        mosaic_face = cv2.resize(small_face, (w, h), interpolation=cv2.INTER_NEAREST)

        # 원본 이미지에 모자이크 처리된 얼굴 이미지 덮어쓰기
        image[y:y+h, x:x+w] = mosaic_face

    # 모자이크 처리된 이미지 파일 생성 및 표시
    result_path = "../result/mosaic.jpg"
    cv2.imwrite(result_path, image)
    cv2.imshow("Mosaic Image", image)

else:
    print("얼굴을 검출할 수 없습니다.")

# 입력 대기 (결과 창이 바로 닫히지 않도록 설정)
cv2.waitKey(0)
cv2.destroyAllWindows()
