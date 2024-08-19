import cv2

# 이미지와 이모티콘 이미지 불러오기
image = cv2.imread("../image/izone.jpg", cv2.IMREAD_COLOR)
emoticon_image = cv2.imread("../image/emoticon.png", cv2.IMREAD_COLOR)

# 이미지 로드 실패 시 예외 발생
if image is None or emoticon_image is None:
    raise Exception("이미지를 불러올 수 없습니다.")

# 얼굴 검출을 위한 그레이스케일 이미지 생성 및 히스토그램 평활화
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.equalizeHist(gray)

# 학습된 얼굴 정면 검출기 사용
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml")

# 얼굴 검출 수행
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

# 얼굴이 검출되었다면 이모티콘을 얼굴에 적용
if len(faces) > 0:
    for (x, y, w, h) in faces:
        # 얼굴 영역 크기에 맞게 이모티콘 이미지 크기 조정
        resized_emoticon = cv2.resize(emoticon_image, (w, h), interpolation=cv2.INTER_AREA)

        # 이모티콘 이미지를 그레이스케일로 변환 및 마스크 생성
        emoticon_gray = cv2.cvtColor(resized_emoticon, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(emoticon_gray, 240, 255, cv2.THRESH_BINARY_INV)

        # 마스크 반전 생성
        mask_inv = cv2.bitwise_not(mask)

        # 이모티콘에서 이모티콘 부분만 추출
        emoticon_fg = cv2.bitwise_and(resized_emoticon, resized_emoticon, mask=mask)

        # 원본 이미지에서 얼굴 부분만 추출
        face_bg = cv2.bitwise_and(image[y:y+h, x:x+w], image[y:y+h, x:x+w], mask=mask_inv)

        # 얼굴 부분에 이모티콘 합성
        image[y:y+h, x:x+w] = cv2.add(face_bg, emoticon_fg)

    # 이모티콘 처리된 이미지 저장 및 표시
    result_path = "../result/emoticon_result.jpg"
    cv2.imwrite(result_path, image)
    cv2.imshow("Emoticon Applied Image", image)

else:
    print("얼굴을 검출할 수 없습니다.")

# 입력 대기 (결과 창이 바로 닫히지 않도록 설정)
cv2.waitKey(0)
cv2.destroyAllWindows()
