import cv2  # OpenCV 라이브러리 불러오기

# 얼굴 탐지를 위한 Haar Cascade 모델 로드
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml")

# 나이 예측을 위한 Caffe 모델 로드
age_net = cv2.dnn.readNetFromCaffe("../model/deploy_age.prototxt", "../model/age_net.caffemodel")

# 성별 예측을 위한 Caffe 모델 로드
gender_net = cv2.dnn.readNetFromCaffe("../model/deploy_gender.prototxt", "../model/gender_net.caffemodel")

# Caffe 모델의 학습 데이터에 사용된 평균값 (RGB 순서)
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

# 나이 예측 결과로 사용할 연령대 리스트 (모델 출력 값에 해당)
age_list = ["(0-2)", "(4-6)", "(8-12)", "(15-20)", "(25-32)", "(38-43)", "(48-53)", "(60-100)"]

# 성별 예측 결과로 사용할 리스트 (모델 출력 값에 해당)
gender_list = ["Male", "Female"]

# 웹캠 영상 캡처 시작 (기본 웹캠 장치 사용)
cam = cv2.VideoCapture(0)

# 실시간 비디오 프레임 처리를 위한 무한 루프
while True:
    # 웹캠에서 프레임 읽기 (ret: 성공 여부, frame: 읽어온 프레임)
    ret, frame = cam.read()
    if not ret:  # 프레임 읽기 실패 시 루프 종료
        break

    # 프레임을 흑백으로 변환하고 히스토그램 평활화 (얼굴 탐지 성능 향상)
    gray = cv2.equalizeHist(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

    # 얼굴을 탐지하여 얼굴 영역 좌표 리스트 반환
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

    # 탐지된 모든 얼굴에 대해 반복
    for (x, y, w, h) in faces:
        # 탐지된 얼굴 영역을 자름
        face_img = frame[y:y+h, x:x+w]

        # 얼굴 이미지를 (227, 227) 크기로 변환하고, 평균값으로 정규화
        blob = cv2.dnn.blobFromImage(face_img, scalefactor=1.0, size=(227, 227), mean=MODEL_MEAN_VALUES, swapRB=False)

        # 성별 예측을 위해 blob 입력 설정
        gender_net.setInput(blob)
        # 성별 예측 결과 얻기 (확률이 가장 높은 인덱스를 선택)
        gender = gender_list[gender_net.forward().argmax()]

        # 나이 예측을 위해 blob 입력 설정
        age_net.setInput(blob)
        # 나이 예측 결과 얻기 (확률이 가장 높은 인덱스를 선택)
        age = age_list[age_net.forward().argmax()]

        # 얼굴 영역에 사각형 그리기
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        # 얼굴 위에 성별과 나이 예측 결과 텍스트 표시
        cv2.putText(frame, f"{gender}, {age}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2, cv2.LINE_AA)

    # 실시간 결과 화면 출력
    cv2.imshow("Webcam - Gender and Age Detection", frame)

    # ESC 키를 누르면 루프 종료
    if cv2.waitKey(1) & 0xFF == 27:
        break

# 웹캠 해제
cam.release()
# 모든 OpenCV 창 닫기
cv2.destroyAllWindows()
