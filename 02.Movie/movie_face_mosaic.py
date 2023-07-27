# 설치한 OpenCV 패키지 불러오기
import cv2

# 학습된 얼굴 정면검출기 사용하기
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml")

cam = cv2.VideoCapture("../movie/bts.mp4")

# 동영상은 시작부터 종료될때까지 프레임을 지속적으로 받아야 하기 때문에 while문을 계속 반복함
while True:
    ret, movie_image = cam.read()

    # 동영상으로부터 프레임(이미지)를 잘 받았으면 실행함
    if ret is True:

        # 동영상의 프레임을 얼굴인식율을 높이기 위해 흑백으로 변경함
        gray = cv2.cvtColor(movie_image, cv2.COLOR_BGR2GRAY)

        # 변환한 흑백사진으로부터 히스토그램 평활화
        gray = cv2.equalizeHist(gray)

        # 얼굴 인식하기
        faces = face_cascade.detectMultiScale(gray, 1.5, 5, 0, (20, 20))

        for face in faces:
            #얼굴 위치 값을 가져오기
            x, y, w, h = face

            #원본이미지로부터 얼굴영역 가져오기
            face_image = movie_image[y:y + h, x:x + w]

            # 모자이크 비율(픽셀크기 증가로 모자이크 만들기)
            mosaic_rate = 30

            # 얼굴 영역의 픽셀을 mosaic_rate에 따라 나눠서 픽셀 확대
            face_image = cv2.resize(face_image, (w // mosaic_rate, h // mosaic_rate))

            # 확대한 얼굴 이미지(픽셀)을 얼굴 크기에 덮어쓰기
            face_image = cv2.resize(face_image, (w, h), interpolation=cv2.INTER_AREA)

            # 원본이미지에 모자이크 처리한 얼굴 이미지 붙이기
            movie_image[y:y + h, x:x + w] = face_image;

            #얼굴 검출 사각형 그리기
            cv2.rectangle(movie_image, face, (255, 0, 0), 4)

        # 사이즈 변경된 이미지로 출력하기
        cv2.imshow("movie_mosaic", movie_image)

    # 입력받는 것 대기하기, 작성안하면, 결과창이 바로 닫힘
    if cv2.waitKey(1) > 0:
        break