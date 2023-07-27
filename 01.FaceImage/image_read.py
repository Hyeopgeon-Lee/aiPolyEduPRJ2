import cv2

# 불러올 이미지 경로
image_file = "../image/my_face.jpg"

# cv2.IMREAD_COLOR : RGB 값만 불러서 이미지 출력
rgb = cv2.imread(image_file, cv2.IMREAD_COLOR)

# cv2.IMREAD_GRAYSCALE : 흑백사진으로 수정해서 이미지 읽기
gray = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)

# cv2.IMREAD_UNCHANGED : 이미지 파일의 알파 채널(Alpha Channel)까지 포함해서 읽기
alpha = cv2.imread(image_file, cv2.IMREAD_UNCHANGED)

# 이미지 보여주기
cv2.imshow("IMREAD_COLOR", rgb)  # RGB만 포함
cv2.imshow("IMREAD_GRAYSCALE", gray)  # 흑백
cv2.imshow("IMREAD_UNCHANGED", alpha)  # RGB + 알파 채널 추가(사진 그대로 읽기)

cv2.waitKey(0)

