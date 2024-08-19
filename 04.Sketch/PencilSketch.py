import cv2

# 이미지 불러오기 (그레이스케일로 바로 불러오기)
image = cv2.imread("../image/my_face.jpg", cv2.IMREAD_GRAYSCALE)

# 이미지 반전
inverted_image = cv2.bitwise_not(image)

# 가우시안 블러 적용 (자동으로 최적화된 커널 사이즈와 시그마 사용)
# 스케치 구현할때, 블러 크기가 클 수록 좀 더 스케치 느낌남
blurred = cv2.GaussianBlur(inverted_image, (21, 21), 0)

# 블러 이미지를 다시 반전
inverted_blurred = cv2.bitwise_not(blurred)

# 원본 이미지와 반전된 블러 이미지를 나눠서 스케치 효과 적용
sketch = cv2.divide(image, inverted_blurred, scale=256.0)

# 결과 출력
cv2.imshow("Sketch", sketch)
cv2.waitKey(0)
cv2.destroyAllWindows()
