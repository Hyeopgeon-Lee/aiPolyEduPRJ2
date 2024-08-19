import cv2
import matplotlib.pyplot as plt

# 이미지 불러오기
image_path = "../image/my_face.jpg"
image = cv2.imread(image_path, cv2.IMREAD_COLOR)

if image is None:
    raise Exception("이미지 읽기 실패")

# 가우시안 블러 적용
gaussian_blur = cv2.GaussianBlur(image, (15, 15), 10)

# 평균 블러 적용
average_blur = cv2.blur(image, (15, 15))

# 원본 이미지와 블러 처리된 이미지 출력
titles = ['Original Image', 'Gaussian Blur', 'Average Blur']
images = [image, gaussian_blur, average_blur]

for i in range(3):
    plt.subplot(1, 3, i + 1)
    plt.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])

plt.show()

