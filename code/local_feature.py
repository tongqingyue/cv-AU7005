import cv2
import numpy as np

# 读取图像
img = cv2.imread("data/12.jpg") # 图像路径
# 转化为灰度图像
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# 构造sift函数
sift = cv2.SIFT_create()
# 找出关键点5
kp = sift.detect(gray, None)
# 使用关键点找出sift特征向量
kp, des = sift.compute(gray, kp)

#在图像中绘制关键点
img2 = cv2.drawKeypoints(gray, kp, None, flags=0)
cv2.imwrite("local_feature.jpg", img2)
 
#显示绘制了关键点的图像
cv2.imshow("Keypoints", img2)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.waitKey(1)