import cv2
import pytesseract

# 读取图像和模板
img = cv2.imread('./output_img/mosaiced_image_full.png')
template_word = "SUMMER"

# 将目标图像转换为灰度图像
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 使用pytesseract进行OCR识别
# 配置tesseract路径（如果需要）
# pytesseract.pytesseract.tesseract_cmd = r'<full_path_to_tesseract_executable>'

# 获取OCR识别的文本及其位置数据
data = pytesseract.image_to_data(img_gray, output_type=pytesseract.Output.DICT)

# 绘制矩形框以标记匹配的单词位置
n_boxes = len(data['level'])
for i in range(n_boxes):
    if data['text'][i].strip().lower() == template_word.lower():
        (x, y, w, h) = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
        img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

# 显示结果
cv2.imshow('Detected', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.waitKey(1)
