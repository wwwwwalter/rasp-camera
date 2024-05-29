
import cv2
import numpy as np
import sklearn.metrics as mr


def mutual_information(template_img, campture_area):
    nmi = mr.normalized_mutual_info_score(template_img.reshape(-1),campture_area.reshape(-1))
    return nmi
# 加载目标图像和模板图像
# target_img_path = r"data\img\3-4-006.jpeg"
# target_img_path = r"data\img\3-12-002.jpeg"
target_img_path = r"data\img\3-4-006.jpeg"

# template_img_path = r"data\template\耵聍栓塞\3.jpeg"
template_img_path = r"data\template\1\3.jpeg"

target_img = cv2.imread(target_img_path)
template_img = cv2.imread(template_img_path)
 
# 获取目标图像和模板图像的宽高
target_h, target_w = target_img.shape[:2]
template_h, template_w = template_img.shape[:2]
 
# 使用平方差匹配算法
result = cv2.matchTemplate(target_img, template_img, cv2.TM_SQDIFF)
 
# 获取最匹配的位置
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
top_left = min_loc
bottom_right = (top_left[0] + template_w, top_left[1] + template_h)
 
# 在目标图像上绘制矩形框
cv2.rectangle(target_img, top_left, bottom_right, (0, 255, 0), 2)
 
# 显示结果图像
cv2.namedWindow('Result', cv2.WINDOW_FREERATIO)
campture_area = target_img[
        top_left[1]:top_left[1]+template_h,
        top_left[0]:top_left[0]+template_w
    ] 

print(mutual_information(template_img.reshape(-1),campture_area.reshape(-1)))
cv2.imshow('Result', campture_area)

# cv2.imshow('Result', target_img )
cv2.waitKey(0)
cv2.destroyAllWindows()