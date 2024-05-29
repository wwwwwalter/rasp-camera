import cv2
import numpy as np
import sklearn.metrics as mr
import os
import threading
#感知哈希算法
def pHash(image): 
    image = cv2.resize(image,(32,32), interpolation=cv2.INTER_CUBIC) 
    image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) 
    # 将灰度图转为浮点型，再进行dct变换 
    dct = cv2.dct(np.float32(image))
#     print(dct)
    # 取左上角的8*8，这些代表图片的最低频率 
    # 这个操作等价于c++中利用opencv实现的掩码操作 
    # 在python中进行掩码操作，可以直接这样取出图像矩阵的某一部分 
    dct_roi = dct[0:8,0:8]  
    avreage = np.mean(dct_roi) 
    hash = [] 
    for i in range(dct_roi.shape[0]): 
        for j in range(dct_roi.shape[1]): 
            if dct_roi[i,j] > avreage: 
                hash.append(1) 
            else: 
                hash.append(0) 
    return hash
 
#均值哈希算法
def aHash(image):
    #缩放为8*8
    image=cv2.resize(image,(8,8),interpolation=cv2.INTER_CUBIC)
    #转换为灰度图
    image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    avreage = np.mean(image) 
    hash = [] 
    for i in range(image.shape[0]): 
        for j in range(image.shape[1]): 
            if image[i,j] > avreage: 
                hash.append(1) 
            else: 
                hash.append(0) 
    return hash
 
#差值感知算法
def dHash(image):
    #缩放9*8
    image=cv2.resize(image,(9,8),interpolation=cv2.INTER_CUBIC)
    #转换灰度图
    image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    hash=[]
    #每行前一个像素大于后一个像素为1，相反为0，生成哈希
    for i in range(8):
        for j in range(8):
            if image[i,j]>image[i,j+1]:
                hash.append(1)
            else:
                hash.append(0)
    return hash
 
#计算汉明距离
def Hamming_distance(hash1,hash2): 
    num = 0
    for index in range(len(hash1)): 
        if hash1[index] != hash2[index]: 
            num += 1
    return num
# if __name__ == "__main__":
#     image_file1 = './data/cartoon1.jpg'
#     image_file2 = './data/cartoon3.jpg'
#     img1 = cv2.imread(image_file1)
#     img2 = cv2.imread(image_file2)
#     hash1 = pHash(img1)
#     hash2 = pHash(img2)
#     dist = Hamming_distance(hash1, hash2)
#     #将距离转化为相似度
#     similarity = 1 - dist * 1.0 / 64 
#     print(dist)
#     print(similarity)
def mutual_information(template_img, campture_area):
    nmi = mr.normalized_mutual_info_score(template_img.reshape(-1),campture_area.reshape(-1))
    return nmi

def similarity(template_img, campture_area, method=None):
    pass

def feature_match(target_img, template_img, method=None):
    # 获取目标图像和模板图像的宽高
    target_h, target_w = target_img.shape[:2]
    template_h, template_w = template_img.shape[:2]

    # 使用平方差匹配算法
    result = cv2.matchTemplate(target_img, template_img, cv2.TM_SQDIFF)

    # 获取最匹配的位置
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    top_left = min_loc
    bottom_right = (top_left[0] + template_w, top_left[1] + template_h)

    campture_area = target_img[
        top_left[1]:top_left[1]+template_h,
        top_left[0]:top_left[0]+template_w
    ] 
    return campture_area

def softmax(z):
    # 计算指数
    exp_z = np.exp(z)
    
    # 计算softmax
    softmax_output = exp_z / np.sum(exp_z)
    
    return softmax_output
def decimal_to_percentage(decimal_number, precision=2):
    return f"{decimal_number:.{precision}%}"

template_img_root = r"./data/template"
disease_category_num = os.listdir(template_img_root)[:-1]
disease_category_name = []
with open(template_img_root+"/disease", 'r', encoding='utf-8') as file:
    for line in file:
        disease_category_name.append(line.strip())


target_img_path = r"data\img\source\3-4-002.jpeg"
target_img = cv2.imread(target_img_path)
disease_probability = []

for disease in disease_category_num:
    score = 0.0
    for i in os.listdir(template_img_root + "/" + disease):
        template_img_path = template_img_root + "/" + disease + '/' + i
        print(template_img_path+" **** "+target_img_path)
        template_img = cv2.imread(template_img_path)
        campture_area = feature_match(target_img,template_img)
        score += mutual_information(template_img, campture_area)
    disease_probability.append(score)

disease_probability = np.array(disease_probability)
disease_probability = softmax(disease_probability)
disease_probability_indx = np.argsort(disease_probability)[::-1]
print("可能疾病为：")
for i in  disease_probability_indx:
    print(disease_category_name[i], decimal_to_percentage(disease_probability[i]))

