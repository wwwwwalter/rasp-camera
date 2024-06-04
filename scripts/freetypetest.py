import cv2  
import numpy as np  
import freetype  
import time
def draw_chinese_text(image, text, position, font_path, font_size, color=(0, 0, 255), thickness=-1):  
    """  
    在给定的图片上绘制中文字符。  
      
    :param image: OpenCV图像对象  
    :param text: 要绘制的文本  
    :param position: 文本的起始位置 (x, y)  
    :param font_path: 字体文件的路径（例如：'simhei.ttf'）  
    :param font_size: 字体大小  
    :param color: 字体颜色，默认为白色  
    :param thickness: 线条粗细，-1表示填充，默认为-1  
    """  
    # 加载字体  
    face = freetype.Face(font_path)  
    face.set_char_size(font_size*64)  # 设置字体大小  
  
    # 绘制文本  
    x, y = position  
    start_in = time.time()
    for char in text:  
        
        face.load_char(char)  
        
        bitmap = face.glyph.bitmap  
        left = face.glyph.bitmap_left  
        top = face.glyph.bitmap_top  

        # 将bitmap转换为OpenCV可以识别的格式  
        glyph_image = np.array(bitmap.buffer, dtype=np.uint8).reshape(bitmap.rows, bitmap.width)  
        # 去除边缘
        _, glyph_image_binary = cv2.threshold(glyph_image, 127, 255, cv2.THRESH_BINARY)

          

        # cv2.RETR_EXTERNAL: 只检索最外层的轮廓。
        # cv2.RETR_LIST: 检索所有的轮廓，并将其保存到列表中，不建立父子关系。
        # cv2.RETR_CCOMP: 检索所有的轮廓，并将它们组织为两层的层次结构：顶层是外部边界，第二层是边界内的孔。
        # cv2.RETR_TREE: 检索所有的轮廓，并重建嵌套轮廓的完整层次结构。
        
        # cv2.CHAIN_APPROX_NONE: 存储轮廓上的所有点。也就是说，它会保存轮廓上的每一个点，包括所有冗余的点。
        # cv2.CHAIN_APPROX_SIMPLE: 仅保存轮廓的端点。对于垂直、水平和对角线分割，即轮廓上的线段只需要两个端点就可以定义，中间点不会被保存。这种方法可以大大减少存储的数据量，并且对于后续的轮廓处理（如绘制、填充等）也更加高效。
        

        # 找到非零（即前景）像素的坐标 
        contours, hierarchy = cv2.findContours(glyph_image_binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        

        # cv2.drawContours(image, contours, -1, (0, 0, 255), 1,hierarchy=hierarchy,maxLevel=1)
        # cv2.drawContours(image, contours, -1, color, 1, hierarchy=hierarchy, maxLevel=3, offset=(x + left, y - top))  

        cv2.drawContours(image, contours, -1, color, -1, offset=(x + left, y - top)) 

        
        x += face.glyph.advance.x >> 6  
    print(f'use:{(time.time()-start_in)*1000}ms') 
    return image  



    # 创建opencv主窗口
# cv2.namedWindow("debug",cv2.WINDOW_KEEPRATIO)
# cv2.namedWindow('Text', cv2.WINDOW_KEEPRATIO) 


#字体路径
font_path='/usr/share/fonts/truetype/google/NotoSansCJKsc.ttf'

# 示例图片路径
image_path = 'capture.jpg'  

# 加载图片  
image = cv2.imread(image_path)  

# 在图片上绘制中文字符  
image_with_text = draw_chinese_text(image, 'hello,world', (50, 50), font_path, 30)  
# 显示图片  
cv2.imshow('Text', image)  
cv2.waitKey(5*1000)  




cv2.destroyAllWindows()