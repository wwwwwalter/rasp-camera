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
    print(x)
    print(y) 

    for char in text:  
        
        face.load_char(char)  
        
        bitmap = face.glyph.bitmap  
        left = face.glyph.bitmap_left  
        top = face.glyph.bitmap_top
        print(f'\n')
        print(f'char:{char}')  
        print(f'top:{top}')
        print(f'left{left}')
        print(f'width:{bitmap.width}')
        print(f'rows:{bitmap.rows}')

        # 将bitmap转换为OpenCV可以识别的格式,当通道灰度图 
        gray_image = np.array(bitmap.buffer, dtype=np.uint8).reshape(bitmap.rows, bitmap.width)

        # 将单通道图像转换为三通道图像  
        bgr_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
        
        # 创建一个与原图像相同大小的零数组  
        zeros = np.zeros(bgr_image.shape[:2], dtype=bgr_image.dtype)  
        bgr_image[:, :, 0] = zeros  
        bgr_image[:, :, 1] = zeros 
        
        print(f'shape:{bgr_image.shape[:2]}')
        
        



        start_in = time.time()
        
        image[y - top : y - top + bitmap.rows, x - left : x - left + bitmap.width]=bgr_image
        
        print(f'use:{(time.time()-start_in)*1000}ms')


        x += face.glyph.advance.x >> 6  

    return image  



# cv2.namedWindow("image",cv2.WINDOW_FREERATIO)
# cv2.setWindowProperty("image", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

#字体路径
font_path='/usr/share/fonts/truetype/google/NotoSansCJKsc.ttf'

# 示例图片路径
image_path = './scripts/stack.jpg'  

# 加载图片  
image = cv2.imread(image_path)  

# 在图片上绘制中文字符  
image_with_text = draw_chinese_text(image, 'hello.。', (5, 40), font_path, 40)  
# 显示图片  
cv2.imshow('image', image_with_text)  
cv2.waitKey(10*1000)  




cv2.destroyAllWindows()