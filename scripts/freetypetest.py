import cv2  
import numpy as np  
import freetype  
import time
def draw_chinese_text(image, text, position, font_path, font_size, color=(0, 0, 255), thickness=-1):  
    # 加载字体  
    face = freetype.Face(font_path)  
    face.set_char_size(font_size*64)  # 设置字体大小  
    
  
    # 绘制文本  
    x, y = position 


    for char in text:
        if char == '\n':  
            x = position[0]  
            y += font_size  
            continue
        face.load_char(char)    
        bitmap = face.glyph.bitmap  
        left = face.glyph.bitmap_left  
        top = face.glyph.bitmap_top

        # 将bitmap转换为OpenCV可以识别的格式,当通道灰度图 
        gray_image = np.array(bitmap.buffer, dtype=np.uint8).reshape(bitmap.rows, bitmap.width)

        # 创建一个与原图像相同大小的零数组（三通道）  
        bgr_image = np.zeros((bitmap.rows, bitmap.width, 3), dtype=np.uint8)  
        
        # 将灰度图像复制到新图像的红色通道  
        bgr_image[:, :, 2] = gray_image 

        mask=gray_image!=0


        start_in = time.time()
        
        image[y - top : y - top + bitmap.rows, x  : x + bitmap.width][mask]=bgr_image[mask]

        # image[y - top : y - top + bitmap.rows, x  : x + bitmap.width]=bgr_image



        
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