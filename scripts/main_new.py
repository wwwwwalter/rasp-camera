import os
import cv2
import glob  
import time
import json  
import torch
import threading
import freetype
import numpy as np
import sklearn.metrics as mr
from networks import LightNet
from PIL import Image, ImageDraw, ImageFont
from weasyprint import HTML, CSS
from bs4 import BeautifulSoup




def decimal_to_percentage(decimal_number, precision=2):
    return f"{decimal_number:.{precision}%}"

def cv2ImgAddText(img, text, left, top, textColor=(0, 255, 0), textSize=20):
    # 判断是否为opencv图片类型
    if (isinstance(img, np.ndarray)):
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    fontText = ImageFont.truetype('simsun.ttc', textSize, encoding="utf-8")
    draw.text((left, top), text, textColor, font=fontText)
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)




def classify_disease(target_img):
    target_img = torch.from_numpy(cv2.cvtColor(target_img,cv2.COLOR_BGR2GRAY).astype(np.float32).reshape((1,1,270,480)))
    with torch.no_grad():
        global model
        model.eval()
        output = model(target_img)
        
        disease_probability = output.numpy()[0]
        disease_probability_indx = np.argsort(disease_probability)[::-1]

        # print("可能疾病为：")
        # local
        disease_probability_info = ""
        for i in  disease_probability_indx:
            # print(disease_category_name[i], decimal_to_percentage(disease_probability[i]))
            disease_probability_info += disease_category_name[i] + decimal_to_percentage(disease_probability[i]) + "\n"

        return disease_probability_info,disease_probability

def img_resize(image):
    height, width = image.shape[0], image.shape[1]
    # 设置新的图片分辨率框架
    width_new = 640
    height_new = 480
    # 判断图片的长宽比率
    if width / height >= width_new / height_new:
        img_new = cv2.resize(image, (width_new, int(height * width_new / width)))
    else:
        img_new = cv2.resize(image, (int(width * height_new / height), height_new))
    return img_new

# 使用算法处理帧
def handle_AI(frame):
    # start_in=time.time()
    frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25, interpolation=cv2.INTER_NEAREST)
    global assistant_flag,disease_probability_info,report_simplified_info,treatment_simplified_info,disease_category_name
    if assistant_flag:
        # 计算最新一张照片的概率
        disease_probability_info,score = classify_disease(frame)
        # 生成最大值序列
        disease_probability_indx = np.argsort(score)[::-1]
        # 取最大概率名称
        name=disease_category_name[disease_probability_indx[0]]
        # 取最大概率精简报告
        name = all_case_info_dict[name][0]['名称']
        report = all_case_info_dict[name][0]['检查结果']
        reason = all_case_info_dict[name][0]['病因']
        complication =  all_case_info_dict[name][0]['并发症']
        treatment = all_case_info_dict[name][0]['治疗建议']
        report_simplified_info = all_case_info_dict[name][0]['简化版结果']
        treatment_simplified_info = all_case_info_dict[name][0]['简化版建议']
        
        
    else:
        disease_probability_info=""
        report_simplified_info=""
        treatment_simplified_info=""

    global handle_AI_flag
    handle_AI_flag = False
    
    # print(f'use:{(time.time()-start_in)*1000} ms')
    # 控制抽帧的频率
    time.sleep(3)

def load_freetype():
    font_path='/usr/share/fonts/truetype/google/NotoSansCJKsc.ttf'
    font_size=30
    # 加载字体
    face=freetype.Face(font_path)
    face.set_char_size(font_size*64)
    return face
    
def draw_chinese_text(image, text, position, font_size=30, color=(0, 0, 255), thickness=-1):  
    global face
    face.set_char_size(font_size*64)
    x, y = position  
    for char in text:
        if char=='\n':
            # 回车
            x=position[0]
            # 换行
            y=y+35 # 行间距
            continue
        face.load_char(char)  
        
        bitmap = face.glyph.bitmap  
        left = face.glyph.bitmap_left  
        top = face.glyph.bitmap_top  

        # 将bitmap转换为OpenCV可以识别的格式  
        glyph_image = np.array(bitmap.buffer, dtype=np.uint8).reshape(bitmap.rows, bitmap.width)  
        
        # 去除边缘，第二个参数阈值
        _, glyph_image_binary = cv2.threshold(glyph_image, 100, 255, cv2.THRESH_BINARY)



        # 找到非零（即前景）像素的坐标 
        contours, hierarchy = cv2.findContours(glyph_image_binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image, contours, -1, color, -1, cv2.LINE_AA, offset=(x + left, y - top)) 

        
        x += face.glyph.advance.x >> 6  



def update_ui_info(frame):

    global camera_status
    # 正常情况
    if camera_status:

        
        # 左下：显示菜单
        global menu_info,assistant_flag,capture_count,pdf_count
        menu_info = "AI辅助诊断:%s\n已采集图像:%2d张\n已生成报告:%2d份\n" % ("开启" if assistant_flag else "关闭",capture_count,pdf_count)
        draw_chinese_text(frame,menu_info,(50,900),font_size=30,color=(0,255,0))
        
        if assistant_flag:
            # 右上：显示概率
            global disease_probability_info
            draw_chinese_text(frame,disease_probability_info,(1700,60),font_size=25,color=(0,0,255))

            
            # 右下：显示精简版报告和建议
            global report_simplified_info
            global treatment_simplified_info
            report_simplified = report_simplified_info + treatment_simplified_info
            # 使用join()和换行符来打印多行
            chunk_size = 16  
            chunks = [report_simplified[i:i+chunk_size] for i in range(0, len(report_simplified), chunk_size)]  
            multi_line_string = "\n".join(chunks)
            draw_chinese_text(frame,multi_line_string,(1500,700),font_size=25,color=(0,0,255))
        
    # 相机离线
    else:
        draw_chinese_text(frame,"请检查相机是否正常连接",(60,60),font_size=30,color=(0,255,0))
    

    return frame

def init_camera():
    global video,frame,camera_status,key_value
    try:
        video = cv2.VideoCapture(0,cv2.CAP_V4L2)
        video.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter.fourcc('M','J','P','G'))
        if not video.isOpened():
            raise BaseException
    except BaseException:
        camera_status = False
        frame = update_ui_info(white_img)
        cv2.imshow("image",frame)
        cv2.waitKey(500)
        print("faild open camera!")
    else:
        camera_status = True
        
        
        video.set(cv2.CAP_PROP_FRAME_WIDTH, weight)
        video.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        video.set(cv2.CAP_PROP_FPS,30)
        
        fps = video.get(cv2.CAP_PROP_FPS)
        size = (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        print(fps)
        print(size)
        
        # 获取FourCC  
        fourcc = int(video.get(cv2.CAP_PROP_FOURCC))  
        # 将FourCC转换为对应的字符  
        fourcc_char = chr((fourcc >> 0) & 0xFF) + chr((fourcc >> 8) & 0xFF) + chr((fourcc >> 16) & 0xFF) + chr((fourcc >> 24) & 0xFF)  
        print(f"Current FourCC: {fourcc_char}") 



    

def check_path_isexist(path_s):
    return os.path.isdir(path_s)

# 获取当天保存图片目录中最新的一张照片的名称 
def get_latest_jpg_file(directory):  
    # 初始化最新文件的信息
    latest_file = None
    latest_file_time = 0  
  
    # 遍历指定目录下的所有.jpg文件  
    for filename in glob.glob(os.path.join(directory, '*.jpg')):  
        if os.path.isfile(filename):  # 确保是一个文件，而不是目录  
            # 获取文件的修改时间  
            file_time = os.path.getmtime(filename)  
  
            # 如果当前文件的修改时间比已知的最新文件时间更新，则更新最新文件信息  
            if file_time > latest_file_time:  
                latest_file_time = file_time  
                latest_file = filename
  
    return latest_file

# 更换文件拓展名后缀
def replace_extension(filename, new_extension):  
    # 使用os.path.splitext()分割文件名和扩展名  
    base_name, extension = os.path.splitext(filename)  
      
    # 确保新的扩展名以'.'开头，如果不是，则添加它  
    if not new_extension.startswith('.'):  
        new_extension = '.' + new_extension  
      
    # 重新组合文件名和新的扩展名  
    new_filename = base_name + new_extension  
      
    return new_filename



# 保存图片
def handle_capture(frame):
    print('handle_capture')
    global capture_count
    now = int(time.time())
    timeArray = time.localtime(now)
    nowtime_str = time.strftime("%Y-%m-%d-%H-%M-%S", timeArray)
    year,month,day = nowtime_str.split('-')[:3]
    capture_path = "output/capture/%s/%s/%s" % (year,month,day)
    dataset_path = "output/dataset/%s/%s/%s" % (year,month,day)

    if camera_status and assistant_flag:
        if not check_path_isexist(capture_path):
            os.makedirs(capture_path)
        if not check_path_isexist(dataset_path):
            os.makedirs(dataset_path)
        
        # 保存完整屏幕照片
        cv2.imwrite(capture_path+"/"+nowtime_str+".jpg",frame)
        
        
        # 裁切中心正方形,计算裁剪区域的左上角坐标  
        start_x = (1920 - 1080) // 2 - 25
        start_y = 0 
    
        # 进行裁剪  
        crop_img = frame[start_y:start_y+1080, start_x:start_x+1080]
        
        # 保存裁切后的中心画面
        cv2.imwrite(dataset_path+"/"+nowtime_str+".jpg",crop_img)
        print("save img " + nowtime_str + ".jpg on %s" % capture_path)
        print("save img " + nowtime_str + ".jpg on %s" % dataset_path)
        # 更新保存帧数
        capture_count+=1
        
    else:
        print("camera_status orassistant_flag is False!")

# 生成pdf文件并保存
def handle_pdf():
    print('handle_pdf')
    global pdf_count,capture_count
    # 如果本次开机没有捕获照片，则不生成pdf报告
    if capture_count == 0:
        return
    now = int(time.time())
    timeArray = time.localtime(now)
    nowtime_str = time.strftime("%Y-%m-%d-%H-%M-%S", timeArray)
    year,month,day = nowtime_str.split('-')[:3]
    dataset_path = "output/dataset/%s/%s/%s" % (year,month,day)
    capture_path = "output/capture/%s/%s/%s" % (year,month,day)
    pdf_path="output/pdf/%s/%s/%s" % (year,month,day)
    
    
    if not check_path_isexist(pdf_path):
        os.makedirs(pdf_path)
    
    # 获取当天dataset目录中保存的最新一张照片，用于生成pdf文件
    latest_dataset_jpg_file = get_latest_jpg_file(dataset_path)
    if latest_dataset_jpg_file:
        pass
    else:
        return
    
    # 获取当天capture目录中保存的最新一张照片，用于生成pdf文件
    latest_jpg_file = get_latest_jpg_file(capture_path)
    if latest_jpg_file:
        pass
    else:
        return
    
    # 生成pdf文件名和保存路径
    pdf_file_name=replace_extension(os.path.basename(latest_jpg_file),'pdf')
    pdf_file_path = pdf_path + "/" + pdf_file_name 



    # 读取最新一张图片，参数0表示以灰度模式读取，参数1表示以彩色模式读取  
    img = cv2.imread(latest_jpg_file, 1)  
    
    # 检查图片是否成功读取  
    if img is None:  
        print("Error: Could not read image")  
        return
    else:  
        # 计算最新一张照片的概率
        img = cv2.resize(img, (0, 0), fx=0.25, fy=0.25, interpolation=cv2.INTER_NEAREST)
        _,score = classify_disease(img)
        # 生成最大值序列
        disease_probability_indx = np.argsort(score)[::-1]

        
        # 取最大概率名称
        name=disease_category_name[disease_probability_indx[0]]
        # 取最大概率报告
        name = all_case_info_dict[name][0]['名称']
        report = all_case_info_dict[name][0]['检查结果']
        reason = all_case_info_dict[name][0]['病因']
        complication =  all_case_info_dict[name][0]['并发症']
        treatment = all_case_info_dict[name][0]['治疗建议']
        report_simplified = all_case_info_dict[name][0]['简化版结果']
        treatment_simplified = all_case_info_dict[name][0]['简化版建议']
        
        
        # print(name)
        # print(report)
        # print(reason)
        # print(complication)
        # print(treatment)
        # print(report_simplified)
        # print(treatment_simplified)
        
    

 
            
    
    

    # 指定 HTML 文件和输出的 PDF 文件名
    html_file = 'scripts/html2pdf.html'


    
    # 读取HTML文件并使用BeautifulSoup解析
    with open(html_file, 'r', encoding='utf-8') as f:  
        soup = BeautifulSoup(f.read(), 'html.parser')
   
    
    # 找到div_container标签
    div_container = soup.find('div',class_='div-container')
    
    # 创建两张图片标签  
    img1 = soup.new_tag('img')  
    img1['src'] = latest_dataset_jpg_file  # 替换为你的图片路径或URL  
    img1['alt'] = 'real picture'  # 添加alt属性，描述图片内容（可选）  

        
    # 对比图片文件
    case_path="case"
    case_pic_path=case_path+"/"+name+".jpg"
    img2 = soup.new_tag('img')  
    img2['src'] = case_pic_path  # 替换为你的图片路径或URL  
    img2['alt'] = 'contrast picture'  # 添加alt属性，描述图片内容（可选）  
    
    # 将图片添加到div_container中  
    div_container.append(img1)  
    div_container.append(img2)  
    
    
    # 找到div_result标签
    div_result = soup.find('div',class_='div-result')
    # 创建4段文本
    paragraph_1 = soup.new_tag('h3')
    paragraph_1.string='检查结果:'
    content_1 = soup.new_tag('p')
    content_1.string=report
    
    
    paragraph_2 = soup.new_tag('h3')
    paragraph_2.string='病因:'
    content_2 = soup.new_tag('p')
    content_2.string=reason
    
    paragraph_3 = soup.new_tag('h3')
    paragraph_3.string='并发症:'
    content_3 = soup.new_tag('p')
    content_3.string=complication
    
    
    paragraph_4 = soup.new_tag('h3')
    paragraph_4.string='治疗建议:'
    content_4 = soup.new_tag('p')
    content_4.string=treatment
    
    
    div_result.append(paragraph_1)
    div_result.append(content_1)
    div_result.append(paragraph_2)
    div_result.append(content_2)
    div_result.append(paragraph_3)
    div_result.append(content_3)
    div_result.append(paragraph_4)
    div_result.append(content_4)
    
    
    # 将修改后的 HTML 转换回字符串  
    modified_html = str(soup)

    
    # 将 HTML 转换为 PDF
    HTML(string=modified_html,base_url='./').write_pdf(pdf_file_path)
    print(f'save pdf: {pdf_file_path}')
    # 更新pdf张书
    pdf_count+=1


def handle_videowriter(frame):
    
    pass
    



# 加载病例json文件
all_case_info_dict={}
disease_category_name=[]
disease_category_num=0
def load_case():
    print('load case json file')
    global all_case_info_dict,disease_category_name,disease_category_num
    # 遍历case目录
    for filename in glob.glob(os.path.join('case', '*.json')): 
        # 读取json文件
        # case=0
        with open(filename, 'r') as file: 
            try: 
                case = json.load(file)
            except:
                pass
            else:
                keys = case.keys()
                name=list(keys)[0]
                all_case_info_dict[name]=case[name]
    # 所有病例名称列表
    disease_category_name=list(all_case_info_dict.keys())
    # 病例名称种类个数
    disease_category_num = len(disease_category_name)
    print(disease_category_name)
    print(disease_category_num)



if __name__ == "__main__":
    key_value=""
    capture_count=0
    pdf_count=0
    
    # 概率信息
    disease_probability_info=""
    menu_info="" # 菜单信息
    report_simplified_info=""       # 精简报告信息
    treatment_simplified_info=""    # 精简版建议

    weight = 1920
    height = 1080
    
    camera_status = False
    assistant_flag = False
    handle_AI_flag = False
    



    # font25 = ImageFont.truetype('/usr/share/fonts/truetype/google/NotoSansCJKsc.ttf', 25)
    # font24 = ImageFont.truetype('/usr/share/fonts/truetype/google/NotoSansCJKsc.ttf', 24)
    white_img = np.zeros((height, weight, 3), np.uint8)
    white_img.fill(255)

    video = None
    frame = None
    
    # 加载模型model 
    model = LightNet(6)

    # 加载病例case
    load_case()
    
    # 加载字体
    face=load_freetype()


    # 创建opencv主窗口
    cv2.namedWindow("image",cv2.WINDOW_FREERATIO)
    cv2.setWindowProperty("image", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
    # 记录开始时间和帧计数  
    start_time = time.time()  
    frame_count = 0 

    while True:       
        # 检查相机状态
        if camera_status:
            try:
                ret, frame = video.read()
                if (not ret) or (frame is None):
                    raise BaseException
            except BaseException:
                camera_status = False
                
            else:
                # 如果AI被打开
                if(not handle_AI_flag) and assistant_flag:
                    # 每隔一段时间抽取一帧进行AI计算
                    handleAI = threading.Timer(0,handle_AI,(frame,))
                    handleAI.start()
                    handle_AI_flag = True
                
                if not assistant_flag:
                    disease_probability_info=""
                    report_simplified_info=""
                    treatment_simplified_info=""
                    
                frame_count+=1
                
                # # 运行一定时间后退出循环，以避免无限循环  
                # if (time.time() - start_time) > 20:  # 例如，运行10秒钟  
                #     break 
                    
                
            
                # 给帧添加文字信息
                start_in=time.time()
                frame = update_ui_info(frame)
                print(f'use:{(time.time()-start_in)*1000} ms')
                
                cv2.imshow("image", frame)
                key_value=cv2.waitKey(1)
   
                if key_value == -1:
                    pass
                else:
                    # 监听遥控命令
                    if key_value == 27:# 退出程序
                        break
                    elif key_value == 24 or key_value==49:# 开关AI
                        assistant_flag = not assistant_flag
                        if assistant_flag:
                            print('open AI')
                        else:
                            print('close AI')


                    elif key_value == 13 or key_value==50: # 保存照片
                        handleCapture = threading.Timer(0,handle_capture,(frame,))
                        handleCapture.start()
                        
                    elif key_value == 103 or key_value==51:# 生成pdf文件
                        
                        handlePdf = threading.Timer(0,handle_pdf)
                        handlePdf.start()
                    elif key_value == 52: # 保存视频
                        handleVideoWriter = threading.Timer(0,handle_videowriter,(frame,))
                        handleVideoWriter.start()
                
                
        else:
            print("check camera!")
            init_camera()

           


    # 计算并打印估算的帧率  
    elapsed_time = time.time() - start_time  
    estimated_fps = frame_count / elapsed_time  
    print(f"input FPS: {estimated_fps:.2f} FRAMES:{frame_count}")


        
                
    if video:
        video.release()
    cv2.destroyAllWindows()
