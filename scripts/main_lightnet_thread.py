import cv2
import numpy as np
import sklearn.metrics as mr
import os
from PIL import Image, ImageDraw, ImageFont
import threading
import time
import torch

from networks import LightNet
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.platypus import Image as REImage 


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

template_img_root = r"./data/template"
disease_category_num = os.listdir(template_img_root)[:-1]
disease_category_name = []
with open(template_img_root+"/disease", 'r', encoding='utf-8') as file:
    for line in file:
        disease_category_name.append(line.strip())


def classify_disease(target_img):
   target_img = torch.from_numpy(cv2.cvtColor(target_img,cv2.COLOR_BGR2GRAY).astype(np.float32).reshape((1,1,270,480)))

   with torch.no_grad():
    global model
    model.eval()
    output = model(target_img)
    
    disease_probability = output.numpy()[0]
    disease_probability_indx = np.argsort(disease_probability)[::-1]

    print("可能疾病为：")
    res = ""
    for i in  disease_probability_indx:
        print(disease_category_name[i], decimal_to_percentage(disease_probability[i]))
        res += disease_category_name[i] + decimal_to_percentage(disease_probability[i]) + "\n"
    return res,disease_probability

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

def do(frame):

    # frame = img_resize(frame)
    frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25, interpolation=cv2.INTER_NEAREST)
    global res,assistant_flag
    if assistant_flag:
        res,_ = classify_disease(frame)
    else:
        res=""
    print("do")

    global cls_flag,res_flag
    cls_flag = False
    res_flag = True
    
    time.sleep(3)
    res_flag = False

def put_text(frame, text, locxy=(60,60), color=(255,0,0)):
    img_PIL = Image.fromarray(frame[..., ::-1])  # 转成 PIL 格式
    draw = ImageDraw.Draw(img_PIL)  # 创建绘制对象
    draw.text(xy=locxy, text=text[0], font=font, fill=color)
    draw.text(xy=(60,900), text=text[1], font=font, fill=(0,255,0))
    frame = cv2.cvtColor(np.asarray(img_PIL), cv2.COLOR_RGB2BGR)  # 再转成 OpenCV 的格式，记住 OpenCV 中通道排布是 BGR
    return frame

def init_camera():
    global video,res,frame,camera_status,status_menu
    try:
        video = cv2.VideoCapture(0)
        if not video.isOpened():
            raise BaseException
    except BaseException:
        camera_status = False
        res = "请检查摄像头是否正常安装！"
        frame = put_text(white_img,[res,status_menu])
        print("faild open camera!")
    else:
        camera_status = True
        res=""
        video.set(cv2.CAP_PROP_FPS,30)
        video.set(3, weight)  # width=1920
        video.set(4, height)  # height=1080
        fps = video.get(cv2.CAP_PROP_FPS)
        print(fps)
        size = (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        print(size)
        cv2.namedWindow("image",cv2.WINDOW_FREERATIO)
        cv2.setWindowProperty("image", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

def check_path_isexist(path_s):
    return os.path.isdir(path_s)

def unlock_push():
    global push_space,push_esc,push_q,push_w
    push_space = False
    push_esc = False
    push_q = False
    push_w = False
    
    print("unlock push!")

def lock_push():
    global push_space,push_esc,push_q,push_w
    push_space = True
    push_esc = True
    push_q = True
    push_w = True

    print("lock push!")

def out_pdf(path_s,text,img_path):
    # 调用模板，创建指定名称的PDF文档
    doc = SimpleDocTemplate(path_s)
    pdfmetrics.registerFont(TTFont('SimSun', '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc'))  #注册字体

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(fontName='SimSun', name='Song', leading=20, fontSize=12))  #自己增加新注册的字体
    story =[]


    text = Paragraph("根据以上采集图片的分析，您可能患有%s。"%text[1]+text[0], styles['Song'])#使用新字体
    
    story.append(REImage(img_path))
    story.append(text)
    doc.build(story)

def update_status_menu():
    global assistant_flag,pic_pool,status_menu

    pic_len = len(pic_pool)

    status_menu = "AI辅助诊断：%s     已采集图像：%d" % ("开启" if assistant_flag else "关闭", pic_len)

if __name__ == "__main__":

    model = LightNet(6)
    model_path = ""
    # model.load_state_dict(torch.load(model_path))

    weight = 1920
    height = 1080

    assistant_flag = False
    cls_flag = False
    res_flag = False
    camera_status = False

    # push lock
    push_space = False
    push_esc = False
    push_q = False
    push_w = False

    frame = None
    video = None
    res = ""
    status_menu = "AI辅助诊断：关闭     已采集图像：0"
    pic_pool = []

    # font = ImageFont.truetype('STZHONGS.TTF', 20)  # 字体设置，Windows系统可以在 "C:\Windows\Fonts" 下查找
    font = ImageFont.truetype('/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc', 22)  # 字体设置，Windows系统可以在 "C:\Windows\Fonts" 下查找
    white_img = np.zeros((height, weight, 3), np.uint8)
    white_img.fill(255)
    report_tmp = {
        "耳膜穿孔": "耳膜穿孔，常由感染、外伤或压力变化引起。治疗方式视其原因、大小、位置及是否影响听力决定。常见方法：观察等待：许多小的耳膜穿孔能自愈，需避免水进入耳朵及复发感染"+
            "药物治疗：由中耳炎引发的穿孔可能需抗生素等药物治疗，有时还需同时治疗伴随疾病如过敏、鼻窦问题等。耳膜修补术：较大、无法愈合或影响听力的穿孔可能需手术修补。保持耳朵清洁和干燥：在治疗过程中很重要，防止感染加重。",
        "耵聍结石": "耵聍结石通常由耵聍积聚硬化形成。治疗方式包括：药物软化：用3-5%碳酸氢钠溶液软化耵聍，随后用温水冲出。采耳器：在医院用采耳器吸引耳结石。耵聍钩取出：使用专门工具钩取。"+
            "外耳道吸引或冲洗：药物软化后，用生理盐水冲洗出耳结石。同时，保持耳部清洁，避免灰尘环境，佩戴耳塞。维持均衡饮食，提高免疫力，定期检查耳部，防止疾病",
        "耵聍栓塞": "耵聍栓塞的治疗通常包括以下几种方法：耳朵冲洗：使用温生理盐水冲洗外耳道，清除耵聍。药物软化：用5%-10%的碳酸氢钠溶液滴耳软化耵聍，每天3-5次，随后冲洗。耵聍钩取出：用钩子等工具直接取出耵聍。"+
            "滴注碳酸氢钠甘油：适用于耵聍紧致且位置较深的情况，软化后进行冲洗。自洁观察：在些情况下，耳垢会自然脱落，无需处理。"+
            "在处理耵聍栓塞时，推荐到正规医疗机构进行，避免自行清洗耳朵导致进一步伤害。医生会根据实际情况选择适合的治疗方式。在治疗后，进行耳部检查来确保耵聍已被彻底清除。同时，保持耳道卫生，定期检查，预防耵聍栓塞的再次发生。",
        "外耳道异物": "外耳道异物的治疗方法主要包括：使用甘油、食用油、酒精软化或杀死昆虫类异物后再取出。用亮光诱使飞虫出来。避免用镊子取圆形异物，以免推深伤鼓膜。冲洗法适用于小昆虫等，但不适用于易膨胀或锐利异物。"+
                "民间方法（使用植物油、酒等）需谨慎使用。面对耳道异物，建议前往医院由专业医生处理，避免自行处理造成伤害。",
        "外耳道炎": "外耳道炎的治疗方法主要包括：使用滴耳液：对于轻度的外耳道炎，可以使用像硼酸这样的药物进行局部治疗，滴耳后用氢化可的松或地塞米松减轻炎症。局部用药：对于一些普遍的外耳道炎，常用局部抗菌药物滴耳，如左氧氟沙星滴耳液、妥布霉素滴耳液、新霉素滴耳液等。"+
            "全身应用抗生素：在发生严重的外耳道炎或蜂窝织炎等情况时，可能需要全身应用抗生素进行治疗，如口服头孢氨苄。清耳：在耳镜的帮助下清除耳道内的耵聍、脓性分泌物或皮屑。疼痛管理：可服用镇痛药以减轻疼痛。"+
            "需要注意的是，严重的外耳道炎需要及时就医，并按照医生的指示进行治疗。而在日常生活中，我们要注意保持耳朵清洁，尽量避免湿水，这样可以预防外耳道炎的发生。",
        "油耳": "油耳是体质性问题，一般不需特殊治疗。如引起不适或耵聍栓塞，治疗方式包括：保持耳道清洁，可用3%双氧水、7%酒精或4%硼酸酒精清理。耳道冲洗，用生理盐水冲洗，需医生指导。"+
            "平时注意耳部卫生，避免过度清理。若耵聍严重阻塞，可考虑手术治疗。",
    }

    while True:
        
        if camera_status:

            try:
                ret, frame = video.read()
                if (not ret) or (frame is None):
                    raise BaseException
            except BaseException:
                camera_status = False
                res = "请检查摄像头是否正常安装！"
               
                frame = put_text(white_img,[res,status_menu])
                print("not frame!")
            else:
                if(not cls_flag) and assistant_flag:
                    cls_t = threading.Timer(1,do,(frame,))
                    cls_t.start()
                    cls_flag = True
                    
                if not assistant_flag:
                    res=""
                frame = put_text(frame,[res,status_menu])
        else:
            print("check camera!")
            time.sleep(3)
            init_camera()
        if frame is not None:
            
            # cv2.resizeWindow("image", 1440, 790)
            
            # cv2.moveWindow("image",100,100)
            cv2.imshow("image", frame)
            c = cv2.waitKey(1)
            if c == 27:
                break
            elif c == 32:
                if not push_space:
                    lock_push()
                    assistant_flag = not assistant_flag
                    update_status_menu()
                    print("assistant_flag:"+str(assistant_flag))
                    if not assistant_flag:
                        res=""
                    push_t = threading.Timer(3.5,unlock_push)
                    push_t.start()
            elif c == 119:

                if not push_w :
                    lock_push()
                     # 获得当前时间时间戳
                    now = int(time.time())
                    #转换为其他日期格式,如:"%Y-%m-%d %H:%M:%S"
                    timeArray = time.localtime(now)
                    nowtime_str = time.strftime("%Y-%m-%d-%H-%M-%S", timeArray)
                    year,month,day = nowtime_str.split('-')[:3]
                    save_path = "output/%s/%s/%s" % (year,month,day)

                    if camera_status and assistant_flag:
                        if len(pic_pool) >=3 :
                            total_score = 0
                            white_mask = np.zeros((50, 640, 3), np.uint8)
                            white_mask.fill(255)
                            sample_img = np.zeros((50*2+360*3,640,3),np.uint8)
                            pic_pool_resize = []
                            for img_it in pic_pool:
                                # img_it = img_resize(img_it)
                                img_it = cv2.resize(img_it, (0, 0), fx=0.25, fy=0.25, interpolation=cv2.INTER_NEAREST)
                                _,score = classify_disease(img_it)
                                pic_pool_resize.append(img_it)
                                total_score +=score


                            sample_img[0:360,:,:]=pic_pool_resize[0]
                            sample_img[360:360+50,:,:]=white_mask
                            sample_img[360+50:360*2+50,:,:]=pic_pool_resize[1]
                            sample_img[360*2+50:360*2+50*2,:,:]=white_mask
                            sample_img[360*2+50*2:360*3+50*2,:,:]=pic_pool_resize[2]

                            disease_probability_indx = np.argsort(total_score)[::-1]
                            # sample_img = img_resize(sample_img)
                            sample_img = cv2.resize(sample_img, (0, 0), fx=0.25, fy=0.25, interpolation=cv2.INTER_NEAREST)
                            cv2.imwrite(save_path+"/"+nowtime_str+".jpg",sample_img)
                            out_pdf(
                                save_path+"/"+nowtime_str+".pdf",
                                [
                                    report_tmp[disease_category_name[disease_probability_indx[0]]],
                                    disease_category_name[disease_probability_indx[0]]
                                ],
                                save_path+"/"+nowtime_str+".jpg"
                            )
                            pic_pool = []
                            print("save img " + nowtime_str + ".pdf on %s" % save_path)
                        else:
                            pic_pool.append(frame)
                            if not check_path_isexist(save_path):
                                os.makedirs(save_path)
                            
                            else:
                                cv2.imwrite(save_path+"/"+nowtime_str+".jpg",frame)
                                print("save img " + nowtime_str + ".jpg on %s" % save_path)
                        update_status_menu()
                    else:
                        print("camera_status orassistant_flag is False!")
                    push_t = threading.Timer(3.5,unlock_push)
                    push_t.start()

                
    if video:
        video.release()
    cv2.destroyAllWindows()
# 可能疾病为：
# 耵聍栓塞 30.77%
# 耳膜穿孔 29.10%
# 外耳道炎 10.65%
# 耵聍结石 10.26%
# 油耳 9.80%
# 耳结石 9.42%
