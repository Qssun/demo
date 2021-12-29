import os
import cv2
import math
import numpy as np
from numpy.core.fromnumeric import shape
from paddleocr import PaddleOCR
from yolo_detect import my_detect_application, detect
from PIL import Image, ImageDraw, ImageFont


min_length_line = 5

def reco_text(img_path, objects_box, ocr_model, save_img_path, show=False, save_img=False):
    # 文字信息识别
    big_img = cv2.imread(img_path, 0)
    file_name = os.path.split(img_path)[1]
    name, suffix = os.path.splitext(file_name)
    text_info = []
    for object in objects_box:
        if object[0] == 0 and ocr_model:
            x_center = object[1]
            y_center = object[2]
            width = object[3]
            height = object[4]
            small_img = big_img[int(y_center-height/2):math.ceil(y_center+height/2), int(x_center-width/2):math.ceil(x_center+width/2)]
            if small_img.shape[0]>small_img.shape[1]:
                small_img = np.rot90(small_img, 3)

            small_img_empty = np.ones(shape=(small_img.shape[0]+min_length_line*8, small_img.shape[1]+min_length_line*8))*255
            small_img_empty[min_length_line*4:(small_img.shape[0]+min_length_line*4), min_length_line*4:(small_img.shape[1]+min_length_line*4)] = small_img[:, :]
            small_img_empty = small_img_empty.astype(np.uint8)
        
            result = ocr_model.ocr(small_img_empty)
            if len(result)>0:
                text, text_probability = result[0][1][0], result[0][1][1]
            else:
                text, text_probability = 'Unknow', 0

            if text_probability >= 0:
                text_info.append(text)
            else:
                text_info.append('Unknow')
        else:
            text_info.append('')

    add_text_object_box = []
    for pos,text in zip(objects_box, text_info):
        pos.append(text)
        add_text_object_box.append(pos)

    if show or save_img and ocr_model:
        big_img = cv2.cvtColor(big_img, cv2.COLOR_GRAY2RGB)
        big_img = Image.fromarray(big_img)
        font = ImageFont.truetype('../data/Fonts_all/simfang.ttf', 18)
        draw = ImageDraw.Draw(big_img)
        
        for object in add_text_object_box:
            if object[0] == 0:
                x_center = object[1]
                y_center = object[2]
                width = object[3]
                height = object[4]
                text = object[6]
                draw.text((x_center-width/2, y_center-height/2-18), text, fill='red', font=font)
                draw.rectangle( (x_center-width/2, y_center-height/2, x_center+width/2, y_center+height/2), outline='red')
        if save_img:
            big_img.save(r'{}/text_{}{}'.format(save_img_path, name, suffix))
        if show:
            big_img.show()
        
    return add_text_object_box
        

def load_ocr_model(enhance, det_model_dir=r'pdmodel/ch_ppocr_server_v2.0_det_infer',\
                    rec_model_dir=r'pdmodel/ch_ppocr_server_v2.0_rec_infer', \
                    cls_model_dir=r'pdmodel/ch_ppocr_mobile_v2.0_cls_infer'):
    if enhance:
        ocr_model = PaddleOCR(det_model_dir=det_model_dir, rec_model_dir=rec_model_dir, cls_model_dir=cls_model_dir, use_angle_cls=True)
    else:
        ocr_model = PaddleOCR(use_angle_cls=True)

    return ocr_model

if __name__=='__main__':
    img_size = 5000
    img_path='../data/sample/labelme/002.png'
    model_path = 'yolov5s.pt'
    config_path = 'data/config_new.json'

    big_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    file_name = os.path.split(img_path)[1]
    name, suffix = os.path.splitext(file_name)

    model, stride, img_size, device = detect(weights=model_path, img_size=img_size) # 加载yolo检测模型
    object_box = my_detect_application(img_path=img_path, img_size=img_size, model=model, stride=stride, device=device)

    objects_list, big_img_ori, no_cell_sign_img, thin_line_img = \
        get_cell_conn_detail_info(object_box, big_img, config_path, file_name=name, show=False, save_small_img=False) # 建立训练集

    ocr_model = load_ocr_model() # 加载ocr识别模型
    objects_list = reco_text(objects_list, big_img_ori, ocr_model, name, show=True)
