from functools import total_ordering
import json
import os
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFilter
import random
import numpy as np
import time

import json
import yaml
from multiprocessing import Pool
import cv2

Image.MAX_IMAGE_PIXELS = 2300000000

def add_noise(img):
    # 图像预处理
    p_0 = np.random.uniform(0, 1)
    if p_0 < 0.01:
        img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)) # 定义结构元素的形状和大小
        img_gray = cv2.erode(img_gray, kernel) # 腐蚀操作
        img = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    p_1 = np.random.uniform(0, 1)
    if p_1 < 0.005: # 模糊
        img = img.filter(ImageFilter.GaussianBlur(radius=3))

    p_2 = np.random.uniform(0, 1)
    if p_2 < 0.3:
        img = np.array(img)
        rows, cols, _ = img.shape
        for i in range(int(rows*cols*0.01)):
            x = np.random.randint(0, rows)
            y = np.random.randint(0, cols)
            p_3 = np.random.uniform(0, 1)
            if p_3 < 0.5:
                img[x, y, :] = 255
            else:
                img[x, y, :] = 0
        img = Image.fromarray(img)
    
    p_3 = np.random.uniform(0, 1)
    if p_3 < 0.3:
        img = np.array(img)
        random_noise = np.random.normal(1, 0.1, size=img.shape)
        img = img * random_noise
        img = np.clip(img, 0, 255)
        img = img.astype(np.uint8)
        img = Image.fromarray(img)

    img = np.array(img)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    img = Image.fromarray(img)

    return img


def crop_one_subplot(big_img, item_0, crop_width, crop_height, save_template_img, save_label_parent_path, label_info, save_img_parent_path, file_name, index, suffix, color_list, t, imageHeight, imageWidth, random_repeat_num):
    # 以一个子图为中心，截取【crop_width, crop_height】区域内的图像
    center_label, x_center_0, y_center_0, width_0, height_0 = item_0
    symbol_template_img = big_img.crop(
        (x_center_0-width_0/2, y_center_0-height_0/2, x_center_0+width_0/2, y_center_0+height_0/2))
    if not os.path.exists(os.path.join(save_template_img, '{}'.format(center_label))):
        os.makedirs(os.path.join(save_template_img, '{}'.format(center_label)))
    number = len([x for x in os.listdir(os.path.join(
        save_template_img, '{}'.format(center_label)))])
    symbol_template_img.save(os.path.join(save_template_img, '{}'.format(
        center_label), '{}{}'.format(number, suffix)))

    for i in range(random_repeat_num):  # 单个目标选取20个中心，数据增强
        random_x, random_y = random.randint(-int(0.5*(crop_width-width_0)), int(0.5*(
            crop_width-width_0))), random.randint(-int(0.5*(crop_height-height_0)), int(0.5*(crop_height-height_0)))
        x_center_0, y_center_0 = x_center_0+random_x, y_center_0+random_y
        x_center_0, y_center_0 = np.clip(x_center_0, crop_width/2, imageWidth), np.clip(
            y_center_0, crop_height/2, imageHeight) 
        small_img = big_img.crop((x_center_0-crop_width/2, y_center_0 -
                                  crop_height/2, x_center_0+crop_width/2, y_center_0+crop_height/2))
        draw = ImageDraw.Draw(small_img)
        if i>=min(random_repeat_num-1, random_repeat_num*0.8):
            save_label_parent_path = save_label_parent_path.replace('train', 'val')
            save_img_parent_path = save_img_parent_path.replace('train', 'val')
        if not os.path.exists(save_label_parent_path):
            os.makedirs(save_label_parent_path)
        with open(os.path.join(save_label_parent_path, '{}_{}.txt'.format(file_name, index)), 'wb') as f:
            # small_img = add_noise(small_img)  # 添加噪声
            for item in label_info:  # 在该子图区域内的其他目标
                label, x_center, y_center, width, height = item
                if label==0:
                    choose = choose_region((x_center, y_center, width, height),
                        (x_center_0, y_center_0, crop_width, crop_height), 0.1)
                else:
                    choose = choose_region((x_center, y_center, width, height),
                        (x_center_0, y_center_0, crop_width, crop_height), t)
                if choose:
                    new_x_center, new_y_center = x_center-x_center_0 + \
                        crop_width/2, y_center-y_center_0+crop_height/2
                    f.write('{} {} {} {} {}\n'.format(label, new_x_center/crop_width, new_y_center /
                                                      crop_height, width/crop_height, height/crop_height).encode('utf-8'))
                    # draw.rectangle((new_x_center-width/2, new_y_center-height/2, new_x_center+width/2, new_y_center+height/2), \
                    #     fill=None, outline=color_list[int(label)], width=1)
            if not os.path.exists(save_img_parent_path):
                os.makedirs(save_img_parent_path)
            small_img.save(os.path.join(save_img_parent_path, '{}_{}{}'.format(file_name, index, suffix)))
        index = index + 1


def mesh_data_train(json_label_path, big_img_parent_path, crop_width, crop_height, save_img_parent_path, save_label_parent_path, color_list, t, save_template_img='../data/samples'):
    # 将Labelme的标注文件转换为yolo的标准格式,并将标签存为txt文件，用于模型训练
    label_info = []
    with open(json_label_path, 'r', encoding='utf-8') as f:
        ret_dic = json.load(f)
        imageHeight = ret_dic['imageHeight']
        imageWidth = ret_dic['imageWidth']
        imagePath = os.path.split(big_img_parent_path)[1]
        big_img = Image.open(big_img_parent_path)

        for _, item in enumerate(ret_dic['shapes']):
            label = item['label']
            [[x_0, y_0], [x_1, y_1]] = item['points']
            x_center, y_center = (x_0+x_1)/2, (y_0+y_1)/2
            width, height = x_1-x_0, y_1-y_0
            assert x_center > 0 and y_center > 0 and width > 0 and height > 0
            label_info.append([label, x_center, y_center, width, height])
            if width > crop_width or height > crop_height and label==0:
                print('\033[1;31单个目标的尺寸大于子图大小，建议调大子图的大小！单目标大小：{:.0f}-{:.0f}，子图大小：{:.0f}-{:.0f}，图编号：{}.\033[0m'.format(width, height,
                                crop_width, crop_height, big_img_parent_path))
        
        pbar = tqdm(total=len(label_info))
        # pbar.set_description(' Flow ')
        update = lambda *args: pbar.update()
        index = 0
        file_name, suffix = os.path.splitext(imagePath)
        p = Pool(min(24, int(os.cpu_count()*0.85)))
        random_repeat_num = 5
        for item_0 in label_info:  # 选定单个目标，将其中心作为子图的中心（添加噪声偏量）
            p.apply_async(crop_one_subplot, args=(big_img, item_0, crop_width, crop_height, save_template_img,
                save_label_parent_path, label_info, save_img_parent_path, file_name, index, suffix, color_list, t, imageHeight, imageWidth, random_repeat_num, ), callback=update)
            index += random_repeat_num
        p.close()
        p.join()

def choose_region(rectangle_1, rectangle_2, t):
    # 计算目标占据整个子图的百分比,当占比大于t时候返回True
    iou = calc_iou(rectangle_1, rectangle_2)
    if iou > t:
        return True
    else:
        return False


def calc_iou(rectangle_1, rectangle_2):
    # 计算两个区域的合并比
    x_1_center, y_1_center, width_1, height_1 = rectangle_1
    x_2_center, y_2_center, width_2, height_2 = rectangle_2
    x_1_0, y_1_0, x_1_1, y_1_1 = x_1_center-width_1/2, y_1_center - \
        height_1/2, x_1_center+width_1/2, y_1_center+height_1/2
    x_2_0, y_2_0, x_2_1, y_2_1 = x_2_center-width_2/2, y_2_center - \
        height_2/2, x_2_center+width_2/2, y_2_center+height_2/2

    x_0 = max(x_1_0, x_2_0)
    y_0 = max(y_1_0, y_2_0)
    x_1 = min(x_1_1, x_2_1)
    y_1 = min(y_1_1, y_2_1)

    s_small_rectangle_1 = (x_1_1-x_1_0) * (y_1_1-y_1_0)  # 第一个矩形的面积
    if (x_1-x_0) > 0 and (y_1-y_0) > 0:
        s_share = (x_1-x_0) * (y_1-y_0)
        iou = s_share/s_small_rectangle_1
        return iou
    else:
        return 0

def build_dataset(label_img_parent_path, crop_width, crop_height, save_img_parent_path, save_label_parent_path, t, save_template_parent_path):
    labelme_json_list = [x for x in os.listdir(label_img_parent_path) if '.json' in x]
    for labelme_json_name in labelme_json_list:
        time.sleep(5)
        print(labelme_json_name)
        file_name, suffix = os.path.splitext(labelme_json_name)
        img_file_path = os.path.join(label_img_parent_path, file_name+'.png')
        json_label_path = os.path.join(label_img_parent_path, labelme_json_name)
        if os.path.exists(img_file_path):
            mesh_data_train(json_label_path, img_file_path, crop_width, crop_height, \
                save_img_parent_path, save_label_parent_path, COLOR_LIST, t, save_template_parent_path)
        else:
            print('未找到文件：{}'.format(file_name+'.png'))


def config_yaml(config_json_path, config_yolo_dataset, train_dataset_path, val_dataset_path, template_img_path):
    # 依据Labelme标记的json文件，生成各组件的相关信息，生成信息保存在config_json_path, config_yolo_dataset里面
    with open(config_json_path, 'r', encoding='utf-8') as json_f:  # 加载组件信息
        component_dict = json.load(json_f)
        for component in component_dict.keys():
            if component != 'demo':
                label = component_dict[component]['label']
                template_img_path_one_class = os.path.join(template_img_path, str(label))
                if os.path.exists(template_img_path_one_class):
                    sample_path = [os.path.join(template_img_path_one_class, x)
                        for x in os.listdir(template_img_path_one_class)]  # 组件模板图片的路径
                    height_width = calc_height_width(sample_path)
                else:
                    sample_path = []
                    height_width = []
                # 保存组件模板图片的路径
                component_dict[component]['samples_path'] = sample_path
                # 保存组件模板的长宽比（均值、最大值、最小值）
                component_dict[component]['height_width'] = height_width

    with open(config_json_path, 'w', encoding='utf-8') as json_f:  # 加载组件信息
        json.dump(component_dict, json_f, ensure_ascii=False, indent=4)

    # with open(config_yolo_dataset, 'r', encoding='utf-8') as yaml_f:  # 加载训练数据集信息
    #     dataset_dict = yaml.safe_load(yaml_f)
    #     dataset_dict['nc'] = len(set(list(component_dict.keys()))) - 1
    #     component_list = []
    #     for index in range(dataset_dict['nc']):
    #         used = 0
    #         for component in component_dict.keys():
    #             if component_dict[component]['label'] == index:
    #                 component_list.append(component)
    #                 used = 1
    #         assert used == 1, '{}中未找到标签{}'.format(config_json_path, index)
    #     dataset_dict['names'] = component_list  # label从0到nc排序后的列表
    #     dataset_dict['train'] = train_dataset_path  # label从0到nc排序后的列表
    #     dataset_dict['val'] = val_dataset_path  # label从0到nc排序后的列表

    with open(config_yolo_dataset, 'r', encoding='utf-8') as yaml_f:  # 加载训练数据集信息
        dataset_dict = yaml.safe_load(yaml_f)
        component_list = []

        total_label = []
        for component in component_dict.keys():
            label = component_dict[component]['label']
            if label<1000: # 1000以上为特殊元件标记不采用yolo-v5识别
                total_label.append(label)
        
        max_label, min_label = max(total_label), min(total_label)
        for index in range(min_label, max_label+1):
            used = 0
            for component in component_dict.keys():
                if component_dict[component]['label'] == index:
                    name = component_dict[component]['no']
                    component_list.append(name)
                    used = 1
                    break
            if used==0:
                print('\033[1;31m元件配置文件【{}】中未找到标签{} \033[0m'.format(config_json_path, index))
                component_list.append('undefined_label_{}'.format(index))

        dataset_dict['nc'] = len(component_list)
        dataset_dict['names'] = component_list  # label从0到nc排序后的列表
        dataset_dict['train'] = train_dataset_path  # label从0到nc排序后的列表
        dataset_dict['val'] = val_dataset_path  # label从0到nc排序后的列表

    with open(config_yolo_dataset, 'w', encoding='utf-8') as yaml_f:  # 保存训练数据集信息
        yaml.safe_dump(dataset_dict, yaml_f)


def calc_height_width(imgs_path):
    # 计算组件模板图片的长/宽比值
    result_temp = []
    for img_path in imgs_path:
        img = Image.open(img_path).convert('RGB')
        long_side, short_side = max(img.size), min(img.size)
        result_temp.append(long_side/short_side)
    result_temp = np.array(result_temp)
    return np.round(np.mean(result_temp), 3), np.round(np.max(result_temp), 3), np.round(np.min(result_temp), 3)


if __name__=='__main__':
    class_type = 1100
    COLOR_LIST = ['{:0<7}'.format('#'+str(hex(int(256*256*256/(class_type+2)*x)))[
                                  2:].upper()) for x in range(class_type)][1:-1]
    
    # 功能1测试，根据标记数据生成训练集
    t = 0.5
    crop_width = 1000
    crop_height = 1000
    label_img_parent_path = r'../data/sample/labelme/train_dataset'
    save_img_parent_path = r'../data/dataset/train/images/mesh'
    save_label_parent_path = r'../data/dataset/train/labels/mesh'
    save_template_parent_path = r'../data/sample/labelme/train_dataset/crop_img'
    build_dataset(label_img_parent_path, crop_width, crop_height, save_img_parent_path, save_label_parent_path, t, save_template_parent_path)
    
    # 功能2测试，依据配置文件信息转换为coco数据集配置信息
    train_dataset_path = r'../data/dataset/train/images/mesh'
    val_dataset_path = r'../data/dataset/val/images/mesh'
    template_img_path = r'../data/sample/labels'
    config_json_path = r'data/config_process.json'
    config_yolo_dataset = r'data/coco128.yaml'
    # config_yaml(config_json_path, config_yolo_dataset, train_dataset_path, val_dataset_path, template_img_path)