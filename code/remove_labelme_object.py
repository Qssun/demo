import json
import copy
import cv2
import numpy as np
import os

def read_labelme_json(label_json_path):
    # 读取labelme标记的json文件
    # 返回每个标记框的标签和位置
    label_info = []
    with open(label_json_path, 'r', encoding='utf-8') as f:
        ret_dic = json.load(f)
        for _, item in enumerate(ret_dic['shapes']):
            label = int(item['label'])
            [[x_0, y_0], [x_1, y_1]] = item['points']
            x_0_temp = copy.deepcopy(x_0)
            x_1_temp = copy.deepcopy(x_1)
            y_0_temp = copy.deepcopy(y_0)
            y_1_temp = copy.deepcopy(y_1)
            x_0 = min(x_0_temp, x_1_temp)
            x_1 = max(x_0_temp, x_1_temp)
            y_0 = min(y_0_temp, y_1_temp)
            y_1 = max(y_0_temp, y_1_temp)
            x_center, y_center = (x_0+x_1)/2, (y_0+y_1)/2
            width, height = x_1-x_0, y_1-y_0
            assert x_center > 0 and y_center > 0 and width > 0 and height > 0
            label_info.append([label, x_center, y_center, width, height, 1.0])
    return label_info

def edit_img(img_path, label_json_path):
    file_name = os.path.split(img_path)[1]
    name, suffix = os.path.splitext(file_name)
    big_img = cv2.imread(img_path, 0)
    big_img_text = copy.deepcopy(big_img)
    big_img_cell = copy.deepcopy(big_img)

    all_label_object = read_labelme_json(label_json_path)

    big_img = cv2.cvtColor(big_img, cv2.COLOR_GRAY2RGB)
    for index, position_info in enumerate(all_label_object):
        label = int(position_info[0])
        start_x = int(position_info[1]) - int(position_info[3]/2)
        start_y = int(position_info[2]) - int(position_info[4]/2)
        stop_x = int(position_info[1]) + int(position_info[3]/2)
        stop_y = int(position_info[2]) + int(position_info[4]/2)
        small_img = big_img[start_y:stop_y, start_x:stop_x]
        save_small_img_path = os.path.split(img_path)[0]
        save_small_img_path = os.path.join(save_small_img_path, r'small_img/{}'.format(label))
        if not os.path.exists(save_small_img_path):
            os.makedirs(save_small_img_path)
        cv2.imwrite(os.path.join(save_small_img_path, '{}_{}.png'.format(name, index)), small_img)
        
        big_img[start_y:stop_y, start_x:stop_x, :] = np.ones((stop_y-start_y, stop_x-start_x, 3))*255
        if label==0:
            big_img_text[start_y:stop_y, start_x:stop_x] = np.ones((stop_y-start_y, stop_x-start_x))*255
        elif label!=200:
            big_img_cell[start_y:stop_y, start_x:stop_x] = np.ones((stop_y-start_y, stop_x-start_x))*255
    
    for index, position_info in enumerate(all_label_object):
        label = int(position_info[0])
        start_x = int(position_info[1]) - int(position_info[3]/2)
        start_y = int(position_info[2]) - int(position_info[4]/2)
        stop_x = int(position_info[1]) + int(position_info[3]/2)
        stop_y = int(position_info[2]) + int(position_info[4]/2)
        
        cv2.rectangle(big_img, (start_x, start_y), (stop_x, stop_y), (0, 0, 255), 1)
        cv2.putText(big_img, str(index), (start_x, start_y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            
    img_path_text = img_path.replace('.png', '_no_text.png')
    cv2.imwrite(img_path_text, big_img_text)
    img_path_cell = img_path.replace('.png', '_no_cell.png')
    cv2.imwrite(img_path_cell, big_img_cell)
    img_path_mark = img_path.replace('.png', '_mark.png')
    cv2.imwrite(img_path_mark, big_img)


if __name__=="__main__":
    img_path_list = [
        "../data/sample/labelme/one_labelme/001_M.png",
        "../data/sample/labelme/one_labelme/002.png",
        "../data/sample/labelme/one_labelme/003.png",
        "../data/sample/labelme/one_labelme/005.png",
        "../data/sample/labelme/one_labelme/006.png",
        "../data/sample/labelme/one_labelme/007.png",
        "../data/sample/labelme/one_labelme/008.png",
        "../data/sample/labelme/one_labelme/009.png",
        "../data/sample/labelme/one_labelme/010.png",
        "../data/sample/labelme/one_labelme/011.png",
        "../data/sample/labelme/one_labelme/012.png",
        "../data/sample/labelme/one_labelme/013.png",
        "../data/sample/labelme/one_labelme/014.png",
        "../data/sample/labelme/one_labelme/015.png",
        "../data/sample/labelme/one_labelme/016.png",
        "../data/sample/labelme/one_labelme/017.png",
        "../data/sample/labelme/one_labelme/018.png"
    ]

    for img_path in img_path_list:
        label_json_path = img_path.replace('.png', '.json')
        if not os.path.exists(label_json_path):
            print('\033[1;31m 图片【{}】内未找到标记的json文件【{}】 \033[0m'.format(img_path, label_json_path))  # 有高亮
            continue

        edit_img(img_path, label_json_path)
