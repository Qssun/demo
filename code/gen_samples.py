from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageEnhance
import os
import random
import numpy as np
import math
from mesh_data import add_noise, config_yaml
from multiprocessing import Pool
from tqdm import tqdm


def random_choose_font(all_font_type):
    # 随机选择一种字体
    num_font_type = len(all_font_type)
    font_type_index = random.randint(0, num_font_type-1)
    return all_font_type[font_type_index]

def random_choose_sign(label_info):
    # 随机选择一种符号
    num_sign_type = len(label_info.keys())
    sign_type_index = random.randint(0, num_sign_type-1)
    p = random.random()
    if p>0.5 and 10<=sign_type_index<=35:
        text = label_info[sign_type_index].upper()
    elif p<=0.5 and 10<=sign_type_index<=35:
        text = label_info[sign_type_index].lower()
    else:
        text = label_info[sign_type_index]
    text = label_info[sign_type_index]
    return text, sign_type_index

def random_sign_num(min_num, max_mun):
    # 随机确定一张图片中符号的个数
    sign_num = random.randint(min_num, max_mun)
    return sign_num

def random_sign_position(sign_num):
    # 随机确定符号的位置
    width_height_num = int(math.sqrt(sign_num*5))
    per_mesh = int(IMG_WIDTH_HEIGHT / width_height_num)
    fill_index = np.random.choice(np.arange(width_height_num**2), size=sign_num, replace=False)
    position = []
    for index in fill_index:
        position.append([index//width_height_num*per_mesh, index%width_height_num*per_mesh])
    return position

def img_split(label_info, parent_path):
    img_path = {}
    for label in label_info.keys():
        img_path[int(label)] = []
    for img_name in [x for x in os.listdir(parent_path)]:
        index = int(img_name.split('_')[0])
        img_path[index].append(os.path.join(parent_path, img_name))
    
    return img_path

def choose_img(label, img_path):
    # 依据标签，随机选择一张图片
    num = len(img_path[label])
    choose_img = random.randint(0, num-1)

    return img_path[label][choose_img]


def worker(item, save_label_parent_path, save_img_parent_path, IMG_WIDTH_HEIGHT, file_name, FONT_SIZE_RANGE, TOTAL_NUM):
    if item>0.98*(TOTAL_NUM):
        save_label_parent_path = save_label_parent_path.replace('train', 'val')
        save_img_parent_path = save_img_parent_path.replace('train', 'val')
    if not os.path.exists(save_label_parent_path):
        os.makedirs(save_label_parent_path)
    if not os.path.exists(save_img_parent_path):
        os.makedirs(save_img_parent_path)

    with open(os.path.join(save_label_parent_path, '{}_{}.txt'.format(file_name, item)), 'wb') as f:
        img = Image.new('RGB', (IMG_WIDTH_HEIGHT, IMG_WIDTH_HEIGHT), (255, 255, 255))
        draw = ImageDraw.Draw(img)
        if item>0.98*(TOTAL_NUM):
            sign_num = random_sign_num(min_num=30, max_mun=50)
        else:
            sign_num = random_sign_num(min_num=50, max_mun=80)
        position = random_sign_position(sign_num)
        for x,y in position:
            FONT_SIZE = random.randint(FONT_SIZE_RANGE[0], FONT_SIZE_RANGE[1])
            label, index = random_choose_sign(label_info)
            try_time = True
            while try_time:
                font_type = random_choose_font(all_font_type)
                try:
                    setFont = ImageFont.truetype(font_type, FONT_SIZE)
                    try_time = False
                except:
                    font_type = random_choose_font(all_font_type)
                    try_time = True
            p = random.random()
            if p>0.5:
                # 字体库
                while True:
                    if 'sign' in label or 'conn' in label or 'cell' in label or '.' in label:
                        label, index = random_choose_sign(label_info)
                    else:
                        break
                draw.text((x+2, y+2), '{}'.format(label), font=setFont, fill='black', direction=None)
                if label=='Q' or label=='q':
                    y = min(max(y+1, 0), IMG_WIDTH_HEIGHT)
                if label=='I' or label=='i' or label=='J' or label=='j':
                    x = min(max(x-3, 0), IMG_WIDTH_HEIGHT)
                img_array = np.array(img)
                for line_index in range(int((x+FONT_SIZE/2)), int((x+FONT_SIZE))):
                    if np.min(img_array[y:y+FONT_SIZE, line_index:line_index+2])>=250:
                        break
                draw.rectangle((x, y+5, line_index+3, y+FONT_SIZE+1), fill=None, outline='red', width=1)
                x_center = (x + line_index+3)/2
                y_center = (y+5 + y+FONT_SIZE+1)/2
                width = line_index+3 - x
                height = y+FONT_SIZE+1 - (y+5)
                f.write('{} {} {} {} {}\n'.format(index, x_center/IMG_WIDTH_HEIGHT, y_center/IMG_WIDTH_HEIGHT, width/IMG_WIDTH_HEIGHT, height/IMG_WIDTH_HEIGHT).encode('utf-8'))
                
            else:
                # 自定义图片图标
                while True:
                    if '_reverse' in label or 'conn' in label or 'cell' in label:
                        label, index = random_choose_sign(label_info)
                    else:
                        break
                choose_img_path = choose_img(index, img_path)
                add_img = Image.open(choose_img_path)
                width_height = add_img.size[0]/add_img.size[1]
                if 'cell' in label or 'conn' in label:
                    FONT_SIZE = int(add_img.size[1]/6)
                add_img = add_img.resize((int(FONT_SIZE*width_height), FONT_SIZE), Image.ANTIALIAS)


                bright_enhancer = ImageEnhance.Brightness(add_img)
                add_img = bright_enhancer.enhance(random.uniform(0.99,1.01))
                contrast_enhancer = ImageEnhance.Contrast(add_img)
                add_img = contrast_enhancer.enhance(random.uniform(0.99,1.5))
                # color_enhancer = ImageEnhance.Color(add_img)
                # add_img = color_enhancer.enhance(1.05)
                add_img = add_noise(add_img)


                p_1 = random.random()
                if p_1>0.5:
                    add_img = np.transpose(np.array(add_img), (1,0,2))
                    add_img = np.flipud(add_img)
                    add_img = Image.fromarray(add_img)
                    width = FONT_SIZE+4
                    height = int(FONT_SIZE*width_height)+4
                else:
                    width = int(FONT_SIZE*width_height)+4
                    height = FONT_SIZE+4

                img.paste(add_img, (x+2, y+2))
                # draw.rectangle((x, y, x+width, y+height), fill=None, outline='red', width=1)
                x_center = (x + x + width)/2
                y_center = (y + y + height)/2
                f.write('{} {} {} {} {}\n'.format(index, x_center/IMG_WIDTH_HEIGHT, y_center/IMG_WIDTH_HEIGHT, width/IMG_WIDTH_HEIGHT, height/IMG_WIDTH_HEIGHT).encode('utf-8'))
                
            if not os.path.exists(os.path.join(save_template_img, '{}'.format(index))):
                os.makedirs(os.path.join(save_template_img, '{}'.format(index)))
            number = len([x for x in os.listdir(os.path.join(
                save_template_img, '{}'.format(index)))])
            if number<=20:
                symbol_template_img = img.crop((x_center-width/2, y_center-height/2, x_center+width/2, y_center+height/2))
                symbol_template_img.save(os.path.join(save_template_img, '{}'.format(index), '{}.png'.format(number)))

        img.save(os.path.join(save_img_parent_path, '{}_{}.png'.format(file_name, item)))


label_info = {
    0: '0',
    1: '1',
    2: '2',
    3: '3',
    4: '4',
    5: '5',
    6: '6',
    7: '7',
    8: '8',
    9: '9',
    10: 'A',
    11: 'B',
    12: 'C',
    13: 'D',
    14: 'E',
    15: 'F',
    16: 'G',
    17: 'H',
    18: 'I',
    19: 'J',
    20: 'K',
    21: 'L',
    22: 'M',
    23: 'N',
    24: 'O',
    25: 'P',
    26: 'Q',
    27: 'R',
    28: 'S',
    29: 'T',
    30: 'U',
    31: 'V',
    32: 'W',
    33: 'X',
    34: 'Y',
    35: 'Z',
    36: '.',
    37: '*',
    38: 'x',
    39: '-',
    40: chr(934).upper(),
    41: 'sign_reverse_1',
    42: 'sign_reverse_2',
    43: 'sign_reverse_3',
    44: 'sign_reverse_4',
    45: 'sign_reverse_5',
    46: 'sign_reverse_6',
    47: 'sign_reverse_7',
    48: 'sign_reverse_8',
    49: 'sign_reverse_9',
    50: 'sign_reverse_10',
    51: 'sign_reverse_11',
    52: 'sign_reverse_12',
    53: 'sign_reverse_13',
    54: 'sign_reverse_14',
    55: 'sign_reverse_15',
    56: 'sign_reverse_16',
    57: 'sign_reverse_17',
    58: 'sign_reverse_18',
    59: 'sign_reverse_19',

    60: 'conn_1_J',
    61: 'conn_1_W',
    62: 'conn_1_K',
    63: 'conn_1_Y',
    64: 'conn_2_WS',
    65: 'conn_2_ES',
    66: 'conn_2_WN',
    67: 'conn_2_EN',
    68: 'conn_2_NS',
    69: 'conn_3_WES',
    70: 'conn_3_WNS',
    71: 'conn_3_WNE',
    72: 'conn_3_NES',
    73: 'conn_4',
    74: 'conn_reverse_0',
    75: 'conn_reverse_1',
    76: 'conn_reverse_2',
    77: 'conn_reverse_3',
    78: 'conn_reverse_4',
    79: 'conn_reverse_5',
    80: 'conn_reverse_6',
    81: 'conn_reverse_7',
    82: 'conn_reverse_8',
    83: 'conn_reverse_9',
    84: 'conn_reverse_10',
    85: 'conn_reverse_11',
    86: 'conn_reverse_12',
    87: 'conn_reverse_13',
    88: 'conn_reverse_14',
    89: 'conn_reverse_15',

    90: 'cell_0',
    91: 'cell_1',
    92: 'cell_2',
    93: 'cell_3',
    94: 'cell_4',
    95: 'cell_5',
    96: 'cell_6',
    97: 'cell_7',
    98: 'cell_8',
    99: 'cell_9',
    100: 'cell_10',
    101: 'cell_11',
    102: 'cell_12',
    103: 'cell_13',
    104: 'cell_14',
    105: 'cell_15',
    106: 'cell_16',
    107: 'cell_17',
    108: 'cell_18',
    109: 'cell_19',
    110: 'cell_20',
    111: 'cell_21',
    112: 'cell_22',
    113: 'cell_23',
    114: 'cell_24',
    115: 'cell_25',
    116: 'cell_26',
    117: 'cell_27',
    118: 'cell_28',
    119: 'cell_29',
    120: 'cell_30',
    121: 'cell_31',
    122: 'cell_32',
    123: 'cell_33',
    124: 'cell_34'

}


TOTAL_NUM = 10
IMG_WIDTH_HEIGHT = 640
FONT_SIZE_RANGE = [20, 30]
save_label_parent_path = r'../data/train/labels/mesh'
save_img_parent_path = r'../data/train/images/mesh'
save_template_img = r'../data/samples'
all_font_type = [os.path.join(r'../data/Fonts', x) for x in os.listdir(r'../data/Fonts') if '.fon' not in x]
img_path = img_split(label_info, parent_path=r'../data/005')

if __name__=='__main__':
    file_name = 'gen'
    p = Pool(min(6, int(os.cpu_count()*0.8)))
    pbar = tqdm(total=TOTAL_NUM)

    def update(*a):
        pbar.update()

    for item in range(TOTAL_NUM):
        p.apply_async(worker, args=(item, save_label_parent_path, save_img_parent_path, IMG_WIDTH_HEIGHT, file_name, FONT_SIZE_RANGE, TOTAL_NUM,), callback=update)
    p.close()
    p.join()

    config_yaml(r'data/config.json', r'data/coco128.yaml')