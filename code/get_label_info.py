import json
import os


def get_label_info(config_json_path, statistics_data):
    # 依据config_json_path,返回以label为键的字典
    with open(config_json_path, 'r', encoding='utf-8') as json_f:  # 加载组件信息
        component_dict = json.load(json_f)
        for item in component_dict['shapes']:
            label = int(item['label'])
            statistics_data[label] += 1
    return statistics_data


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
    38: 'x(multiple)',
    39: chr(934).upper(),
    40: '/',
    41: '-'
}


if __name__ == '__main__':
    parent_path = r'../data/label2' # 将该处修改为标注的文件夹名称
    expect_num = 30

    statistics_data = {}
    for i in range(len(label_info.keys())):
        statistics_data[i] = 0
    config_json_path_list = [os.path.join(parent_path, x) for x in os.listdir(parent_path) if '.json' in x]
    for config_json_path in config_json_path_list:
        get_label_info(config_json_path, statistics_data)
    
    print('{:-^20}{:-^20}{:-^20}'.format('-', '-', '-'))
    print('{:^18}{:^15}{:^15}'.format('符号', '已标记数目', '仍需标记数目'))
    for index, label in enumerate(statistics_data.keys()):
        if index % 5==0:
            print('{:-^20}{:-^20}{:-^20}'.format('-', '-', '-'))
        print('{:^20}{:^20}{:^20}'.format(label_info[label], statistics_data[label], max(0, expect_num-statistics_data[label])))
    print('{:-^20}{:-^20}{:-^20}'.format('-', '-', '-'))
