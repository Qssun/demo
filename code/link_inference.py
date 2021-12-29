import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt

from my_utils import opposite_direction, read_csv_data

import cv2
import os
import copy
import math
from PIL import Image

min_length_line = 5

def inference_confirm_link(all_object_list, big_img, render):
    # link_probability的结果准确
    has_update = False
    total_object_num = len(all_object_list)
    for index in range(total_object_num):
        object = all_object_list[index]
        if object.type in ['cell', 'conn']:
            if render and index%50==0:
                inference_result = statistics_link_info(all_object_list, show=False)
                plot_link_line(img_path='', big_img=big_img, inference_result=inference_result, \
                    save_img_path='', show=False, save_img=False, render=True)
            object.update_object_info() # 更新该元件的连接信息和元件信息
            possible_link_object = [x[0] for x in object.inference_link_object if x not in object.inference_confirm_link_object]
            # for index_add in range(index+1, total_object_num):
            for index_add in range(len(possible_link_object)):
                # link_object = all_object_list[index_add]
                link_object = possible_link_object[index_add]
                if link_object.type in ['cell', 'conn']:

                    has_update_temp = object.estimate_confirm_link_line(link_object)
                    if has_update_temp:
                        has_update = True
    return all_object_list, has_update


def inference_confirm_link_twice(twice_object_list, all_object_list):
    # 考虑到link_probability的结果可能不准确
    has_update = False
    total_object_num_twice = len(twice_object_list)
    total_object_num = len(all_object_list)
    for index in range(total_object_num_twice):
        object = twice_object_list[index]
        if object.type in ['cell', 'conn']:
            for index_add in range(total_object_num):
                link_object = all_object_list[index_add]
                if link_object.type in ['cell', 'conn']:

                    has_update_temp = object.estimate_confirm_link_line(link_object, first=False)
                    if has_update_temp:
                        has_update = True
            object.update_object_info() # 更新该元件的连接信息和元件信息
    return all_object_list, has_update


def choose_unconfirm_object(all_object_list, big_img, object_box, img_ori_RGB, img_path, show, save_img, save_img_path):
    # 从第一轮确定性识别结果中选择出仍未识别的元素
    total_object_num = len(all_object_list)
    twice_object_list = []
    for index in range(total_object_num):
        object = all_object_list[index]
        if object.type in ['cell', 'conn'] and not object.has_confirm_all_link:
            twice_object_list.append(object)

    # object_box = [[x.label, x.x_center, x.y_center, x.width, x.height] for x in twice_object_list]
    twice_object_list = update_cell_conn_object_info(twice_object_list, big_img, object_box, img_ori_RGB, img_path, show, save_img, save_img_path, first=False, next_object=all_object_list)
    return twice_object_list


def inference_unsure_link(all_object_list):
    has_update = False
    total_object_num = len(all_object_list)
    for index in range(total_object_num):
        object = all_object_list[index]
        if object.type in ['cell', 'conn']:
            object.update_object_info() # 更新该元件的连接信息和元件信息
            possible_link_object = [x[0] for x in object.inference_link_object if x not in object.inference_confirm_link_object]
            for index_add in range(len(possible_link_object)):
            # for index_add in range(index+1, total_object_num):
                link_object = possible_link_object[index_add]
                # link_object = all_object_list[index_add]
                if link_object.type in ['cell', 'conn']:
                    has_update_temp = object.estimate_unsure_link_line(link_object)
                    if has_update_temp:
                        has_update = True
    return all_object_list, has_update

def update_cell_conn_object_info(all_object_list, big_img, object_box, img_ori_RGB, img_path, show, save_img, save_img_path, first=True, next_object=None):
    # 选择出每个元件所有可能的连接，排除掉一条直线上的多个连接，选择最邻近的连接
    # 更新元件的连接信息，更新拐点检测结果
    # big_img = Image.fromarray(big_img)
    # big_img.show()

    # 筛选出相互包含的元件，即某元件在另一个元件的内部
    object_box = [x[:6] for x in object_box]
    for index, object in enumerate(all_object_list):
        # if object.object_num==779:
        #     print(object.type, object.label, '+================')
        if object.type in ['cell', 'conn']:
            # 选择出可能存在连接的元件
            if first:
                # all_object_list_temp = all_object_list[index+1:]
                all_object_list_temp = all_object_list
                # object_box_temp = object_box[index+1:]
                object_box_temp = object_box
                all_object_list_temp = select_possible_link_object(object, all_object_list_temp, object_box_temp)
            else:
                all_object_list_temp = next_object
            for one_direction in object.link_direction: # 每个确定方向
                other_object_list, p_dis_list = [], []
                for link_object in all_object_list_temp: # 每个元件
                    if link_object.type in ['cell', 'conn'] and link_object!=object:
                        if include_cell_conn(object, link_object):
                            if link_object not in object.include_cell_sign:
                                object.include_cell_sign.append(link_object)
                            if object not in link_object.is_inside:
                                link_object.is_inside.append(object)
    

    # 依据元件的位置（方位和连接点）选出所有可能存在的连接
    object_box = [x[:6] for x in object_box]
    for index, object in enumerate(all_object_list):
        if object.type in ['cell', 'conn']:
            # 选择出可能存在连接的元件
            if first:
                # all_object_list_temp = all_object_list[index+1:]
                all_object_list_temp = all_object_list
                # object_box_temp = object_box[index+1:]
                object_box_temp = object_box
                all_object_list_temp = select_possible_link_object(object, all_object_list_temp, object_box_temp)
            else:
                all_object_list_temp = next_object
            
            for one_direction in object.link_direction: # 每个确定方向
                other_object_list, p_dis_list = [], []
                for link_object in all_object_list_temp: # 每个元件
                    if link_object.type in ['cell', 'conn'] and link_object!=object and len(link_object.is_inside)==0:
                        p_dis = two_object_link_probability_in_direction(object, link_object, one_direction, big_img, first)
                        if p_dis[0]!=0.0:
                            other_object_list.append(link_object)
                            p_dis_list.append(p_dis)
                p_dis_link_objects_list = delete_impossible_link_v2(object, other_object_list, p_dis_list, one_direction, first)
                update_link_info(object, p_dis_link_objects_list, one_direction)
    
    # 元件之间应该彼此间存在连接，删除并非彼此间连接的元件连接信息
    for main_object in all_object_list:
        if main_object.type in ['cell', 'conn']:
            main_link_object_temp = [x[0] for x in main_object.inference_link_object]
            for other_object in main_link_object_temp:
                other_link_object_temp = [x[0] for x in other_object.inference_link_object]
                if main_object not in other_link_object_temp:
                    will_delete_link_main = [x for x in main_object.inference_link_object if x[0]==other_object][0]
                    main_object.inference_link_object.remove(will_delete_link_main)
        
                # if main_object.object_num==1905:
                #     print(will_delete_link_main[0])
                

    # 当与某个拐点相连的所有元件均是拐点时，删除该拐点
    will_remove_object = [] # 将要删除的拐点
    together_object_cell = [] # 存在设备的元件集合
    together_object_no_cell = [] # 不存在设备的元件集合
    for index, object in enumerate(all_object_list):
        if object.type in ['conn']:
            # if object in together_object_cell:
            #     continue
            # elif object in together_object_no_cell:
            #     will_remove_object.append(object)
            if len(object.inference_link_object)==0:
                will_remove_object.append(object)
            else:
                has_cell_connect, together_object_this_time = find_cell_connect(object)
                # if object.object_num==1320:
                #     print(has_cell_connect, [[x[0].object_num, x[0].type] for x in object.inference_link_object])
                if not has_cell_connect:
                    together_object_no_cell.extend(together_object_this_time)
                    will_remove_object.append(object)
                else:
                    together_object_cell.extend(together_object_this_time)

    for object in will_remove_object:
        all_object_list.remove(object)

    if show or save_img:
        name, suffix = os.path.splitext(os.path.split(img_path)[1])
        # 更新拐点信息后绘制拐点位置图片
        img_temp = copy.deepcopy(img_ori_RGB)
        # img_temp = cv2.imread(r'{}\thin_{}{}'.format(save_img_path, name, suffix), 0)
        for object in all_object_list:
            if object.type in ['conn']:
                x_center = object.x_center
                y_center = object.y_center
                width = object.width
                height = object.height
                # if object.lin
                rectangle_color = (255, 0, 0)
                cv2.rectangle(img_temp, (int(x_center-width/2), int(y_center-height/2)), (int(x_center+width/2), int(y_center+height/2)), rectangle_color, 1)
        if save_img:
            cv2.imwrite(r'{}\conn_{}{}'.format(save_img_path, name, suffix), img_temp)
        if show:
            img_temp = Image.fromarray(img_temp)
            img_temp.show()
            img_temp = np.array(img_temp)

        # 绘制元件与外界连接点
        img_temp = copy.deepcopy(img_ori_RGB)
        for object in all_object_list:
            if object.type in ['cell', 'conn']:
                for one_direction in ['E', 'S', 'W', 'N']:
                    corner = object.link_position[one_direction]
                    for points in corner:
                        cv2.line(img_temp, (int(points[0][0]), int(points[0][1])), (int(points[1][0]), int(points[1][1])), (0, 0, 255), 2)
                if object.type=='cell':
                    rectangle_color = (0, 255, 0)
                elif object.type=='conn':
                    rectangle_color = (255, 0, 0)
                cv2.rectangle(img_temp, (int(object.x_center-object.width/2), int(object.y_center-object.height/2)), (int(object.x_center+object.width/2), int(object.y_center+object.height/2)), rectangle_color, 1)
        if save_img:
            cv2.imwrite(r'{}\edge_{}{}'.format(save_img_path, name, suffix), img_temp)
        if show:
            img_temp = Image.fromarray(img_temp)
            img_temp.show()
            img_temp = np.array(img_temp)
    
    return all_object_list

def include_cell_conn(main_object, other_object):
    # 判断other_object是否在main_object内部
    main_width_range_min, main_width_range_max = main_object.x_center-main_object.width/2, main_object.x_center+main_object.width/2
    main_height_range_min, main_height_range_max = main_object.y_center-main_object.height/2, main_object.y_center+main_object.height/2
    other_width_range_min, other_width_range_max = other_object.x_center-other_object.width/2, other_object.x_center+other_object.width/2
    other_height_range_min, other_height_range_max = other_object.y_center-other_object.height/2, other_object.y_center+other_object.height/2
    if main_width_range_min<other_width_range_min and main_width_range_max>other_width_range_max and \
        main_height_range_min<other_height_range_min and main_height_range_max>other_height_range_max:
        # print('某元件在主元件内部', main_object.object_num, other_object.object_num)
        return True
    else:
        return False

def find_cell_connect(conn_objects):
    # 判断与object相连的元件是否为设备，若全部为连接点，则将该连接点删除
    def analy_one_layer_object(conn_object, together_object):
        new_add_object = []
        for other_object in conn_object.inference_link_object:
            if other_object[0].type == 'cell':
                # if conn_objects.object_num==1533:
                #     print(other_object[0].type, other_object[0].object_num, conn_object.object_num)
                return together_object, 'has_cell' # 该连接链上存在元件
            else:
                if other_object[0] not in together_object:
                    together_object.append(other_object[0])
                    new_add_object.append(other_object[0])

        return together_object, new_add_object

    together_object = [conn_objects]
    total_new_add_object_last = [conn_objects]
    for index in range(10): # 迭代10次
        total_new_add_object_now = []
        for conn_object in total_new_add_object_last:
            # if conn_objects.object_num==1533:
            #     print(index, conn_object.object_num, [x.object_num for x in total_new_add_object_last])
            together_object, new_add_object = analy_one_layer_object(conn_object, together_object)
            if new_add_object == 'has_cell':
                return True, together_object # 该连接链上存在元件
            total_new_add_object_now.extend(new_add_object)
        if len(total_new_add_object_now)==0:
            return False, together_object # 该连接链上不存在元件
        
        total_new_add_object_last = total_new_add_object_now
    
    return False, together_object


def update_link_info(main_object, p_dis_link_objects_list, one_direction):
    # 将元件的连接信息写入每个元件的实例化对象
    p_dis_link_objects_list = sorted(p_dis_link_objects_list, key= lambda x: x[2])
    for p_dis_link_object in p_dis_link_objects_list:
        two_object_link_probability = p_dis_link_object[1]
        two_object_link_dis = p_dis_link_object[2]
        line_p = p_dis_link_object[3]
        link_object = p_dis_link_object[4]
        main_object.update(link_object, one_direction, two_object_link_probability, two_object_link_dis, line_p, float("inf"))
        # link_object.update(main_object, opposite_direction(one_direction), two_object_link_probability, two_object_link_dis, line_p, float("inf"))
    # if main_object.object_num==1227:
    #     print([[x[0].object_num, x[1], x[2], x[3],x[4], x[5]] for x in main_object.inference_link_object])


def select_possible_link_object(main_object, all_object_list, object_box):
    # 选择出与main_object可能存在连接的其他object
    # 方法，main_object四个方向上才存在连接的可能性
    assert len(object_box)==len(all_object_list)
    object_box = np.array(object_box).reshape((-1, 6))
    space_x_y = abs(object_box[:, 1:3]-[main_object.x_center, main_object.y_center]) - \
                    (object_box[:, 3:5]/2+[main_object.width/2, main_object.height/2])
    min_x_y_space = np.min(space_x_y, axis=1)
    condition_index = np.where(min_x_y_space<=0)[0]
    condition_index_no_sign = [x for x in condition_index if object_box[x][0]!=0]
    result = np.array(all_object_list)[condition_index_no_sign]

    # if main_object.object_num==1054:
    #     print(main_object.x_center, main_object.y_center, main_object.width, main_object.height)
    #     print([x.object_num for x in result])

    return result


def two_object_link_probability_in_direction(main_object, other_object, one_direction, big_img, first):
    # 确定性推理
    # 计算两个元件在某一方向上连接的可能性
    # 连接方向以main_object记
    # 返回值：[a,b,c]：
    # a为连接的概率，0代表不可能连接，1代表可能连接
    # b为两者间在同一直线上的的可能性大小，b越小越可能在同一条直线上，连接的可能性越小
    # c为两者之间的距离，c越小连接的可能性越大
    x_diff = main_object.x_center - other_object.x_center # 水平距离
    y_diff = main_object.y_center - other_object.y_center # 竖直距离
    total_width = main_object.width + other_object.width # 两个元件的总的宽度
    total_height = main_object.height + other_object.height # 两个元件总的高度

    def two_list_has_share_space(main_link_position, other_link_position, first):
        # 判断两个列表是否存在交集
        # if main_object.object_num==1054 and other_object.object_num==1071:
        #     print(main_link_position, other_link_position, one_direction)
        if not first:
            return True
        for main_item in main_link_position:
            main_min, main_max = main_item
            for other_item in other_link_position:
                other_min, other_max = other_item
                # if main_object.object_num==939 and other_object.object_num==1624:
                #     print(main_min, main_max, other_min, other_max)
                if main_min<=other_min<=main_max or main_min<=other_max<=main_max:
                    return True
                if other_min<=main_min<=other_max or other_min<=main_max<=other_max:
                    return True
        return False

    link_opposite_direction = opposite_direction(one_direction)
    main_link_position_list = np.array(main_object.link_position[one_direction])
    other_link_position_list = np.array(other_object.link_position[link_opposite_direction])


    # if main_object.object_num==113 and other_object.object_num==262 and not first:
    #     print(main_object.x_center, main_object.y_center, main_object.width, main_object.height, main_link_position_list, one_direction)
    #     print(other_object.x_center, other_object.y_center, other_object.width, other_object.height, other_link_position_list)

    if len(main_link_position_list)>0 and len(other_link_position_list)>0:
        if one_direction=='E' and -x_diff>total_width*0.0: # 在东边且水平距离大于两个元件宽度和的0.4倍
            main_link_position = np.array(main_link_position_list)[:, :, 1] # 东边和西边，看竖直方向坐标，南边和北边看水平方向坐标
            other_link_position = np.array(other_link_position_list)[:, :, 1]
            if two_list_has_share_space(main_link_position, other_link_position, first):
                if abs(y_diff) < total_height/2: # 竖直距离小于两个元件高度和的一半
                    p_dis = [1.0, abs(y_diff), -x_diff-total_width/2]
                else:
                    p_dis = [0.0, float("inf"), float("inf")]
            else:
                p_dis = [0.0, float("inf"), float("inf")]
        
        elif one_direction=='W' and x_diff>total_width*0.0: # 在西边且水平距离大于两个元件宽度和的0.4倍
            main_link_position = np.array(main_link_position_list)[:, :, 1]
            other_link_position = np.array(other_link_position_list)[:, :, 1]
            if two_list_has_share_space(main_link_position, other_link_position, first):
                if abs(y_diff) < total_height/2: # 竖直距离小于两个元件高度和的一半
                    p_dis = [1.0, abs(y_diff), x_diff-total_width/2]
                else:
                    p_dis = [0.0, float("inf"), float("inf")]
            else:
                p_dis = [0.0, float("inf"), float("inf")]

        elif one_direction=='S' and -y_diff>total_height*0.0: # 在南边且竖直距离大于两个元件高度和的0.4倍
            main_link_position = np.array(main_link_position_list)[:, :, 0]
            other_link_position = np.array(other_link_position_list)[:, :, 0]
            if two_list_has_share_space(main_link_position, other_link_position, first):
                if abs(x_diff) < total_width/2: # 水平距离小于两个元件宽度和的一半
                    p_dis = [1.0, abs(x_diff), -y_diff-total_height/2]
                else:
                    p_dis = [0.0, float("inf"), float("inf")]
            else:
                p_dis = [0.0, float("inf"), float("inf")]

        elif one_direction=='N' and y_diff>total_height*0.0: # 在北边且竖直距离大于两个元件宽度和的0.4倍
            main_link_position = np.array(main_link_position_list)[:, :, 0]
            other_link_position = np.array(other_link_position_list)[:, :, 0]
            if two_list_has_share_space(main_link_position, other_link_position, first):
                if abs(x_diff) < total_width/2: # 水平距离小于两个元件宽度和的一半
                    p_dis = [1.0, abs(x_diff), y_diff-total_height/2]
                else:
                    p_dis = [0.0, float("inf"), float("inf")]
            else:
                p_dis = [0.0, float("inf"), float("inf")]
        else:
            p_dis = [0.0, float("inf"), float("inf")]

    else:
        p_dis = [0.0, float("inf"), float("inf")]

    assert p_dis[0]>=0 and p_dis[1]>=0 # p_dis 内元素均应大于0
    # if main_object.object_num==113 and other_object.object_num==262 and not first:
    #     print(p_dis, one_direction)

    # 当两个元件之间存在连接可能性的时候，检测两个元件间是否存在直线，若不存在直线则不存在两个元件间不存在连接
    if p_dis[0]==1.0:
        line_length_p = detect_line(main_object, other_object, big_img, one_direction)
    else:
        line_length_p = 0
    p_dis.append(line_length_p)
    if line_length_p == 0:
        p_dis[0] = 0.0

    # if not first and p_dis[0]!=0:
    #     print(p_dis)
    
    # if main_object.object_num==113 and other_object.object_num==262 and not first:
    #     print(p_dis, one_direction, x_diff, y_diff, total_width, total_height)
    #     print(main_object.x_center, main_object.y_center, main_object.width, main_object.height)
    #     print(other_object.x_center, other_object.y_center, other_object.width, other_object.height)

    return p_dis


def delete_impossible_link_v2(main_object, other_object_list, p_dis_list, one_direction, first):
    # 确定性推理
    # 当两个元件间的连线上存在其他元件时，删除该连接
    p_dis_link_objects_list = [] 
    for other_object, p_dis_list in zip(other_object_list, p_dis_list): # 选择出可能存在连接的元件，以便下一步做进一步筛选
        if p_dis_list[0]==1.0:
            p_dis_list.append(other_object) # 将object与距离、连接概率列表合并
            p_dis_link_objects_list.append(p_dis_list)

    # if main_object.object_num==118:
        # print([[x[-1].object_num, x[0], x[1], x[2], x[3]] for x in p_dis_link_objects_list], one_direction, '++++++1111+++++')
    
    soretd_list_dis = sorted(p_dis_link_objects_list, key=lambda x: x[2]) # 依据两个元件间的距离进行排序

    if first:
        connect_position = np.array(main_object.link_position[one_direction])  # 该元件在该方向上的连接点位置
    else:
        if one_direction=='E':
            connect_position = np.array([[[main_object.x_center+main_object.width/2, main_object.y_center-main_object.height/2], [main_object.x_center+main_object.width/2, main_object.y_center+main_object.height/2]]])  # 该元件在该方向上的连接点位置,整个标记框的边
        elif one_direction=='S':
            connect_position = np.array([[[main_object.x_center-main_object.width/2, main_object.y_center+main_object.height/2], [main_object.x_center+main_object.width/2, main_object.y_center+main_object.height/2]]])  # 该元件在该方向上的连接点位置,整个标记框的边
        elif one_direction=='W':
            connect_position = np.array([[[main_object.x_center-main_object.width/2, main_object.y_center-main_object.height/2], [main_object.x_center-main_object.width/2, main_object.y_center+main_object.height/2]]])  # 该元件在该方向上的连接点位置,整个标记框的边
        elif one_direction=='N':
            connect_position = np.array([[[main_object.x_center-main_object.width/2, main_object.y_center-main_object.height/2], [main_object.x_center+main_object.width/2, main_object.y_center-main_object.height/2]]])  # 该元件在该方向上的连接点位置,整个标记框的边
    if len(connect_position)>0:
        if one_direction in ['E', 'W']:
            position_list = connect_position[:, :, 1] # 竖直方向的高度
        else:
            position_list = connect_position[:, :, 0] # 水平方向的宽度
        all_connect_point = []
        main_object_true_min_line_width = 10000 # 与main_object连接的管线的最小宽度
        for one_position in position_list:
            start_point, stop_point = one_position
            conn_points = [x for x in range(int(start_point), math.ceil(stop_point+1))]
            all_connect_point.extend(conn_points)
            main_object_true_min_line_width = min(main_object_true_min_line_width, len(conn_points))
        
        # if main_object.object_num==118 or main_object.object_num==128:
        #     print(main_object.object_num, all_connect_point, one_direction, true_min_line_width)

        for index, other_object_p_dis in enumerate(soretd_list_dis):
            if len(all_connect_point)<main_object_true_min_line_width:
                p_dis_link_objects_list.remove(other_object_p_dis)
                continue
            other_object = other_object_p_dis[-1]
            other_object_connect_position = np.array(other_object.link_position[opposite_direction(one_direction)])  # 该元件在该方向上的连接点位置
            
            # if main_object.object_num==118 and other_object.object_num==130:
            #     print(other_object_connect_position, '++++++++++222222+++++++++', one_direction)
            
            if len(other_object_connect_position)>0:
                if one_direction in ['E', 'W']:
                    other_object_position_list = other_object_connect_position[:, :, 1] # 竖直方向的高度
                else:
                    other_object_position_list = other_object_connect_position[:, :, 0] # 水平方向的宽度
                other_all_point = []
                other_object_true_min_line_width = 10000 # 与other_object连接的管线的最小宽度
                for other_one_position in other_object_position_list:
                    start_point, stop_point = other_one_position
                    conn_points = [x for x in range(int(start_point), math.ceil(stop_point+1))]
                    other_all_point.extend(conn_points)
                    other_object_true_min_line_width = min(other_object_true_min_line_width, len(conn_points))
                
                # if main_object.object_num==958 and other_object.object_num==928:
                #     print(all_connect_point, other_all_point, main_object.object_num, other_object.object_num)

                if len([x for x in all_connect_point if x in other_all_point])<min(main_object_true_min_line_width, other_object_true_min_line_width):
                    p_dis_link_objects_list.remove(other_object_p_dis)
                
                all_connect_point = [x for x in all_connect_point if x not in other_all_point]

            else:
                p_dis_link_objects_list.remove(other_object_p_dis)

    # if main_object.object_num==1142:
    #     print([[x[-1].object_num, x[0], x[1], x[2], x[3]] for x in p_dis_link_objects_list], one_direction, '++++++2222+++++\n')

    # if not first:
    #     print(p_dis_link_objects_list)

    return p_dis_link_objects_list


def delete_impossible_link(main_object, other_object_list, p_dis_list, one_direction):
    # 确定性推理
    # 删除大元件背后面的元件
    p_dis_link_objects_list = [] 
    for other_object, p_dis_list in zip(other_object_list, p_dis_list): # 选择出可能存在连接的元件，以便下一步做进一步筛选
        if p_dis_list[0]==1.0:
            p_dis_list.append(other_object) # 将object与距离、连接概率列表合并
            p_dis_link_objects_list.append(p_dis_list)

    # if main_object.object_num==1868:
    #     print([[x[-1].object_num, x[0], x[1], x[2], x[3]] for x in p_dis_link_objects_list], one_direction, '++++++1111+++++')
    
    soretd_list_dis = sorted(p_dis_link_objects_list, key=lambda x: x[2]) # 依据两个元件间的距离进行排序
    main_object_min_height = main_object.y_center - main_object.height/2
    main_object_max_height = main_object.y_center + main_object.height/2
    main_object_min_width = main_object.x_center - main_object.width/2
    main_object_max_width = main_object.x_center + main_object.width/2
    for index, other_object_p_dis in enumerate(soretd_list_dis):
        other_object = other_object_p_dis[-1]
        other_object_min_height = other_object.y_center - other_object.height/2
        other_object_max_height = other_object.y_center + other_object.height/2
        other_object_min_width = other_object.x_center - other_object.width/2
        other_object_max_width = other_object.x_center + other_object.width/2
        
        if one_direction in ['E', 'W']: # 竖直距离
            commom_link_dis = None
            if index==0: # 水平距离最近的元素不会删除
                min_height_list = [other_object_min_height] # 竖直范围的最小值
                max_height_list = [other_object_max_height] # 竖直范围的最大值
                commom_link_dis = calc_commom_dis(main_object_max_height, main_object_min_height, \
                        other_object_max_height, other_object_min_height, [], [])
                p_dis_link_objects_list[p_dis_link_objects_list.index(other_object_p_dis)][1] = min(1, commom_link_dis/ min(main_object.height, other_object.height))
            else:
                max_height_list_temp, min_height_list_temp = copy.deepcopy(max_height_list), copy.deepcopy(min_height_list)
                for max_height, min_height in zip(max_height_list_temp, min_height_list_temp):
                    if min_height<=main_object_min_height and max_height>=main_object_max_height:
                        p_dis_link_objects_list.remove(other_object_p_dis) # 如果max_min竖直范围超过元件竖直范围，则它不可能与其余元件连接，删除
                        commom_link_dis = 0
                        break
                    elif other_object_min_height>=min_height and other_object_max_height<=max_height:
                        p_dis_link_objects_list.remove(other_object_p_dis) # 如果距离更近元件的竖直范围大于它，则它不可能与其连接，删除
                        commom_link_dis = 0
                        break
                if commom_link_dis==None:
                    commom_link_dis = calc_commom_dis(main_object_max_height, main_object_min_height, \
                        other_object_max_height, other_object_min_height, max_height_list, min_height_list)
                    p_dis_link_objects_list[p_dis_link_objects_list.index(other_object_p_dis)][1] = min(1, commom_link_dis/min(main_object.height, other_object.height))

                min_height_list.append(other_object_min_height)
                max_height_list.append(other_object_max_height)
                min_height_list, max_height_list = union_list(min_height_list, max_height_list)

        elif one_direction in ['S', 'N']: # 水平距离
            commom_link_dis = None
            if index==0: # 竖直距离最近的元素不会删除
                min_width_list = [other_object_min_width] # 水平范围的最小值
                max_width_list = [other_object_max_width] # 水平范围的最大值
                commom_link_dis = calc_commom_dis(main_object_max_width, main_object_min_width, \
                        other_object_max_width, other_object_min_width, [], [])
                p_dis_link_objects_list[p_dis_link_objects_list.index(other_object_p_dis)][1] = min(1, commom_link_dis/min(main_object.width, other_object.width))
            else:
                max_width_list_temp, min_width_list_temp = copy.deepcopy(max_width_list), copy.deepcopy(min_width_list)
                for max_width, min_width in zip(max_width_list_temp, min_width_list_temp):
                    if min_width<=main_object_min_width and max_width>=main_object_max_width:
                        p_dis_link_objects_list.remove(other_object_p_dis) # 如果max_min水平范围超过元件水平范围，则它不可能与其余元件连接，删除
                        commom_link_dis = 0
                        break
                    elif other_object_min_width>=min_width and other_object_max_width<=max_width:
                        p_dis_link_objects_list.remove(other_object_p_dis) # 如果距离更近元件的水平范围大于它，则它不可能与其连接，删除
                        commom_link_dis = 0
                        break
                if commom_link_dis==None:
                    commom_link_dis = calc_commom_dis(main_object_max_width, main_object_min_width, \
                            other_object_max_width, other_object_min_width, max_width_list, min_width_list)
                    p_dis_link_objects_list[p_dis_link_objects_list.index(other_object_p_dis)][1] = min(1, commom_link_dis/min(main_object.width, other_object.width))

                min_width_list.append(other_object_min_width)
                max_width_list.append(other_object_max_width)
                min_width_list, max_width_list = union_list(min_width_list, max_width_list)

    # if main_object.object_num==1868:
    #     print([[x[-1].object_num, x[0], x[1], x[2], x[3]] for x in p_dis_link_objects_list], one_direction, '++++++2222+++++\n')

    return p_dis_link_objects_list


def detect_line(main_object, other_object, big_img, direction):
    # 检测两个元件之间是否存在直线,返回两个元件间距离占两元件距离的百分比
    if direction=='E':
        start_point_height = max((main_object.y_center-main_object.height/2), (other_object.y_center-other_object.height/2))
        start_point_width = main_object.x_center+main_object.width/2
        stop_point_height = min((main_object.y_center+main_object.height/2), (other_object.y_center+other_object.height/2))
        stop_point_width = other_object.x_center-other_object.width/2
    
    elif direction=='W':
        start_point_height = max((main_object.y_center-main_object.height/2), (other_object.y_center-other_object.height/2))
        start_point_width = other_object.x_center+other_object.width/2
        stop_point_height = min((main_object.y_center+main_object.height/2), (other_object.y_center+other_object.height/2))
        stop_point_width = main_object.x_center-main_object.width/2
    
    elif direction=='S':
        start_point_height = main_object.y_center+main_object.height/2
        start_point_width = max((main_object.x_center-main_object.width/2), (other_object.x_center-other_object.width/2))
        stop_point_height = other_object.y_center-other_object.height/2
        stop_point_width = min((main_object.x_center+main_object.width/2), (other_object.x_center+other_object.width/2))
    
    elif direction=='N':
        start_point_height = other_object.y_center+other_object.height/2
        start_point_width = max((main_object.x_center-main_object.width/2), (other_object.x_center-other_object.width/2))
        stop_point_height = main_object.y_center-main_object.height/2
        stop_point_width = min((main_object.x_center+main_object.width/2), (other_object.x_center+other_object.width/2))
    
    # 两个标记框存在重叠的可能性,因此start_point_height不一定小于stop_point_height，start_point_width不一定小于stop_point_width
    height_list = [start_point_height, stop_point_height]
    width_list = [start_point_width, stop_point_width]

    small_img = big_img[int(min(height_list)):math.ceil(max(height_list)), int(min(width_list)):math.ceil(max(width_list))]
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # if main_object.object_num==1054 and other_object.object_num==1071:
    #     print(small_img.shape, small_img)
    if min(small_img.shape)>=1:
        # edges = cv2.Canny(small_img, 80, 150)
        edges = small_img
        # lines = cv2.HoughLinesP(edges, 1, np.pi/180, min_length_line, maxLineGap=min_length_line/2)
        if direction in ['E', 'W']:
            line_p = 1- min(np.mean(edges/255, axis=1))
            # print(edges.shape, np.mean(edges/255, axis=1), direction, '\n')
        elif direction in ['S', 'N']:
            line_p = 1- min(np.mean(edges/255, axis=0))
            # print(edges.shape, np.mean(edges/255, axis=0), direction, '\n')
            # line_length = abs(line[1] - line[3])
        # if main_object.object_num==1178 and other_object.object_num==1380:
        #     print(line_p)
        #     cv2.imshow('aaa', edges)
        #     cv2.waitKey(5000)


        # # 虚线识别，单端点间应该为空白，且间距不能过大
        # # if main_object.object_num==1054 and other_object.object_num==1071:
        # #     print(line_p, edges)
        # if main_object.type=='conn' and other_object.type=='conn' and line_p==0:
        #     if direction in ['E', 'W']:
        #         dis_space = edges.shape[1] # 水平间距的大小
        #         if dis_space < 4*(main_object.width+other_object.width): # 经验确定
        #             line_p = 1.0
        #     elif direction in ['S', 'N']:
        #         dis_space = edges.shape[0] # 竖直间距的大小
        #         if dis_space < 4*(main_object.height+other_object.height): # 经验确定
        #             line_p = 1.0

        # if main_object.object_num==773 and other_object.object_num==1178:
        #     print(line_p, edges)
        # 虚线判断
        if 0.5<=line_p<=0.98:
            if direction in ['E', 'W']:
                x_mean = np.mean(edges/255, axis=1)
                index = int(np.mean(np.where(x_mean==min(x_mean))[0]))
                max_black_line = edges[index, :] # 最黑的线
            elif direction in ['S', 'N']:
                y_mean = np.mean(edges/255, axis=0)
                index = int(np.mean(np.where(y_mean==min(y_mean))[0]))
                max_black_line = edges[:, index] # 最黑的线
            point_index_white = [index for index in range(len(max_black_line)-1) if \
                (max_black_line[index]==0 and max_black_line[index+1]==255)]
            point_index_black = [index for index in range(len(max_black_line)-1) if \
                (max_black_line[index]==255 and max_black_line[index+1]==0)]
            
            # 白色起始点和黑色起始点的个数大于1
            if min(len(point_index_black), len(point_index_white))>=1:
                 # 以白色起始点为起点，因此当白色起始点的索引大于黑色起始点的索引时候，去掉第一个黑色点
                if point_index_white[0] > point_index_black[0] and len(point_index_black)>=2:
                    point_index_black = point_index_black[1:]
                black_space = [y-x for x,y in zip(point_index_white, point_index_black)] # 白色区域的长度（像素）
                
                # 规则1，白色线段的长度基本一致，白色线段不能过长(以白色线段长度为3分情况讨论)
                if len(black_space)>=3 and (max(black_space)-min(black_space)) <= 2: # 白色区域的长度基本一致，2依据经验给定
                    line_p = 1
                    return line_p
                elif len(black_space)<3 and (max(black_space)-min(black_space)) <= 2 and max(black_space)<2*min_length_line:
                    line_p = 1
                    return line_p
                
                # 规则2，筛选出白色间隔长度基本一致的线段，认为该部分白色线段为虚线，将其看做黑色线段（以最小白色线段为起点）
                if len(black_space)>=2:
                    line_p += np.sum([x for x in black_space if x<=(min(black_space)+2)])/len(max_black_line)

            # point_index = [index for index in range(len(max_black_line)-1) if \
            #     (max_black_line[index]==0 and max_black_line[index+1]==255) or \
            #     (max_black_line[index]==255 and max_black_line[index+1]==0)]
            # point_index_0 = point_index[:len(point_index)//2*2-1]
            # point_index_1 = point_index[1:][:len(point_index[1:])//2*2-1]
            # line_length_0 = [point_index_0[index+1]-point_index_0[index] for index in range(0, len(point_index_0)-1, 2)]
            # line_length_1 = [point_index_1[index+1]-point_index_1[index] for index in range(0, len(point_index_1)-1, 2)]
            # if min(len(line_length_0), len(line_length_1))>=2:
            #     # if main_object.object_num==1169 and other_object.object_num==1373:
            #     #     print(line_length_0, line_length_1)
            #     if (max(line_length_0)-min(line_length_0)) < min_length_line and \
            #         (max(line_length_1)-min(line_length_1)) < min_length_line:
            #         # print(line_length_0, line_length_1, direction)
            #         # cv2.imshow('aaa', edges)
            #         # cv2.waitKey(1000)
            #         # print(main_object.object_num, other_object.object_num)
            #         line_p = 1
            #         return line_p

            # white_point_index = [index for index in range(len(max_black_line)-1) if max_black_line[index]==255 and max_black_line[index+1]==0]
            # black_point_index = black_point_index[:(len(black_point_index)//2)*2-1] # 取偶数个
            # white_point_index = white_point_index[:(len(white_point_index)//2)*2-1] # 取偶数个
            # black_line_length = np.array([black_point_index[index+1]-black_point_index[index] for index in range(0, len(black_point_index)-1, 2)])
            # white_line_length = np.array([white_point_index[index+1]-white_point_index[index] for index in range(0, len(white_point_index)-1, 2)])
            # num_1 = len(black_line_length)
            # num_2 = len(white_line_length)
            # if max(num_1, num_2)>=2 and abs(num_1-num_2)<2 and min(num_1, num_2)>=1:
            #     if (max(black_line_length)-min(black_line_length))<min_length_line and \
            #        (max(white_line_length)-min(white_line_length))<min_length_line:
            #         print(black_line_length, white_line_length, direction)
            #         cv2.imshow('aaa', edges)
            #         cv2.waitKey(50000)

        if line_p<0.5: # line_p < 0.5
            line_p = 0
            return line_p
        else:
            return line_p
    else:
        line_p = 1
        return line_p


def statistics_link_info(all_object_list, show=True):
    # 统计推理结果
    total_confirm_link_num = 0 # 已确定连接关系的数量，越大越好
    confirm_cell_conn_num = 0  # 已确定连接关系元件的数量，越大越好
    confirm_direction_cell_conn_num = 0  # 已确定连接方向信息的元件的数量，越大越好
    total_possible_link_num = 0  # 可能连接的数量，越小越好
    total_cell_conn_num = 0  # 总的元件数量，定值
    confirm_link = []  # 已确定的连线集合
    unsure_link = []  # 可能连线的集合
    possible_link = [] # 所有可能的连线集合
    confirm_objects = []  # 已确定连接关系的元件
    unsure_objects = [] # 未确定连接关系的元件
    for object in all_object_list:
        if object.type in ['cell', 'conn']:
            total_confirm_link_num += len(object.inference_confirm_link_object)
            confirm_link.extend([[object, x[0], x[5]] for x in object.inference_confirm_link_object])
            unsure_link.extend([[object, x[0], x[5]] for x in object.inference_unsure_link_object])
            # unsure_link.extend([[object, x[0], x[5]] for x in object.inference_link_object])
            possible_link.extend([[object, x[0], x[5]] for x in object.inference_link_object])
            
            if object.link_probability==1.0:
                confirm_cell_conn_num += 1
            total_possible_link_num += len(object.inference_link_object)

            if object.has_confirm_all_link:
                confirm_direction_cell_conn_num += 1
                confirm_objects.append(object)
            else:
                unsure_objects.append(object)
            
            total_cell_conn_num += 1

    assert len(confirm_link)==total_confirm_link_num
    assert len(confirm_objects)+len(unsure_objects)==total_cell_conn_num
    assert len(confirm_objects)==confirm_direction_cell_conn_num
    
    inference_result = {}
    inference_result['total_confirm_link_num'] = total_confirm_link_num
    inference_result['confirm_cell_conn_num'] = confirm_cell_conn_num
    inference_result['confirm_direction_cell_conn_num'] = confirm_direction_cell_conn_num
    inference_result['total_possible_link_num'] = total_possible_link_num
    inference_result['total_cell_conn_num'] = total_cell_conn_num
    inference_result['confirm_link'] = confirm_link
    inference_result['unsure_link'] = unsure_link
    inference_result['possible_link'] = possible_link
    inference_result['confirm_objects'] = confirm_objects
    inference_result['unsure_objects'] = unsure_objects

    if show:
        print('+' ,'-'*179, '+')
        print('| {}{:<4} | {}{:<4} | {}{:<4} {}{:<5.2f} % ||| {}{:<4} | {}{:<4} | {}{:<4} {}{:5<.2f} % |'.format('总元件数：', \
            total_cell_conn_num, '已确定连接关系元件数：', len(confirm_objects), '未确定连接关系元件数：', len(unsure_objects), \
            '完成百分比：', round(len(confirm_objects)/total_cell_conn_num, 4)*100,'所有可能连线数：', \
            total_possible_link_num, '已确定连线数：', total_confirm_link_num,  '未确定连线数：', \
            total_possible_link_num-total_confirm_link_num, '完成百分比：', round(total_confirm_link_num/max(total_possible_link_num, 0.1),4)*100))
        print('+' ,'-'*179, '+')

    return inference_result
    
def union_list(min_list, max_list):
    # 将重叠区域合并
    new_list = []
    for index, (min_value, max_value) in enumerate(zip(min_list, max_list)):
        new_list_temp = [x for x in range(int(min_value), math.ceil(max_value+1))]
        new_list.extend(new_list_temp)

    new_list = sorted(list(set(new_list)))
    new_array = np.array(new_list)
    split_index_list = np.where(np.diff(new_array)>2)[0]
    split_index_list = list(split_index_list)
    split_index_list.append(new_list[-1])

    return_list_min = []
    return_list_max = []
    start_index = 0
    for index in split_index_list:
        temp = new_list[start_index:index+1]
        return_list_min.append(temp[0])
        return_list_max.append(temp[-1])
        start_index = index+1
    
    return return_list_min, return_list_max


def calc_commom_dis(main_object_max_height, main_object_min_height, other_object_max_height, other_object_min_height, space_max, space_min):
    # 计算两个元件间可能存在连线的区域长度
    main_list = [x for x in range(int(main_object_min_height), math.ceil(main_object_max_height))]
    other_list = [x for x in range(int(other_object_min_height), math.ceil(other_object_max_height))]
    commom_range_list = [x for x in main_list if x in other_list]
    for start, stop in zip(space_min, space_max):
        space_list = [x for x in range(int(start), math.ceil(stop))]
        commom_range_list = [x for x in commom_range_list if x not in space_list]

    return len(commom_range_list)


def plot_link_line(img_path, big_img, inference_result, save_img_path, show, save_img, render):
    # 绘制识别结果
    file_name = os.path.split(img_path)[1]
    name, suffix = os.path.splitext(file_name)
    for index, confirm_possible_object_list in enumerate([inference_result['confirm_objects'], inference_result['unsure_objects']]):
        for object in confirm_possible_object_list:
            if index==0:
                rectangle_color = (255, 0, 0)
            else:
                rectangle_color = (0, 0, 255)
            if object.type in ['conn', 'cell']:
                # if object.type=='cell':
                #     rectangle_color = (0, 0, 255)
                # else:
                #     rectangle_color = (255, 0, 0)
                if object.object_num==1779 or object.object_num==1285:
                    rectangle_color = (0,255,0)
                cv2.rectangle(big_img, (int(object.x_center-object.width/2), int(object.y_center-object.height/2)), \
                    (int(object.x_center+object.width/2), int(object.y_center+object.height/2)), rectangle_color, 1)
                # cv2.putText(big_img, str(object.object_num), (int(object.x_center-object.width/2), int(object.y_center-object.height/2)), \
                # cv2.putText(big_img, str(round(object.probability,2)), (int(object.x_center-object.width/2), int(object.y_center-object.height/2-5)), \
                #    cv2.FONT_ITALIC, 0.4, rectangle_color, 1)
                # if object.link_probability==1.0:
                #     cv2.circle(big_img, (int(object.x_center), int(object.y_center)), 2, (0,0,255), 2)
                # else:
                #     cv2.circle(big_img, (int(object.x_center), int(object.y_center)), 2, (0,0,255), 2)

    # for index, confirm_possible_link in enumerate([inference_result['confirm_link'], inference_result['unsure_link']]):
    for index, confirm_possible_link in enumerate([inference_result['confirm_link']]):
        for object, link_object, rule_num in confirm_possible_link:
            if index==0:
                rectangle_color = (255, 0, 0)
            else:
                rectangle_color = (255, 0, 0)
            cv2.circle(big_img, (int(object.x_center), int(object.y_center)), math.ceil(max(big_img.shape)/3000), rectangle_color, 1)
            cv2.line(big_img, (int(object.x_center), int(object.y_center)), (int(link_object.x_center),int(link_object.y_center)), rectangle_color, 2)
            # cv2.putText(big_img, str(rule_num), (int((object.x_center+link_object.x_center)/2), int((object.y_center+link_object.y_center)/2)), \
            #     cv2.FONT_HERSHEY_SIMPLEX, 0.7, rectangle_color, 1)
    
    if save_img:
        cv2.imwrite(r'{}/line_{}{}'.format(save_img_path, name, suffix), big_img)
    
    if show:
        big_img = Image.fromarray(big_img)
        big_img.show()
    
    if render:
        big_img = cv2.resize(big_img, (int(big_img.shape[1]/2), int(big_img.shape[0]/2)))        
        mngr = plt.get_current_fig_manager()  # 获取当前figure manager
        mngr.window.wm_geometry("+100+0")  # 调整窗口在屏幕上弹出的位置
        plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95)
        plt.axis('off')
        plt.imshow(big_img)
        plt.pause(0.005)
        plt.show()

if __name__=='__main__':
    min_length_line = 4
    file_name = '001_M'

    big_img = cv2.imread(r'..\data\sample\labelme\one_labelme\{}.png'.format(file_name), cv2.IMREAD_GRAYSCALE)
    config_path = r'data\config_new.json'
    # cell_sign_info = read_labelme_json(r'..\data\sample\labelme\one_labelme\005.json')
    cell_sign_info = read_csv_data(r'..\data\sample\labelme\one_labelme\{}.xlsx'.format(file_name))
    objects_list, big_img_ori, no_cell_sign_img, thin_line_img = get_cell_conn_detail_info(cell_sign_info, big_img, config_path, file_name=file_name, show=show, save_small_img=False)

    objects_list = update_cell_conn_object_info(objects_list, no_cell_sign_img) # 依据元件的位置和连接方向确定所有可能的连接方向

    # 确定性连接
    has_update = True
    while True:
        inference_result = statistics_link_info(objects_list, show=True)
        if has_update:
            objects_list, has_update = inference_confirm_link(objects_list)
        if not has_update:
            break
    
    # 第二轮确定性连接
    # has_update = True
    # twice_object_list = choose_unconfirm_object(objects_list, big_img)
    # while True:
    #     objects_list, has_update = inference_confirm_link_twice(twice_object_list, objects_list)
    #     if not has_update:
    #         break

    # 可能性连接
    has_update = True
    while True:
        inference_result = statistics_link_info(objects_list, show=True)
        if  has_update:
            objects_list, has_update = inference_unsure_link(objects_list)
        if not has_update:
            break

    big_img_ori = cv2.cvtColor(big_img_ori, cv2.COLOR_GRAY2BGR)
    inference_result = statistics_link_info(objects_list, show=False)
    plot_link_line(big_img_ori, inference_result, file_name)

