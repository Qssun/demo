import os
import time
import shutil
import cv2
from tqdm import tqdm
from tqdm.contrib import tzip
import torch
import matplotlib
matplotlib.use('TkAgg')
from PIL import Image
from matplotlib import pyplot as plt
import pickle
from PIL import Image

from multiprocessing import Pool, Manager

from my_utils import get_config_parameter, img_process, save_reco_info, narrow_cell_posi_img, adjust_cell_posi_img

from yolo_detect import my_detect_application, detect, plot_result
from reco_text import reco_text, load_ocr_model

from link_inference import update_cell_conn_object_info, statistics_link_info, inference_confirm_link, inference_unsure_link, plot_link_line, choose_unconfirm_object, inference_confirm_link_twice

def work(object_box, img_path, config_path, save_result_path, show, save_img, save_crop_img, render):
    # 元件识别，并将识别结果中的每个元件转为实例化的识别对象
    # object_box = my_detect_application(img_path=img_path, img_size=img_size, model=yolo_model, stride=stride, device=device)
    # plot_result(big_img, object_box)
    img_peocess_ins = img_process(object_box, img_path, config_path, show, save_img, save_result_path, save_crop_img) # 建立训练集
    objects_list = img_peocess_ins.add_conn_info(img_process_for_thin=False, use_thin_img=True)

    img_ori_RGB = img_peocess_ins.img_ori_RGB # 原始图片
    img_bin_no_cell_conn = img_peocess_ins.bin2value(img_ori_RGB) # 去掉元件的图片
    new_object_box = img_peocess_ins.object_box
    objects_list = update_cell_conn_object_info(objects_list, img_bin_no_cell_conn, new_object_box, img_ori_RGB, img_path, show, save_img, save_result_path) # 依据元件的位置和连接方向确定所有可能的连接方向

    # 确定性连接
    has_update = True
    while True:
        inference_result = statistics_link_info(objects_list, show=False)
        if has_update:
            objects_list, has_update = inference_confirm_link(objects_list, img_peocess_ins.img_ori_RGB, render)
        if not has_update:
            break

    # # 第二轮确定性连接
    # has_update = True
    # twice_object_list = choose_unconfirm_object(objects_list, img_bin_no_cell_conn, new_object_box, img_ori_RGB, img_path, show, save_img, save_result_path)
    # while True:
    #     objects_list, has_update = inference_confirm_link_twice(twice_object_list, objects_list)
    #     if not has_update:
    #         break

    # # 可能性连接
    # has_update = True
    # while True:
    #     inference_result = statistics_link_info(objects_list, show=True)
    #     if  has_update:
    #         objects_list, has_update = inference_unsure_link(objects_list)
    #     if not has_update:
    #         break

    # 绘图，保存实验结果
    inference_result = statistics_link_info(objects_list, show=False)
    plot_link_line(img_path, img_peocess_ins.img_ori_RGB, inference_result, save_result_path, show, save_img, render=False)
    save_reco_info(objects_list, img_path, save_result_path)


if __name__=='__main__':
    start_time = time.perf_counter()
    config_info = get_config_parameter('config_parameter.json')
    img_size = config_info['max_img_size']
    min_length_line = config_info['min_length_line']
    test_img_path = config_info['test_img_path']
    model_path = config_info['yolo_model_path']
    config_path = config_info['config_path']
    calc_device = config_info['device']
    ocr = config_info['ocr']
    save_result_path = config_info['save_result_path']
    show = config_info['show']
    save_img = config_info['save_img']
    save_crop_img = config_info['save_crop_img']
    render = config_info['render']
    mul_process = config_info['mul_process']
    if render: # 当需要实时显示结果时，不采用多进程
        mul_process = False

    # 检查保存结果的文件是否存在，若不存在则创建
    if not os.path.exists(save_result_path):
        os.makedirs(save_result_path)
    
    # 检查测试数据是否存在
    for img_path in test_img_path:
        if not os.path.exists(img_path):
            print('\033[1;31未找到待识别文件【{}】！！！ \033[0m'.format(img_path))
        shutil.copy2(img_path, save_result_path)

    # 加载yolo模型和ocr模型
    if ocr:
        ocr_model = load_ocr_model(enhance=False) # 加载ocr识别模型
    else:
        ocr_model = None
    yolo_model, stride, img_size, device = detect(weights=model_path, img_size=img_size, device=calc_device)

    # yolo模型识别
    print('正在进行元件检测与定位....')
    object_box_list = my_detect_application(img_path_list=test_img_path, img_size=img_size, model=yolo_model, stride=stride, device=device)
    torch.cuda.empty_cache()
    # plot_result(big_img, object_box)
    print('元件检测与定位完成！')
    stop_time = time.perf_counter()
    print("耗时：{} 秒".format(stop_time-start_time))

    # 缩小元件框的大小
    object_box_list_narrow = []
    for img_path, object_box in zip(tqdm(test_img_path), object_box_list):
        object_box_temp = narrow_cell_posi_img(img_path, object_box)
        # object_box_temp = adjust_cell_posi_img(img_path, object_box_temp)
        object_box_list_narrow.append(object_box_temp)

    # 文字识别
    start_time = time.perf_counter()
    print('正在识别流程图参数文字....')
    object_box_list_text = []
    for img_path, object_box in zip(tqdm(test_img_path), object_box_list_narrow):
        object_box = reco_text(img_path, object_box, ocr_model, save_result_path, show=show, save_img=save_img)
        object_box_list_text.append(object_box)
    print('流程图参数文字识别完成!')
    stop_time = time.perf_counter()
    print("耗时：{} 秒".format(stop_time-start_time))

    # 连接关系识别
    start_time = time.perf_counter()
    print('正在计算连接关系....')
    pbar = tqdm(total=len(test_img_path))
    update = lambda *args: pbar.update()
    p = Pool(min(len(test_img_path), int(os.cpu_count()*0.75)))
    for img_path, object_box in zip(test_img_path, object_box_list_text):
        if render:
            fig = plt.figure(img_path, figsize=(11, 7))
            fig.tight_layout()
            plt.ion()
        if mul_process:
            p.apply_async(work, args=(object_box, img_path, config_path, save_result_path, show, save_img, save_crop_img, render), callback=update)
        else:
            work(object_box, img_path, config_path, save_result_path, show, save_img, save_crop_img, render)
        
        if render:
            plt.ioff()
            plt.close('all')
    p.close()
    p.join()
    print('连接关系计算完成！')
    stop_time = time.perf_counter()
    print("耗时：{} 秒".format(stop_time-start_time))

    img_path_0, img_path_1 = os.path.split(img_path)
    line_img_path = os.path.join(save_result_path, 'line_'+img_path_1)
    text_img_path = os.path.join(save_result_path, 'text_'+img_path_1)
    im = Image.open("{}".format(line_img_path))
    im.show()
    im_text = Image.open("{}".format(text_img_path))
    im_text.show()