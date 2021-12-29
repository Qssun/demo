from PIL import Image
import math
import os
import pickle

def mesh_data_test(test_img_path, crop_width, crop_height, save_small_img_parent_path=None, moving_radio=0.5):
    # 将输入的大图片切割为多个小图片，用于模型测试
    big_img = Image.open(test_img_path).convert('RGB')
    total_width, total_height = big_img.size[0], big_img.size[1]
    moving_step_width, moving_step_height = int(crop_width*moving_radio), int(crop_height*moving_radio)  # 近似的滑动步长
    num_width, num_height = math.ceil((total_width-crop_width)/moving_step_width) + 1, math.ceil(
        (total_height-crop_height)/moving_step_height) + 1  # 宽度和高度上分别划分为多少个子图
    moving_step_width, moving_step_height = math.ceil((total_width-crop_width)/(
        num_width-1)), math.ceil((total_height-crop_height)/(num_height-1))  # 实际滑动步长
    top_left_points = [[x*moving_step_width, y*moving_step_height]
                       for y in range(num_height) for x in range(num_width)]

    save_small_imgs = []
    mesh_info = {}
    mesh_info['total_width'] = total_width
    mesh_info['total_height'] = total_height
    mesh_info['num_width'] = num_width
    mesh_info['num_height'] = num_height
    mesh_info['moving_step_width'] = moving_step_width
    mesh_info['moving_step_height'] = moving_step_height
    mesh_info['crop_width'] = crop_width
    mesh_info['crop_height'] = crop_height
    reality_top_left_points = []
    for index, (top_left_x, top_left_y) in enumerate(top_left_points):
        bottom_right_x = min(total_width, top_left_x+crop_width)
        bottom_right_y = min(total_height, top_left_y+crop_height)
        top_left_x = bottom_right_x - crop_width
        top_left_y = bottom_right_y - crop_height
        small_img = big_img.crop(
            (top_left_x, top_left_y, bottom_right_x, bottom_right_y))
        save_small_imgs.append(small_img)
        file_name, suffix = os.path.split(
            test_img_path)[-1].split('.')[0], os.path.split(test_img_path)[-1].split('.')[1]
        if save_small_img_parent_path:
            if not os.path.exists(save_small_img_parent_path):
                os.makedirs(save_small_img_parent_path)
            small_img.save(os.path.join(save_small_img_parent_path,
                                        '{}_{}.{}'.format(file_name, index, suffix)))
        reality_top_left_points.append([top_left_x, top_left_y])

    mesh_info['top_left_points'] = reality_top_left_points
    # mesh_info['data'] = save_small_imgs
    if save_small_img_parent_path:
        with open(os.path.join(save_small_img_parent_path, 'top_left_point.pkl'), 'wb') as f:
            pickle.dump(mesh_info, f)

    return big_img, save_small_imgs, mesh_info

if __name__=='__main__':
    test_img_path = '../data/sample/labelme/one_labelme/005.png'
    crop_width = 640
    crop_height = 640
    save_small_img_parent_path='../data/crop_img'
    moving_radio = 0.5
    mesh_data_test(test_img_path, crop_width, crop_height, save_small_img_parent_path, moving_radio=0.5)