from genericpath import exists
import os
import cv2
import torch
import shutil

from PIL import Image

from models.experimental import attempt_load
from utils.datasets import LoadImages
from utils.general import check_img_size, non_max_suppression, set_logging, scale_coords
from utils.torch_utils import select_device


@torch.no_grad()
def detect(weights='yolov5s.pt',  # model.pt path(s)
           img_size=640,  # inference size (pixels)
           device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
           half=False,  # use FP16 half-precision inference
           ):

    # Initialize
    set_logging()
    device = select_device(device)
    half &= device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    img_size = check_img_size(img_size, s=stride)  # check image size
    if half:
        model.half()  # to FP16

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, img_size, img_size).to(device).type_as(next(model.parameters())))  # run once
    
    return model, stride, img_size, device

def my_detect_application(img_path_list, img_size, model, stride,
    conf_thres=0.65,  # confidence threshold
    iou_thres=0.45,  # NMS IOU threshold
    max_det=100000,  # maximum detections per image
    device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    classes=None,  # filter by class: --class 0, or --class 0 2 3
    agnostic_nms=False,  # class-agnostic NMS
    augment=False,  # augmented inference
    half=False,  # use FP16 half-precision inference
    temp_path = '__temp_path__',
    ):

    try:
        shutil.rmtree(temp_path)
    except:
        pass
    finally:
        os.mkdir(temp_path)
        [shutil.copy2(x, temp_path) for x in img_path_list]
    all_img_path_name = [os.path.split(x)[1] for x in img_path_list] # 将测试数据移动到临时缓存文件夹

    det_list = [[]]*len(img_path_list)

    dataset = LoadImages(temp_path, img_size=img_size, stride=stride)
    for path, img, im0s, _ in dataset:
        print('\n')
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        pred = model(img, augment=augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Process detections
        assert len(pred)==1
        det = pred[0]
    
        im0 = im0s.copy()
        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
        det = det[:, (5, 0,1,2,3, 4)]

        det = det.cpu()
        det = list(det)
        det = [[int(x[0]), float(x[1]/2+x[3]/2), float(x[2]/2+x[4]/2), float(x[3]-x[1]), float(x[4]-x[2]), float(x[5])] for x in det]
        index = all_img_path_name.index( os.path.split(path)[1] )
        det_list[index] = det
    
    try:
        shutil.rmtree(temp_path)
    except:
        pass

    return det_list

def plot_result(big_img, object_box):
    for one_box in object_box:
        label = one_box[0]
        x_center = one_box[1]
        y_center = one_box[2]
        width = one_box[3]
        height = one_box[4]
        probability = one_box[5]
        cv2.rectangle(big_img, (int(x_center-width/2), int(y_center-height/2)), (int(x_center+width/2), int(y_center+height/2)), (255,0,0), 1)
        cv2.putText(big_img, str(int(label)), (int(x_center-width/2), int(y_center-height/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 1)

    big_img = Image.fromarray(big_img)
    big_img.show()


if __name__ == '__main__':
    img_size = 5000
    img_path='data/images/005.png'
    model_path = 'yolov5s.pt'

    big_img = cv2.imread(img_path)
    model, stride, img_size, device = detect(weights=model_path, img_size=img_size)
    object_box = my_detect_application(img_path=img_path, img_size=img_size, model=model, stride=stride, device=device)
    plot_result(big_img, object_box)
