from detr import detr_inf
from torchvision_models import tv_inf
from yolov8_inf import yolov8_inf
import os


if __name__ == '__main__':
    root = 'object_detection_imgs'
    file_list = os.listdir(root)
    i = 0
    for f in file_list:
        file_pth = os.path.join(root,f)

        detr_inf(file_pth, i)
        tv_inf(file_pth, i)
        yolov8_inf(file_pth, i)
        i+=1

