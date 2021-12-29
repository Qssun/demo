import cv2
import os
from PIL import Image
from paddleocr import PPStructure, save_structure_res

save_path = 'result'
img = cv2.imread('034.png')

try:
    os.remove(os.path.join(save_path, 'test.xlsx'))
except:
    pass
# 表格内容识别
table_engine = PPStructure(show_log=True)
result = table_engine(img)
# 识别结果保存
# print(result)
save_structure_res(result, '', os.path.basename(save_path).split('.')[0], )

# 识别结果显示，弹出打开xlsx和图片
xlsx_file_path = [os.path.join(save_path, x)
                  for x in os.listdir(save_path) if 'xlsx' in x]
if len(xlsx_file_path) > 0:
    xlsx_path = xlsx_file_path[0]

os.rename(xlsx_path, os.path.join(save_path, 'test.xlsx'))
xlsx_file_path = [os.path.join(save_path, x)
                  for x in os.listdir(save_path) if 'xlsx' in x]
if len(xlsx_file_path) > 0:
    xlsx_path = xlsx_file_path[0]
os.popen('start {}'.format(os.path.join(os.getcwd(), xlsx_path)))

img = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
img.show()
