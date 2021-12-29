import numpy as np
import cv2

# direction_E = [round(width_height/2), round(width_height/2):]
# direction_S = [round(width_height/2):, round(width_height/2)]
# direction_W = [round(width_height/2), :round(width_height/2+1)]
# direction_N = [:round(width_height/2+1), round(width_height/2)]

def gen_one_conn(width_height):
    if width_height%2==0:
        width_height -= 1
    empty_img_E = np.ones(shape=(width_height, width_height), dtype=np.uint8)*255
    empty_img_S = np.ones(shape=(width_height, width_height), dtype=np.uint8)*255
    empty_img_W = np.ones(shape=(width_height, width_height), dtype=np.uint8)*255
    empty_img_N = np.ones(shape=(width_height, width_height), dtype=np.uint8)*255
    empty_img_E[round(width_height/2), round(width_height/2):] = 0
    empty_img_S[round(width_height/2):, round(width_height/2)] = 0
    empty_img_W[round(width_height/2), :round(width_height/2+1)] = 0
    empty_img_N[:round(width_height/2+1), round(width_height/2)] = 0

    return empty_img_E, empty_img_S, empty_img_W, empty_img_N

def gen_two_conn(width_height):
    if width_height%2==0:
        width_height -= 1
    empty_img_ES = np.ones(shape=(width_height, width_height), dtype=np.uint8)*255
    empty_img_SW = np.ones(shape=(width_height, width_height), dtype=np.uint8)*255
    empty_img_WN = np.ones(shape=(width_height, width_height), dtype=np.uint8)*255
    empty_img_NE = np.ones(shape=(width_height, width_height), dtype=np.uint8)*255
    empty_img_ES[round(width_height/2), round(width_height/2):] = 0
    empty_img_ES[round(width_height/2):, round(width_height/2)] = 0

    empty_img_SW[round(width_height/2):, round(width_height/2)] = 0
    empty_img_SW[round(width_height/2), :round(width_height/2+1)] = 0

    empty_img_WN[round(width_height/2), :round(width_height/2+1)] = 0
    empty_img_WN[:round(width_height/2+1), round(width_height/2)] = 0
    
    empty_img_NE[:round(width_height/2+1), round(width_height/2)] = 0
    empty_img_NE[round(width_height/2), round(width_height/2):] = 0

    return empty_img_ES, empty_img_SW, empty_img_WN, empty_img_NE

def gen_three_conn(width_height):
    if width_height%2==0:
        width_height -= 1
    empty_img_ESW = np.ones(shape=(width_height, width_height), dtype=np.uint8)*255
    empty_img_SWN = np.ones(shape=(width_height, width_height), dtype=np.uint8)*255
    empty_img_WNE = np.ones(shape=(width_height, width_height), dtype=np.uint8)*255
    empty_img_NES = np.ones(shape=(width_height, width_height), dtype=np.uint8)*255
    empty_img_ESW[round(width_height/2), round(width_height/2):] = 0
    empty_img_ESW[round(width_height/2):, round(width_height/2)] = 0
    empty_img_ESW[round(width_height/2), :round(width_height/2+1)] = 0

    empty_img_SWN[round(width_height/2):, round(width_height/2)] = 0
    empty_img_SWN[round(width_height/2), :round(width_height/2+1)] = 0
    empty_img_SWN[:round(width_height/2+1), round(width_height/2)] = 0

    empty_img_WNE[round(width_height/2), :round(width_height/2+1)] = 0
    empty_img_WNE[:round(width_height/2+1), round(width_height/2)] = 0
    empty_img_WNE[round(width_height/2), round(width_height/2):] = 0
    
    empty_img_NES[:round(width_height/2+1), round(width_height/2)] = 0
    empty_img_NES[round(width_height/2), round(width_height/2):] = 0
    empty_img_NES[round(width_height/2):, round(width_height/2)] = 0

    return empty_img_ESW, empty_img_SWN, empty_img_WNE, empty_img_NES

def gen_four_conn(width_height):
    if width_height%2==0:
        width_height -= 1
    empty_img_ESWN = np.ones(shape=(width_height, width_height), dtype=np.uint8)*255
    empty_img_ESWN[round(width_height/2), round(width_height/2):] = 0
    empty_img_ESWN[round(width_height/2):, round(width_height/2)] = 0
    empty_img_ESWN[round(width_height/2), :round(width_height/2+1)] = 0
    empty_img_ESWN[:round(width_height/2+1), round(width_height/2)] = 0

    return empty_img_ESWN

def gen_all_kinks_conn(width_height):
    # 所有拐点的模板图像
    all_conn_template = {}
    empty_img_E, empty_img_S, empty_img_W, empty_img_N = gen_one_conn(width_height)
    empty_img_ES, empty_img_SW, empty_img_WN, empty_img_NE = gen_two_conn(width_height)
    empty_img_ESW, empty_img_SWN, empty_img_WNE, empty_img_NES = gen_three_conn(width_height)
    empty_img_ESWN = gen_four_conn(width_height)
    all_conn_template['E'] = empty_img_E
    all_conn_template['S'] = empty_img_S
    all_conn_template['W'] = empty_img_W
    all_conn_template['N'] = empty_img_N
    all_conn_template['ES'] = empty_img_ES
    all_conn_template['SW'] = empty_img_SW
    all_conn_template['WN'] = empty_img_WN
    all_conn_template['NE'] = empty_img_NE
    all_conn_template['ESW'] = empty_img_ESW
    all_conn_template['SWN'] = empty_img_SWN
    all_conn_template['WNE'] = empty_img_WNE
    all_conn_template['NES'] = empty_img_NES
    all_conn_template['ESWN'] = empty_img_ESWN

    return all_conn_template


if __name__=="__main__":
    all_conn_template = gen_all_kinks_conn(50)
    for item in all_conn_template:
        cv2.imshow(item, all_conn_template[item])
        cv2.waitKey(1000)
    