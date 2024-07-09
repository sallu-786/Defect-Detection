import numpy as np
import pandas as pd
import cv2


def loadPolyxy(pathCsvxy):    
    """load polygon dimension and create a dictionary for segmentation
    of image into small parts"""
    df = pd.read_csv(pathCsvxy, encoding='utf-8')
    pieces = np.asarray(df.loc[:, ['piece']]).astype('unicode')
    pieces = np.unique(pieces)
    polyxyDict = dict()

    for _, piece in enumerate(pieces):
        poly_xy = df[df['piece'].isin([piece])]
        poly_xy = poly_xy.loc[:, ['x', 'y']]
        poly_xy = np.asarray(poly_xy).astype(np.int32)
        polyxyDict[piece] = poly_xy.astype(int)

    return polyxyDict


def shrink_poly(poly_xy, level):
    """ 縮小ポリゴン座標取得 """
    min_x = int(min(poly_xy[:, 0])) - 1
    max_x = int(max(poly_xy[:, 0])) + 1
    min_y = int(min(poly_xy[:, 1])) - 1
    max_y = int(max(poly_xy[:, 1])) + 1

    poly_xy_tmp = poly_xy - [min_x, min_y]

    width = max_x - min_x
    height = max_y - min_y

    mask = np.zeros((height, width), np.uint8)
    cv2.fillConvexPoly(mask, poly_xy_tmp, (255, 255, 255))

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=level)
   
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = np.array(contours)
    poly_xy_tmp = np.squeeze(contours, (0, 2))
    
    poly_xy = poly_xy_tmp + [min_x, min_y]

    return poly_xy


def check_crop_area_x(w,x): #fix image coordinates
    if x < 0:
        x = 0
    if x > w:
        x = w
    return x

def check_crop_area_y(h,y):
    if y < 0:
        y = 0
    if y > h:
        y = h
    return y

def get_crop_area(poly_xy, w, h):  #crop the area of interest with some extra are around it 

    # クロップ領域拡張率
    ex_rate_h = 0.3             #extra area selection rate
    ex_rate_v = 0.5
    # 画像切り出し
    min_x = int(np.min(poly_xy.T[0]))
    min_y = int(np.min(poly_xy.T[1]))
    max_x = int(np.max(poly_xy.T[0]))
    max_y = int(np.max(poly_xy.T[1]))
    min_x_ex = int(min_x - ((max_x - min_x) * ex_rate_h) * 0.5)
    max_x_ex = int(max_x + ((max_x - min_x) * ex_rate_h) * 0.5)
    min_y_ex = int(min_y - ((max_y - min_y) * ex_rate_v) * 0.5)
    max_y_ex = int(max_y + ((max_y - min_y) * ex_rate_v) * 0.5)

    min_x_ex = check_crop_area_x(w, min_x_ex)
    max_x_ex = check_crop_area_x(w, max_x_ex)

    min_y_ex = check_crop_area_y(h, min_y_ex)
    max_y_ex = check_crop_area_y(h, max_y_ex)

    crop_area = [min_x_ex, min_y_ex, max_x_ex, max_y_ex]
    return crop_area


def make_mask(base_img, poly_xy):
    """ ガウシアンマスク生成 """
    h, w= base_img.shape
    crop_area=get_crop_area(poly_xy, w, h)
    poly_xy_area=poly_xy - np.array([crop_area[0], crop_area[1]])

    mask=np.ones_like(base_img) * 32

    cv2.fillConvexPoly(mask, poly_xy_area, 255)

    kernel=np.array(
        [[0, 0, 0, 1, 0, 0, 0],
         [0, 1, 1, 1, 1, 1, 0],
         [0, 1, 1, 1, 1, 1, 0],
         [1, 1, 1, 1, 1, 1, 1],
         [0, 1, 1, 1, 1, 1, 0],
         [0, 1, 1, 1, 1, 1, 0],
         [0, 0, 0, 1, 0, 0, 0]], np.uint8)

    mask=cv2.dilate(mask, kernel, iterations = 16)
    mask=cv2.GaussianBlur(mask, (127, 127), 20)

    return mask


def part_crop(src, part_poly_xy):
    """ パーツ座標+マージンを切り出す """
    h, w, _ = src.shape
    part_area_margine = get_crop_area(part_poly_xy, w, h)
    dst = src[part_area_margine[1]:part_area_margine[3],part_area_margine[0]:part_area_margine[2]]

    return dst

def preprocess(src):
    tmp = src.copy()
    if len(tmp.shape) == 3:
        tmp = cv2.cvtColor(tmp, cv2.COLOR_BGR2GRAY)
    ori = tmp.copy()
    tmp = cv2.GaussianBlur(tmp, (0, 0), 0.05)
    tmp = cv2.equalizeHist(tmp)

    dst = tmp
    return dst


def get_phase(base_img, target_img, mask):
    """ ズレたぶんの位相を取得 """

    base_img = preprocess(base_img)
    target_img = preprocess(target_img)

    mask=mask.astype(np.float32) / 255.0
    base_img=base_img.astype(np.float32) / 255.0
    target_img=target_img.astype(np.float32) / 255.0

    base_img=base_img * mask
    target_img=target_img * mask
    # 位相限定相関法
    phase=cv2.phaseCorrelate(base_img, target_img)
    return phase


def img_phase_correlate(src, phase):
    M=np.float32([[1, 0, -phase[0][0]], [0, 1, -phase[0][1]]])
    rows, cols=src.shape[:2]
    dst=cv2.warpAffine(src, M, (cols, rows))
    return dst


def stabilizer_method(base_img, target_img, mask = None):
    """ 位置合わせ（単一画像） """

    # 位相限定相関法
    phase=get_phase(base_img, target_img, mask)

    # 位置合わせ
    stab_img=img_phase_correlate(target_img, phase)

    shift_xy=phase[0]

    return stab_img, shift_xy, phase


def fillout_color(img, poly_xy, offset_xy, fill_outer_color):
    """ ポリゴンの外側を塗りつぶす """
    poly_xy_tmp = poly_xy - offset_xy
    src = img.astype(np.float32) / 255.0

    back = np.empty((img.shape[0], img.shape[1], 3), np.uint8)
    back[:] = fill_outer_color
    back = back.astype(np.float32) / 255.0

    mask = np.zeros_like(img, np.float32)
    cv2.fillConvexPoly(mask, poly_xy_tmp, (1.0, 1.0, 1.0))

    # cv2.imshow('mask', mask)
    # cv2.waitKey(0)

    dst = src * mask + back * (1.0 - mask)
    dst = (dst * 255).astype(np.uint8)

    return dst


def bounding_crop(src, poly_xy, offset_xy):   #get boundry for area of interest
    """ ピースカット画像トリミング """
    min_x = int(min(poly_xy[:, 0])) - 1 - offset_xy[0]
    max_x = int(max(poly_xy[:, 0])) + 1 - offset_xy[0]
    min_y = int(min(poly_xy[:, 1])) - 1 - offset_xy[1]
    max_y = int(max(poly_xy[:, 1])) + 1 - offset_xy[1]

    dst = src[min_y:max_y, min_x:max_x]

    return dst