import math
import os
import time

import cv2 as cv
import cv2
import numpy as np


# 显示图片
def show(window_name, img_name, size=(600, 800)):
    cv2.namedWindow(str(window_name), cv2.WINDOW_NORMAL)
    cv2.resizeWindow(str(window_name), size[0], size[1])  # 改变窗口大小
    cv2.imshow(str(window_name), img_name)


# 图片预处理，返回二值化图片
def preprogress(img):
    gray = cv.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 黑帽运算
    kernel = np.ones((20, 20), np.uint8)
    balck_hat = cv.morphologyEx(gray, cv.MORPH_BLACKHAT, kernel)
    show('balck_hat', balck_hat)

    # 二值化
    _, threshold = cv.threshold(balck_hat, 1, 255, cv.THRESH_OTSU)
    show('threshold', threshold)

    # 二值化反转黑白
    _, threshold1 = cv.threshold(threshold, 150, 255, cv.THRESH_BINARY_INV)
    show('threshold1', threshold1)
    return threshold1


# 对四边形四个点进行排序 https://blog.csdn.net/littlezhuhui/article/details/100567965
def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect.astype(int)


# 对轮廓大小进行排序
def cnt_area(cnt):
    area = cv2.contourArea(cnt)
    return area


# 获取拟合边界的四边形框
def bounding_box(c):
    epsilon = 1
    while True:
        approxBox = cv2.approxPolyDP(c, epsilon, True)
        if len(approxBox) < 4:
            return None
        if len(approxBox) > 4:
            epsilon += 1
            continue
        else:           # approx的长度为4，表明已经拟合成矩形了
            approxBox = approxBox.reshape((4, 2))                   # 转换成4*2的数组
            return approxBox


# 投影变换，截取四边形区域内的图像并变换为矩形
def perspective_transform(img, points):
    width, height = 1800, 1000
    target_points = [(0, 0), (width, 0), (width, height), (0, height)]
    points, target_points = np.array(points, dtype=np.float32), np.array(target_points, dtype=np.float32)
    #  透视变换矩阵
    M = cv.getPerspectiveTransform(points, target_points)
    result = cv.warpPerspective(img, M, (0, 0))
    result = result[:height, :width]
    # show('transform', result, (900, 500))
    return result


# 将网格内的每个数字截取出来
def slice_image(img, img_name):
    for i in range(10):
        if not os.path.exists('./data/number/' + str(i)):  # 判断是否存在文件夹如果不存在则创建为文件夹
            os.makedirs('./data/number/' + str(i))
        for j in range(17):
            number = img[i * 100 + 10:i * 100 + 95, j * 100 + 100 + 10:j * 100 + 100 + 90]
            cv.imwrite('./data/number/{}/{}_{}_{}.jpg'.format(i, img_name, i, j), number)


def main():
    image_list = os.listdir('data/raw_image')
    if len(image_list) == 0:
        print('请创建\'data/raw_image\'文件夹，并将原始图片放入这个文件夹')
    for i in range(len(image_list)):
        raw_image = cv.imread('data/raw_image/' + image_list[i])
        if raw_image.shape[0] < raw_image.shape[1]:  # 部分图像方向错误，需左转
            rotate_image = cv.rotate(raw_image, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)
            cv.imwrite('data/raw_image/' + image_list[i], rotate_image)
            raw_image = rotate_image
        bianry_image = preprogress(raw_image)

        contours, hierarchy = cv2.findContours(bianry_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        contours.sort(key=cnt_area, reverse=True)
        contours_image = raw_image.copy()

        for j in range(1, 3):  # 获取上下两部分的两个大框
            box = bounding_box(contours[j])
            box = order_points(box)

            cv.drawContours(contours_image, [box], -1, (0, 0, 255), 10)
            trans = perspective_transform(bianry_image, box)
            show('transform' + str(j), trans, (900, 500))
            slice_image(trans, image_list[i][11:17])

        cv.waitKey(500)
        time.sleep(5)
        cv.destroyAllWindows()
        print(image_list[i], 'finished.')


if __name__ == '__main__':
    main()
