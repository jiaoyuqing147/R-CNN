# -*- coding: utf-8 -*-

"""
@author: zj
@file:   selectivesearch.py
@time:   2020-02-25
"""

import sys
import cv2


def get_selective_search():
    gs = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    return gs


def config(gs, img, strategy='q'):
    gs.setBaseImage(img)

    if (strategy == 's'):
        gs.switchToSingleStrategy()
    elif (strategy == 'f'):
        gs.switchToSelectiveSearchFast()
    elif (strategy == 'q'):
        gs.switchToSelectiveSearchQuality()
    else:
        print(__doc__)
        sys.exit(1)


def get_rects(gs):
    rects = gs.process()#获取矩形框，但是这个矩形框是(x, y, width, height)形式表示的矩形坐标
    rects[:, 2] += rects[:, 0]#矩形框转换，(x, y, width, height)->(x1, y1, x2, y2)
    rects[:, 3] += rects[:, 1]#矩形框转换，(x, y, width, height)->(x1, y1, x2, y2)
    return rects


def rect_img(img, color, rects):
    for x1,y1,x2,y2 in rects[0:2000]:
        cv2.rectangle(img, (x1,y1),(x2,y2),color,thickness=2)

        cv2.putText(img, 'objectname', (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), thickness=2)

        #cv2.imshow( 'lena_rect', img)




if __name__ == '__main__':
    """
    选择性搜索算法操作
    """
    img = cv2.imread('./data/lena.jpg', cv2.IMREAD_COLOR)
    gs = get_selective_search()
    config(gs, img, strategy='q')
    rects = get_rects(gs)
    print(len(rects))#lena.jpg这幅图总共可以生成6203个矩形
    color = (255,255,255)

    rect_img(img,color,rects)#这个函数中只取了2000个矩形
    cv2.imshow('head', img)
    cv2.imwrite("E:/R-CNN/py/data/lena_rect.jpg", img)
    cv2.waitKey()