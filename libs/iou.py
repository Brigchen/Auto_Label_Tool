# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 23:00:49 2023

@author: brigc
"""

# encoding: utf-8
def compute_IOU(rec1,rec2):
    """
    计算两个矩形框的交并比。
    :param rec1: (x0,y0,x1,y1)      (x0,y0)代表矩形左上的顶点，（x1,y1）代表矩形右下的顶点。下同。
    :param rec2: (x0,y0,x1,y1)
    :return: 交并比IOU.
    """
    left_column_max  = max(rec1[0],rec2[0])
    right_column_min = min(rec1[2],rec2[2])
    up_row_max       = max(rec1[1],rec2[1])
    down_row_min     = min(rec1[3],rec2[3])
    #两矩形无相交区域的情况
    if left_column_max>=right_column_min or down_row_min<=up_row_max:
        return 0
    # 两矩形有相交区域的情况
    else:
        S1 = (rec1[2]-rec1[0])*(rec1[3]-rec1[1])
        S2 = (rec2[2]-rec2[0])*(rec2[3]-rec2[1])
        S_cross = (down_row_min-up_row_max)*(right_column_min-left_column_max)
        return S_cross/(S1+S2-S_cross)
#测试样例1
# r1=(2,3,10,12)
# r2=(12,5,20,24)
# IOU = compute_IOU(r1,r2)
# print("测试样例1，IOU：%f"%IOU)
# #测试样例2
# r1=(2,2,4,4)
# r2=(3,3,5,5)
# IOU = compute_IOU(r1,r2)
# print("测试样例2，IOU：%f"%IOU)
