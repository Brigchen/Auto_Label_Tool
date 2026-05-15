# -*- coding: utf-8 -*-
"""
Created on Thu Jul 14 08:45:14 2022

@author: Brig
"""

import os
import random
import shutil
def make_train_val(path_img,path_label,path_out, train_percent=0.7):
    # path_out = './data'
    if os.path.exists(path_img) and os.path.exists(path_label):
        total_img = os.listdir(path_img)
        num = len(total_img)
        tr = int(num * train_percent)
        train = random.sample(range(num), tr)
        sets = ['train', 'val']#, 'val']
        
        for image_set in sets:
            path_out_img = os.path.join(path_out,'images/%s'%image_set)
            if not os.path.exists(path_out_img):
                os.makedirs(path_out_img)
            path_out_label =  os.path.join(path_out,'labels/%s'%image_set)   
            if not os.path.exists(path_out_label):
                os.makedirs(path_out_label)
        
        
        for i in range(num):
            image_id = os.path.splitext(total_img[i])[0]
            # print(image_id)
            try:
                if i in train:
                    if os.path.exists(os.path.join(path_label,image_id+'.txt')):
                        shutil.copy(os.path.join(path_img,total_img[i]), os.path.join(path_out,'images/train'))
                        shutil.copy(os.path.join(path_label,image_id+'.txt'), os.path.join(path_out,'labels/train'))
                    # print(os.path.exists(os.path.join(path_img,image_id+'.jpg')))
                    # print(os.path.join(path_out,'images/train'))
                else:
                    if os.path.exists(os.path.join(path_label,image_id+'.txt')):
                        shutil.copy(os.path.join(path_img,total_img[i]),  os.path.join(path_out,'images/val'))
                        shutil.copy(os.path.join(path_label, image_id+'.txt'), os.path.join(path_out,'labels/val'))
            except Exception as e:
                print(e)
                # ftest.write(name)
    

