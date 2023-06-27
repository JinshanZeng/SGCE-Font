import os
import cv2
import random
img_path = './temp'
for i in os.listdir(img_path):
    yuanshi = os.path.join(img_path, i)
    img = cv2.imread(yuanshi)
    img_fanzhuan = cv2.flip(img, random.randint(-1, 1))
    out_img = img_path + '/fanzhuan' + i
    # print(out_img)
    cv2.imwrite(out_img, img_fanzhuan)
    # delete = 'copy'
    # if delete in i:
    #     delete_name = os.path.join(img_path, i)
    #     os.remove(delete_name)