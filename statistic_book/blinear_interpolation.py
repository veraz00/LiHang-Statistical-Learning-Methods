# https://meghal-darji.medium.com/implementing-bilinear-interpolation-for-image-resizing-357cbb2c2722
import cv2, os
import numpy as np
import math
path = 'D:\zenglinlin\statistic_book\lantern-6826687_960_720.png'
img = cv2.imread(path)
print('img', img)
# cv2.imwrite(aim_path, img)
# cv2.imshow('white', np.zeros((224, 224, 3)))
print(img.shape) # Print image shape (640, 960, 3)

# cv2.imshow("original", img)
def bl_resize(original_img, new_h, new_w):
    old_h, old_w, c = original_img.shape
    resized = np.zeros((new_h, new_w, c))
    w_scale_factor = old_w/new_w if new_w != 0 else 0
    h_scale_factor = old_h/new_h if new_h != 0 else 0
    for i in range(new_h):
        for j in range(new_w):
            x = i * h_scale_factor 
            y = j * w_scale_factor  
            x_floor = math.floor(x)
            x_ceil = min(old_h-1, math.ceil(x))
            y_floor = math.floor(y)
            y_ceil = min(old_w-1, math.ceil(y))
            
            if x_ceil == x_floor and y_ceil == y_floor:
                q = original_img[int(x), int(y), :]
            elif x_ceil == x_floor:
                q1 = original_img[int(x), int(y_floor), :]
                q2 = original_img[int(x), int(y_ceil), :]
                q = q1 * (y_ceil-y) + q2 * (y-y_floor) # ??
            elif y_ceil == y_floor:
                q1 = original_img[int(x_floor), int(y), :]
                q2 = original_img[int(x_ceil), int(y), :]
                q = q1*(x_ceil-x) + q2*(x-x_floor)
            else:
                v1 = original_img[x_floor, y_floor, :]
                v2 = original_img[x_ceil, y_floor, :] 
                v3 = original_img[x_ceil, y_ceil, :]
                v4 = original_img[x_floor, y_ceil, :]
                q1 = v1*(x_ceil - x) + v3 * (x-x_floor)
                q2 = v3*(x-x_floor) + v4 * (x_ceil - x) 
                q = q2 * (y-y_floor) * q1 * (y_ceil - y)
            resized[i, j, :] = q
    return resized.astype(np.uint8)    
new = bl_resize(img, 224, 224)
cv2.imwrite('new.png', new)