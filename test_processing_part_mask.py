import cv2
import numpy as np

T_IMG = 1

def get_img(path: str):
    return cv2.imread(fr"{path}", cv2.IMREAD_GRAYSCALE)  # загрузка изображения


def show_img(img):
    cv2.imshow("Binary", img)
    cv2.waitKey(0)

def resizing(img):
    res_img = cv2.resize(
        img, (img.shape[0]*3, img.shape[1]*2), cv2.INTER_LINEAR)
    return res_img


def cropping(img, st_w, end_w, st_h, end_h):
    return img[st_w:end_w, st_h:end_h]

def notOutBorder(border_y, y):
    return not (-1 < y < border_y)


def find_sf(img, i, first, last, direction, checking) -> tuple:

    for j in range(first, last, direction):

        if (img[i][j]):
            img[i][j] = 0
            T_IMG[i][j] = 100
        else:
            break

    i += checking  # checking - сверху или снизу
    if (direction == -1):
        for j in range(j - 1, j - 1 + 10):
            # T_IMG[i][j] = 100
            if (img[i][j]):
                return i, j
    else:
        for j in range(j + 1*notOutBorder(img.shape[1], j), j - 1 - 10, -1):
            # T_IMG[i][j] = 100
            if (img[i][j]):
                return i, j
    return -1, -1

def isColorObject(color_object: int):
    color_field = 0 # цвет фона
    return color_field < color_object

def find_sf_hor(img, j, first, last, direction, checking):
    
    for i in range(first, last, direction):
        if (isColorObject(img[i][j])):
            img[i][j] = 0
            T_IMG[i][j] = 100
        else:
            break
        
    j += checking # checking - слева или справа
    if (direction == -1):
        for i in range(i - 1, i + 9):
            if (isColorObject(img[i][j])):
                return i, j
    else:
        for i in range(i + 1*notOutBorder(img.shape[0], i), i - 10, -1):
            if (isColorObject(img[i][j])):
                return i, j
    return -1, -1

def del_flowers(img):
    im_height = img.shape[0]
    im_width = img.shape[1]
    
    # сверху
    
    for j in range(im_width):
        last = im_width
        i = 0
        dir = 1
        while (i > -1 and isColorObject(img[i][j])):
            i, j = find_sf(img, i, j, last, dir, 1)
            dir = -dir
            if (dir == 1):
                last = im_width
            else:
                last = -1
        
    #снизу
    
    for j in range(im_width):
        dir = 1
        last = im_width
        i = im_height - 1
        while (i > -1 and isColorObject(img[i][j])):
            i, j = find_sf(img, i, j, last, dir, -1)
            dir = -dir
            if (dir == 1):
                last = im_width
            else:
                last = -1
        
    
    #слева
    for i in range(im_height):
        dir = 1
        last = im_height
        j = 0
        while (j > -1 and isColorObject(img[i][j])):
            i, j = find_sf_hor(img, j, i, last, dir, 1)
            dir = -dir
            if (dir == 1):
                last = im_height
            else:
                last = -1
    #слева
    for i in range(im_height):
        dir = 1
        last = im_height
        j = im_width-1
        while (j > -1 and isColorObject(img[i][j])):
            i, j = find_sf_hor(img, j, i, last, dir, -1)
            dir = -dir
            if (dir == 1):
                last = im_height
            else:
                last = -1
        
def prog(img):
    global T_IMG 
    T_IMG = img
    # img = get_img(path)
    # T_IMG = get_img(t_path)
    show_img(img)
    del_flowers(img)
    show_img(T_IMG)

    print("Done")
    

if (__name__ == "__main__"):
    photo = "sf.png"
    img = get_img(photo)
    # programm("final.png", "t_final.png")
    # programm(photo, "t_" + photo)
    step = 300
    for i in range(0, img.shape[0]-step, step):
        for j in range(0, img.shape[1] - step, step):
            part_img = resizing(cropping(img, i, i+step, j, j+step))
            prog(part_img)
