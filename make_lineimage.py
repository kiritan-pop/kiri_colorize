# coding: utf-8

import os,glob,sys,random
from PIL import Image, ImageOps, ImageFile, ImageChops, ImageFilter, ImageEnhance
from multiprocessing import Process, Queue
import numpy as np
ImageFile.LOAD_TRUNCATED_IMAGES = True

color_dir = './class_col/'
line_dir = './line_images5/'

def new_convert(img, mode):
    if img.mode == "RGBA":
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[3])
    elif img.mode == "LA":
        bg = Image.new("L", img.size, (255,))
        bg.paste(img, mask=img.split()[1])
    else:
        bg = img
    return bg.convert(mode)


def image_to_line(img): # img:RGBモード
    # 線画化
    gray = new_convert(img, "RGB") #グレイスケール
    gray2 = gray.filter(ImageFilter.MaxFilter(5))
    senga_inv = ImageChops.difference(gray, gray2)
    senga_inv = ImageOps.invert(senga_inv)

    senga_inv = new_convert(senga_inv, "L") #グレイスケール
    # senga_inv = senga_inv.filter(ImageFilter.MinFilter(3))
    # en = ImageEnhance.Contrast(senga_inv)
    # senga_inv = en.enhance(3.0)
    # senga_inv = senga_inv.filter(ImageFilter.MaxFilter(3))
    return senga_inv

def image_to_line2(img): # img:RGBモード
    # 線画化
    gray = new_convert(img, "RGB") #グレイスケール
    gray2 = gray.filter(ImageFilter.CONTOUR)
    gray2 = new_convert(gray2, "L") #グレイスケール
    return gray2

def image_to_line_prc(num,readQ):
    print(f'--Process({num}) start')
    cnt = 0
    try:
        while True:
            img, output_path = readQ.get(timeout=10) #キューからトゥートを取り出すよー！
            img = Image.fromarray(img)
            line = image_to_line(img)
            line.save(output_path, optimize=True)
            cnt += 1
            if cnt%100 == 0:
                print(f'--executing image_to_line({num}) {cnt}images--')
    except Exception as e:
        print(e)
        print(f'--Finish image_to_line({num}) {cnt}images--')
        return

def reader(readQ):
    print('--Start reading--')
    cnt = 0
    for p in os.listdir(color_dir):
        for f in os.listdir(color_dir + p + '/'):
            input_path = color_dir + p + '/' + f
            output_path = line_dir + p + '/' + f
            if not os.path.exists(output_path):
                img = Image.open(input_path)
                img = np.asarray(img, dtype=np.uint8)
                readQ.put((img,output_path))
                cnt += 1
                if cnt % 100 == 0:
                    print(f'reading {cnt}images...')
    print(f'--Finish reading {cnt}--')


if __name__ == "__main__":
    if not os.path.exists(color_dir):
        exit()

    for p in os.listdir(color_dir):
        os.makedirs(os.path.join(line_dir,p), exist_ok=True)

    readQ = Queue()

    p_r = Process(target=reader, args=(readQ,))
    p_r.start()

    p_n = []
    for num in range(0,10):
        tmp  = Process(target=image_to_line_prc, args=(num,readQ))
        tmp.start()
        p_n.append(tmp)
   
