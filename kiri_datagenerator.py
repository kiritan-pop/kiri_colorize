# coding: utf-8

import os,glob,sys,json,random,cv2,threading,pprint,queue
from time import sleep
import numpy as np
import math
from PIL import Image, ImageOps, ImageFile, ImageChops, ImageFilter, ImageEnhance
import tensorflow as tf
graph = tf.get_default_graph()

STANDARD_SIZE_S1 = (128, 128)
STANDARD_SIZE_S2 = (512, 512)

Colors = {}
Colors['red']    = 0
Colors['blue']   = 1
Colors['green']  = 2
Colors['purple'] = 3
Colors['brown']  = 4
Colors['pink']   = 5
Colors['blonde'] = 6
Colors['white']  = 7
Colors['black']  = 8

Colors_rev = {v:k for k,v in Colors.items()}

def image_to_line_open(path):
    tmp = path.split("/")
    tmp[-3] = "line_images"
    line_path = "/".join(tmp)
    return Image.open(line_path).convert("L")

def line_enhance(img): # img:RGBモード
    gray = img.convert("L") #グレイスケール
    # gray = gray.filter(ImageFilter.MinFilter(3))
    gray = gray.convert("RGB") #グレイスケール
    return gray

def image_arrange(path, resize=(128,128)):
    img = Image.open(path).convert('RGB')
    #画像加工ランダム値（重いので）
    rotate_rate = random.choice([0,90,180,270]) #回転
    mirror =  random.randint(0,100) % 2

    #線画（線画化してからリサイズ）
    line = image_to_line_open(path)

    img = img.resize(resize, Image.BICUBIC).rotate(rotate_rate, expand=False, resample=Image.NEAREST)
    line = line.resize(resize, Image.BICUBIC).rotate(rotate_rate, expand=False, resample=Image.NEAREST)

    if mirror ==0:
        img = ImageOps.mirror(img)
        line = ImageOps.mirror(line)

    return img, line


class D_Datagenerator():
    def __init__(self, color_path, batch_size, g_model, val=0):
        # コンストラクタ
        self.color_path = color_path
        self.batch_size = batch_size
        self.color_images = []
        for p in os.listdir(color_path):
            for f in os.listdir(color_path + p + '/'):
                self.color_images.append(color_path + p + '/' + f)
        random.shuffle(self.color_images)
        self.val = val
        self.valid_images = random.sample(self.color_images,batch_size)
        for d in self.valid_images:
            self.color_images.remove(d)
        self.g_model = g_model
        self.old_fake = {}

        # エラー回避のため、１回空振りさせる
        img = np.zeros((STANDARD_SIZE_S1[0],STANDARD_SIZE_S1[1]))
        with graph.as_default():
            g_model.predict_on_batch([np.array([img]), np.array([Colors['red']])])
        sleep(2)

    def __getitem__(self, idx):
        xy  = queue.Queue()
        # データの取得実装
        if self.val == 999:
            ytmp1 = self.valid_images
        else:
            ytmp1 = self.color_images[self.batch_size*idx:self.batch_size*(idx+1)]

        def func(path):
            img, line = image_arrange(path, resize=STANDARD_SIZE_S1)
            line = (np.asarray(line)-127.5)/127.5
            img = (np.asarray(img)-127.5)/127.5

            # rand = random.randint(0,100)    # BartchNormalization しない場合はこっち
            if self.val == 999:
                rand = random.randint(0,1000) 
            else:
                rand = self.val

            selcol = path.split('/')[-2]
            selvec = Colors[selcol]

            if rand%3 == 0:
                #本物
                x = img
                retvecs = selvec
                lines = line
                r = random.uniform(0.0, 0.2)
                y = np.asarray([1.0 - r, 0.0 + r])

            elif rand%3 == 1:
                #本物だけど、色間違い
                x = img
                # カラーラベルランダム取得（正解以外）
                tmpcol = list(Colors.keys())
                tmpcol.remove(selcol) 
                selcol2 = random.choice(tmpcol)
                selvec2 = Colors[selcol2]
                retvecs = selvec2
                lines = line
                r = random.uniform(0.0, 0.2)
                y = np.asarray([0.0 + r, 1.0 - r])

            else:
                #偽物
                selcol3 = random.choice(list(Colors.keys()))
                selvec3 = Colors[selcol3]

                if random.randint(0,10) == 0 and path in self.old_fake:
                    #過去の生成画像を使用する
                    x = self.old_fake[path]
                else:
                    with graph.as_default():
                        ret = self.g_model.predict_on_batch([np.array([line]), np.array([selvec3])])
                    x = ret[0]

                retvecs = selvec3
                lines = line
                r = random.uniform(0.0, 0.2)
                y = np.asarray([0.0 + r, 1.0 - r])

                r = random.randint(0,self.batch_size*5)
                if r == 0:
                    filename = f'temp/gen_{selcol3}.png'
                    tmp = (x*127.5+127.5).clip(0, 255).astype(np.uint8)
                    Image.fromarray(tmp).save(filename,'png', optimize=True)

                    filename = f'temp/gen_{selcol3}_in.png'
                    tmp = (line*127.5+127.5).clip(0, 255).astype(np.uint8)
                    Image.fromarray(tmp).save(filename,'png', optimize=True)

                    #偽物画像を保存しておく（次のDiscriminater用の学習へ）
                    self.old_fake[path] = x

            xy.put([x, lines, retvecs, y])

        threads = []
        for path in ytmp1:
            threads.append( threading.Thread(target=func, args=(path,) ))

        for th in threads:
            th.start()
        for th in threads:
            th.join()

        #順序がバラバラにならないように
        x = []
        lines = []
        retvecs = []
        y = []
        while xy.qsize() > 0:
            a,b,c,d = xy.get()
            x.append(a)
            lines.append(b)
            retvecs.append(c)
            y.append(d)

        # return [np.asarray(x), np.asarray(lines), np.asarray(retvecs)], np.asarray(y)
        return [np.asarray(x), np.asarray(retvecs)], np.asarray(y)

    def __len__(self):
        # 全データ数をバッチサイズで割って、何バッチになるか返すよー！
        sample_per_epoch = math.ceil(len(self.color_images)/self.batch_size)
        return sample_per_epoch

class Comb_Datagenerator():
    def __init__(self, color_path, batch_size, val=0):
        # コンストラクタ
        self.color_path = color_path
        self.batch_size = batch_size
        self.val = val
        self.color_images = {}
        self.color_images_flat = []
        self.valid_images = []
        for p in os.listdir(color_path):
            tmp = []
            for f in os.listdir(color_path + p + '/'):
                tmp.append(color_path + p + '/' + f)

            valid_tmp = random.sample(tmp, math.ceil(batch_size/len(Colors)))
            self.valid_images.extend(valid_tmp)
            for d in valid_tmp:
                tmp.remove(d)
            random.shuffle(tmp)
            self.color_images[p] = tmp
            self.color_images_flat.extend(tmp)

        random.shuffle(self.color_images_flat)

    def __getitem__(self, idx):
        # データの取得実装
        if self.val == 999:
            ytmp1 = random.sample(self.valid_images, self.batch_size)
        else:
            ytmp1 = self.color_images_flat[self.batch_size*idx:self.batch_size*(idx+1)]
        xy  = queue.Queue()
        def func(path):
            img, line = image_arrange(path, resize=STANDARD_SIZE_S1)
            line = (np.asarray(line)-127.5)/127.5
            img = (np.asarray(img)-127.5)/127.5
            color_name = path.split("/")[-2]
            selvec = Colors[color_name]

            # r = random.uniform(0.0, 0.2)
            r = 0
            y = [1.0 - r, 0.0 + r]

            xy.put([line, selvec, y, img])

        threads = []
        for path in ytmp1:
            #本物
            threads.append( threading.Thread(target=func, args=(path,) ))

        for th in threads:
            th.start()
        for th in threads:
            th.join()

        #順序がバラバラにならないように
        x = []
        selvecs = []
        y = []
        y_imgs = []
        while xy.qsize() > 0:
            a,b,c,d = xy.get()
            x.append(a)
            selvecs.append(b)
            y.append(c)
            y_imgs.append(d)

        return [np.asarray(x), np.asarray(selvecs)], [np.asarray(y), np.asarray(y_imgs)]

    def __len__(self):
        # 全データ数をバッチサイズで割って、何バッチになるか返すよー！

        return math.ceil(len(self.color_images_flat)/self.batch_size)


class D_DatageneratorS2():
    def __init__(self, color_path, batch_size, g_model, val=0):
        # コンストラクタ
        self.color_path = color_path
        self.batch_size = batch_size
        self.color_images = []
        for p in os.listdir(color_path):
            for f in os.listdir(color_path + p + '/'):
                self.color_images.append(color_path + p + '/' + f)
        random.shuffle(self.color_images)
        self.val = val
        self.valid_images = random.sample(self.color_images,batch_size)
        for d in self.valid_images:
            self.color_images.remove(d)
        self.g_model = g_model
        self.old_fake = {}

        # エラー回避のため、１回空振りさせる
        img_s = np.zeros((STANDARD_SIZE_S1[0],STANDARD_SIZE_S1[1],3))
        img = np.zeros((STANDARD_SIZE_S2[0],STANDARD_SIZE_S2[1]))
        with graph.as_default():
            g_model.predict_on_batch([np.array([img]), np.array([img_s])])
        sleep(2)

    def __getitem__(self, idx):
        xy  = queue.Queue()
        # データの取得実装
        if self.val == 999:
            ytmp1 = self.valid_images
        else:
            ytmp1 = self.color_images[self.batch_size*idx:self.batch_size*(idx+1)]

        def func(path):
            img, line = image_arrange(path, resize=STANDARD_SIZE_S2)
            img_small = img.resize(STANDARD_SIZE_S1, Image.BICUBIC)
            line = (np.asarray(line)-127.5)/127.5
            img = (np.asarray(img)-127.5)/127.5
            img_small = (np.asarray(img_small)-127.5)/127.5

            # rand = random.randint(0,100)    # BartchNormalization しない場合はこっち
            if self.val == 999:
                rand = random.randint(0,1000) 
            else:
                rand = self.val

            selcol = path.split('/')[-2]
            selvec = Colors[selcol]

            if rand%2 == 0:
                #本物
                x = img
                retvecs = selvec
                lines = line
                r = random.uniform(0.0, 0.2)
                y = np.asarray([1.0 - r, 0.0 + r])

            else:
                #偽物
                selcol3 = random.choice(list(Colors.keys()))
                selvec3 = Colors[selcol3]

                if random.randint(0,10) == 0 and path in self.old_fake:
                    #過去の生成画像を使用する
                    x = self.old_fake[path]
                else:
                    with graph.as_default():
                        ret = self.g_model.predict_on_batch([np.array([line]), np.array([img_small])])
                    x = ret[0]

                retvecs = selvec3
                lines = line
                r = random.uniform(0.0, 0.2)
                y = np.asarray([0.0 + r, 1.0 - r])

                r = random.randint(0,self.batch_size*5)
                if r == 0:
                    filename = f'temp/gen2.png'
                    tmp = (x*127.5+127.5).clip(0, 255).astype(np.uint8)
                    Image.fromarray(tmp).save(filename,'png', optimize=True)

                    filename = f'temp/gen2_in.png'
                    tmp = (line*127.5+127.5).clip(0, 255).astype(np.uint8)
                    Image.fromarray(tmp).save(filename,'png', optimize=True)

                    #偽物画像を保存しておく（次のDiscriminater用の学習へ）
                    self.old_fake[path] = x

            xy.put([x, lines, retvecs, y])

        threads = []
        for path in ytmp1:
            threads.append( threading.Thread(target=func, args=(path,) ))

        for th in threads:
            th.start()
        for th in threads:
            th.join()

        #順序がバラバラにならないように
        x = []
        y = []
        while xy.qsize() > 0:
            a,b,c,d = xy.get()
            x.append(a)
            # lines.append(b)
            # retvecs.append(c)
            y.append(d)

        return np.asarray(x), np.asarray(y)

    def __len__(self):
        # 全データ数をバッチサイズで割って、何バッチになるか返すよー！
        sample_per_epoch = math.ceil(len(self.color_images)/self.batch_size)
        return sample_per_epoch


class Comb_DatageneratorS2():
    def __init__(self, color_path, batch_size, val=0):
        # コンストラクタ
        self.color_path = color_path
        self.batch_size = batch_size
        self.val = val
        self.color_images = {}
        self.color_images_flat = []
        self.valid_images = []
        for p in os.listdir(color_path):
            tmp = []
            for f in os.listdir(color_path + p + '/'):
                tmp.append(color_path + p + '/' + f)

            valid_tmp = random.sample(tmp, math.ceil(batch_size/len(Colors)))
            self.valid_images.extend(valid_tmp)
            for d in valid_tmp:
                tmp.remove(d)
            random.shuffle(tmp)
            self.color_images[p] = tmp
            self.color_images_flat.extend(tmp)

        random.shuffle(self.color_images_flat)

    def __getitem__(self, idx):
        # データの取得実装
        if self.val == 999:
            ytmp1 = random.sample(self.valid_images, self.batch_size)
        else:
            ytmp1 = self.color_images_flat[self.batch_size*idx:self.batch_size*(idx+1)]
        xy  = queue.Queue()
        def func(path):
            img, line = image_arrange(path, resize=STANDARD_SIZE_S2)
            img_small = img.resize(STANDARD_SIZE_S1, Image.BICUBIC)
            img_small = img_small.filter(ImageFilter.GaussianBlur(random.randint(0,2)))
            line = (np.asarray(line)-127.5)/127.5
            img = (np.asarray(img)-127.5)/127.5
            img_small = (np.asarray(img_small)-127.5)/127.5

            y = [1.0, 0.0]

            xy.put([line, img_small, y, img])

        threads = []
        for path in ytmp1:
            #本物
            threads.append( threading.Thread(target=func, args=(path,) ))

        for th in threads:
            th.start()
        for th in threads:
            th.join()

        #順序がバラバラにならないように
        x = []
        # selvecs = []
        s1gen = []
        y = []
        y_imgs = []
        while xy.qsize() > 0:
            a,e,c,d = xy.get()
            x.append(a)
            # selvecs.append(b)
            s1gen.append(e)
            y.append(c)
            y_imgs.append(d)

        return [np.asarray(x), np.asarray(s1gen)], [np.asarray(y), np.asarray(y_imgs)]

    def __len__(self):
        # 全データ数をバッチサイズで割って、何バッチになるか返すよー！

        return math.ceil(len(self.color_images_flat)/self.batch_size)
