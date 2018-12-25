# coding: utf-8
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import backend

import os,glob,sys,json,random
from time import sleep
import numpy as np
import math
from PIL import Image, ImageOps, ImageFile, ImageChops, ImageFilter
import argparse

from kiri_datagenerator import Colors, STANDARD_SIZE_S1, STANDARD_SIZE_S2, Colors_rev


ImageFile.LOAD_TRUNCATED_IMAGES = True

total_epochs = 999999

test_dir = './test/'
test_colored_dir = './test_colored/'
test_short_dir = './test_short/'
test_short_colored_dir = './test_short_colored/'
g_model_s1_path = 'g_model_s1.h5'

def gan_s1(GPUs, start_idx, batch_size):
    def g_on_epoch_end(epoch):
        g_model_s1.save(g_model_s1_path)
        # g_model_s1.save_weights(g_model_s1_path)
        generator_test(g_model_s1, test_dir, epoch, result_path)

    def g_on_epoch_end_sub(epoch):
        generator_test(g_model_s1, test_short_dir, epoch, None, True)

    def d_on_epoch_end(epoch):
        d_model_s1.save(d_model_s1_path)
        # d_model_s1.save_weights(d_model_s1_path)
        discrimin_test(d_model_s1, epoch, result_path, q_valid_d)

    #######################################################
    # STAGE-1
    if os.path.exists(g_model_s1_path):
        # g_model_s1.load_weights(g_model_s1_path, by_name=False)
        g_model_s1 = load_model(g_model_s1_path)
    else:
        g_model_s1 = build_generator()

    if GPUs > 1:
        g_model_s1_tr = multi_gpu_model(g_model_s1, gpus=GPUs)
    else:
        g_model_s1_tr = g_model_s1

    def summary_write(line):
        f.write(line+"\n")
    with open('g_model_s1.txt', 'w') as f:
        g_model_s1.summary(print_fn=summary_write)
    plot_model(g_model_s1, to_file='g_model.png')

    if os.path.exists(d_model_s1_path):
        # d_model_s1.load_weights(d_model_s1_path, by_name=False)
        d_model_s1 = load_model(d_model_s1_path)
    else:
        d_model_s1 = build_discriminator()

    if GPUs > 1:
        d_model_s1_tr = multi_gpu_model(d_model_s1, gpus=GPUs)
    else:
        d_model_s1_tr = d_model_s1
    d_model_s1_tr.compile(loss='categorical_crossentropy',   #binary_crossentropy
                    # optimizer=Adam(lr=2e-5) #, beta_1=0.5) #, decay=1e-5)
                    # optimizer=Adam(lr=1e-4), #1e-4でよさそう Nadam
                    # optimizer=Adadelta(),
                    optimizer=Nadam(),
                    )
    with open('d_model_s1.txt', 'w') as f:
        d_model_s1.summary(print_fn=summary_write)

    frozen_d = build_frozen_discriminator(d_model_s1)

    _tmpmdl = build_combined(g_model_s1, frozen_d)
    if GPUs > 1:
        combined = multi_gpu_model(_tmpmdl, gpus=GPUs)
    else:
        combined = _tmpmdl

    combined.compile(loss=['categorical_crossentropy', 'mean_squared_error'],  #, 'categorical_crossentropy'], #, 'mean_squared_error'],
                    loss_weights=[1.0, 0.05],  # 重みにn倍差をつけてみる 20倍でよさそう。40倍だと暴れるかも。
                    # optimizer=Adam(lr=4e-4, beta_1=0.5),
                    # optimizer=Adam(lr=1e-4) #, beta_1=0.5) #, decay=1e-5),
                    # optimizer=Adam(lr=1e-4),  #1e-4でよさそう
                    # optimizer=Adadelta(),
                    optimizer=Nadam(),
                    )

    #
    g_model_s1_tr._make_predict_function()
    d_model_s1_tr._make_predict_function()
    d_model_s1_tr._make_train_function()
    combined._make_predict_function()
    combined._make_train_function()


    # discriminator用のデータジェネレータ、データを格納するキュー
    ddgens = []
    q1s = []
    for i in range(3):
        ddgen = D_Datagenerator(color_path=img_dir, batch_size=batch_size, g_model=g_model_s1_tr ,val=i)
        q1 = dataQ(ddgen, args.queue_size)
        ddgens.append(ddgen)
        q1s.append(q1)
        sleep(args.queue_size/4)

    ddgen_valid = D_Datagenerator(color_path=img_dir, batch_size=batch_size, g_model=g_model_s1_tr, val=999)
    q_valid_d = dataQ(ddgen_valid, args.queue_size)
    sleep(args.queue_size/4)

    # 結合モデル（combined）用のデータジェネレータ、データを格納するキュー
    cdgen = Comb_Datagenerator(color_path=img_dir, batch_size=batch_size)
    q2 = dataQ(cdgen, args.queue_size, MP=True)
    sleep(args.queue_size/4)

    # cdgens = []
    # q2s = []
    # for i in range(len(Colors)):
    #     cdgen = Comb_Datagenerator(color_path=img_dir, batch_size=batch_size, val=i)
    #     q2 = dataQ(cdgen, args.queue_size, MP=True)
    #     cdgens.append(cdgen)
    #     q2s.append(q2)
    #     sleep(args.queue_size/4)

    cdgen_valid = Comb_Datagenerator(color_path=img_dir, batch_size=batch_size, val=999)
    q_valid_g = dataQ(cdgen_valid, args.queue_size, MP=True)
    sleep(args.queue_size/4)

    pre_d_loss = 1000.0
    pre_g_loss = 1000.0
    for epoch in range(start_idx+1, total_epochs+1):
        print(f'\repochs={epoch:6d}/{total_epochs:6d}:', end='')
        if pre_d_loss * 1.5 >= pre_g_loss or epoch < 10 or epoch%10 == 0:
            d_losses = []
            for i in range(3):
                x,y = q1s[i].get()
                d_loss = d_model_s1_tr.train_on_batch(x, y)
                d_losses.append(d_loss)

            pre_d_loss = sum(d_losses)/3
        print(f'D_loss={pre_d_loss:.3f}  ',end='')

        # if pre_d_loss <= pre_g_loss or epoch < 10:
        # 色別にバッチを分けて実施
        x,y = q2.get()
        g_loss = combined.train_on_batch(x, y)
        pre_g_loss = g_loss[1]
        print(f'G_loss={g_loss[0]:.3f}({g_loss[1]:.3f},{g_loss[2]:.3f})',end='')

        # 100epoch毎にテスト画像生成、validuetion
        if epoch%100 == 0:
            g_on_epoch_end_sub(epoch)
            # validuate
            print(f'\tValidation :', end='')
            x,y = q_valid_d.get()
            ret = d_model_s1_tr.test_on_batch(x, y)
            print(f'D_loss={ret:.3f},',end='')

            x,y = q_valid_g.get()
            ret = combined.test_on_batch(x, y)
            print(f'G_loss={ret[0]:.3f}({ret[1]:.3f},{ret[2]:.3f})')

        # 1000epoch毎にモデルの保存、テスト実施
        if epoch%1000 == 0:
            result_path = 'results/'
            if not os.path.exists(result_path):
                os.mkdir(result_path)
            result_path += f'{epoch:05d}/'
            if not os.path.exists(result_path):
                os.mkdir(result_path)

            g_on_epoch_end(epoch)
            d_on_epoch_end(epoch)


def gan_s2(GPUs, start_idx, batch_size):
    def g_on_epoch_end(epoch):
        g_model_s2.save(g_model_s2_path)
        generator_testS2(g_model_s2_tr, test_dir, epoch, result_path)

    def g_on_epoch_end_sub(epoch):
        generator_testS2(g_model_s2_tr, test_short_dir, epoch, None, True)

    def d_on_epoch_end(epoch):
        d_model_s2.save(d_model_s2_path)
        discrimin_test(d_model_s2_tr, epoch, result_path, q_valid_d)

    #######################################################
    # STAGE-2
    # if os.path.exists(g_model_s1_path):
    #     g_model_s1 = load_model(g_model_s1_path)
    # else:
    #     g_model_s1 = build_generator()
    # if GPUs > 1:
    #     g_model_s1_tr = multi_gpu_model(g_model_s1, gpus=GPUs)
    # else:
    #     g_model_s1_tr = g_model_s1

    if os.path.exists(g_model_s2_path):
        g_model_s2 = load_model(g_model_s2_path)
    else:
        g_model_s2 = build_generatorS2()
    if GPUs > 1:
        g_model_s2_tr = multi_gpu_model(g_model_s2, gpus=GPUs)
    else:
        g_model_s2_tr = g_model_s2

    def summary_write(line):
        f.write(line+"\n")
    with open('g_model_s2.txt', 'w') as f:
        g_model_s2.summary(print_fn=summary_write)
    plot_model(g_model_s2, to_file='g_model.png')

    if os.path.exists(d_model_s2_path):
        d_model_s2 = load_model(d_model_s2_path)
    else:
        d_model_s2 = build_discriminatorS2()

    if GPUs > 1:
        d_model_s2_tr = multi_gpu_model(d_model_s2, gpus=GPUs)
    else:
        d_model_s2_tr = d_model_s2
    d_model_s2_tr.compile(loss='categorical_crossentropy',
                    optimizer=Nadam(),
                    )
    with open('d_model_s2.txt', 'w') as f:
        d_model_s2.summary(print_fn=summary_write)

    frozen_d = build_frozen_discriminator(d_model_s2)

    _tmpmdl = build_combinedS2(g_model_s2, frozen_d)
    if GPUs > 1:
        combined = multi_gpu_model(_tmpmdl, gpus=GPUs)
    else:
        combined = _tmpmdl

    combined.compile(loss=['categorical_crossentropy', 'mean_squared_error'], 
                    loss_weights=[1.0, 0.05],  # 重みにn倍差をつけてみる 20倍でよさそう。40倍だと暴れるかも。
                    optimizer=Nadam(),
                    )

    #
    g_model_s1_tr._make_predict_function()
    g_model_s2_tr._make_predict_function()
    d_model_s2_tr._make_predict_function()
    d_model_s2_tr._make_train_function()
    combined._make_predict_function()
    combined._make_train_function()

    # discriminator用のデータジェネレータ、データを格納するキュー
    ddgens = []
    q1s = []
    for i in range(3):
        ddgen = D_DatageneratorS2(color_path=img_dir, batch_size=batch_size, g_model=g_model_s2_tr ,val=i)
        q1 = dataQ(ddgen, args.queue_size)
        ddgens.append(ddgen)
        q1s.append(q1)
        sleep(args.queue_size/4)

    ddgen_valid = D_DatageneratorS2(color_path=img_dir, batch_size=batch_size, g_model=g_model_s2_tr, val=999)
    q_valid_d = dataQ(ddgen_valid, args.queue_size)
    sleep(args.queue_size/4)

    # 結合モデル（combined）用のデータジェネレータ、データを格納するキュー
    cdgen = Comb_DatageneratorS2(color_path=img_dir, batch_size=batch_size)
    q2 = dataQ(cdgen, args.queue_size, MP=True)
    sleep(args.queue_size/4)

    cdgen_valid = Comb_DatageneratorS2(color_path=img_dir, batch_size=batch_size, val=999)
    q_valid_g = dataQ(cdgen_valid, args.queue_size, MP=True)
    sleep(args.queue_size/4)

    pre_d_loss = 1000.0
    pre_g_loss = 1000.0
    for epoch in range(start_idx+1, total_epochs+1):
        print(f'\repochs={epoch:6d}/{total_epochs:6d}:', end='')
        if pre_d_loss * 1.5 >= pre_g_loss or epoch < 10 or epoch%10 == 0:
            d_losses = []
            for i in range(3):
                x,y = q1s[i].get()
                d_loss = d_model_s2_tr.train_on_batch(x, y)
                d_losses.append(d_loss)

            pre_d_loss = sum(d_losses)/3
        print(f'D_loss={pre_d_loss:.3f}  ',end='')

        # if pre_d_loss <= pre_g_loss or epoch < 10:
        # 色別にバッチを分けて実施
        x,y = q2.get()
        g_loss = combined.train_on_batch(x, y)
        pre_g_loss = g_loss[1]
        print(f'G_loss={g_loss[0]:.3f}({g_loss[1]:.3f},{g_loss[2]:.3f})',end='')

        # 100epoch毎にテスト画像生成、validuetion
        if epoch%100 == 0:
            g_on_epoch_end_sub(epoch)
            # validuate
            print(f'\tValidation :', end='')
            x,y = q_valid_d.get()
            ret = d_model_s2_tr.test_on_batch(x, y)
            print(f'D_loss={ret:.3f},',end='')

            x,y = q_valid_g.get()
            ret = combined.test_on_batch(x, y)
            print(f'G_loss={ret[0]:.3f}({ret[1]:.3f},{ret[2]:.3f})')

        # 1000epoch毎にモデルの保存、テスト実施
        if epoch%1000 == 0:
            result_path = 'results/'
            if not os.path.exists(result_path):
                os.mkdir(result_path)
            result_path += f'{epoch:05d}/'
            if not os.path.exists(result_path):
                os.mkdir(result_path)

            g_on_epoch_end(epoch)
            d_on_epoch_end(epoch)


def generator_test(g_model, test_dir, epoch, result_path, short=False):
    i_dirs = []
    for f in os.listdir(test_dir):
            i_dirs.append(test_dir + f)
    # for image_path in random.sample(i_dirs,min([len(i_dirs),10])):
    for image_path in random.sample(i_dirs,min([len(i_dirs),3])):
        img = Image.open(image_path)
        img = img.convert('RGB')
        img = img.resize(STANDARD_SIZE_S1,Image.BICUBIC)
        img = (np.asarray(img)-127.5)/127.5
        # selcol = random.choice(list(Colors.keys()))
        # colorvec = Colors[selcol]
        for selcol, colorvec in Colors.items():
            ret = g_model.predict_on_batch([np.array([img]), np.array([colorvec])])

            if short:
                exet = image_path.rsplit('.',1)[-1]
                tmp_name = image_path.rsplit('.',1)[0].rsplit('/',1)[-1]
                save_path = f'temp2/{epoch:06d}/'
                if not os.path.exists(save_path):
                    os.mkdir(save_path)

                filename = f"{save_path}testgen1_{epoch:06}_{tmp_name}_{selcol}.{exet}"
                tmp = (ret[0]*127.5+127.5).clip(0, 255).astype(np.uint8)
                Image.fromarray(tmp).save(filename, optimize=True)
            else:
                save_path = result_path + f'g1_{epoch:06}/'
                if not os.path.exists(save_path):
                    os.mkdir(save_path)

                filename = image_path.rsplit('/',1)[-1]
                filename = selcol + '_' + filename
                tmp = (ret[0]*127.5+127.5).clip(0, 255).astype(np.uint8)
                Image.fromarray(tmp).save(save_path + filename, optimize=True)


def generator_testS2(g_model_s2, test_dir, epoch, result_path, short=False):
    i_dirs = []
    for f in os.listdir(test_dir):
            i_dirs.append(test_dir + f)
    # for image_path in random.sample(i_dirs,min([len(i_dirs),10])):
    for image_path in random.sample(i_dirs,min([len(i_dirs),3])):
        img = Image.open(image_path)
        img = img.convert('RGB')
        line_s1 = img.resize(STANDARD_SIZE_S1,Image.BICUBIC)
        line_s1 = (np.asarray(line_s1)-127.5)/127.5
        line_s2 = img.resize(STANDARD_SIZE_S2,Image.BICUBIC)
        line_s2 = (np.asarray(line_s2)-127.5)/127.5
        for selcol, colorvec in Colors.items():
            gens1 = g_model_s1.predict_on_batch([np.array([line_s1]), np.array([colorvec])])[0]
            ret = g_model_s2.predict_on_batch([np.array([line_s2]), np.array([colorvec]), np.array([gens1]) ])

            if short:
                exet = image_path.rsplit('.',1)[-1]
                tmp_name = image_path.rsplit('.',1)[0].rsplit('/',1)[-1]
                save_path = f'temp2/{epoch:06d}/'
                if not os.path.exists(save_path):
                    os.mkdir(save_path)

                filename = f"{save_path}testgen2_{epoch:06}_{tmp_name}_{selcol}.{exet}"
                tmp = (ret[0]*127.5+127.5).clip(0, 255).astype(np.uint8)
                Image.fromarray(tmp).save(filename, optimize=True)
            else:
                save_path = result_path + f'g2_{epoch:06}/'
                if not os.path.exists(save_path):
                    os.mkdir(save_path)

                filename = image_path.rsplit('/',1)[-1]
                filename = selcol + '_' + filename
                tmp = (ret[0]*127.5+127.5).clip(0, 255).astype(np.uint8)
                Image.fromarray(tmp).save(save_path + filename, optimize=True)


def discrimin_test(d_model, epoch, result_path, queue, stage=1):
    #判定
    (imgs, colornums),_ = queue.get()
    results = d_model.predict_on_batch([imgs, colornums])
    #確認用保存
    save_path = result_path + f'd{stage}_{epoch:06}/'

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    for i in range(results.shape[0]):
    # for i,(img,num,result) in enumerate(zip(imgs,colvecs,results)):
        cl = Colors_rev[colornums[i]]        
        filename = f'd{i:02}[{cl}][{results[i,0]:1.2f}][{results[i,1]:1.2f}].png'
        tmp = (imgs[i]*127.5+127.5).clip(0, 255).astype(np.uint8)
        Image.fromarray(tmp).save(save_path + filename, 'png', optimize=True)


if __name__ == "__main__":
    #GPU設定
    config = tf.ConfigProto(
                gpu_options=tf.GPUOptions(allow_growth=False, visible_device_list="2"),
                allow_soft_placement=True, 
                log_device_placement=False
                )
    session = tf.Session(config=config)
    backend.set_session(session)

    gens1_model = load_model(g_model_s1_path)

    for dir,outdir in zip([test_dir,test_short_dir],[test_colored_dir,test_short_colored_dir]):
        if not os.path.exists(outdir):
            os.mkdir(outdir)
        for f in os.listdir(dir):
            img = Image.open(dir+f).convert('RGB')
            line_s1 = img.resize(STANDARD_SIZE_S1,Image.BICUBIC)
            line_s1 = (np.asarray(line_s1)-127.5)/127.5

            for color,val in Colors.items():

                save_path = outdir + color + '/'
                if not os.path.exists(save_path):
                    os.mkdir(save_path)

                ret = gens1_model.predict_on_batch([np.asarray([line_s1]), np.asarray([val])])

                filename = save_path + f
                tmp = (ret[0]*127.5+127.5).clip(0, 255).astype(np.uint8)
                Image.fromarray(tmp).save(filename, optimize=True)








