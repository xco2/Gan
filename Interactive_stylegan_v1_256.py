from keras.applications.inception_v3 import InceptionV3
from keras import Model
from keras.layers import Input
import numpy as np
import os, scipy, time
from stylegan_ada_v1_256 import *


def is_int(n):
    try:
        int(n)
        return True
    except:
        return False


class model(StyleGan):
    def __init__(self):
        self.cha = 32

        self.input_shape = (256, 256, 3)  # 原图输入
        self.input_shape_decoder = (512,)  # decoder输入

        self.k = 71  # 预计算的中间值
        # 初始Bnoise
        self.inp_noises = []
        for i in range(7):
            shape = (self.k + 1, 4 * 2 ** i, 4 * 2 ** i, 1)
            self.inp_noises.append(self.create_noise(0.0, 1.0, shape, threshold=1))
            self.inp_noises.append(self.create_noise(0.0, 1.0, shape, threshold=1))
        self.inp_noises[0] = np.tile(self.create_noise(0.0, 1.0, (4, 4, 1), threshold=1), (self.k + 1, 1, 1, 1))

        # 选择的风格
        self.choiced_w = None
        self.choiced_noiseB = None

        self.G_mapping = self.build_G_mapping(self.input_shape_decoder)
        self.G_syn = self.build_G_syn(self.input_shape_decoder)
        self.load_model(None)

    def load_model(self, V):
        if V:
            save_dir = "./StyleGan_MODEL/v" + str(V) + "/"
            print("加载 v" + str(V))
        else:
            save_dir = "./StyleGan_MODEL/FID 54(G后4层相加32开始/FID 34/"
        v_epoch = 1000
        self.G_mapping.load_weights(save_dir + 'G_mapping_{0}.h5'.format(v_epoch))
        self.G_syn.load_weights(save_dir + 'G_syn_{0}.h5'.format(v_epoch), by_name=True)

        print("加载完成")

    def change_last_noiseB(self, n):
        self.inp_noises[0] = np.concatenate(
            [np.expand_dims(self.inp_noises[0][0], 0), self.create_noise(0.0, 1.0, (n - 1, 4, 4, 1), threshold=1)])

    def change_noiseB(self, n, last):
        self.inp_noises = []
        for i in range(7):
            shape = (n, 4 * 2 ** i, 4 * 2 ** i, 1)
            self.inp_noises.append(self.create_noise(0.0, 1.0, shape, threshold=1))
            self.inp_noises.append(self.create_noise(0.0, 1.0, shape, threshold=1))
        if last is not None:
            self.inp_noises[0] = np.tile(last, (n, 1, 1, 1))
        else:
            self.inp_noises[0] = np.tile(self.create_noise(0.0, 1.0, (4, 4, 1), threshold=1), (self.k + 1, 1, 1, 1))

    def first_choic(self):
        w = np.load(r'./StyleGan_MODEL/FID 54(G后4层相加32开始/FID 34/w_mid{}.npy'.format(self.k))

        w_avg = self.get_w_avg()
        w = w_avg + 0.7 * (w - w_avg)
        w = np.concatenate([np.array([w_avg]), w], axis=0)

        while True:
            img = self.G_syn.predict([w] + self.inp_noises)
            img = img * 0.5 + 0.5
            img = self.create_image(img, 9, 8, self.input_shape[0], 3)
            img.show()
            index = input("序号:")
            if index != 'r' and is_int(index):
                self.choiced_w = w[int(index) - 1]
                break
            elif index == 'n':
                noise = self.create_noise(0.0, 1.0, (self.k + 1, self.input_shape_decoder[0]))
                w = self.G_mapping.predict(noise)
                w = w_avg + 0.7 * (w - w_avg)
            else:
                self.inp_noises[0] = np.tile(self.create_noise(0.0, 1.0, (4, 4, 1), threshold=1), (self.k + 1, 1, 1, 1))

    def choic_waifu(self, num):
        self.first_choic()  # 第一次选择
        # 选择风格
        while True:
            det_w = self.create_noise(0.0, 1.0, (15, self.input_shape_decoder[0]))
            det_w = self.G_mapping.predict(det_w)
            w = self.choiced_w + 0.3 * (det_w - self.choiced_w)
            w = np.concatenate([np.expand_dims(self.choiced_w, 0), w], axis=0)

            img = self.G_syn.predict([w] + self.inp_noises)
            img = img * 0.5 + 0.5
            img = self.create_image(img, 4, 4, self.input_shape[0], 3)
            img.show()
            index = input("序号:")
            if index != 'r' and is_int(index):
                self.choiced_w = w[int(index) - 1]
                break

        # 选择构图
        w = np.tile(self.choiced_w, (16, 1))
        while True:
            self.change_last_noiseB(16)
            img = self.G_syn.predict([w] + self.inp_noises)
            img = img * 0.5 + 0.5
            img = self.create_image(img, 4, 4, self.input_shape[0], 3)
            img.show()
            index = input("序号:")
            if index != 'r' and is_int(index):
                self.choiced_noiseB = self.inp_noises[0][int(index) - 1]
                break

        # 选择细节
        while True:
            self.change_noiseB(16, self.choiced_noiseB)
            img = self.G_syn.predict([w] + self.inp_noises)
            img = img * 0.5 + 0.5
            img = self.create_image(img, 4, 4, self.input_shape[0], 3)
            img.show()
            index = input("序号:")
            if index != 'r' and is_int(index):
                self.choiced_noiseB = []
                for i in range(14):
                    self.choiced_noiseB.append(np.expand_dims(self.inp_noises[i][int(index) - 1], 0))
                break

        w = np.expand_dims(self.choiced_w, 0)
        img = self.G_syn.predict([w] + self.choiced_noiseB)
        img = (img[0] * 0.5 + 0.5) * 255
        img = Image.fromarray(np.uint8(img))
        img.save("./gan_images/头像/choic{}.png".format(num))
        self.change_noiseB(self.k + 1, None)
        img.show()


if __name__ == "__main__":
    m = model()
    i = 0
    while os.path.exists("./gan_images/头像/choic{}.png".format(i)):
        i += 1
    while True:
        print(i)
        m.choic_waifu(i)
        i += 1
