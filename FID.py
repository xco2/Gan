from keras.applications.inception_v3 import InceptionV3
from keras import Model
from keras.layers import Input
import numpy as np
import os, scipy, time
from stylegan_ada_v1_256 import *


# if not os.path.exists("./use"):
#     os.environ['CUDA_VISIBLE_DEVICES'] = " "
#     print("已禁用GPU" + "=" * 20)

class model(StyleGan):
    def __init__(self, data_path):
        # 图片数据文件夹
        self.data_path = data_path
        self.cha = 32

        self.input_shape = (256, 256, 3)  # 原图输入
        self.input_shape_decoder = (512,)  # decoder输入

        self.G_mapping = self.build_G_mapping(self.input_shape_decoder)
        self.G_syn = self.build_G_syn(self.input_shape_decoder)

    def load_model(self, V):
        if V:
            save_dir = "./StyleGan_MODEL/v" + str(V) + "/"
            print("加载 v" + str(V))
        else:
            save_dir = "./StyleGan_MODEL/"
        v_epoch = 1000
        self.G_mapping.load_weights(save_dir + 'G_mapping_{0}.h5'.format(v_epoch))
        self.G_syn.load_weights(save_dir + 'G_syn_{0}.h5'.format(v_epoch), by_name=True)

        print("加载完成")


s_t = time.time()

data_path = r"E:\p站图片\database\face2head"
s = model(data_path)
s.load_model(None)

num = 2500

print("产生图片..")
gan_images = []
for i in range(3):
    gan_images.append(s.predict_Gan(num))
    print(i, end="")
gan_images = np.concatenate(gan_images)
print("完成")
del (s)

model = InceptionV3(weights='./VGG/inception_v3_weights_tf_dim_ordering_tf_kernels.h5')
m_inp = Input(shape=(256, 256, 3))
model = Model(model.input, model.get_layer('avg_pool').output)
out = model(m_inp)
model = Model(m_inp, out)
# model.summary()

print("计算假图片")
gan_vector = model.predict(gan_images)
mu_fake = np.mean(gan_vector, axis=0)
sigma_fake = np.cov(gan_vector, rowvar=False)
del (gan_images)

print("计算真图片")
data = dataGenerator(data_path.split("\\")[-1], 256, data_path, flip=True)
real_img = data.get_batch(5000)
del (data)
real_vector = model.predict(real_img)
mu_real = np.mean(real_vector, axis=0)
sigma_real = np.cov(real_vector, rowvar=False)

print("FID")
m = np.square(mu_fake - mu_real).sum()
s, _ = scipy.linalg.sqrtm(np.dot(sigma_fake, sigma_real), disp=False)  # pylint: disable=no-member
dist = m + np.trace(sigma_fake + sigma_real - 2 * s)

print(dist)
dist = np.real(dist)  # 48
print(dist)

e_t = time.time()
print(e_t - s_t, "s")
