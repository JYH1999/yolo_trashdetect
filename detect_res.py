import tensorflow as tf
import re
import os
import json
import numpy as np
from tools.data_gen import preprocess_img
from models.resnet50 import ResNet50
from keras.layers import Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from keras.models import Model
from keras import regularizers
from PIL import Image, ImageTk
import random
from models.resnet50 import preprocess_input
import cv2
from keras.backend import set_session
# from client import receive_video


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.2
sess = tf.Session(config=config)
graph = tf.get_default_graph()
# 全局配置文件
tf.app.flags.DEFINE_integer('num_classes', 40, '垃圾分类数目')
tf.app.flags.DEFINE_integer('input_size', 224, '模型输入图片大小')
tf.app.flags.DEFINE_integer('batch_size', 16, '图片批处理大小')
FLAGS = tf.app.flags.FLAGS
h5_weights_path = './resmodel/best.h5'



def add_new_last_layer(base_model, num_classes):    # 增加最后输出层
    x = base_model.output
    x = GlobalAveragePooling2D(name='avg_pool')(x)
    x = Dropout(0.5, name='dropout1')(x)
    # x = Dense(1024, activation='relu', kernel_regularizer = regularizers.l2(0.0001), name='fc1')(x)
    # x = BatchNormalization(name='bn_fc_00')(x)
    x = Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.0001), name='fc2')(x)
    x = BatchNormalization(name='bn_fc_01')(x)
    x = Dropout(0.5, name='dropout2')(x)
    x = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=x)
    return model


def model_fn(FLAGS):    # 加载模型
    # K.set_learning_phase(0)
    # setup model
    global sess
    set_session(sess)
    base_model = ResNet50(weights="imagenet",
                          include_top=False,
                          pooling=None,
                          input_shape=(FLAGS.input_size, FLAGS.input_size, 3),
                          classes=FLAGS.num_classes)
    for layer in base_model.layers:
        layer.trainable = False
    model = add_new_last_layer(base_model, FLAGS.num_classes)
    model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy'])
    #model._make_predict_function()
    #model.graph=tf.get_default_graph()
    return model


def init_artificial_neural_network():    # 暴露模型初始化
    global sess,graph
    with graph.as_default():
        set_session(sess)
        model = model_fn(FLAGS)
        model.load_weights(h5_weights_path, by_name=True)
        #model._make_predict_function()
        #model.graph=tf.get_default_graph()
        return model

def img_process(img_in,img_size):
    try:
        image_temp = cv2.cvtColor(img_in, cv2.COLOR_BGR2RGB)
        img=Image.fromarray(image_temp)
        # if img.format:
        # resize_scale = img_size / max(img.size[:2])
        # img = img.resize((int(img.size[0] * resize_scale), int(img.size[1] * resize_scale)))
        img = img.resize((256, 256))
        img = img.convert('RGB')
        # img.show()
        img = np.array(img)
        imgs = []
        for _ in range(10):
            i = random.randint(0, 32)
            j = random.randint(0, 32)
            imgg = img[i:i + 224, j:j + 224]
            imgg = preprocess_input(imgg)
            imgs.append(imgg)
        return imgs
    except Exception as e:
        print('发生了异常data_process：', e)
        return 0

def prediction_result_from_img(model, image_in):    # 测试图片
    # 加载分类数据
    global sess,graph
    with graph.as_default():
        set_session(sess)
        with open("./res_classify_rule.json", 'r', encoding='utf-8') as load_f:
            load_dict = json.load(load_f)
            test_data = img_process(image_in, FLAGS.input_size)
        tta_num = 5
        predictions = [0 * tta_num]
        for i in range(tta_num):
            x_test = test_data[i]
            x_test = x_test[np.newaxis, :, :, :]
            prediction = model.predict(x_test)[0]
            predictions += prediction
        pred_label = np.argmax(predictions, axis=0)
        print_str=str( load_dict[str(pred_label)])+':1.000'
        return_data=[]
        return_data.append(print_str)
        #print(print_str)
        #print('')
        #return pred_label, load_dict[str(pred_label)]
        return return_data,image_in


if __name__ == "__main__":
    global flag
    global pre_label
    global picture
    global name
    flag = 0
    picture = 'test.jpg'
    name = 0
    pre_label = 0
    model_load=init_artificial_neural_network()
    print(prediction_result_from_img(model_load, picture))
