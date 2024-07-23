import tensorflow as tf
import gc
import os
from PIL import Image
import numpy as np
import autokeras as ak

# GPU 메모리 설정
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
        print("메모리 증가 설정 중 에러 발생")

dataset_path = "./new_dataset/new_MVTecAD"

def load_data_all(class_name, size):
    train_img = []
    train_label = []      
    test_img = []
    test_label = []

    for file in os.listdir(dataset_path):
        if file == class_name:
            tmp_path = f"{dataset_path}/{file}"
            for class_folder in os.listdir(tmp_path):
                img_list = os.listdir(os.path.join(tmp_path, class_folder))
                np.random.shuffle(img_list)
                if class_folder == "0":
                    for i, img_name in enumerate(img_list):
                        img = Image.open(os.path.join(tmp_path, class_folder, img_name))
                        img = img.resize((size, size))
                        img = np.array(img)
                        if len(img.shape) != 3:
                            img = np.stack((img,) * 3, axis=-1)
                        img = img / 255.0
                        img = np.transpose(img, (2, 0, 1))
                        if i < 15:
                            train_img.append(img)
                            train_label.append(0)
                        else:
                            test_img.append(img)
                            test_label.append(0)
                else:
                    for i, img_name in enumerate(img_list):
                        img = Image.open(os.path.join(tmp_path, class_folder, img_name))
                        img = img.resize((size, size))
                        img = np.array(img)
                        if len(img.shape) != 3:
                            img = np.stack((img,) * 3, axis=-1)
                        img = img / 255.0
                        img = np.transpose(img, (2, 0, 1))
                        if i < 15:
                            train_img.append(img)
                            train_label.append(1)
                        else:
                            test_img.append(img)
                            test_label.append(1)

    train_img, train_label, test_img, test_label = np.array(train_img), np.array(train_label), np.array(test_img), np.array(test_label)
    return (train_img, train_label), (test_img, test_label)

path = './new_dataset/new_MVTecAD'
size = 150
except_list = ['bottle','leather','carpet','grid', 'pill','metal_nut','tile','cable','transistor','capsule', 'toothbrush','wood','zipper','hazelnut','screw']

count_time = 0
for class_name in os.listdir(path):
    count_time += 1
    print(f'[System] 데이터셋을 로드합니다! Test Class: {class_name}')
    if class_name in except_list:
        print(f"[System] 제외 목록의 항목 발견: {class_name}")
        continue

    (train_img, train_label), (test_img, test_label) = load_data_all(class_name, size)

    print("모델 생성")
    model = ak.ImageClassifier(max_trials=10, overwrite=True)

    print("훈련 시작")
    model.fit(train_img, train_label, validation_split=0.3, batch_size=5, epochs=50)

    print("모델 평가 시작!")
    eva_result = model.evaluate(test_img, test_label)
    print(eva_result)

    if count_time == 1:
        with open('./result/result_report.txt', "w", encoding='UTF-8') as log:
            log.write(f"[{count_time}] Test Class: {class_name} Loss: {eva_result[0]} Accuracy: {eva_result[1]}\n")
    else:
        with open('./result/result_report.txt', "a", encoding='UTF-8') as log:
            log.write(f"[{count_time}] Test Class: {class_name} Loss: {eva_result[0]} Accuracy: {eva_result[1]}\n")

    model = None
    train_img = None
    train_label = None
    test_img = None
    test_label = None
    gc.collect()
    tf.keras.backend.clear_session()
    del model
