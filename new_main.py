# 필요한 라이브러리를 가져옵니다.
import tensorflow as tf
# GPU 메모리를 한번에 할당하는 것을 방지하여, 메모리 초과를 방지함 ( 점진적으로 메모리 증가함 )
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    print(e)
    print("메모리 증가 설정 중 에러 발생")
    
    
import autokeras as ak
import os
from PIL import Image
import numpy as np
# 데이터셋 로드 및 전처리
# 여기서는 예제를 위해 TensorFlow에서 제공하는 데이터셋을 사용합니다.
# 실제 데이터셋으로 대체하여 사용하세요.


dataset_path = "./dataset/new_MVTecAD"

def load_data():
    train_path = os.path.join(dataset_path, "train")
    test_path = os.path.join(dataset_path, "test")
    train_img = []
    train_label = []
    for file in os.listdir(train_path):
        if file == "0":
            for img_name in os.listdir(os.path.join(train_path, "0")):
                img = Image.open(os.path.join(train_path, "0", img_name))
                img = img.resize((150,150))
                img = np.array(img)
                if len(img.shape) != 3:
                    img = np.stack((img,) *3, axis=-1)
                print(img.shape, img_name)
                img = img / 255.0
                img = np.transpose(img, (2,0,1))
                train_img.append(img)
                train_label.append(0)
        else:
            for img_name in os.listdir(os.path.join(train_path, "1")):
                img = Image.open(os.path.join(train_path, "1", img_name))
                img = img.resize((150,150))
                img = np.array(img)
                if len(img.shape) != 3:
                    img = np.stack((img,) *3, axis=-1)
                img = img / 255.0
                img = np.transpose(img, (2,0,1))
                train_img.append(img)
                train_label.append(1)
                
    test_img = []
    test_label = []
    for file in os.listdir(test_path):
        if file == "0":
            for img_name in os.listdir(os.path.join(test_path, "0")):
                img = Image.open(os.path.join(test_path, "0", img_name))
                img = img.resize((150,150))
                img = np.array(img)
                if len(img.shape) != 3:
                    img = np.stack((img,) *3, axis=-1)
                img = img / 255.0
                img = np.transpose(img, (2,0,1))
                test_img.append(img)
                test_label.append(0)
        else:
            for img_name in os.listdir(os.path.join(test_path, "1")):
                img = Image.open(os.path.join(test_path, "1", img_name))
                img = img.resize((150,150))
                img = np.array(img)
                if len(img.shape) != 3:
                    img = np.stack((img,) *3, axis=-1)
                img = img / 255.0
                img = np.transpose(img, (2,0,1))
                test_img.append(img)
                test_label.append(1)
    print(len(train_img), len(train_label), len(test_img), len(test_label))
    train_img, train_label, test_img, test_label = np.array(train_img),np.array(train_label),np.array(test_img),np.array(test_label),    
    return (train_img, train_label), (test_img, test_label)

# def load_data_as_dataset(batch_size=32):
#     (train_images, train_labels), (test_images, test_labels) = load_data()  # 위에서 정의한 load_data 함수 사용

#     # 훈련 데이터셋과 테스트 데이터셋 생성
#     train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
#     test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels))

#     # 데이터셋 배치 처리 및 섞기
#     train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)
#     test_dataset = test_dataset.batch(batch_size)

#     return train_dataset, test_dataset

# # 모델 생성 및 훈련 코드는 동일하게 유지
# # 데이터 로딩 부분을 수정
# batch_size = 8
# train_dataset, test_dataset = load_data_as_dataset(batch_size=batch_size)

(train_img, train_label), (test_img, test_label) = load_data()
                

# 모델 생성
# ResNet50 모델을 사용합니다. ImageClassifier를 사용하면 AutoKeras가 자동으로 최적의 모델을 탐색합니다.
# input_node에 이미지 형태를 명시하고, output_node에는 분류기를 설정합니다.
# 사전 훈련된 ResNet50을 사용하기 위해 preprocess_input를 설정합니다.
input_node = ak.ImageInput()
output_node = ak.Normalization()(input_node)
output_node = ak.ImageAugmentation()(output_node)
# 여기에서는 ResNet50을 사용하지만 AutoKeras에서는 pretrained=True 파라미터를 사용할 수 없으므로
# 추후에 ImageBlock 내부의 코드를 수정하거나 사전 훈련된 모델의 가중치를 직접 로드해야 합니다.
output_node = ak.ResNetBlock(version='v2')(output_node)
output_node = ak.ClassificationHead()(output_node)

# AutoModel을 구성합니다.
model = ak.AutoModel(
    inputs=input_node, outputs=output_node, overwrite=True, max_trials=10)

# 모델 훈련
print("훈련 시작")
model.fit(train_img, train_label, validation_split=0.3,batch_size=16, epochs=100)

# 모델 평가
print("모델 평가 시작!")
print(test_img, test_label)
print(model.evaluate(test_img,test_label))
