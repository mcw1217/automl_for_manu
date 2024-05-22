import numpy as np
import shutil
import os
from PIL import Image

path = './dataset/MVTecAD'
move_path = './dataset/new_MVTecAD'
count = 0
c = 0
d = 0
f =0 
for file in os.listdir(path):
    count += 1
    c = 0
    d = 0
    f =0 
    print(f" {count} / {len(os.listdir(path))}")
    os.mkdir(move_path +"/" + file)
    os.mkdir(move_path +"/"+ file +"/0")
    os.mkdir(move_path +"/"+ file +"/1")
    tmp_path_0 = f"{move_path}/{file}/0"
    tmp_path_1 = f"{move_path}/{file}/1"
    
    new_path = os.path.join(path,file,"train","good")
    for file2 in os.listdir(new_path):
        c += 1
        img = Image.open(os.path.join(new_path, file2))
        img = img.resize((224,224))
        file2 = f"{file}_{c}.jpg"
        img.save(os.path.join(tmp_path_0,file2))
        
    new_path = os.path.join(path,file,"test")
    for file3 in os.listdir(new_path):
        if file3 != "good":
            
            d += 1
            last_path = os.path.join(new_path, file3)
            for file4 in os.listdir(last_path):
                f += 1
                img = Image.open(os.path.join(last_path, file4))
                img = img.resize((224,224))
                file4 = f"{file}_{d}_{f}.jpg"
                img.save(os.path.join(tmp_path_1,file4))
    
        
        
