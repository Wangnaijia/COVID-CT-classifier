

import os
import numpy as np

def read_txt(txt_path):
    with open(txt_path) as f:
        lines = f.readlines()
    txt_data = [line.strip() for line in lines]
    return txt_data

def move(path):
    '该文件夹下所有的文件（包括文件夹）'
    source_path = img_path + '/'+read_txt
    shutil.copy(source_path,target_path)
    FileList = os.listdir(path)
    '遍历所有文件'
    for files in FileList:
        '原来的文件路径'
        oldDirPath = os.path.join(path, files)
        '如果是文件夹则递归调用'
        if os.path.isdir(oldDirPath):
            rename(oldDirPath)
        '文件名'
        fileName = os.path.splitext(files)[0]
        '文件扩展名'
        fileType = os.path.splitext(files)[1]
        '新的文件路径'
        newDirPath = os.path.join(path, "sun"+fileName+ fileType)
        '重命名'
        os.rename(oldDirPath, newDirPath)
def create_txt(name, path, file_image):
    txt_path = path + name + '.txt'
    txt = open(txt_path, 'w')
    for i in file_image:
        image_dir = str(i)
        txt.write(image_dir)
        txt.write('\n')

def read_file(path1):
    filelist1 = os.listdir(path1)
    file_image = np.array([file for file in filelist1], dtype=object)
    # filelist2 = os.listdir(path2)
    # file_label = np.array([file for file in filelist2 if file.endswith('.png')], dtype=object)
    return file_image


path1 = './test_folder/'

# path2 = './dataset/test_label/'

file_pos = read_file(path1+'positive')
file_neg = read_file(path1+'negative')
create_txt('test_final_covid', './Data-split/', file_pos)
create_txt('test_final_noncovid', './Data-split/', file_neg)