import numpy as np
import pandas as pd
import glob
from PIL import Image
from model import deep_auto_encoder_model

def read_imgs_list(path_list):
    """ 画像読み込みと前処理 """
    input_size = (128, 128)
    imgs = list()
    for img_path in path_list:
        img = Image.open(img_path)
        img = img.convert('RGB')
        img = img.resize(input_size)
        img = np.asarray(img).astype(np.float32) / 255.0
        imgs.append(img)
    return np.asarray(imgs)


def load_test_images(folder):
    """ test画像読み込み """
    ok_test_files = list()
    ok_test_files = glob.glob(folder + '/*.png')
    ok_test_files = ok_test_files + glob.glob(folder + '/*.jpg')
    imgs_ok_test = read_imgs_list(ok_test_files)
    return imgs_ok_test, ok_test_files


def load_judge_thresh(thresh_csv_path):       #get the threshold values for each piece
    """ threshold読み込み """
    df = pd.read_csv(thresh_csv_path, encoding='utf-8')
    pieces = np.asarray(df.loc[:, ['piece']]).astype('unicode')
    pieces = np.unique(pieces)
    thresh_dic = dict()
    for _, piece in enumerate(pieces):
        threshold = df[df['piece'].isin([piece])]
        threshold = threshold.loc[:, 'threshold']
        threshold = np.asarray(threshold).astype(np.float32)
        thresh_dic[piece] = threshold.astype(float)
    return thresh_dic

def test(spec, part_names):
    """ テスト実施 """

    input_shape = (128, 128, 3)
    model = deep_auto_encoder_model(input_shape)
        
    thresh_file_path = './image/' + spec + '/base/threshold.csv'
    thresh_dic = load_judge_thresh(thresh_file_path)

    listResult = []
    
    for part_name in part_names:
        
        judge_thresh = thresh_dic[part_name]
 
        test_folder      = './image/' + spec + '/' + part_name + '/stabcut'
        weight_file_path = './image/' + spec + '/base/' + part_name + '.hdf5'
        model.load_weights(weight_file_path)
        
        img_test,_ = load_test_images(test_folder)

        img_test_pred = model.predict(img_test, batch_size=32)        
        index_test    = np.mean(np.abs(img_test - img_test_pred), axis=(1, 2, 3))

        print(f"comparing threshold for {part_name}")
        
        if index_test <= judge_thresh:
            print('Part '+part_name+' result = ok')
            print(f'Difference score: {index_test[0]:.4f}')
            print(f'Threshold : {judge_thresh}')
            listResult.append('ok')

        elif index_test > judge_thresh:
            print('Part '+part_name+' result = ng')
            print(f'Difference score: {index_test[0]:.4f}')
            print(f'Threshold : {judge_thresh} \n\n')
            listResult.append('ng')

    return listResult 
