import PySimpleGUI as sg
import tensorflow as tf
import pyautogui
import numpy as np
import glob
import datetime
import os
import shutil
import time
import cv2
import warnings
import ctypes 
import utils.tisgrabber as tis
from testing import test
from camera_config import setup_camera, trigger_camera, stop_camera
from segmentation import loadPolyxy,shrink_poly,get_crop_area,make_mask,part_crop,img_phase_correlate,\
    stabilizer_method,fillout_color,bounding_crop


# GPUメモリ制限     allow Memory growth if GPU out overflows
tf.config.set_soft_device_placement(True)
devices = tf.config.experimental.list_physical_devices('GPU')
for device in devices:
    tf.config.experimental.set_memory_growth(device, True)


# 警告表示抑制
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=Warning)
tf.get_logger().setLevel('INFO')
tf.autograph.set_verbosity(1)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
widthScreen, heightScreen = pyautogui.size()


#-----------------------------------------------------------------------------------------------


def imgToBytes(img):     #convert img to png and then to bytes
    _, imgbytes = cv2.imencode('.png', img)
    imgbytes = imgbytes.tobytes()
    return imgbytes


def imgToPred(img, inputShape): 
    """resize the image according to inputshape, normalize (0 to 1) and 
    #expand dimesnion for batch processing"""
    imgPred = cv2.resize(img, inputShape)
    imgPred = np.asarray(imgPred).astype(np.float32) / 255.0
    imgPred = np.expand_dims(imgPred, axis=0)
    return imgPred


def getResultImg(img, dictGuixy, listPiece, listTestResult):
    for piece, testResult in zip(listPiece, listTestResult):
        guixy = dictGuixy[piece]                               #polygon dimension for each piece
        if testResult == 'ng':                                #If test failed
            img = cv2.fillConvexPoly(img, guixy, (0,0,255))   #fill with red color

    return img   #retun modified image


def base_crop_save(spec, part_names, deviceName):
    """ 基準パーツ切り出し保存 """

    # パーツ座標csv読み込み----read csv file for polygon dimensions
    xy_csv_path = './image/' + spec + '/base/xy.csv'
    poly_xy_dic = loadPolyxy(xy_csv_path)
    
    
    fill_outer_color = (128, 128, 128)           
    
    #カメラ画像の読み込み   loading camera images
    cam_folder_path = './' + deviceName 
    cam_file_path = glob.glob(cam_folder_path + '/*.jpg')
    src = cv2.imread(cam_file_path[0])
    file_name = os.path.basename(cam_file_path[0])
    file_name = file_name[:-4]
    
    # 画像ファイルの移動   move image to source folder
    src_folder_path = './image/' + spec + '/src'
    shutil.move(cam_file_path[0], src_folder_path)
    
    
    for part_name in part_names:
        # パーツ座標
        print('Processing part ' + part_name)

        basecut_file_path = './image/' + spec + '/base/base_img_' + part_name + '.jpg'

        basecut_img       = cv2.imread(basecut_file_path)    #acts as reference on how to segment images

        poly_xy = poly_xy_dic[part_name]

        poly_xy_sh = shrink_poly(poly_xy, 10)
        h, w, _  = src.shape

        crop_area=get_crop_area(poly_xy, w, h)
        offset_xy = crop_area[0:2]
        
        # 切り抜きポリゴンを縮小
        # poly_xy_sh = shrink_poly(poly_xy, 10)

        cut_folder_path     = './image/' + spec + '/' + part_name   #create folder for cut parts
        stabcut_folder_path = cut_folder_path + '/stabcut'
        

        if os.path.isdir(stabcut_folder_path):
            shutil.rmtree(stabcut_folder_path)
            
        os.makedirs(cut_folder_path, exist_ok=True)        
        os.makedirs(stabcut_folder_path, exist_ok=True)    

        basecut_img = cv2.cvtColor(basecut_img,cv2.COLOR_RGB2GRAY)    #convert to grayscale

        mask = make_mask(basecut_img, poly_xy)                        #remove background info
        dst  = part_crop(src, poly_xy)                                #cut source according to polygon                       

        src_cut = cv2.cvtColor(dst, cv2.COLOR_RGB2GRAY)               
        stab_img, shift_xy, phase = stabilizer_method(basecut_img, src_cut, mask)
        
        stab_img = np.array(stab_img)
        shift_xy = np.array(shift_xy)
        shift_xy = shift_xy.astype(np.int64)

        stab_imgC = img_phase_correlate(dst, phase)

        stabcut_img = fillout_color(stab_imgC, poly_xy_sh, offset_xy, fill_outer_color)
        stabcut_img = bounding_crop(stabcut_img, poly_xy_sh, offset_xy)

        save_file_path = f'{stabcut_folder_path}/{file_name}_{part_name}.jpg'
        cv2.imwrite(save_file_path, stabcut_img)


#-----------------------------------------------------------------------------------------------



#tisgrabber_x64.dllをインポートする
ic = ctypes.cdll.LoadLibrary("C:/PyEnv/suleman_test/utils/tisgrabber_x64.dll")
tis.declareFunctions(ic)

#ICImagingControlクラスライブラリを初期化しますp。
#この関数は、このライブラリの他の関数が呼び出される前に1回だけ呼び出す必要があります。

ic.IC_InitLibrary(0)

class CallbackUserdata(ctypes.Structure):
    """ コールバック関数に渡されるユーザーデータの例 """
    def __init__(self, ):
        self.unused = ''
        self.Value1 = 0
        self.fName  = ''

def moveFile(filename, dst):
    os.makedirs(dst, exist_ok = True)
    shutil.move(filename, dst + '/')


#-----Camera_config-------------------------------------------------------------------------------

# コールバック関数を定義
# 関数ポインタを作成

def frameReadyCallbackDevice00(hGrabber, pBuffer, framenumber, pData):
     #これはコールバック関数の例です
     #：param：hGrabber：これはグラバーオブジェクトへの実際のポインター(使用禁止)
     #：param：pBuffer：最初のピクセルのバイトへのポインタ
     #：param：framenumber：ストリームが開始されてからのフレーム数
     #：param：pData：追加のユーザーデータ構造へのポインター

    #CallbackUserdataが機能しているか確認
    # print("コールバック関数呼び出し", pData.Value1)
    # print(pData.fName)
    pData.Value1 = pData.Value1 + 1

    Width = ctypes.c_long()
    Height = ctypes.c_long()
    BitsPerPixel = ctypes.c_int()
    colorformat = ctypes.c_int()

    dstCam0 = 'C:/BoxDrive/Box/(P).24_AI_IoTプロジェクト/(P).24_AI_IoTプロジェクト_画像保存/3ARr/Cam00/'

    # 画像データの解像度・ピクセルごとの使用するビット数、カラーフォーマットを取得
    ic.IC_GetImageDescription(hGrabber, Width, Height, BitsPerPixel,
                              colorformat)

    # バッファサイズを計算
        
    bpp = int(BitsPerPixel.value/8.0)
    buffer_size = Width.value * Height.value * bpp

    
    if buffer_size > 0:
        image = ctypes.cast(pBuffer,
                            ctypes.POINTER(
                                ctypes.c_ubyte * buffer_size))

        cvMat = np.ndarray(buffer=image.contents,
                           dtype=np.uint8,
                           shape=(Height.value,
                                  Width.value,
                                  bpp))
        
        cvMat = cv2.rotate(cvMat, cv2.ROTATE_180)
        cvMat = cv2.flip(cvMat, 1)

        now  = datetime.datetime.now()
        dNow = now.strftime('%Y%m%d')
        sNow = now.strftime('%Y%m%d_%H%M%S')
        dst  = dstCam0 + dNow

        cv2.imwrite('./device00/device00_'+sNow+'.jpg',cvMat)
        pData.fName = 'device00_'+sNow+'.jpg'


def frameReadyCallbackDevice01(hGrabber, pBuffer, framenumber, pData):

    pData.Value1 = pData.Value1 + 1

    Width = ctypes.c_long()
    Height = ctypes.c_long()
    BitsPerPixel = ctypes.c_int()
    colorformat = ctypes.c_int()

    dstCam1 = 'C:/BoxDrive/Box/(P).24_AI_IoTプロジェクト/(P).24_AI_IoTプロジェクト_画像保存/3ARr/Cam01/'

    # 画像データの解像度・ピクセルごとの使用するビット数、カラーフォーマットを取得
    ic.IC_GetImageDescription(hGrabber, Width, Height, BitsPerPixel,
                              colorformat)

    # バッファサイズを計算
    
    
    bpp = int(BitsPerPixel.value/8.0)
    buffer_size = Width.value * Height.value * bpp

    if buffer_size > 0:
        image = ctypes.cast(pBuffer,
                            ctypes.POINTER(
                                ctypes.c_ubyte * buffer_size))

        cvMat = np.ndarray(buffer=image.contents,
                           dtype=np.uint8,
                           shape=(Height.value,
                                  Width.value,
                                  bpp))


        cvMat = cv2.rotate(cvMat, cv2.ROTATE_180)
        cvMat = cv2.flip(cvMat, 1)

        now  = datetime.datetime.now()
        dNow = now.strftime('%Y%m%d')
        sNow = now.strftime('%Y%m%d_%H%M%S')
        dst  = dstCam1 + dNow
        #if not pData.fName == '':
        #    moveFile(pData.fName, dst)
        #    print('move')

        cv2.imwrite('./device01/device01_'+sNow+'.jpg',cvMat)
        pData.fName = 'device01_'+sNow+'.jpg'


frameReadyCallbackfuncDevice00 = ic.FRAMEREADYCALLBACK(frameReadyCallbackDevice00)
userdataDevice00 = CallbackUserdata()

frameReadyCallbackfuncDevice01 = ic.FRAMEREADYCALLBACK(frameReadyCallbackDevice01)
userdataDevice01 = CallbackUserdata()


#hGrabber = ic.IC_ShowDeviceSelectionDialog(None)
#ダイアログ画面を表示
#hGrabberDevice00 = tis.openDevice(ic)
device_file00="device00.xml"
device_file01="device01.xml"

hGrabberDevice00 = setup_camera(ic,device_file00,frameReadyCallbackfuncDevice00,userdataDevice00)
hGrabberDevice01 = setup_camera(ic,device_file01,frameReadyCallbackfuncDevice01,userdataDevice01)

#-----------------------------

pathDevice00 = './device00'
pathDevice01 = './device01'

#-------Gui_config---------------------------------------------------------------------------
sg.theme('Darkgrey13')
locationWindow = (0, 0)
nameWindow     = 'InspectionSystem'
nameFont       = 'Meiryo UI'
sizeFont       = 20

imgAnalysis = cv2.imread('./imgs/imgAnalysis.png')
imgOk       = cv2.imread('./imgs/imgOK.png')
imgNG       = cv2.imread('./imgs/imgNG.png')
imgError    = cv2.imread('./imgs/imgError.png')


#size = 900*150
imgResult = sg.Image('./imgs/imgStart.png',
                       pad = ((500, 65), (20,20)),
                       size = (950, 130),
                       key = '-textResult-')


imgRB = sg.Image('./imgs/RB01.png', pad = ((80, 80), (0,10)), size = (900, 650), key = '-RB-')
imgRC = sg.Image('./imgs/RC01.png', pad = ((80, 80), (0,0)), size = (900, 500), key = '-RC-')


TextSpec   = sg.Text('社内記号：-----',   #internal code
                     font = (nameFont, sizeFont), text_color ='#ffffff',
                     pad = ((0,0),(30,0)), key = '-Spec-',)

TextSerial = sg.Text('連番-----', font = (nameFont, sizeFont), text_color ='#ffffff',
                     pad = ((100,0),(20,0)), key = '-Serial-',)

buttonStart = sg.Button('Exit', font = (nameFont, 16), pad = ((900,0),(0,0)), size = (8,0))


layout = [
    [imgResult],
    [imgRB, imgRC],  # Place imgRB and imgRC in the same row
    [TextSpec, TextSerial],
    [buttonStart]
]



window = sg.Window(nameWindow, layout, size = (1920, 1080), location = locationWindow,\
                    return_keyboard_events=True)

if os.path.isdir('device00'):
    shutil.rmtree('device00')
os.makedirs('device00', exist_ok=True)

if os.path.isdir('device01'):
    shutil.rmtree('device01')
os.makedirs('device01', exist_ok=True)



while True:
    event, values = window.read(timeout = 20)

    if event in (None, 'Exit'):
        break

    elif event == 'a':
        if os.path.isdir(pathDevice00):
            shutil.rmtree(pathDevice00)
        os.makedirs(pathDevice00)

        if os.path.isdir(pathDevice01):
            shutil.rmtree(pathDevice01)
        os.makedirs(pathDevice01)

#trigger camera------------------------------------------------
      
        
        trigger_camera(ic,hGrabberDevice00)
        trigger_camera(ic,hGrabberDevice01)
        
      
        dispTextResult = imgToBytes(imgAnalysis)
        window['-textResult-'].update(data = dispTextResult)

        sg.popup_no_buttons(auto_close=True,
                            auto_close_duration=0,
                            no_titlebar=True)
        time.sleep(1)

        #-------------------------------------------------

        flgPicLoad = False
        for i in range(11):                                     #ten seconds time-slot
            Pics00 = len(glob.glob(pathDevice00 + '/*.jpg'))
            Pics01 = len(glob.glob(pathDevice01 + '/*.jpg'))
            
            if Pics00 >= 1 and Pics01 >= 1:                      #if an image exists 
                flgPicLoad = True
                print('Image loading successful')               #Load was successful
                break
    
            if i >= 10:        
                print('Error_timeout')

                break        

            time.sleep(1)
        
        if flgPicLoad == False:
            dispTextResult = imgToBytes(imgError)             #Otherwise display error image
            window['-textResult-'].update(data = dispTextResult)
            continue


        #------Load_Spec-----------
        specRB = 'CAMRY12'
        specRC = 'CAMRY44'


        listPieceRB = ['A', 'C', 'D', 'E', 'G', 'H'] #List of pieces for camera 1
        listPieceRC = ['A','B', 'C', 'D','F', 'G', 'H', 'I','J','K','L'] #List of pieces for camera 2

        
        #-----PieceCut------
        print("\n \n *******Camera 1 image processing******* \n \n")
        base_crop_save(specRB, listPieceRB, 'device00')  #cut into pieces for camera 1
        print("\n \n *******Camera 2 image processing******* \n \n")
        base_crop_save(specRC, listPieceRC, 'device01')   #cut into pieces for camera 2
        

        #-----test-----
        print("\n \n *******Camera 1 testing******* \n \n")
        listTestResultRB = test(specRB, listPieceRB)   # perform testing on B
        print("\n \n *******Camera 2 testing******* \n \n")
        listTestResultRC = test(specRC, listPieceRC)   #perform testing on C
        

        resultRB = 'ok' in listTestResultRB            #if ok exists in list or not
        resultRC = 'ok' in listTestResultRC            #check if ok 
        if resultRB == True and resultRC == False:     #only if RB is True and RC false then result will be OK
            result = 'ok'
        
        else:
            result = 'ng'

        #--------------------------------------------

        imgResultRB = cv2.imread('./imgs/RB01.png')
        imgResultRC = cv2.imread('./imgs/RC01.png')

        pathCSVguixyRB  = './disp/guixyRB.csv'
        pathCSVguixyRC  = './disp/guixyRC.csv'

        dictGuixyRB = loadPolyxy(pathCSVguixyRB)
        dictGuixyRC = loadPolyxy(pathCSVguixyRC)


        imgResultRB = getResultImg(imgResultRB, dictGuixyRB, listPieceRB, listTestResultRB)
        imgResultRC = getResultImg(imgResultRC, dictGuixyRC, listPieceRC, listTestResultRC)

        #-----------------------------------------

        if result == 'ok':
            dispTextResult = imgToBytes(imgOk)
            window['-textResult-'].update(data = dispTextResult)

        if result == 'ng':
            dispTextResult = imgToBytes(imgNG)
            window['-textResult-'].update(data = dispTextResult)

            dispImgResultRB = imgToBytes(imgResultRB)
            dispImgResultRC = imgToBytes(imgResultRC)
            window['-RB-'].update(data = dispImgResultRB)
            window['-RC-'].update(data = dispImgResultRC)

#stop camera--------------------
stop_camera(ic,hGrabberDevice00)
stop_camera(ic,hGrabberDevice01)

window.close()
