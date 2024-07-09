import ctypes 
import tisgrabber as tis
import cv2 as cv2
import numpy as np

#tisgrabber_x64.dllをインポートする
ic = ctypes.cdll.LoadLibrary("./tisgrabber_x64.dll")
tis.declareFunctions(ic)

#ICImagingControlクラスライブラリを初期化します。
#この関数は、このライブラリの他の関数が呼び出される前に1回だけ呼び出す必要があります。
ic.IC_InitLibrary(0)

class CallbackUserdata(ctypes.Structure):
    """ コールバック関数に渡されるユーザーデータの例 """
    def __init__(self, ):
        self.unused = ""
        self.Value1=0


def frameReadyCallback(hGrabber, pBuffer, framenumber, pData):
     #これはコールバック関数の例です
     #：param：hGrabber：これはグラバーオブジェクトへの実際のポインター(使用禁止)
     #：param：pBuffer：最初のピクセルのバイトへのポインタ
     #：param：framenumber：ストリームが開始されてからのフレーム数
     #：param：pData：追加のユーザーデータ構造へのポインター

    #CallbackUserdataが機能しているか確認
    print("コールバック関数呼び出し", pData.Value1)
    pData.Value1 = pData.Value1 + 1

    Width = ctypes.c_long()
    Height = ctypes.c_long()
    BitsPerPixel = ctypes.c_int()
    colorformat = ctypes.c_int()

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
        # 二値化の閾値の設定
        threshold = 80
        # OpenCV処理
        cvMat = cv2.flip(cvMat, 0) #画像を反転させない
        cvMat = cv2.cvtColor(cvMat, cv2.COLOR_BGR2GRAY) #モノクロ化
        ret, img_THRESH_BINARY = cv2.threshold(cvMat, threshold, 255, cv2.THRESH_BINARY) #二値化
        resized_img = cv2.resize(img_THRESH_BINARY,(640, 480)) #表示を縮小
        cv2.imshow('Window', resized_img) #Windowに表示
        cv2.waitKey(10)


# コールバック関数を定義
# 関数ポインタを作成
frameReadyCallbackfunc = ic.FRAMEREADYCALLBACK(frameReadyCallback)
userdata = CallbackUserdata()

#hGrabber = ic.IC_ShowDeviceSelectionDialog(None)

#ダイアログ画面を表示
hGrabber = tis.openDevice(ic)

if ic.IC_IsDevValid(hGrabber):
    #コールバック関数を使用する宣言
    ic.IC_SetFrameReadyCallback(hGrabber, frameReadyCallbackfunc, userdata)
    #連続モードでは、フレームごとにコールバックが呼び出されます。
    ic.IC_SetContinuousMode(hGrabber, 0) #コールバック関数を使用するときには必ず定義

    #トリガーモードをON
    ic.IC_SetPropertySwitch(hGrabber, tis.T("Trigger"), tis.T("Enable"), 1)
    ic.IC_StartLive(hGrabber, 1) #ライブスタート開始　引数：0の時非表示、引数：1の時表示
    key = ""
    while key != "q":
        print("pキー: ソフトウェアトリガーを実行")
        print("外部トリガーでも実行可能です。")
        print("qキー: プログラムを終了")
        key = input('pキーかqキーを押下してください。:')
        if key == "p":
            ic.IC_PropertyOnePush(hGrabber, tis.T("Trigger"), tis.T("Software Trigger"))

    ic.IC_StopLive(hGrabber)
else:
    ic.IC_MsgBox(tis.T("No device opened"), tis.T("Simple Live Video"))

ic.IC_ReleaseGrabber(hGrabber)