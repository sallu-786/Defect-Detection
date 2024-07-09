概要:
このコードは、カメラ（device00とdevice01）を使用して部品の画像をキャプチャし、
座標（xy.csvファイル）で指定された部品の検査と解析を行うシステムを実装しています。
画像処理にはOpenCV、機械学習操作にはTensorFlow、グラフィカルユーザーインターフェース(GUI)
の作成にはPySimpleGUIを使用しています。

主なパーツ:

1. 初期化と設定
    i) Python 3.9.13で環境をセットアップしています。最新のPythonを使用すると問題が発生する
       可能性があります。必要なファイルは2024年7月現在の更新がされています。

    ii) GPUのメモリ成長をサポートするように設定されています。

    iii) カメラとのやり取り:
        tisgrabberライブラリ（import utils.tisgrabber as tis）をインポートしてカメラとのやり取りを統合
	しています。カメラ（hGrabberDevice00、hGrabberDevice01）によってトリガーされるイメージ
	キャプチャイベントを処理する コールバック関数
	（frameReadyCallbackDevice00、frameReadyCallbackDevice01）を定義しています。

2. 画像処理関数:

    i) base_crop_save(spec, part_names, deviceName):
        カメラ（device00とdevice01）によってキャプチャされた画像を処理します。
        ポリゴンの座標（xy.csv）に基づいて特定の部品（part_names）を切り出します。
        ポリゴン操作（loadPolyxy、shrink_poly）、マスキング（make_mask）、
	クロッピング（part_crop）、安定化（stabilizer_method）のための関数を使用しています。

3. テストと解析関数:

    i) test(specRB, listPieceRB):
        カメラによってキャプチャされた画像の部品（specRBとspecRC）に対してテストを実施します。
        テスト結果（listTestResultRBとlistTestResultRC）を評価し、部品が合格（ok）か不良（ng）
	かを判断します。個々の部品の結果を集計して全体の検査結果を決定します。

    ii) getResultImg(img, dictGuixy, listPiece, listTestResult):
        キャプチャされた画像（img）をテスト結果（listTestResult）に基づいて修正し、
        各部品（listPiece）に対して失敗した部分を赤色（(0, 0, 255)）でハイライトします。

4. GUIとユーザーインタラクション グラフィカルユーザーインターフェース(GUI):

        システムの状態を示す検査結果（imgResult）を表示します
	（imgStart.png、imgOk.png、imgNG.png、imgError.png）。
        検査された部品に関連する特定の部品画像（imgRB、imgRC）とテキスト情報
	（TextSpec、TextSerial）を表示します。ユーザーがインスペクションを開始し、
	カメラのキャプチャをトリガーするためのボタン（buttonStart）を介した
        ユーザーインタラクションを可能にします。

5. システム制御とエラーハンドリング:

        キャプチャされた画像（device00、device01）の保存のためのファイルパス（os、shutil）
	を管理します。イメージの取得と処理中に堅牢性を確保するためにエラーハンドリング
	（if flgPicLoad == False）やタイムアウトメカニズム（for i in range(11)）を実装します。
        インスペクションの完了後にリソースを解放するために適切にカメラ操作を停止します
	（stop_camera）。システム操作の終了時にGUIウィンドウを閉じます（window.close）。

