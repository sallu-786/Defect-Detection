import utils.tisgrabber as tis

def setup_camera(ic,device_file, frameReadyCallbackfunc, userdata):
    hGrabber = ic.IC_LoadDeviceStateFromFile(None, tis.T(device_file))
    ic.IC_SetFrameReadyCallback(hGrabber, frameReadyCallbackfunc, userdata)
    ic.IC_SetContinuousMode(hGrabber, 0)     #連続モードでは、フレームごとにコールバックが呼び出されます。
    ic.IC_SetPropertySwitch(hGrabber, tis.T("Trigger"), tis.T("Enable"), 1)      #トリガーモードをON
    ic.IC_StartLive(hGrabber, 0)
    return hGrabber

def trigger_camera(ic,hGrabber):
    ic.IC_PropertyOnePush(hGrabber, tis.T("Trigger"), tis.T("Software Trigger"))

def stop_camera(ic,hGrabber):
    ic.IC_StopLive(hGrabber)
    ic.IC_ReleaseGrabber(hGrabber)