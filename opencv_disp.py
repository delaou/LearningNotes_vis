import cv2
import numpy as np

def SGBM_disp(img_l, img_r, blockSize=3, down_scale=False, mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY):#视差计算
    
    if len(img_l.shape) == 2:
        img_dim = 1
    else:
        img_dim = 3#读入图像通道数
    SGBM_pam_l = {#设定SGBM参数
            #预处理参数
            'preFilterCap': 1,#映射滤波器大小
            #代价计算参数
            'minDisparity': 0,#最小视差，决定左图像素点在右图匹配起点
            'numDisparities': 64,#视差搜索范围，16的整数倍
            'blockSize': blockSize,#代价计算窗口大小，根据实际状况设置，值越大视差图越平滑
            #动态规划参数
            'P1': 8*img_dim*blockSize*blockSize,#相邻像素点视差增减1时的惩罚系数
            'P2': 32*img_dim*blockSize*blockSize,#相邻像素点视差变化大于1时的惩罚系数
            #优化处理参数
            'uniquenessRatio': 10,#唯一性检测参数(区分误匹配，值越大检测区分越强)
            'disp12MaxDiff': 1,#一致性检测最大允许误差
            'speckleWindowSize': 100,#视差联通区域最小允许像素点个数
            'speckleRange': 100,#视差联通最大允许视差变化值
            'mode': mode#不同模式下速度与效果不同
            }
    
    SGBM_matcher_l = cv2.StereoSGBM_create(**SGBM_pam_l)
    SGBM_pam_r = SGBM_pam_l
    SGBM_pam_r['minDisparity'] = -SGBM_pam_l['numDisparities']
    SGBM_matcher_r = cv2.StereoSGBM_create(**SGBM_pam_r)
    
    if down_scale == False:
        disp_l = SGBM_matcher_l.compute(img_l, img_r)
        disp_r = SGBM_matcher_l.compute(img_r, img_l)
        
    else:
        img_l_down = cv2.pyrDown(img_l)
        img_r_down = cv2.pyrDown(img_r)
        factor = img_l_down.shape[1]/img_l_down.shape[1]
        disp_l_h = SGBM_matcher_l.compute(img_l_down, img_r_down)
        disp_r_h = SGBM_matcher_r.compute(img_r_down, img_l_down)
        disp_l = cv2.resize(disp_l_h, img_l.shape[:2][::-1],interpolation=cv2.INTER_AREA)
        disp_r = cv2.resize(disp_r_h, img_r.shape[:2][::-1],interpolation=cv2.INTER_AREA)
        disp_l = factor*disp_l
        disp_r = factor*disp_r
        
    disp_l = disp_l.astype(np.float32)/1.
    disp_r = disp_r.astype(np.float32)/1.
    
    return disp_l, disp_r