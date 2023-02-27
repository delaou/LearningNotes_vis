import numpy as np
import cv2
import glob


def Calibrate(sampt,adset,limnum=30):
    #单目标定
    img_adset=glob.glob(adset)
    sterpt_arr=[]
    cornpt_arr=[]
    fal_read=[]
    count=0
    sterpt=np.zeros((sampt[0]*sampt[1],3),np.float32)
    x,y=np.mgrid[0:sampt[0],0:sampt[1]]
    sterpt[:,:2]=cv2.merge((np.transpose(x),np.transpose(y))).reshape(-1,2)#生成世界坐标
    for img_ad in img_adset:
        img=cv2.imread(img_ad,0)#依次读入图片
        ret,cornpt=cv2.findChessboardCorners(img,sampt)#寻找棋盘点
        if ret==True:
            cornpt=cv2.cornerSubPix(img,cornpt,(11,11),(-1,-1),(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,30,0.001))#亚像素角点检测
            sterpt_arr.append(sterpt)
            cornpt_arr.append(cornpt)
            count+=1
            if count>=limnum:
                break
        else:
            fal_read.append(img_ad)#记录未找到棋盘点的图片
    img_size=img.shape[::-1]
    ret,camMat,dist,rvecs,tvecs=cv2.calibrateCamera(sterpt_arr,cornpt_arr,img_size,None,None)#相机标定，计算参数
    return sterpt_arr,cornpt_arr,camMat,dist


def RectifyMaps(img_size, class_params):
    #立体校正和畸变矫正
    rec_rvecs_l, rec_rvecs_r, rec_tvecs_l, rec_tvecs_r, Q, validPixROI1,validPixROI2 = \
        cv2.stereoRectify(class_params.camMat_l, class_params.dist_l, class_params.camMat_r, class_params.dist_r,
                          img_size[:2], class_params.rvecs, class_params.tvecs, alpha=0)
        
    l_map1, l_map2 = cv2.initUndistortRectifyMap(class_params.camMat_l, class_params.dist_l, rec_rvecs_l, rec_tvecs_l,
                                              img_size[:2], cv2.CV_16SC2)
    r_map1, r_map2 = cv2.initUndistortRectifyMap(class_params.camMat_r, class_params.dist_r, rec_rvecs_r, rec_tvecs_r,
                                              img_size[:2], cv2.CV_16SC2)
    
    return l_map1, l_map2, r_map1, r_map2, Q
