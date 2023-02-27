import cv2
import numpy as np
import Camera_parameters  # 导入相机标定的参数
from opencv_preprocess import RectifyMaps
from opencv_disp import SGBM_disp
# import pcl
# import pcl.pcl_visualization

mode = cv2.STEREO_SGBM_MODE_HH

def mouse_click(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        print('(%d, %d) 3D_coord: (%f, %f, %f)' %(
        x, y, points_3d[y, x, 0], points_3d[y, x, 1], points_3d[y, x, 2]))
        distance = ((points_3d[y, x, 0]**2 + points_3d[y, x, 1]**2 + points_3d[y, x, 2]**2)**0.5)/1000
        print('(%d, %d): %0.3f m from the left camera' %(x, y, distance)) 
        
if __name__ == '__main__':
    img_l = cv2.imread(r"D:\Filea\Camera_calibrate\left\left0017.bmp", 1)
    img_r = cv2.imread(r"D:\Filea\Camera_calibrate\right\right0017.bmp", 1)
    img_size = img_l.shape[:2][::-1]
    img_lg = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
    img_rg = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)
    
    cam_params = Camera_parameters.Parameters()
    
    img_lg = cv2.undistort(img_lg, cam_params.camMat_l, cam_params.dist_l)
    img_rg = cv2.undistort(img_rg, cam_params.camMat_r, cam_params.dist_r)
    
    l_map1, l_map2, r_map1, r_map2, Q = RectifyMaps(img_size, cam_params)
    
    # img_l_e = cv2.equalizeHist(img_lg)
    # img_r_e = cv2.equalizeHist(img_rg)
    
    rectified_img_lg = cv2.remap(img_l, l_map1, l_map2, cv2.INTER_LINEAR)
    rectified_img_rg= cv2.remap(img_r, r_map1, r_map2, cv2.INTER_LINEAR)
    # rectified_img_l = cv2.cvtColor(rectified_img_lg, cv2.COLOR_GRAY2BGR)
    # rectified_img_r = cv2.cvtColor(rectified_img_rg, cv2.COLOR_GRAY2BGR)
    
    disp_l, disp_r = SGBM_disp(rectified_img_lg, rectified_img_rg, blockSize=3, down_scale=True, mode=mode)
    
    # disp_l = cv2.normalize(disp_l, disp_l, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)# gray
    
    disp_l_color = cv2.normalize(disp_l, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    disp_l_color = cv2.applyColorMap(disp_l_color, 2)
    
    points_3d = cv2.reprojectImageTo3D(disp_l, Q, handleMissingValues=True)
    
    points_3d = points_3d*16

    cv2.namedWindow("disparity", 0)
    cv2.imshow("disparity", disp_l_color)
    cv2.setMouseCallback("disparity", mouse_click, 0)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
