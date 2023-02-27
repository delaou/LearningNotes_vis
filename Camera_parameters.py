import numpy as np

class Parameters():
    
    def __init__(self):
        
        self.camMat_l = np.array([[5.347859801204778e+03, 0, 0], 
                                  [0, 5.350574841506704e+03, 0], 
                                  [9.748426007966679e+02, 6.236733312910876e+02, 1]]).T
        
        self.camMat_r = np.array([[7.098767860712695e+03, 0, 0], 
                                  [0, 7.102553579387220e+03, 0], 
                                  [9.097081750340408e+02, 6.244755014399113e+02, 1]]).T
    
        self.dist_l = np.array([-0.111163865731777, 2.753625470972829, 0, 0, 0])
               
        self.dist_r = np.array([-0.027357011977946, -1.985581188272772, 0, 0, 0])
        
        self.rvecs = np.array([[0.999416738570253, 0.032328680731352, 0.011001775664386],
                               [-0.032352835976846, 0.999474459232070, 0.002024684424001], 
                               [-0.010930538406429, -0.002379442147193, 0.999937428835032]]).T
        
        self.tvecs = np.array([-99.717260083345440, -1.853295331713976, 9.290465463503027]).T
        