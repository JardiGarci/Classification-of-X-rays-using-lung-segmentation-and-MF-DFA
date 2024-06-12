import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
import cv2


def trend_surface(surface , grade = 1, show = False):
    x,y = surface.shape
    Sub_IM = np.array(surface)
    TX, TY = np.meshgrid(range(x),range(y))
    x1, y1, z1 = TX.flatten(), TY.flatten(), Sub_IM.flatten()
    if grade == 1:
        ####
        X_data = np.array([x1, y1]).T
        Y_data = z1

        reg = linear_model.LinearRegression().fit(X_data, Y_data)
        a1 = reg.coef_[0]; a2 = reg.coef_[1]; c = reg.intercept_

        # ZZ = FF(TX, TY, a1, a2, c)
        ZZ = a1*TX + a2*TY + c
        ####
        
    elif grade == 2:
        ###
        x1y1, x1x1, y1y1 = x1*y1, x1*x1, y1*y1
        X_data = np.array([x1, y1, x1y1, x1x1, y1y1]).T  
        Y_data = z1

        reg = linear_model.LinearRegression().fit(X_data, Y_data)
        a1 = reg.coef_[0]; a2 = reg.coef_[1]; a3 = reg.coef_[2]; a4 = reg.coef_[3]; a5 = reg.coef_[4]; c = reg.intercept_

        # ZZ = func(TX, TY, a1, a2, a3, a4, a5, c)
        ZZ = a1*TX + a2*TY + a3*TX*TY + a4*TX*TX + a5*TY*TY + c
        ###
    else: print("\n Orden incorrecto \n")

    if show == True:
        fig = plt.figure()
        ax = plt.axes(projection ='3d')
        ax.plot3D(x1, y1, z1, '.r')
        ax.plot_surface(TX, TY, ZZ)

    return ZZ



def dentred_flutation_function_2D(img, size, mask = [], umbral = 0.85, grade = 2, axis = 0, show = False):

    if len(mask) > 0:
        img_mask = mask
        umbral = umbral * np.max(mask)
    else:
        img_mask = np.array(img)
        umbral = np.min(img) - 1
    y,x = img.shape

    X = np.arange(0,x - size,size)
    X2 =  np.arange(x , 0 + size, -size) - size

    Y = np.arange(0,y - size,size) 
    Y2 =  np.arange(y , 0 + size, -size) - size

    TX,TY = np.meshgrid(X,Y)
    TX2,TY2 = np.meshgrid(X2,Y2)

    TX = list(TX.flatten())
    TY = list(TY.flatten())

    TX2 = list(TX2.flatten())
    TY2 = list(TY2.flatten())

    axis_boxes = [[TX,TY],[TX + TX2 ,TY + TY],[TX + TX2 + TX ,TY + TY + TY2],[TX + TX2 + TX + TX2 , TY + TY + TY2 + TY2]]
    axis = axis_boxes[axis]

    
    new_img = np.array(img)
    
    F = []
    points = []

    for i,j in zip(axis[0],axis[1]):
        i2 = i + size
        j2 = j + size
        box = np.array(img[j : j2 , i : i2])
        if np.mean(np.array(img_mask[j : j2 , i : i2])) > umbral:
            new_img = cv2.rectangle(new_img, (i, j), (i2, j2), (255, 255, 255), 1)
            surface = trend_surface(box, grade=grade, show=False)    
            residual_matrix = box - surface
            F.append((np.mean(np.square(residual_matrix)))**(1/2))
            points.append([i,j])


    if show == True:
        return new_img
    else:
        return F,points

def multidim_cumsum(a):
        out = a[...,:].cumsum(-1)[...,:]
        for i in range(2,a.ndim+1):
            np.cumsum(out, axis=-i, out=out)
        return out

class MF_DFA_2D():

    def __init__(self, img = [],
                 mask = [],
                 FF = [],
                 box_sizes = []
                 ):
        self.img = img
        self.mask = mask
        self.FF = FF
        self.box_sizes = box_sizes
        self.box_sizes_log = np.log10(self.box_sizes)
        self.a = []
        self.f = []
        self.F_q = []


    def grade(self,grade):
        val = 2
        for i in range(1,grade + 1):
            val = np.sqrt(val)
        return val
    
    def img_mean(self):
        if len(self.img) == 0:
            print("There is no image to analyze, so it must be added manually or the class must be recreated.")
        else:
            # Substract mean
            if len(self.mask) >1:
                media = np.mean(self.img[self.mask == 1])
                self.img = np.array(self.img) - media
                self.img[self.mask != 1] = 0
            else:
                media = np.mean(self.img)
                self.img = np.array(self.img) - media
    
    def img_cumsum(self):
        if len(self.img) == 0:
            print("There is no image to analyze, so it must be added manually or the class must be recreated.")
        else:
            # Accumulated sum
            if len(self.mask) > 1:
                self.img = multidim_cumsum(self.img) 
                self.img[self.mask != 1] = 0
            else:
                self.img = multidim_cumsum(self.img) 


    def img_to_FF(self, 
                 lim_box_sizes = [6, False],    # If False, max size = min(M,N)/4
                 step_size = 'bineo',       # 'Bineo' or a int
                 bineo_grade = 2,
                 grade = 2,
                 threshold = 0.85,
                 axis = 0   
                 ):
        self.F_q = []

        if len(self.img) == 0:
            print("There is no image to analyze, so it must be added manually or the class must be recreated.")
        else:
            
            if lim_box_sizes[1] == False:
                max_size = int(np.min(self.img.shape) / 4)
            elif lim_box_sizes[1] == 'Max' :  
                max_size = np.min(self.img.shape) 
            else:
                max_size = lim_box_sizes[1]
    
            if step_size == 'bineo':
                val_bin = self.grade(bineo_grade)
                s = lim_box_sizes[0]
                box_sizes = []
                while s < max_size:
                    box_sizes.append(s)
                    s = int(s * val_bin ) + 1
            else:
                box_sizes = list(range(lim_box_sizes[0],max_size, step_size))

            self.FF = []
            self.Points = []
            self.box_sizes = []
            for size in box_sizes:
                DFF,points = dentred_flutation_function_2D(img = self.img,
                                                mask = self.mask,
                                                size = size,
                                                umbral = threshold,
                                                axis=axis,
                                                grade = grade,
                                                show = False)
                if len(DFF) < 4:
                    break
                self.box_sizes.append(size)
                self.FF.append(DFF)
                self.Points.append(points)
            self.box_sizes_log = np.log10(self.box_sizes)
        
  
    def FF_to_spectrum(self, lim_q = [-5,5], dq = 0.25, min_size = False, max_size = False ):
        if len(self.FF) == 0:
            print("There is no functions for fluctuations (self.FF),  they need to be added manually or run img_to_FF")
        elif len(self.box_sizes) == 0:
            print("There is no box sizes (self.box_sizes),  they need to be added manually or run img_to_FF")
        else:
            min_size = list(np.abs(np.array(self.box_sizes) - min_size))
            self.index_min = min_size.index(min(min_size))

            if max_size == False:
                self.index_max = len(self.box_sizes) + 1
            else:
                max_size = list(np.abs(np.array(self.box_sizes) - max_size))
                self.index_max = max_size.index(min(max_size)) + 1

            if len(lim_q) == 2:
                self.Q = np.arange( lim_q[0]-dq, lim_q[1] + 2*dq, dq)
            else:
                self.Q = lim_q
                
            self.F_q_log = [[] for i in self.Q]
            self.F_q = [[] for i in self.Q]
            for F in self.FF[self.index_min:self.index_max]:
                for j,q in enumerate(self.Q):
                    if q == 0:
                        dentred_fluctuation = np.exp(np.mean(np.log(np.array(F))))
                        self.F_q[j].append(dentred_fluctuation) 
                        self.F_q_log[j].append(np.log10(dentred_fluctuation))
                        
                    else:
                        dentred_fluctuation = np.mean(np.array(F)**(q))**(1.0/q)
                        self.F_q[j].append(dentred_fluctuation)
                        self.F_q_log[j].append(np.log10(dentred_fluctuation))

            self.holder = []
            self.tau = []
            for q, F_q_log in zip(self.Q, self.F_q_log):
                h_q = np.polyfit(self.box_sizes_log[self.index_min:self.index_max], F_q_log, 1)[0]
                self.holder.append(h_q)
                self.tau.append(q * h_q - 2)
            
            self.a = []
            self.f = []
            for i in range(1 , len(self.Q) -1):
                if len(lim_q) != 2:
                    a = (self.tau[i+1] - self.tau[i-1]) / (2 * dq[i+1])
                else:
                    a = (self.tau[i+1] - self.tau[i-1]) / (2 * dq)
                self.a.append(a)
                f = self.Q[i] * a - self.tau[i]
                self.f.append(f)

    def Features(self):
        if len(self.a) == 0:
            self.FF_to_spectrum()
            
        C2,C1,C0 = np.polyfit(self.Q,self.tau,2)
        self.features_vals = [self.a[self.f.index(max(self.f))],            # a_star
                            min(self.a),                                  # a_min
                            max(self.a),                                  # a_max
                            max(self.a) - min(self.a),                    # width
                            max(self.f) - min(self.f),                    # height
                            sum([np.linalg.norm(np.array((self.a[i-1],self.f[i-1])) - np.array((self.a[i],self.f[i]))) for i in range(1,len(self.a))]),
                            - C0,                     # Lineal function C0 + C1X + C2X^2 of tau
                            C1,
                            - 2 * C2
                            ]
        self.features_names = ['a_star','a_min','a_max','width','height','length', 'C0' ,'C1','C2']
        
        return  self.features_names, self.features_vals
        
    def Show(self, path_save = ''):
        if len(self.F_q) == 0:
            self.FF_to_spectrum()
        plt.ioff()

        fig = plt.figure(constrained_layout=False, figsize=[9,7])
        fig.suptitle('MF-DFA')

        gs1 = fig.add_gridspec(nrows=1, ncols=1, left=0.05, right=0.6, hspace=0.0, wspace= 0.5)
        f_ax1 = fig.add_subplot(gs1[:, :])
        f_ax1.grid(True)
        f_ax1.set_xlabel('s')
        f_ax1.set_ylabel('Fq(s)')
        f_ax1.set_xscale('log')
        f_ax1.set_yscale('log')
        for F_q in self.F_q:
            f_ax1.plot(self.box_sizes[self.index_min:self.index_max], F_q)
  
        gs2 = fig.add_gridspec(nrows=4, ncols=1, left=0.7, right=0.98, hspace=0.00)
        f_ax2 = fig.add_subplot(gs2[0, :])
        f_ax2.grid(True)
        f_ax2.set_ylabel('h(q)')
        f_ax2.scatter(self.Q, self.holder, edgecolors='b', c = 'white', s = 15)

        f_ax3 = fig.add_subplot(gs2[1, :])
        f_ax3.grid(True)
        f_ax3.set_ylabel('τ(q)',)
        # f_ax3.set_xlabel('q')
        f_ax3.scatter(self.Q, self.tau, edgecolors='b', c = 'white', s = 15)

        gs3 = fig.add_gridspec(nrows=4, ncols=1, left=0.7, right=0.98, hspace=0.560)
        f_ax4 = fig.add_subplot(gs3[2:, :])
        f_ax4.grid(True)
        f_ax4.set_ylabel('f(α)')
        f_ax4.set_xlabel('α')
        f_ax4.scatter(self.a,self.f, edgecolors='r', c = 'white', s = 35)
        f_ax4.plot(self.a,self.f, 'r')

        if path_save:
            plt.savefig(path_save, bbox_inches='tight')
        else:
            plt.show()
        plt.close()