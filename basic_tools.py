# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 11:56:52 2018

@author: Farewell
"""

import numpy as np
from matplotlib import pyplot as plt
import LightPipes as LP
import os
import Chinese_Lib
Chinese_Lib.set_ch()

from pylab import mpl
from matplotlib.font_manager import FontProperties
import time

plt.rc('font',family='Times New Roman')

class basic_tools():
    
    def __init__(self, wvl=1550*LP.nm, w0=0.01*LP.m, WindowSize=0.1*LP.m, N=512, Cn2=1e-15, L0=5*LP.m, l0=5*LP.mm, alpha=11./3):
        
        self.wvl = wvl   # wavelength;
        self.w0 = w0     # beam waist;
        self.WindowSize = WindowSize    # plane size of the light field;
        self.N = N       # grid sampling points;
        self.Cn2 = Cn2
        self.L0 = L0
        self.l0 = l0
        self.alpha = alpha

# In[]: data I/O with .txt；(array only)
        
    def write_txt(self, matrix=None, filename='new.txt'):
        
        """
        Write a matrix to the .txt
        Parameters:
            matrix: to be written
            filename: name of the .txt(include expanded-name)
        Returns:
            if write successful,return true.
        """
        
        try:      
            N,M = matrix.shape
            
        except:
            
            matrix = matrix[:,np.newaxis]
        
        finally:
            
            with open(filename,'w') as file_object:    # stored the file &  contents into var file_object
                
                # write the first line (first block)
                file_object.write(str(matrix[0,0]))    # first row, first column
                for j in range(1,np.size(matrix,1)):
                    file_object.write('\t' + str(matrix[0,j]))    # write the back columns
                # write the back rows
                for i in range(1,np.size(matrix,0)):
                    file_object.write('\n'+str(matrix[i,0]))
                    for j in range(1,np.size(matrix,1)):
                        file_object.write('\t'+str(matrix[i,j]))
        return True
    
    def read_txt(self,filename='new.txt'):
        
        """
        Read a matrix from the .txt
        Parameters:
            filename: name of the .txt(include expanded-name)
        Returns:
            if write successful,return the matrix.
        """
        
        matrix = []
        with open(filename, 'r') as file_to_read:
            while True:
                lines = file_to_read.readline()         # Read one line data
                if not lines:
                    break
                    pass
                if lines.split()[0].find('.') == 0:
                    q = [int(i) for i in lines.split()]  
                else:
                    q = [float(i) for i in lines.split()]  
                matrix.append(q)
                pass
            matrix = np.array(matrix)
            pass
        return matrix    
    
# In[]:        
    def show_intensity_image(self, Amp, Amp_Flag=True, gray=True, set_title=None, set_title_Flag=False,
                             save=False, log=False, log_multiple=5, path=os.getcwd(), plot=True, label=True):
                             
        """
        A simple package about imshow function, it can be useful in showing light field intensity.
        Parameters:
            Amp: amplitude of the light field.s
            gray: show the gray-scale map if set True, else show the color map, default set as True.
            set_title: default set as None, it will take effect in map title and filename after set a value.
            set_title_Flag: default set as False, if set True, the gray-scale(color) map will get a title.
            save: default set as False, if set True, the map will be saved as an .png image.
            log: default set as False, if set True, the map will be processed with Logarithmic gray-scale Algorithm.
            log_multiple: default set as 5.
            path: save path, default set as current working directory.
            plot: default set as True, and the image will be displayed.
        
        Returns:
            im: AxisImage.
            save: True or False.
            Intensity: the intensity of the light filed.
        """
        
        if Amp_Flag:
            Intensity = (Amp*Amp.conjugate()).real
        else:
            Intensity = Amp
        if plot:            
            if gray:
                plt.figure(figsize = (8,7))
                if set_title_Flag:       
                    plt.title(set_title,fontsize=25)
                if log:
                    
                    # Logarithmic gray-scale Algorithm.
                    Intensity = Intensity/np.max(Intensity)*255
                    Intensity = np.log(1+log_multiple*Intensity)/np.log2(1+log_multiple)
                    
                    im = plt.imshow(Intensity,cmap='gray',interpolation='bicubic',extent=[-self.WindowSize/self.w0/2,self.WindowSize/self.w0/2,
                                                                                          -self.WindowSize/self.w0/2,self.WindowSize/self.w0/2])
                    if not label:
                        plt.axis('off')
                else:
                    im = plt.imshow(Intensity,cmap='gray',interpolation='bicubic',extent=[-self.WindowSize/self.w0/2,self.WindowSize/self.w0/2,
                                                                                         -self.WindowSize/self.w0/2,self.WindowSize/self.w0/2])
                    if not label:
                        plt.axis('off')
            else:
                if set_title_Flag:    
                    plt.title(set_title,fontsize=25)
                plt.figure(figsize = (8,7))
                im = plt.imshow(Intensity,cmap='jet',interpolation='bicubic',extent=[-self.WindowSize/self.w0/2,self.WindowSize/self.w0/2,
                                                                                     -self.WindowSize/self.w0/2,self.WindowSize/self.w0/2])
                if not label:
                    plt.axis('off')
                    
            ax=plt.gca()
            ax.set_xticks(np.linspace(-self.WindowSize/self.w0/2,self.WindowSize/self.w0/2,5))
            ax.set_yticks(np.linspace(-self.WindowSize/self.w0/2,self.WindowSize/self.w0/2,5))
            plt.xticks(fontsize=26)
            plt.yticks(fontsize=26)
            plt.xlabel(r'${w}_{0}$', fontsize=20)
            plt.ylabel(r'${w}_{0}$', fontsize=20)
            if save:         # save figures 
                save = plt.savefig(path+"\\"+set_title+'.png')
                return save
            else:
                return im           
            if label:
                pass
            else:
                plt.axis('off')
        else:
            return Intensity    
        
# In[]:        
    def show_phase_image(self, Amp, gray=True, set_title = None, set_title_Flag=False,
                         save=False, path=os.getcwd(),label=True):
      
        """
        A simple package for displaying phase map through imshow function.
        Parameters:
            Amp: amplitude of the light field.s
            gray: show the gray-scale map if set True, else show the color map, default set as True.
            set_title: default set as None, it will take effect in map title and filename after set a value.
            set_title_Flag: default set as False, if set True, the gray-scale(color) map will get a title.
            save: default set as False, if set True, the map will be saved as an .png image.
            path: save path, default set as current working directory.
        
        Returns:
        """
        
        plt.figure(figsize = (8,7))

        if set_title_Flag:
            plt.title(set_title)

        phase = np.angle(Amp)

        if gray:
            plt.imshow(phase,cmap='gray',interpolation='bicubic',extent=[-self.WindowSize/self.w0/2,self.WindowSize/self.w0/2,
                                                                         -self.WindowSize/self.w0/2,self.WindowSize/self.w0/2])
            if label:
                pass
            else:
                plt.axis('off')
        else:
            plt.imshow(phase,cmap='jet',interpolation='bicubic',extent=[-self.WindowSize/self.w0/2,self.WindowSize/self.w0/2,
                                                                         -self.WindowSize/self.w0/2,self.WindowSize/self.w0/2])
            if label:
                pass
            else:
                plt.axis('off')
                
        ax=plt.gca()
        ax.set_xticks(np.linspace(-self.WindowSize/self.w0/2,self.WindowSize/self.w0/2,5))
        ax.set_yticks(np.linspace(-self.WindowSize/self.w0/2,self.WindowSize/self.w0/2,5))
        plt.xticks(fontsize=26)
        plt.yticks(fontsize=26)
        plt.xlabel(r'${w}_{0}$', fontsize=20)
        plt.ylabel(r'${w}_{0}$', fontsize=20)
        if save:     # save figures
            plt.savefig(path+"\\"+set_title+'.png')
        else:
            pass
        
        return True
# In[]:
    def show_3D_image(self,x,y,z,color=plt.cm.summer):
        
        """
        display a 3-D image through Axes3D library.
        Parameters:
            x,y,z: the cartesian coordinate(z=f(x,y)).
            color: default set as plt.cm.summer.
        Returns:
            True if displayde successfully.
        """
    
        from mpl_toolkits.mplot3d import Axes3D  
        # create 3-D graphic object
        fig = plt.figure() 
        ax = Axes3D(fig)   
        
        ax.plot_surface(x,y,z,cmap=color)
        
        return True
    
    def Estimate_time(self, Time):
    
        time_start = time.time()
        print('预计程序计算完成时间为',time.strftime('%Y-%m-%d %X',time.localtime(time_start+Time)))
        
        return True