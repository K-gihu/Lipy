# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 11:18:59 2018

@author: Farewell
"""

import numpy as np
import LightPipes as LP
from math import factorial as Fac
from scipy.special import comb,gamma,gammainc
import basic_tools
import basic_math

# =============================================================================
# Vortex array module.
# =============================================================================
class vortex_array(basic_tools.basic_tools,basic_math.basic_math):
   
    """ 
    This class is a derivative of the basic_tools & basic_math.
    """
    
    # Don't define __init__() function here, python will automatically call the initialization-function 
    # of superclass when constructing object.
#    def __init__(self,wvl=1550*LP.nm,w0=0.01*LP.m,
#                 WindowSize=0.1*LP.m,N=512):
#        
##        super(vortex_array,self)._init_()
#        # super()函数是用于调用父类的一个方法，用于解决多重继承问题，直接用类名调用父类方法在使用单继承时不会出现问题，但
#        # 若使用多继承会涉及到查找顺序(MRO)、重复调用等问题。
#        self.wvl = wvl
#        self.w0 = w0
#        self.WindowSize = WindowSize
#        self.N = N
        
    def vortex_array_square(self, Horizontal_beams=3, Vertical_beams=3, Xd=6, Yd=6, TC=1, polar=False):
        
        """
        Generate a square vortex array and return the complex amplitude.
        Parameters:
            
            Horizontal_beams: number of horizontal beams.(odd only temporarily)
            Vertical_beams: number of vertical beams.(odd only temporarily)
            Xd,Yd: distance between adjacent sub-beams.
            w0: beam waist.
            
        Returns:
            ndarray: 2-D complex amplitude.
        """
        
        if not polar:
            
            # Establish the cartesian coordinate.
            x = np.linspace(-self.WindowSize/2,self.WindowSize/2,self.N)
            y = np.linspace(-self.WindowSize/2,self.WindowSize/2,self.N)
            x,y = np.meshgrid(x,y)
            Xd = Xd*self.w0
            Yd = Yd*self.w0
            Amp = np.zeros((self.N,self.N),dtype=complex)
            M = int((Horizontal_beams-1)/2)
            N = int((Vertical_beams-1)/2)
            for i in range(-M,M+1):
                for j in range(-N,N+1):
    
                    Amp_n0 = (((x-i*Xd)+1j*np.sign(TC)*(y-j*Yd))
                    **abs(TC)*np.exp(-((x-i*Xd)**2+(y-j*Yd)**2)/self.w0**2))
                    Amp += Amp_n0
        else:
            # Establish the polar coordinate.
            r = np.linspace(0,self.WindowSize*np.sqrt(2)/2,self.N)
            phi = np.linspace(0,2*np.pi,self.N)
            r,phi = np.meshgrid(r,phi)
            
            x = r*np.cos(phi)
            y = r*np.sin(phi)
            
            Xd = Xd*self.w0
            Yd = Yd*self.w0
            Amp = np.zeros((self.N,self.N),dtype=complex)
            M = int((Horizontal_beams-1)/2)
            N = int((Vertical_beams-1)/2)
            for i in range(-M,M+1):
                for j in range(-N,N+1):
    
                    Amp_n0 = (((x-i*Xd)+1j*np.sign(TC)*(y-j*Yd))
                    **abs(TC)*np.exp(-((x-i*Xd)**2+(y-j*Yd)**2)/self.w0**2))
                    Amp += Amp_n0
            
        return Amp
    
    def vortex_array_spherical(self, Radial_beams=6, r0=6, TC=1, polar=False):
        
        """
        Generate a spherical vortex array and return the complex amplitude.
        Parameters:
            Radial_beams: number of sub-beams.
            r0: distance between sub-beams' center and axis.
            w0: beam waist.
            D: windowsize.
            
        Returns:
            ndarray: 2-D complex amplitude.
        """
        
        if not polar:
            
            # Establish the cartesian coordinate.
            x = np.linspace(-self.WindowSize/2,self.WindowSize/2,self.N)
            y = np.linspace(-self.WindowSize/2,self.WindowSize/2,self.N)
            x,y = np.meshgrid(x,y)
            alpha0 = 2*np.pi/Radial_beams
            Amp = np.zeros((self.N, self.N),dtype=complex)
            for n in range(1,Radial_beams+1):
                
                Xd = r0*self.w0*np.cos(n*alpha0)
                Yd = r0*self.w0*np.sin(n*alpha0)
                Amp_n0 = (((x-Xd)+1j*np.sign(TC)*(y-Yd))**abs(TC)
                *np.exp(-((x-Xd)**2+(y-Yd)**2)/self.w0**2))
                Amp += Amp_n0
            
        else:
            
            # Establish the polar coordinate.
            r = np.linspace(0, self.WindowSize*np.sqrt(2), self.N)
            phi = np.linspace(0, 2*np.pi, self.N)
            r,phi = np.meshgrid(r,phi)
            x = r*np.cos(phi)
            y = r*np.sin(phi)
            
            alpha0 = 2*np.pi/Radial_beams
            Amp = np.zeros((self.N,self.N),dtype=complex)
            for i in range(Radial_beams):
                
                Xd = r0*self.w0*np.cos(i*alpha0)
                Yd = r0*self.w0*np.sin(i*alpha0)
                Amp_n0 = (((r*np.cos(phi)-Xd)+1j*np.sign(TC)*(r*np.sin(phi)-Yd))**abs(TC)
                *np.exp(-((r*np.cos(phi)-Xd)**2+(r*np.sin(phi)-Yd)**2)/self.w0**2))
                Amp += Amp_n0
            
        return Amp     
    
    def vortex_array_spherical_expansion(self, Radial_beams=6, r0=6, TC=1, polar=False):
        
        # Establish the cartesian coordinate.
        x = np.linspace(-self.WindowSize/2,self.WindowSize/2,self.N)
        y = np.linspace(-self.WindowSize/2,self.WindowSize/2,self.N)
        x,y = np.meshgrid(x,y)
        alpha_0 = 2*np.pi/Radial_beams
        r0 *= self.w0
        Sum = np.zeros((self.N, self.N),dtype=complex)
        for n in range(1,Radial_beams+1):
            alpha_n = n*alpha_0
            Cos = r0*np.cos(alpha_n)
            Sin = r0*np.sin(alpha_n)
            for r1 in range(abs(TC)+1):
                C_TC_r1 = comb(abs(TC),r1)
                for r2 in range(abs(TC)-r1+1):
                    C_2 = comb(abs(TC)-r1,r2)
                    for r3 in range(r1+1):
                        
                        Sum += (C_TC_r1*C_2*comb(r1,r3)*(-Cos)**r2*(-Sin)**r3
                        *(1j*np.sign(TC))**r1*x**(abs(TC)-r1-r2)*y**(r1-r3)*np.exp(-x**2/self.w0**2+2*Cos/self.w0**2*x)
                        *np.exp(-y**2/self.w0**2+2*Sin/self.w0**2*y))
                        
        Amp = np.exp(-r0**2/self.w0**2)*Sum
        return Amp
# In[]: 子光束拓扑荷不同的情况.
    def vortex_array_spherical_TC(self, Radial_beams=6, r0=6, TC=None, polar=False):
    
        """
        Generate a spherical vortex array and return the complex amplitude.
        Parameters:
            Radial_beams: number of sub-beams.
            r0: distance between sub-beams' center and axis.
            w0: beam waist.
            D: windowsize.
            
        Returns:
            ndarray: 2-D complex amplitude.
        """
        
        if TC== None:
            TC = np.zeros((Radial_beams),dtype='int')
            for i in range(Radial_beams):
                TC[i] = i+1
        else:
            pass
            
        if not polar:
            
            # Establish the cartesian coordinate.
            x = np.linspace(-self.WindowSize/2, self.WindowSize/2, self.N)
            y = np.linspace(-self.WindowSize/2, self.WindowSize/2, self.N)
            x,y = np.meshgrid(x,y)
            alpha0 = 2*np.pi/Radial_beams
            Amp = np.zeros((self.N,self.N),dtype=complex)
            for i in range(Radial_beams):
                
                Xd = r0*self.w0*np.cos(i*alpha0)
                Yd = r0*self.w0*np.sin(i*alpha0)
                Amp_n0 = (((x-Xd)+1j*np.sign(TC[i])*(y-Yd))**abs(TC[i])
                *np.exp(-((x-Xd)**2+(y-Yd)**2)/self.w0**2)*(np.sqrt(2)/self.w0)**(TC[i]+1)/np.sqrt(gamma(TC[i]+1)))
    
    #            Amp_n0 /= np.max(Amp_n0)     # Normalization            
                Amp += Amp_n0
            
        else:
            
            # Establish the polar coordinate.
            r = np.linspace(0, self.WindowSize*np.sqrt(2), self.N)
            phi = np.linspace(0, 2*np.pi, self.N)
            r,phi = np.meshgrid(r,phi)
            x = r*np.cos(phi)
            y = r*np.sin(phi)
            
            alpha0 = 2*np.pi/Radial_beams
            Amp = np.zeros((self.N,self.N),dtype=complex)
            for i in range(Radial_beams):
                
                Xd = r0*self.w0*np.cos(i*alpha0)
                Yd = r0*self.w0*np.sin(i*alpha0)
                Amp_n0 = (((r*np.cos(phi)-Xd)+1j*np.sign(TC[i])*(r*np.sin(phi)-Yd))**abs(TC[i])
                *np.exp(-((r*np.cos(phi)-Xd)**2+(r*np.sin(phi)-Yd)**2)/self.w0**2)*(np.sqrt(2)/self.w0)**(abs(TC[i])+1)/np.sqrt(gamma(abs(TC[i])+1)))
                
    #            Amp_n0 /= np.max(Amp_n0)     # Normalization
                Amp += Amp_n0
            
        return Amp     
    
    def vortex_array_square_TC(self, Horizontal_beams=3, Vertical_beams=3, Xd=6, Yd=6, TC=None, polar=False):
        
        """
        Generate a square vortex array and return the complex amplitude.
        Parameters:
            
            Horizontal_beams: number of horizontal beams.(odd only temporarily)
            Vertical_beams: number of vertical beams.(odd only temporarily)
            Xd,Yd: distance between adjacent sub-beams.
            w0: beam waist.
            
        Returns:
            ndarray: 2-D complex amplitude.
        """
        
        if TC==None:
            TC = np.zeros((Horizontal_beams,Vertical_beams),dtype='int')
            for i in range(Horizontal_beams):
                for j in range(Vertical_beams):
                    TC[i,j] = i+j+1
    
        if not polar:
            
            # Establish the cartesian coordinate.
            x = np.linspace(-self.WindowSize/2, self.WindowSize/2, self.N)
            y = np.linspace(-self.WindowSize/2, self.WindowSize/2, self.N)
            x,y = np.meshgrid(x,y)
            Xd = Xd*self.w0
            Yd = Yd*self.w0
            Amp = np.zeros((self.N,self.N),dtype=complex)
            M_ = int((Horizontal_beams-1)/2)
            N_ = int((Vertical_beams-1)/2)
            for i in range(-M_,M_+1):
                for j in range(-N_,N_+1):
    
                    Amp_n0 = (((x-i*Xd)+1j*np.sign(TC[i+M_,j+N_])*(y-j*Yd))
                    **abs(TC[i+M_,j+N_])*np.exp(-((x-i*Xd)**2+(y-j*Yd)**2)/self.w0**2)
                    *(np.sqrt(2)/self.w0)**(abs(TC[i+M_,j+N_])+1)/np.sqrt(gamma(abs(TC[i+M_,j+N_])+1)))
                    
                    Amp += Amp_n0
                    
        return Amp
