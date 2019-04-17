# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 10:40:27 2018

@author: FAREWELL
"""
import numpy as np
from matplotlib import pyplot as plt
from scipy.fftpack import fft
import LightPipes as LP
import math

import Chinese_Lib
Chinese_Lib.set_ch()

class basic_math():
    
    def __init__(self,wvl=1550*LP.nm, w0=0.01*LP.m, WindowSize=0.1*LP.m, N=512, Cn2=1e-15, L0=5*LP.m, l0=5*LP.mm, alpha=11./3):
        
        self.wvl = wvl   # wavelength
        self.w0 = w0     # beam waist (beam radius)
        self.WindowSize = WindowSize    # plane size of light field
        self.N = N       # grid sampling points
        self.Cn2 = Cn2
        self.L0 = L0
        self.l0 = l0
        self.alpha = alpha
    
# In[]: Some mathematical tools; 
        
    def Hermitte_Poly(self,n = 1,x = 1):
        
        """Calculate the Hermitte Polynomial"""
        
        if n == 0:
            H = 1
        elif n == 1:
            H = 2*x
        else:
            H0 = 1
            H1 = 2*x
            for i in range(2,n+1):
                H = 2*x*H1 - 2*(i-1)*H0
                H0 = H1
                H1 = H
        return H
        
# In[]:
        
    def Hermitte_Gaussian_Func(self,l,u):
        
        """Calculate the Hermitte Gaussian Function"""
        
        G = self.Hermitte_Poly(l,u)*np.exp(-u**2/2)
        return G
    
    def double_trapz(self, I, dx=1.0, dy=1.0):
    
        """
        Calculate double numerical integration.
        Parameters:
            
            I: 2-D matrix
            dx,dy: interval of the integrand
            
        Return:
            Integral results.
        """
        
        I = np.trapz(I,dx=dx,axis=0)
        Res = np.trapz(I,dx=dy,axis=0)
        
        return Res
    
# In[]: Module containing useful FFT based function and classes
        
    def ft(self,data, delta):
        
        """
        A properly scaled 1-D FFT
    
        Parameters:
            data (ndarray): An array on which to perform the FFT
            delta (float): Spacing between elements
    
        Returns:
            ndarray: scaled FFT
        """
        
        DATA = np.fft.fftshift(
                np.fft.fft(
                        np.fft.fftshift(data, axes=(-1))),
                axes=(-1)) * delta
        return DATA

    def ift(self,DATA, delta_f):
        
        """
        Scaled inverse 1-D FFT
    
        Parameters:
            DATA (ndarray): Data in Fourier Space to transform
            delta_f (ndarray): Frequency spacing of grid
    
        Returns:
            ndarray: Scaled data in real space
        """
    
        data = np.fft.ifftshift(
                np.fft.ifft(
                        np.fft.ifftshift(DATA, axes=(-1))),
                axes=(-1)) * len(DATA) * delta_f
    
        return data


    def ft2(self,data, delta):
        
        """
        A properly scaled 2-D FFT
    
        Parameters:
            data (ndarray): An array on which to perform the FFT
            delta (float): Spacing between elements
    
        Returns:
            ndarray: scaled FFT
        """
    
        DATA = np.fft.fftshift(
                np.fft.fft2(
                        np.fft.fftshift(data, axes=(-1,-2))
                        ), axes=(-1,-2)
                )*delta**2
    
        return DATA

    def ift2(self,DATA, delta_f):
        
        """
        Scaled inverse 2-D FFT
    
        Parameters:
            DATA (ndarray): Data in Fourier Space to transform
            delta_f (ndarray): Frequency spacing of grid
    
        Returns:
            ndarray: Scaled data in real space
        """
        
        N = DATA.shape[0]
        g = np.fft.ifftshift(
                np.fft.ifft2(
                        np.fft.ifftshift(DATA))) * (N * delta_f)**2
    
        return g

    def rft(self,data, delta):
        
        """
        A properly scaled real 1-D FFT
    
        Parameters:
            data (ndarray): An array on which to perform the FFT
            delta (float): Spacing between elements
    
        Returns:
            ndarray: scaled FFT
        """
        
        DATA = np.fft.fftshift(
                np.fft.rfft(
                        np.fft.fftshift(data, axes=(-1))),
                axes=(-1)) * delta
        return DATA

    def irft(self,DATA, delta_f):
        
        """
        Scaled real inverse 1-D FFT
    
        Parameters:
            DATA (ndarray): Data in Fourier Space to transform
            delta_f (ndarray): Frequency spacing of grid
    
        Returns:
            ndarray: Scaled data in real space
        """
    
        data = np.fft.ifftshift(
                np.fft.irfft(
                        np.fft.ifftshift(DATA, axes=(-1))),
                axes=(-1)) * len(DATA) * delta_f
    
        return data

    def rft2(self,data, delta):
        
        """
        A properly scaled, real 2-D FFT
    
        Parameters:
            data (ndarray): An array on which to perform the FFT
            delta (float): Spacing between elements
    
        Returns:
            ndarray: scaled FFT
        """
        
        DATA = np.fft.fftshift(
                np.fft.rfft2(
                        np.fft.fftshift(data, axes=(-1,-2))
                        ), axes=(-1,-2)
                )*delta**2
    
        return DATA

    def irft2(self,DATA, delta_f):
        
        """
        Scaled inverse real 2-D FFT
    
        Parameters:
            DATA (ndarray): Data in Fourier Space to transform
            delta_f (ndarray): Frequency spacing of grid
    
        Returns:
            ndarray: Scaled data in real space
        """
        
        data = np.fft.ifftshift(
                np.fft.irfft2(
                        np.fft.ifft2(DATA), s=(-1,-2)
                        )
                ) * (DATA.shape[0]*delta_f)**2
        return data
    
# In[]: Module containing useful matrix dealing functions;
        
    def matrix_tanh(self,I):
        
        """
        Take the tangent of matrix
        Parameters:
            I: Matrix to be processed
            
        Returns:
            ndarray: Matrix
            
        """
        (raw,column) = I.shape
        for i in range(raw):
            for j in range(column):
                I[i,j] = math.tanh(I[i,j])
        return I
    
    def Matrix_pow(self,I,n):  
        
        """
        Take the power of the matrix
        Parameters:
            I: Matrix to be processed
            n: power exponent
        Returns:
            ndarray: Matrix
        """
        
        x,y = I.shape
        for i in range(0,x):
            for j in range(0,y):
                try:
                    I[i,j] = math.pow(I[i,j],n)
                except:
                    I[i,j] = I[i-1,j]
        return I
    
    def Matrix_Int(self,I):
        
        """
        Take the round-numbers of all the elements of matrix
        Parameters:
            I: Matrix to be processed
        Returns:
            ndarray: Matrix
        """
        x,y = I.shape
        for i in range(0,x):
            for j in range(0,y):
                I[i,j] = int(I[i,j])
        return I
    
    def Laguerre_Poly(self,n,alpha,x):
        
        """
        Calculate the Laguerre Polynomial
        Some error exists,use scipy.special.genlaguerre() instead.
        Parameters:
        Returns:
        """
        
        L = 0
        for k in range(0,n):
            L += ((-1)**k*math.factorial(n+alpha)/math.factorial(k)/
            math.factorial(n-k)/math.factorial(alpha+k)*x**k)
            
        return L
 
    def getPositon(self,I,value=None):    
          
        """
        Find the closest number of the given number (default set as the maximum of the matrix) in the matrix, 
        and return the location.
        """
        
        raw, column = I.shape   # get the raw and column of a matrix  
        
        if value == None:       # default circumstance
            
            _position = np.argmax(I)
            m, n = divmod(_position, column)
               
        else:
            
            _position = np.where(value)      # get the index of max in the matrix
            try:
                m, n, Residual = self.Findproximate(I,value)
            except:
                m, n = (0,0)
                
        return m,n
        
    def Findproximate(self, I, value):
        
        """
        Find the closest number of the given number in the matrix, then return the location and residual.
        """
        
        raw, column = I.shape
        Residual = abs(I[0,0]-value)
        m = 0
        n = 0
        for i in range(raw):
            for j in range(column):
                
                if abs(I[i,j]-value) < Residual:
                    Residual = abs(I[i,j]-value)
                    m = i
                    n = j
                else:
                    pass
                
        if Residual > (1e-3*np.max(I)):
            
            print("Error!value not found!")
            return False
        else:
            return m,n,Residual
        
    def cart2pol(self,x, y):
        
        """
        convert the Cartesian coordinates to Polar coordinates;
        """
        
        theta = np.arctan2(y, x)
        rho = np.sqrt(x**2 + y**2)
        return (theta, rho)

    def complex_randn(self,N):
        
        """
        genenrate a N-dimensional complex random matrix;
        """
        
        rand_Matrix = np.random.randn(N,N)+1j*np.random.randn(N,N)
        return rand_Matrix
    
    def OneDimensional_Integral(self,func,dx):
        
        """
        Compute a simple definite integral, the input func is supposed to be a one-dimensional-array.
        """
        
        I = 0
        for i in func:
            I += i*dx
        return I

    def spectral_factorization(self,signal,plot=False):
        
        """
        Take the spectral decomposition of a one-dimensional signal.
        """
        
        fy = abs(fft(signal))/len(signal)
        fx = np.arange(len(signal))
        if plot:
            plt.bar(left=fx[:10],height=fy[:10],width=0.5,align='center',yerr=0.000001)

    def Horizontal_Addition(self,Matrix):
        
        """
        Add all the numbers to the first number in each line.
        """
        
        (x,y) = Matrix.shape
        New_Matrix = np.zeros((x,1),dtype=Matrix.dtype)
        for i in range(0,x):
            for j in range(0,y):
                New_Matrix[i,0] += Matrix[i,j] 
                
        return New_Matrix

    def Vertical_Addition(self,Matrix):
        
        """
        Add all the numbers to the first number in each column.
        """
        
        (x,y) = Matrix.shape
        New_Matrix = np.zeros((1,y),dtype=Matrix.dtype)
        for i in range(0,y):
            for j in range(0,x):
                New_Matrix[0,i] += Matrix[j,i] 
                
        return New_Matrix

    def Bound_Norm(self,Matrix):
        
        """
        Compute the bound norm of a vector.
        """
        
        (N,) = Matrix.shape
        Res = 0
        for i in range(N):
            Res += Matrix[i]**2
        Res = np.sqrt(Res)/N
        
        return Res
    

