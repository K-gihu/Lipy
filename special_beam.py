# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 08:49:42 2018

@author: FAREWELL
"""

import numpy as np
from matplotlib import pyplot as plt
from scipy import special
import LightPipes as LP
from math import factorial as Fac
import basic_tools
import basic_math

class special_beam(basic_tools.basic_tools,basic_math.basic_math):
    
    # Don't define __init__() function here, python will automatically call the initialization-function 
    # of superclass when constructing object.
    
# In[]:
    
    def Hermitte_GaussianBeam(self, l=0, m=0, z=0, polar=False):
        
        """
        Calculate a Hermitte Gaussian Beam and return the complex amplitude.
        Parameters:
            l,m: coefficients.
            wvl: light wavelength.
            z: propagation distance.
            n: ??(I forgot).
            polar: 
                True, polar coordinate
                False, cartesian coordinate
                (default set as False)
            
        Returns:
            ndarray: 2-D complex amplitude.
        """
        
        if not polar:
            
            x = np.linspace(-self.WindowSize/2,self.WindowSize/2,self.N)
            y = np.linspace(-self.WindowSize/2,self.WindowSize/2,self.N)
            x,y = np.meshgrid(x,y)
            r = np.sqrt(x**2+y**2)
            phi = np.arctan2(y,x)
            
            z0 = np.pi*self.w0**2/self.wvl
            wvn = 2*np.pi/self.wvl
            if z == 0:
#                Amp = self.Hermitte_Gaussian_Func(l,
#                            np.sqrt(2)*x/self.w0)*self.Hermitte_Gaussian_Func(m,
#                            np.sqrt(2)*y/self.w0)
                Amp = self.Hermitte_Gaussian_Func(l,
                            np.sqrt(2)*x/self.w0)*self.Hermitte_Gaussian_Func(m,
                            np.sqrt(2)*y/self.w0)*np.exp(-r**2/self.w0**2)
            else:
                Wz = self.w0*np.sqrt(1+(z/z0)**2)
                Rz = z*(1+(z0/z)**2)
                zeta = np.arctan2(z,z0)
                Amp = self.w0/Wz*self.Hermitte_Gaussian_Func(l,
                            np.sqrt(2)*x/Wz)*self.Hermitte_Gaussian_Func(m,
                            np.sqrt(2)*y/Wz)*np.exp(-1j*wvn*z-1j*wvn*r**2/(2*Rz)
                            +1j*(l+m+1)*zeta)*np.exp(-r**2/self.w0**2)
                
        else:
            r = np.linspace(0,np.sqrt(2)*self.WindowSize/2,self.N)
            phi = np.linspace(0,2*np.pi,self.N)
            r,phi = np.meshgrid(r,phi)
            x = r*np.cos(phi)
            y = r*np.sin(phi)
            
            z0 = np.pi*self.w0**2/self.wvl
            wvn = 2*np.pi/self.wvl
            if z == 0:
                Amp = self.Hermitte_Gaussian_Func(l,
                            np.sqrt(2)*x/self.w0)*self.Hermitte_Gaussian_Func(m,
                            np.sqrt(2)*y/self.w0)*np.exp(-r**2/self.w0**2)
            else:
                Wz = self.w0*np.sqrt(1+(z/z0)**2)
                Rz = z*(1+(z0/z)**2)
                zeta = np.arctan2(z,z0)
                Amp = self.w0/Wz*self.Hermitte_Gaussian_Func(l,
                            np.sqrt(2)*x/Wz)*self.Hermitte_Gaussian_Func(m,
                            np.sqrt(2)*y/Wz)*np.exp(-1j*wvn*z-1j*wvn*r**2/(2*Rz)
                            +1j*(l+m+1)*zeta)*np.exp(-r**2/self.w0**2)
            
        return Amp
       
# In[]:
    def Laguerre_GaussianBeam(self, l=1, p=0, simple=True, polar = False):
        
        """
        Calculate a Laguerre Gaussian Beam in the z = 0 plane and return the complex amplitude.
        Parameters:
            
            l: topological charge.
            p: radial index.
            simple: choose simple form or complex form, default set as True.
            polar: 
                True, polar coordinate
                False, cartesian coordinate
                (default set as False)
        
        Returns:
            ndarray: 2-D complex amplitude.            
        """
        
        if simple:
            if not polar:
                
                x = np.linspace(-self.WindowSize/2,self.WindowSize/2,self.N)
                y = np.linspace(-self.WindowSize/2,self.WindowSize/2,self.N)
                x,y = np.meshgrid(x,y)
                r = np.sqrt(x**2+y**2)
                phi = np.arctan2(y,x)
                
                Amp = (r/self.w0)**abs(l)*np.exp(-r**2/self.w0**2)*np.exp(1j*l*phi)
                
            else:
                r = np.linspace(0,np.sqrt(2)*self.WindowSize,self.N)
                phi = np.linspace(0,2*np.pi,self.N)
                r,phi = np.meshgrid(r,phi)
                
                Amp = (r/self.w0)**abs(l)*np.exp(-r**2/self.w0**2)*np.exp(1j*l*phi)
        else:
            L = special.genlaguerre(p,l)    # declare the Laguerre polynomial
            if not polar:
                
                x = np.linspace(-self.WindowSize/2,self.WindowSize/2,self.N)
                y = np.linspace(-self.WindowSize/2,self.WindowSize/2,self.N)
                x,y = np.meshgrid(x,y)
                r = np.sqrt(x**2+y**2)
                phi = np.arctan2(y,x)
               
                A = np.sqrt(2*Fac(p)/np.pi/Fac(p+np.abs(l)))    # normalized amplitude
                
                Amp = (A/self.w0*(np.sqrt(2)*r/self.w0)**(np.abs(l))*L(2*r**2/self.w0**2)
                *np.exp(-r**2/self.w0**2)*np.exp(1j*l*phi))
                
            else:
                r = np.linspace(0,np.sqrt(2)*self.WindowSize/2,self.N)
                phi = np.linspace(0,2*np.pi,self.N)
                r,phi = np.meshgrid(r,phi)
                
                A = np.sqrt(2*Fac(p)/np.pi/Fac(p+np.abs(l)))    # normalized amplitude
                Amp = (A/self.w0*(np.sqrt(2)*r/self.w0)**(np.abs(l))*L(2*r**2/self.w0**2)
                *np.exp(-r**2/self.w0**2)*np.exp(1j*l*phi))
            
        return Amp
    
    def Bessel_beam(self,order=0, polar=False):
        
        """
        Calculate a Bessel Beam and return the complex amplitude.
        Parameters:
            order: topological charge.
            polar: 
                True, polar coordinate
                False, cartesian coordinate
                (default set as False)
                
        Returns:
            ndarray: 2-D complex amplitude.     
        """
        
        self.WindowSize /= 10000   # Zoom in, otherwise return a strange figure
        if not polar:
            
            x = np.linspace(-self.WindowSize/2,self.WindowSize/2,self.N)
            y = np.linspace(-self.WindowSize/2,self.WindowSize/2,self.N)
            x,y = np.meshgrid(x,y)
            r = np.sqrt(x**2+y**2)
            phi = np.arctan2(y,x)
            
        else:
            
            r = np.linspace(0,np.sqrt(2)*self.WindowSize*self.w0,self.N)
            phi = np.linspace(0,2*np.pi,self.N)
            r,phi = np.meshgrid(r,phi)
            
        wvl = 632.8*LP.nm
        wvn = 2*np.pi/wvl
        Amp = special.jv(order,wvn*r)*np.exp(1j*order*phi)
        
        return Amp 
       
# =============================================================================
# Attention：Bessel beam is different from Bessel gaussian beam.
# Bessel beam is an ideal beam and cannot generate in laboratory because of the divergence of energy integral,
# thus laser beam generated practically is Bessel gaussian beam (BGB), 
# similar with Bessel beam，the BGB is also characterized by diffraction and self-healing.
# =============================================================================
# In[]: 
    def Bessel_Gaussian_beam(self,order = 0,polar = False):
        
        """
        Parameters:
            order: topological charge.
            polar: 
                True, polar coordinate
                False, cartesian coordinate
                (default set as False)
        Returns:
            ndarray: 2-D complex amplitude.
        """
        WindowSize = 0.0005
        if not polar:
            
            x = np.linspace(-WindowSize*self.w0,WindowSize*self.w0,self.N)
            y = np.linspace(-WindowSize*self.w0,WindowSize*self.w0,self.N)
            x,y = np.meshgrid(x,y)
            r = np.sqrt(x**2+y**2)
            phi = np.arctan2(y,x)
            
        else:
            
            r = np.linspace(0,np.sqrt(2)*WindowSize*self.w0,self.N)
            phi = np.linspace(0,2*np.pi,self.N)
            r,phi = np.meshgrid(r,phi)
        
        wvl = 632.8*LP.nm
        wvn = 2*np.pi/wvl
        Amp = special.jv(order,wvn*r)*np.exp(-r**2/self.w0**2)*np.exp(1j*order*phi)
        
        return Amp 
        
    def Airy_Gaussian_beam(self, r0, a=0.05, b=0.3, TC=1, polar=False):
        
        """
        Calculate a Airy gaussian beam and return the complex amplitude.
        Parameters:
            r0: Radius of the primary Airy ring;
            a: 0 <= a < 1 exponential truncation factor which determines the propagation distance along the radial direction;
            b: distribution factor parameter;
        Returns:
            ndarray: 2-D complex amplitude.
        """
        
        if not polar:
            
            x = np.linspace(-self.WindowSize/2,self.WindowSize/2,self.N)
            y = np.linspace(-self.WindowSize/2,self.WindowSize/2,self.N)
            x,y = np.meshgrid(x,y)
            
            r = np.sqrt(x**2+y**2)
            phi = np.arctan2(y,x)
            
        else:
            
            r = np.linspace(0,np.sqrt(2)*self.WindowSize*self.w0,self.N)
            phi = np.linspace(0,2*np.pi,self.N)
            r,phi = np.meshgrid(r,phi)
            
        R = (r0-r)/self.w0
        Amp = special.airy(R/b)[0]*np.exp(a*R/b)*np.exp(-R**2)*r**abs(TC)*np.exp(1j*TC*phi)
        
        return Amp
            
    def single_vorticity(self, m=1, xn=0, yn=0, polar=False):
        
        """
        Calculate a single vorticity vortex beam and return the complex amplitude.
        Parameters:
            m: topological charge.
            xn,yn: vortex centroid coordinate.
            polar:          
                True, polar coordinate
                False, cartesian coordinate
                (default set as False)
        Returns:
            ndarray: 2-D complex amplitude.
        """
        
        if not polar:
            
            x = np.linspace(-self.WindowSize/2,self.WindowSize/2,self.N)
            y = np.linspace(-self.WindowSize/2,self.WindowSize/2,self.N)
            x,y = np.meshgrid(x,y)
            r = np.sqrt(x**2+y**2)
            
        else:
            
            r = np.linspace(0,np.sqrt(2)*self.WindowSize/2,self.N)
            phi = np.linspace(0,2*np.pi,self.N)
            r,phi = np.meshgrid(r,phi)
            x = r*np.cos(phi)
            y = r*np.sin(phi)
            
        Amp = (((x-xn)+1j*np.sign(m)*(y-yn))**np.abs(m)
                *np.exp(-r**2/self.w0**2))
        return Amp

    def double_vortices(self,m1=1, m2=1, d1=1, d2=0, polar=False): 
        
        """
        Calculate the vortex beam with double-vortices and return the complex amplitude.
        Parameters:
            m1,m2: topological charge of vortices.
            d1,d2：distance between vortex centroid and optical axis (unit:self.w0).
            polar: 
                True, polar coordinate
                False, cartesian coordinate
                (default set as False)
        Returns:
            ndarray: 2-D complex amplitude.
        """
       
        if not polar:
            x = np.linspace(-self.WindowSize/2,self.WindowSize/2,self.N)
            y = np.linspace(-self.WindowSize/2,self.WindowSize/2,self.N)
            x,y = np.meshgrid(x,y)
            r = np.sqrt(x**2+y**2)
            x1 = d1*self.w0/2
            x2 = -x1
            y1 = d2*self.w0/2
            y2 = -y1
            Amp = (((x-x1)+1j*np.sign(m1)*(y-y1))**np.abs(m1))*(((x-x2)+1j*np.sign(m2)*(y-y2))**np.abs(m2))*np.exp(-r**2/self.w0**2)
        else:
            r = np.linspace(0,np.sqrt(2)*self.WindowSize,self.N)
            phi = np.linspace(0,2*np.pi,self.N)
            r,phi = np.meshgrid(r,phi)
            x = r*np.cos(phi)
            y = r*np.sin(phi)
            x1 = d1*self.w0/2
            x2 = -x1
            y1 = d2*self.w0/2
            y2 = -y1
            Amp = (((x-x1)+1j*np.sign(m1)*(y-y1))**np.abs(m1))*(((x-x2)+1j*np.sign(m2)*(y-y2))**np.abs(m2))*np.exp(-r**2/self.w0**2)
        
        return Amp   
        
    def multi_ring_vortices(self,m1=1,m2=10,plot=True):
        
        """
        Calculate the vortex beam with multi_ring_vortices and return the complex amplitude.
        Parameters:
            m1,m2: topological charge of vortices.
            plot: set True to display the gray-scale map and return the AxisImage, 
                set False to return the normalized intensity.
        Returns:
            ndarray: 2-D complex amplitude.
        """
		
        x = np.linspace(-self.WindowSize/2,self.WindowSize/2,self.N)
        y = np.linspace(-self.WindowSize/2,self.WindowSize/2,self.N)
        x,y = np.meshgrid(x,y)
        r = np.sqrt(x**2+y**2)
        phi = np.arctan2(y,x)
        		
        Amp1 = np.exp(-r**2/self.w0**2)*(r/self.w0)**m1*np.exp(-1j*m1*phi)
        Amp2 = np.exp(-r**2/self.w0**2)*(r/self.w0)**m2*np.exp(-1j*m2*phi)
        Intensity_1 = (Amp1*Amp1.conjugate()).real
        Intensity_2 = (Amp2*Amp2.conjugate()).real
        Intensity = Intensity_1/np.max(Intensity_1)+Intensity_2/np.max(Intensity_2)
        if plot:
            im = plt.imshow(Intensity,cmap='gray',interpolation='bicubic')
            return im
        else:
            return Intensity
            
    def Gaussian_beam(self,polar=False):
        
        """
        Calculate the simple Gaussian Beam without phase(but with a imaginary part) and return the complex amplitude.
        Parameters:
            polar: 
                True, polar coordinate
                False, cartesian coordinate
                (default set as False)
        Returns:
            ndarray: 2-D complex amplitude.
        """
        
        if not polar:
            
            x = np.linspace(-self.WindowSize/2,self.WindowSize/2,self.N)
            y = np.linspace(-self.WindowSize/2,self.WindowSize/2,self.N)
            x,y = np.meshgrid(x,y)
            r = np.sqrt(x**2+y**2)
            Amp = np.exp(-r**2/self.w0**2)*np.exp(1j*0)
            
        else:
            
            r = np.linspace(0,np.sqrt(2)*self.WindowSize*self.w0,self.N)
            phi = np.linspace(0,2*np.pi,self.N)
            r,phi = np.meshgrid(r,phi)
            Amp = np.exp(-r**2/self.w0**2)*np.exp(1j*0)
            
        return Amp
    
    def Flat_topped_gaussian_beam(self,N0):
        
        """
        N0: beam order
        """
        
        x = np.linspace(-self.WindowSize/2,self.WindowSize/2,self.N)
        y = np.linspace(-self.WindowSize/2,self.WindowSize/2,self.N)
        x,y = np.meshgrid(x,y)
        r = np.sqrt(x**2+y**2)
        Amp = np.zeros((self.N,self.N),dtype='complex')
        Amp = np.exp(-r**2/self.w0**2)
        for n in range(1,N0+1):
            Amp += (-1)**(n-1)/N0*special.comb(N0,n)*np.exp(-n*r**2/self.w0**2)
            
        return Amp
            
            