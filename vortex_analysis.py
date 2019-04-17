# -*- coding: utf-8 -*-
"""
Created on Sun May 20 20:18:46 2018

@author: lionking
"""

import numpy as np
import LightPipes as LP
import basic_tools
import basic_math
from math import factorial as Fac
from scipy.special import comb, gamma, gammainc
from time import time 

# =============================================================================
# This module contains the results of vortex beam dealt with analysis methods.
# =============================================================================
class vortex_analysis(basic_tools.basic_tools,basic_math.basic_math):
    
# =============================================================================
# Compute common integrals: x^n*exp(-px^2+q*x).
# =============================================================================
    def Frequently_used_integral(self, n, p, q):
    
        """
        Compute common integrals: x^n*exp(-px^2+q*x).
        """
        Sum = 0.0
        for i in range(n//2):
            Sum += (p/4/q**2)**i/Fac(n-2*i)/Fac(i)
        result = Fac(n)*np.exp(q**2/p)*np.sqrt(np.pi/p)*(q/p)**n*Sum
        
        return result   
    
    def double_trapz(self, I, dx=1.0, dy=1.0):
    
        """
        Calculate double integral through trapz().
        Parameters:
            I: 2-D matrix to be integrated.
        Returns:
            Integral results.
        """
        
        I = np.trapz(I,dx=dx,axis=0)
        Res = np.trapz(I,dx=dy,axis=0)
        
        return Res

    def Cn2_Slant(self, z, zeta, alpha):
    
        """
        Calculate atmospheric refractive index structure constant in slant path.
        Relative formula refer to Lu Fang's doctoral dissertation(3-6).
        """
    
        wvn = 2*np.pi/self.wvl
        v_rms = 21
        Cn2_0 = 1.7e-14
        A_alpha = gamma(alpha-1)*np.cos(alpha*np.pi/2)/(4*np.pi**2)
        Cn2 = 0.033/A_alpha*(np.sqrt(wvn/z))**(alpha-11/3)*(8.148e-56*v_rms**2*(z*np.cos(zeta))**10
                             *np.exp(-z*np.cos(zeta)/1000)+2.7e-16*np.exp(-z*np.cos(zeta)/1500)+
                             Cn2_0*np.exp(-z*np.cos(zeta)/100))
        
        return Cn2
    
    def M_non_Kolmogorov(self, alpha=11/3, Cn2=1e-15, L0=5*LP.m, l0=0.05*LP.m, z=500*LP.m, zeta=np.pi/2, 
                     SlantFlag=False):
        
        """
        Calculate M-factor in non-Kolmogororv turbulence.
        Parameters:
            
            alpha: generalized exponential factor
            Cn2: atmospheric refractive index structure constant
            L0: turbulence outer scale (1~10m)
            l0: turbulence inner scale (0.01~0.1m)
            z: propagation distance
            zeta: zenith angle (0~np.pi/2)
            wvl: light beam wavelength
            SlantFlag: slant path flag, if True call the Cn2_Slant sub-function,
                       otherwise use the Cn2 given in formal parameter list.
        Return:
            M-factor.
        """
        
        if SlantFlag:
            Cn2_Tilde = self.Cn2_Slant(z,zeta,alpha)        # Cn2 in a slant path
        else:
            Cn2_Tilde = Cn2       # turbulence intensity in a horizontal path
        
        wvn = 2*np.pi/self.wvl         # spatial wave number
        A_alpha = gamma(alpha-1)*np.cos(alpha*np.pi/2)/4/np.pi/np.pi    # A(alpha)；
        C_alpha = (gamma((5-alpha)/2)*2*np.pi*A_alpha/3)**(1/(alpha-5))
        km = C_alpha/l0
        k0 = 2*np.pi/L0
        A = A_alpha*np.pi*np.pi*wvn*wvn*Cn2_Tilde*z/6/(alpha-2)
        Beta = (2*(k0**2-km**2)+alpha*km**2)*km**(2-alpha)
        C = gammainc(k0**2/km**2,2-alpha/2)
        M = A*(Beta*np.exp(k0**2/km**2)*C-2*k0**(4-alpha))
            
        return M
    
    def Astigmatic_Transformation(self, f=0.1*LP.m, TC=1, w=0.5*LP.mm):
                                      
        """
        Measure topological charge through cylindrical lens.
        Parameters:
            wvl: wavelength(nm)
            f: focal length
            TC: topological charge
            w0: beam radius
            
        Return: ampltude (pass through the cylindrical lens and propagate to focal plane).
        """
        
        wvn = 2*np.pi/self.wvl
        
        D = 5*w
        
        x = np.linspace(-D/2,D/2,self.N)
        y = np.linspace(-D/2,D/2,self.N)
        x,y = np.meshgrid(x,y)
        
        p1 = 1/w**2-1j*wvn/2/f
        p2 = 1/w**2
        q1 = -1j*wvn/2/f*x
        q2 = -1j*wvn/2/f*y
        
        Amp = (np.exp(1j*wvn*f)/(1j*self.wvl*f)*(1/w)**abs(TC)
        *np.exp(1j*wvn/2/f*(x**2+y**2))*np.exp(q1**2/p1)
        *np.exp(q2**2/p2)*np.sqrt(np.pi/p1)*np.sqrt(np.pi/p2))
        
        Sum = np.zeros((self.N,self.N),dtype='complex')
        
        for r in range(abs(TC)+1):
            for t1 in range((abs(TC)-r)//2+1):
                for t2 in range(r//2+1):
                    Sum += (comb(abs(TC),r)*(1j*np.sign(TC))**r*Fac(abs(TC)-r)*Fac(r)
                    *(q1/p1)**(abs(TC)-r)*(q2/p2)**r*(p1/4/q1**2)**t1*(p2/4/q2**2)**t2
                    /Fac(abs(TC)-r-2*t1)/Fac(t1)/Fac(r-2*t2)/Fac(t2))
                    
        Amp *= Sum
        
        return Amp
    
    def single_vorticity_freespace(self, z=500*LP.m, TC=1):
        
        """
        Single vortex beam propagation through analytical method.
        """
        x = np.linspace(-self.D/2,self.D/2,self.N)
        y = np.linspace(-self.D/2,self.D/2,self.N)
        x,y = np.meshgrid(x,y)
        wvn = 2*np.pi/self.wvl
        p = 1/(self.w0**2)-1j*wvn/2/z
        q1 = -1j*wvn*x/2/z
        q2 = -1j*wvn*y/2/z
        Sum = np.zeros((self.N,self.N),dtype='complex')
        Amp = np.exp(1j*wvn*z)/(1j*self.wvl*z)*np.exp(1j*wvn/2/z*(x**2+y**2))*np.exp((q1**2+q2**2)/p)*np.pi/p
        for r in range(abs(TC)+1):
            for t in range((abs(TC)-r)//2+1):
                for f in range(r//2+1):
                     Sum += ((comb(abs(TC),r)*(1j*np.sign(TC))**r*Fac(abs(TC)-r)*Fac(r)*
                     (q1/p)**(abs(TC)-r)*(q2/p)**r*(p/4/q1**2)**t*(p/4/q2**2)**f) / (Fac(abs(TC)-r-2*t)*
                     Fac(t)*Fac(r-2*f)*Fac(f)))
        Amp *= Sum
        
        return Amp  
    
    def single_vorticity_turb(self, D=0.2*LP.m, TC=3, z=500*LP.m):
    
        wvn = 2*np.pi/self.wvl
        
        x = np.linspace(-D/2,D/2,self.N)
        y = np.linspace(-D/2,D/2,self.N)
        x,y = np.meshgrid(x,y)
        
        M = self.M_non_Kolmogorov(alpha=self.alpha, Cn2=self.Cn2, L0=self.L0, l0=self.l0, z=z, zeta=np.pi/2, SlantFlag=False)
        
        p1 = 1/self.w0**2+M-1j*wvn/2/z
        p2 = 1/self.w0**2+M+1j*wvn/2/z-M**2/p1
        
        q1 = -1j*wvn/2/z*x
        q2 = -1j*wvn/2/z*y
        q3 = 1j*wvn/2/z*x+M*q1/p1
        q4 = 1j*wvn/2/z*y+M*q2/p1
        
        I = 1/self.wvl**2/z**2*np.pi*np.pi/p1/p2
        Sum = np.zeros((self.N,self.N),dtype='complex')
        for r1 in range(abs(TC)+1):
            for r2 in range(abs(TC)+1):
                for t1 in range((abs(TC)-r1)//2+1):
                    for t2 in range(r1//2+1):
                        for m1 in range(abs(TC)-r1-2*t1+1):
                            for m2 in range(r1-2*t2+1):
                                N3 = 2*abs(TC)-r1-r2-2*t1-m1
                                N4 = r1+r2-2*t2-m2
                                for t3 in range(N3//2+1):
                                    for t4 in range(N4//2+1):
                                        
                                        Sum += ((1j*np.sign(TC))**r1*(-1j*np.sign(TC))**r2*q1**m1*q2**m2*4**(-t1-t2-t3-t4)*M**(abs(TC)-m1-m2-2*t1-2*t2)
                                        *p1**(t1+t2-abs(TC))*p2**(t3+t4-N3-N4)*q3**(N3-2*t3)*q4**(N4-2*t4)*comb(abs(TC),r1)*comb(abs(TC),r2)
                                        *comb(abs(TC)-r1-2*t1,m1)*comb(r1-2*t2,m2)*np.exp((q1**2+q2**2)/p1)*np.exp((q3**2+q4**2)/p2)
                                        *Fac(abs(TC)-r1)*Fac(r1)*Fac(N3)*Fac(N4)/Fac(abs(TC)-r1-2*t1)/Fac(t1)/Fac(r1-2*t2)/Fac(t2)/Fac(N3-2*t3)/Fac(t3)
                                        /Fac(N4-2*t4)/Fac(t4))
        I *= Sum                         
        
        return I.real
    
    def Double_vortices(self, x1=0.8, x2=-0.8, y1=0, y2=-0, w0=1*LP.cm, TC1=1, TC2=-1):
        
        x1 *= w0
        x2 *= w0
        y1 *= w0
        y2 *= w0
        
        x = np.linspace(-self.D/2,self.D/2,self.N)
        y = np.linspace(-self.D/2,self.D/2,self.N)
        x,y = np.meshgrid(x,y)
        Amp = np.exp(-(x**2+y**2)/w0**2)
        Sum = np.zeros((self.N,self.N),dtype='complex')
        for r1 in range(abs(TC1)+1):
            for r2 in range(abs(TC1)-r1+1):
                for r3 in range(r1+1):
                    for r4 in range(abs(TC2)+1):
                        for r5 in range(abs(TC2)-r4+1):
                            for r6 in range(r4+1):
                                Sum += (comb(abs(TC1),r1)*comb(abs(TC1)-r1,r2)*comb(r1,r3)*(-x1)**r2*(-y1)**r3*(1j*np.sign(TC1))**r1*x**(abs(TC1)-r1-r2)*y**(r1-r3)
                                *comb(abs(TC2),r4)*comb(abs(TC2)-r4,r5)*comb(r4,r6)*(-x2)**r5*(-y2)**r6*(1j*np.sign(TC2))**r4*x**(abs(TC2)-r4-r5)*y**(r4-r6))
        Amp = Amp*Sum        
        return Amp
    
    def Double_vortices_freespace(self, z=0, TC1=1, TC2=-1, x1=0.6, x2=-0.6, y1=0.6, y2=0.6):
        
        if z==0:
            z = np.pi*self.w0**2/self.wvl    # 如果z为缺省值则将其设置为Rayleigh距离.
            
        wvn = 2*np.pi/self.wvl
        p = 1/self.w0**2-1j*wvn/2/z
        
        x1 *= self.w0
        y1 *= self.w0
        x2 *= self.w0
        y2 *= self.w0
        
        x = np.linspace(-self.D/2,self.D/2,self.N)
        y = np.linspace(-self.D/2,self.D/2,self.N)
        x,y = np.meshgrid(x,y)
        q1 = -1j*wvn/2/z*x
        q2 = -1j*wvn/2/z*y
        
        Amp = np.exp(1j*wvn*z)/(1j*self.wvl*z)*np.pi/p*np.exp((q1**2+q2**2)/p)*np.exp(1j*wvn/2/z*(x**2+y**2))
        Sum = np.zeros((self.N,self.N),dtype='complex')
        
        for r1 in range(abs(TC1)+1):
            for r2 in range(abs(TC1)-r1+1):
                for r3 in range(r1+1):
                    for r4 in range(abs(TC2)+1):
                        for r5 in range(abs(TC2)-r4+1):
                            for r6 in range(r4+1):
                                N1 = abs(TC1)+abs(TC2)-r1-r2-r4-r5
                                N2 = r1+r4-r3-r6
                                for f1 in range(N1//2+1):
                                    for f2 in range(N2//2+1):
                                        
                                        Sum += ((-x1)**r2*(-y1)**r3*(-x2)**r5*(-y2)**r6*4**(-f1-f2)*p*(f1+f2-N1-N2)*q1**(N1-2*f1)*q2**(N2-2*f2)*(1j*np.sign(TC1))**r1
                                        *(1j*np.sign(TC2))**r4*comb(abs(TC1),r1)*comb(abs(TC1)-r1,r2)*comb(r1,r3)*comb(abs(TC2),r4)*comb(abs(TC2)-r4,r5)*comb(r4,r6)
                                        *Fac(N1)*Fac(N2)/Fac(N1-2*f1)/Fac(f1)/Fac(N2-2*f2)/Fac(f2))
        Amp *= Sum
        
        return Amp                                
        
        
    def Double_vortices_turbulence(self, zeta=np.pi/2, z=0, TC1=1, TC2=-1, x1=0.6, x2=-0.6, y1=0, y2=0):
    
        """
        Calculate the light intensity of vortex beam propagate in non-Kolmogorov turbulence (analytical method).
        """
        
        if z==0:
            z = np.pi*self.w0**2/self.wvl    # 如果z为缺省值则将其设置为Rayleigh距离.
        wvn = 2*np.pi/self.wvl
        
        # without regard to slant, Cn2 is external assignment
#        M = self.M_non_Kolmogorov(alpha=self.alpha,Cn2=self.Cn2,L0=self.L0,l0=self.l0,z=z,zeta=zeta,wvl=self.wvl,SlantFlag=False)
        M = 0
        
        x = np.linspace(-self.D/2,self.D/2,self.N)
        y = np.linspace(-self.D/2,self.D/2,self.N)
        x,y = np.meshgrid(x,y)
        
        p1 = 1/self.w0**2+M-1j*wvn/2/z
        p2 = 1/self.w0**2+M+1j*wvn/2/z-M**2/p1
        
        q1 = -1j*wvn/2/z*x
        q2 = -1j*wvn/2/z*y
        q3 = 1j*wvn/2/z*x+q1*M/p1
        q4 = 1j*wvn/2/z*y+q2*M/p1
        
        I = 1/self.wvl**2/z**2 * np.pi/p1 * np.pi/p2
        Sum = np.zeros((self.N,self.N),dtype='complex')
        
        for r1 in range(abs(TC1)+1):
            for r2 in range(abs(TC1)-r1+1):
                for r3 in range(r1+1):
                    for r4 in range(abs(TC2)+1):
                        for r5 in range(abs(TC2)-r4+1):
                            for r6 in range(r4+1):
                                for t1 in range(abs(TC1)+1):
                                    for t2 in range(abs(TC1)-t1+1):
                                        for t3 in range(t1+1):
                                            for t4 in range(abs(TC2)+1):
                                                for t5 in range(abs(TC2)-t4+1):
                                                    for t6 in range(t4+1):
                                                        N1 = abs(TC1)+abs(TC2)-r1-r2-r4-r5
                                                        N2 = r1+r4-r3-r6
                                                        for f1 in range(N1//2+1):
                                                            for f2 in range(N2//2+1):
                                                                for g1 in range(N1-2*f1+1):
                                                                    for g2 in range(N2-2*f2+1):
                                                                        for f3 in range((N1+g1)//2+1):
                                                                            for f4 in range((N2+g2)//2+1):
                                                                                
                                                                                Sum += (comb(abs(TC1),r1)*comb(abs(TC1)-r1,r2)*comb(r1,r3)*comb(abs(TC2),r4)*comb(abs(TC2)-r4,r5)*comb(r4,r6)
                                                                                *comb(abs(TC1),t1)*comb(abs(TC1)-t1,t2)*comb(t1,t3)*comb(abs(TC2),t4)*comb(abs(TC2)-t4,t5)*comb(t4,t6)
                                                                                *comb(N1-2*f1,g1)*comb(N2-2*f2,g2)*np.exp((q1**2+q2**2)/p1+(q3**2+q4**2)/p2)*(1j*np.sign(TC1))**r1
                                                                                *(1j*np.sign(TC2))**r4*(-1j*np.sign(TC1))**t1*(-1j*np.sign(TC2))**t4*(-x1)**r2*(-y1)**r3*(-x2)**r5*(-y2)**r6
                                                                                *(-x1)**t2*(-y1)**t3*(-x2)**t5*(-y2)**t6*4**(-f1-f2-f3-f4)*M**(g1+g2)*p1**(f1+f2-N1-N2)*p2**(f3+f4-N1-N2-g1-g2)
                                                                                *q1**(N1-2*f1-g1)*q2**(N2-2*f2-g2)*q3**(N1+g1-2*f3)*q4**(N2+g2-2*f4)*Fac(N1)*Fac(N2)*Fac(N1+g1)*Fac(N2+g2)
                                                                                /Fac(N1-2*f1)/Fac(f1)/Fac(N2-2*f2)/Fac(f2)/Fac(N1+g1-2*f3)/Fac(f3)/Fac(N2+g2-2*f4)/Fac(f4))
                                                         
        I *= Sum
        return I.real
    
    def vortex_array_square_freespace(self,z=500*LP.m,D=0.5*LP.m,TC=1,Horizontal_beams=3,Vertical_beams=3,Xd=6,Yd=6):
        """
        Parameters:
            z: propagation distance
            N: sample points
            wvl: wavelength
            Horizontal_beams: number of horizontal beams
            Vertical_beams: number of vertical beams
            Xd,Yd: distance between adjacent sub-beams(unit:w0)
            w0: beam waist
        Return:
            amplitude
        """
        
        x = np.linspace(-D/2,D/2,self.N)
        y = np.linspace(-D/2,D/2,self.N)
        x,y = np.meshgrid(x,y)
        wvn = 2*np.pi/self.wvl    
        p = 1/(self.w0**2)-1j*wvn/2/z
        Xd *= self.w0
        Yd *= self.w0
        Amp = np.exp(1j*wvn*z)/(1j*self.wvl*z)*np.exp(1j*wvn/2/z*(x**2+y**2))*np.pi/p
        Sum = np.zeros((self.N,self.N),dtype='complex')
        
        for n in range(-(Vertical_beams-1)//2,(Vertical_beams-1)//2+1):
            for m in range(-(Horizontal_beams-1)//2,(Horizontal_beams-1)//2+1):
                for r in range(abs(TC)+1):
                    for t in range(r+1):
                        for i in range((r-t)//2+1):
                            for g in range(abs(TC)-r+1):
                                for f in range((abs(TC)-r-g)//2+1):
                                    
                                    q1 = m*Yd/self.w0**2 - 1j*wvn/2/z*y
                                    q2 = n*Xd/self.w0**2 - 1j*wvn/2/z*x
                                    Sum += (np.exp((q1**2+q2**2)/p)*comb(abs(TC),r)*comb(r,t)*comb(abs(TC)-r,g)*(1j*np.sign(TC))**r
                                    *np.exp(-(m**2*Yd**2+n**2*Xd**2)/self.w0**2)*(-m*Yd)**t*(-n*Xd)**g*Fac(r-t)
                                    *Fac(abs(TC)-r-g)*(q1/p)**(r-t)*(q2/p)**(abs(TC)-r-g)*(p/4/q1**2)**i*(p/4/q2**2)**f
                                    /Fac(r-t-2*i)/Fac(i)/Fac(abs(TC)-r-g-2*f)/Fac(f))
        Amp *= Sum
        return Amp
    
    def vortex_array_spherical_freespace(self,z=500*LP.m,D=0.5*LP.m,TC=1,Radial_beams=6,r0=6):
                                         
        """
        Parameters:
            z: propagation distance
            N: sample points
            wvl: wavelength
            D: windowsize
            Radial_beams: number of sub-beams
            w0: beam waist
            r0: distance between sub-beams' center and axis(unit:w0)
        Return:
            amplitude.
        """
        
        # Establish the cartesian coordinate.
        wvn = 2*np.pi/self.wvl
        x = np.linspace(-self.WindowSize/2,self.WindowSize/2,self.N)
        y = np.linspace(-self.WindowSize/2,self.WindowSize/2,self.N)
        x,y = np.meshgrid(x,y)
        r = np.sqrt(x**2+y**2)
        alpha_0 = 2*np.pi/Radial_beams
        r0 *= self.w0
        
        p = 1/(self.w0**2)-1j*wvn/2/z
        Amp = np.exp(1j*wvn*z)/(1j*self.wvl*z)*np.pi/p*np.exp(1j*wvn/2/z*r**2)*np.exp(-r0**2/self.w0**2)
        Sum = np.zeros((self.N,self.N),dtype='complex')
        
        for n in range(1,Radial_beams+1):
            for r1 in range(abs(TC)+1):
                for r2 in range(r1+1):
                    for r3 in range((r1-r2)//2+1):
                        for t1 in range(abs(TC)-r1+1):
                            for t2 in range((abs(TC)-r1-t1)//2+1):
                                
                                alpha_n = n*alpha_0
                                q1 = r0*np.sin(alpha_n)/self.w0**2-1j*wvn/2/z*y
                                q2 = r0*np.cos(alpha_n)/self.w0**2-1j*wvn/2/z*x
                                
                                Sum += (comb(abs(TC),r1)*comb(r1,r2)*comb(abs(TC)-r1,t1)*(1j*np.sign(TC))**r1*(-r0*np.sin(alpha_n))**r2
                                *(-r0*np.cos(alpha_n))**t1*Fac(r1-r2)*Fac(abs(TC)-r1-t1)*np.exp((q1**2+q2**2)/p)*(q1/p)**(r1-r2)*(q2/p)**(abs(TC)-r1-t1)
                                *(p/4/q1**2)**r3*(p/4/q2**2)**t2/Fac(r1-r2-2*r3)/Fac(r3)/Fac(abs(TC)-r1-t1-2*t2)/Fac(t2))
       
        Amp *= Sum
        return Amp
    
    def vortex_array_spherical_freespace_intensity(self,z=500*LP.m,D=0.5*LP.m,TC=1,Radial_beams=3,r0=6):
        
        """
        Calculate light intensity of radial distribution vortex beam array propagate through free space to a given distance
        (analytical method).
        """
        
        # parameters initialization.
        wvn = 2*np.pi/self.wvl
        p = 1/self.w0**2-1j*wvn/2/z
        p_ = 1/self.w0**2+1j*wvn/2/z
        
        alpha_0 = 2*np.pi/Radial_beams
        x = np.linspace(-D/2,D/2,self.N)
        y = np.linspace(-D/2,D/2,self.N)
        x,y = np.meshgrid(x,y)
        r0 *= self.w0
        
        I = (1/self.wvl**2/z**2)*(np.pi**2/p/p_)*np.exp(-2*r0**2/self.w0**2)    # intensity
        Sum = np.zeros((self.N,self.N),dtype='complex')
        
        for n in range(1,Radial_beams+1):
            for r1 in range(abs(TC)+1):
                for r2 in range(abs(TC)-r1+1):
                    for r3 in range(r1+1):
                        for t1 in range((abs(TC)-r1-r2)//2+1):
                            for t2 in range((r1-r3)//2+1):
                                for m in range(1,Radial_beams+1):
                                    for f1 in range(abs(TC)+1):
                                        for f2 in range(abs(TC)-f1+1):
                                            for f3 in range(f1+1):
                                                for g1 in range((abs(TC)-f1-f2)//2+1):
                                                    for g2 in range((f1-f3)//2+1):
                                                        
                                                        alpha_n = n*alpha_0
                                                        alpha_m = m*alpha_0
                                                        
                                                        q1 = r0*np.cos(alpha_n)/self.w0**2-1j*wvn/2/z*x
                                                        q2 = r0*np.sin(alpha_n)/self.w0**2-1j*wvn/2/z*y
                                                        q1_ = r0*np.cos(alpha_m)/self.w0**2+1j*wvn/2/z*x
                                                        q2_ = r0*np.sin(alpha_m)/self.w0**2+1j*wvn/2/z*y
                                                        
                                                        Sum += ((1j*np.sign(TC))**r1*(-1j*np.sign(TC))**f1 * comb(abs(TC),r1)*comb(abs(TC)-r1,r2)
                                                        *comb(r1,r3)*comb(abs(TC),f1)*comb(abs(TC)-f1,f2)*comb(f1,f3) * (-r0*np.cos(alpha_n))**r2
                                                        *(-r0*np.sin(alpha_n))**r3*(-r0*np.cos(alpha_m))**f2*(-r0*np.sin(alpha_m))**f3*
                                                        Fac(abs(TC)-r1-r2)*Fac(r1-r3)*Fac(abs(TC)-f1-f2)*Fac(f1-f3)*np.exp((q1**2+q2**2)/p)*
                                                        np.exp((q1_**2+q2_**2)/p_) * (q1/p)**(abs(TC)-r1-r2)*(q2/p)**(r1-r3)*(q1_/p_)**(abs(TC)-f1-f2)
                                                        *(q2_/p_)**(f1-f3) * (p/4/q1**2)**t1*(p/4/q2**2)**t2*(p_/4/q1_**2)**g1*(p_/4/q2_**2)**g2
                                                        /Fac(abs(TC)-r1-r2-2*t1)/Fac(t1)/Fac(r1-r3-2*t2)/Fac(t2)
                                                        /Fac(abs(TC)-f1-f2-2*g1)/Fac(g1)/Fac(f1-f3-2*g2)/Fac(g2))   
                                                       
        I *= Sum
        I = I.real      
        
        return I
    
# Formula reference C:\Users\FAREWELL\Desktop\径向分布阵列涡旋光束\Doc\Turbulence propagation.docx        
    def Radial_intensity_turbulence(self, z=500*LP.m, zeta=np.pi/2, TC=1, Radial_beams=6, D=0.3*LP.m, r0=6):
        
        """
        Calculate the light intensity of vortex beam propagate in non-Kolmogorov turbulence (analytical method).
        """
         
        wvn = 2*np.pi/self.wvl
        # 避免每次都要对TC求abs值.
        Sgn = np.sign(TC)
        TC = abs(TC)
        # without regard to slant, Cn2 is external assignment
        M = self.M_non_Kolmogorov(alpha=self.alpha,Cn2=self.Cn2,L0=self.L0,l0=self.l0,z=z,zeta=zeta,SlantFlag=False)
        
        p1 = 1/self.w0**2-1j*wvn/2/z+M
        p2 = 1/self.w0**2+1j*wvn/2/z+M
        p3 = p2-M**2/p1
        
        x = np.linspace(-D/2,D/2,self.N)
        y = np.linspace(-D/2,D/2,self.N)
        x,y = np.meshgrid(x,y)
        r0 *= self.w0
        
        alpha_0 = 2*np.pi/Radial_beams
        I = 1/self.wvl**2/z**2 * np.pi/p1 * np.pi/p3 * np.exp(-2*r0**2/self.w0**2)
        Sum = np.zeros((self.N,self.N),dtype='complex')
        
        for n1 in range(1,Radial_beams+1):
            alpha_n1 = n1*alpha_0
            q1 = r0*np.cos(alpha_n1)/self.w0**2-1j*wvn/2/z*x
            q2 = r0*np.sin(alpha_n1)/self.w0**2-1j*wvn/2/z*y
            Exp_1 = np.exp((q1**2+q2**2)/p1)
            for n2 in range(1,Radial_beams+1):
                alpha_n2 = n2*alpha_0
                q3 = r0*np.cos(alpha_n2)/self.w0**2+1j*wvn/2/z*x+2*q1*M/p1
                q4 = r0*np.sin(alpha_n2)/self.w0**2+1j*wvn/2/z*y+2*q2*M/p1
                Exp_2 = np.exp((q3**2+q4**2)/p3)
                for r1 in range(TC+1):
                    C_1 = comb(TC,r1)
                    for r2 in range(TC-r1+1):
                        C_2 = comb(TC-r1,r2)
                        for r3 in range(r1+1):
                            C_3 = comb(r1,r3)
                            for r4 in range(TC+1):
                                C_4 = comb(TC,r4)
                                for r5 in range(TC-r4+1):
                                    C_5 = comb(TC-r4,r5)
                                    for r6 in range(r4+1):
                                        C_6 = comb(r4,r6)
                                        for t1 in range((TC-r1-r2)//2+1):
                                            for t2 in range((r1-r3)//2+1):
                                                for f1 in range(TC-r1-r2-2*t1+1):
                                                    C_7 = comb(TC-r1-r2-2*t1,f1)
                                                    q1_ = q1**(TC-r1-r2-2*t1-f1)
                                                    for f2 in range(r1-r3-2*t2+1):
                                                        N1 = abs(TC)-r4-r5+f1
                                                        N2 = r4-r6+f2
                                                        C_8 = comb(r1-r3-2*t2,f2)
                                                        q2_ = q2**(r1-r3-2*t2-f2)
                                                        for t3 in range(N1//2+1):
                                                            q3_ = q3**(N1-2*t3)
                                                            for t4 in range(N2//2+1):
                                                                
                                                                q4_ = q4**(N2-2*t4)
                                                                Sum += (Exp_1*Exp_2*(1j*Sgn)**r1*(-1j*Sgn)**r4*C_1*C_2*C_3*C_4*C_5*C_6*C_7*C_8
                                                                *(-r0*np.cos(alpha_n1))**r2*(-r0*np.sin(alpha_n1))**r3*(-r0*np.cos(alpha_n2))**r5
                                                                *(-r0*np.sin(alpha_n2))**r6*M**(f1+f2)*4**(-t1-t2-t3-t4)*p1**(t1+t2+r2+r3-TC)*p3**(t3+t4-N1-N2)
                                                                *q1_*q2_*q3_*q4_*Fac(N1)*Fac(N2)*Fac(TC-r1-r2)*Fac(r1-r3)/Fac(N1-2*t3)/Fac(t3)
                                                                /Fac(N2-2*t4)/Fac(t4)/Fac(TC-r1-r2-2*t1)/Fac(t1)/Fac(r1-r3-2*t2)/Fac(t2))
        I *= Sum 
        return I.real
    
    def Radial_intensity_turbulence_multiprocessing(self, z=500*LP.m, zeta=np.pi/2, TC=1, Radial_beams=6, D=0.3*LP.m, r0=6):
        
        """
        Calculate the light intensity of vortex beam propagate in non-Kolmogorov turbulence (analytical method).
        """
         
        wvn = 2*np.pi/self.wvl
        # 避免每次都要对TC求abs值.
        Sgn = np.sign(TC)
        TC = abs(TC)
        # without regard to slant, Cn2 is external assignment
        M = self.M_non_Kolmogorov(alpha=self.alpha,Cn2=self.Cn2,L0=self.L0,l0=self.l0,z=z,zeta=zeta,SlantFlag=False)
        
        p1 = 1/self.w0**2-1j*wvn/2/z+M
        p2 = 1/self.w0**2+1j*wvn/2/z+M
        p3 = p2-M**2/p1
        
        x = np.linspace(-D/2,D/2,self.N)
        y = np.linspace(-D/2,D/2,self.N)
        x,y = np.meshgrid(x,y)
        r0 *= self.w0
        
        alpha_0 = 2*np.pi/Radial_beams
        I = 1/self.wvl**2/z**2 * np.pi/p1 * np.pi/p3 * np.exp(-2*r0**2/self.w0**2)
        Sum = np.zeros((self.N,self.N),dtype='complex')
        
        for n1 in range(1,Radial_beams+1):
            alpha_n1 = n1*alpha_0
            q1 = r0*np.cos(alpha_n1)/self.w0**2-1j*wvn/2/z*x
            q2 = r0*np.sin(alpha_n1)/self.w0**2-1j*wvn/2/z*y
            Exp_1 = np.exp((q1**2+q2**2)/p1)
            for n2 in range(1,Radial_beams+1):
                alpha_n2 = n2*alpha_0
                q3 = r0*np.cos(alpha_n2)/self.w0**2+1j*wvn/2/z*x+2*q1*M/p1
                q4 = r0*np.sin(alpha_n2)/self.w0**2+1j*wvn/2/z*y+2*q2*M/p1
                Exp_2 = Exp_1*np.exp((q3**2+q4**2)/p3)
                for r1 in range(TC+1):
                    C_1 = comb(TC,r1)
                    for r2 in range(TC-r1+1):
                        C_2 = comb(TC-r1,r2)
                        for r3 in range(r1+1):
                            C_3 = comb(r1,r3)
                            for r4 in range(TC+1):
                                C_4 = comb(TC,r4)
                                for r5 in range(TC-r4+1):
                                    C_5 = comb(TC-r4,r5)
                                    for r6 in range(r4+1):
                                        C_6 = comb(r4,r6)
                                        for t1 in range((TC-r1-r2)//2+1):
                                            for t2 in range((r1-r3)//2+1):
                                                for f1 in range(TC-r1-r2-2*t1+1):
                                                    C_7 = comb(TC-r1-r2-2*t1,f1)
                                                    q1_ = q1**(TC-r1-r2-2*t1-f1)
                                                    for f2 in range(r1-r3-2*t2+1):
                                                        N1 = abs(TC)-r4-r5+f1
                                                        N2 = r4-r6+f2
                                                        C_8 = comb(r1-r3-2*t2,f2)
                                                        q2_ = q2**(r1-r3-2*t2-f2)
                                                        for t3 in range(N1//2+1):
                                                            q3_ = q3**(N1-2*t3)
                                                            for t4 in range(N2//2+1):
                                                                
                                                                q4_ = q4**(N2-2*t4)
                                                                Sum += (Exp_2*(1j*Sgn)**r1*(-1j*Sgn)**r4*C_1*C_2*C_3*C_4*C_5*C_6*C_7*C_8
                                                                *(-r0*np.cos(alpha_n1))**r2*(-r0*np.sin(alpha_n1))**r3*(-r0*np.cos(alpha_n2))**r5
                                                                *(-r0*np.sin(alpha_n2))**r6*M**(f1+f2)*4**(-t1-t2-t3-t4)*p1**(t1+t2+r2+r3-TC)*p3**(t3+t4-N1-N2)
                                                                *q1_*q2_*q3_*q4_*Fac(N1)*Fac(N2)*Fac(TC-r1-r2)*Fac(r1-r3)/Fac(N1-2*t3)/Fac(t3)
                                                                /Fac(N2-2*t4)/Fac(t4)/Fac(TC-r1-r2-2*t1)/Fac(t1)/Fac(r1-r3-2*t2)/Fac(t2))
        I *= Sum 
        return I.real

# Formula reference C:\Users\FAREWELL\Desktop\径向分布阵列涡旋光束\Doc\Turbulence propagation.docx     
    def Rectangular_intensity_turbulence(self, z=500*LP.m, TC=1, zeta=np.pi/2, Horizontal_beams=3, Vertical_beams=3, Xd=6, Yd=6):
        
        wvn = 2*np.pi/self.wvl
        Sgn = 1j*np.sign(TC)
        TC = abs(TC)
        
        # without regard to slant, Cn2 is external assignment
        M = self.M_non_Kolmogorov(alpha=self.alpha,Cn2=self.Cn2,L0=self.L0,l0=self.l0,z=z,zeta=zeta,SlantFlag=False)
        
        p1 = 1/self.w0**2-1j*wvn/2/z+M
        p2 = 1/self.w0**2+1j*wvn/2/z+M
        p3 = p2-M**2/p1
        
        x = np.linspace(-self.WindowSize/2, self.WindowSize/2, self.N)
        y = np.linspace(-self.WindowSize/2, self.WindowSize/2, self.N)
        x,y = np.meshgrid(x,y)

        Xd *= self.w0
        Yd *= self.w0
        
        I = np.pi*np.pi/self.wvl**2/z**2/p1/p3
        Sum = np.zeros((self.N,self.N),dtype='complex')
        
        for n1 in range(-(Horizontal_beams-1)//2,(Horizontal_beams-1)//2+1):
            q1 = n1*Xd/self.w0**2-1j*wvn/2/z*x
            q1_2 = q1**2
            for n2 in range(-(Horizontal_beams-1)//2,(Horizontal_beams-1)//2+1):
                q3 = n2*Xd/self.w0**2+1j*wvn/2/z*x+2*q1*M/p1
                q3_2 = q3**2
                for m1 in range(-(Vertical_beams-1)//2,(Vertical_beams-1)//2+1):
                    q2 = m1*Yd/self.w0**2-1j*wvn/2/z*y
                    Exp1 = np.exp((q1_2+q2**2)/p1)
                    Temp1 = np.exp(-(n1**2*Xd**2+m1**2*Yd**2)/self.w0**2)
                    for m2 in range(-(Vertical_beams-1)//2,(Vertical_beams-1)//2+1):
                        q4 = m2*Yd/self.w0**2+1j*wvn/2/z*y+2*q2*M/p1
                        Exp2 = Exp1*np.exp((q3_2+q4**2)/p3)
                        Temp2 = np.exp(-(n2**2*Xd**2+m2**2*Yd**2)/self.w0**2)
                        for r1 in range(TC+1):
                            C_1 = comb(TC,r1)
                            for r2 in range(TC-r1+1):
                                C_2 = C_1*comb(TC-r1,r2)
                                for r3 in range(r1+1):
                                    C_3 = C_2*comb(r1,r3)
                                    for r4 in range(TC+1):
                                        C_4 = C_3*comb(TC,r4)
                                        for r5 in range(TC-r4+1):
                                            C_5 = C_4*comb(TC-r4,r5)
                                            for r6 in range(r4+1):
                                                C_6 = C_5*comb(r4,r6)
                                                for t1 in range((TC-r1-r2)//2+1):
                                                    for t2 in range((r1-r3)//2+1):
                                                        for f1 in range(TC-r1-r2-2*t1+1):
                                                            C_7 = C_6*comb(TC-r1-r2-2*t1,f1)
                                                            q1_ = q1**(TC-r1-r2-2*t1-f1)
                                                            for f2 in range(r1-r3-2*t2+1):
                                                                N1 = TC-r4-r5+f1
                                                                N2 = r4-r6+f2
                                                                C_8 = C_7*comb(r1-r3-2*t2,f2)
                                                                q2_ = q1_*q2**(r1-r3-2*t2-f2)
                                                                for t3 in range(N1//2+1):
                                                                    q3_ = q2_*q3**(N1-2*t3)
                                                                    for t4 in range(N2//2+1):
                                                                        q4_ = q3_*q4**(N2-2*t4)
                                                                        
                                                                        Sum += (Exp2*(Sgn)**r1*(-Sgn)**r4*C_8*(-n1*Xd)**r2*(-m1*Yd)**r3
                                                                        *(-n2*Xd)**r5*(-m2*Yd)**r6*M**(f1+f2)*4**(-t1-t2-t3-t4)*p1**(t1+t2+r2+r3-TC)
                                                                        *p3**(t3+t4-N1-N2)*q4_*Fac(N1)*Fac(N2)*Fac(TC-r1-r2)*Fac(r1-r3)/Fac(N1-2*t3)/Fac(t3)
                                                                        /Fac(N2-2*t4)/Fac(t4)/Fac(TC-r1-r2-2*t1)/Fac(t1)/Fac(r1-r3-2*t2)/Fac(t2)*Temp1*Temp2)
        I *= Sum
        return I.real
    
    def I_turbulence_vortex_array_TC(self, wvl=1550*LP.nm, z=500*LP.m, N=512, Radial_beams=6, r0=6):
        
        wvn = 2*np.pi/wvl
        x = np.linspace(-self.WindowSize/2, self.WindowSize/2, N)
        y = np.linspace(-self.WindowSize/2, self.WindowSize/2, N)
        x,y = np.meshgrid(x,y)
        
        alpha_0 = 2*np.pi/Radial_beams
        r0 *= self.w0
        
        # without regard to slant, Cn2 is external assignment
        M = self.M_non_Kolmogorov(alpha=self.alpha,Cn2=self.Cn2,L0=self.L0,l0=self.l0,z=z,zeta=np.pi/2,SlantFlag=False)
        
        p1 = 1/self.w0**2-1j*wvn/2/z+M
        p2 = 1/self.w0**2+1j*wvn/2/z+M
        p3 = p2-M**2/p1
        I = np.pi*np.pi/self.wvl**2/z**2/p1/p3*np.exp(-2*r0**2/self.w0**2)
        Sum = np.zeros((self.N,self.N),dtype='complex')
#        count = 0   #计数器
        for n1 in range(1,Radial_beams+1):
            alpha_n1 = n1*alpha_0
            q1 = r0*np.cos(alpha_n1)/self.w0**2-1j*wvn/2/z*x
            q2 = r0*np.sin(alpha_n1)/self.w0**2-1j*wvn/2/z*y
            Exp1 = np.exp((q1**2+q2**2)/p1)
            for n2 in range(1,Radial_beams+1):
                alpha_n2 = n2*alpha_0
                q3 = r0*np.cos(alpha_n2)/self.w0**2+1j*wvn/2/z*x+2*q1*M/p1
                q4 = r0*np.sin(alpha_n2)/self.w0**2+1j*wvn/2/z*y+2*q2*M/p1
                Exp2 = Exp1*np.exp((q3**2+q4**2)/p3)
                Correct_term = (np.sqrt(2)/self.w0)**(n1+n2+2)/np.sqrt(gamma(n1+1)*gamma(n2+1))
                for r1 in range(n1+1):
                    C_1 = comb(n1,r1)
                    for r2 in range(n1-r1+1):
                        C_2 = comb(n1-r1,r2)
                        for r3 in range(r1+1):
                            C_3 = comb(r1,r3)
                            for r4 in range(n2+1):
                                C_4 = comb(n2,r4)
                                for r5 in range(n2-r4+1):
                                    C_5 = comb(n2-r4,r5)
                                    for r6 in range(r4+1):
                                        C_6 = comb(r4,r6)
                                        for t1 in range((n1-r1-r2)//2+1):
                                            for t2 in range((r1-r3)//2+1):
                                                for f1 in range(n1-r1-r2-2*t1+1):
                                                    C_7 = comb(n1-r1-r2-2*t1,f1)
                                                    q1_ = q1**(n1-r1-r2-2*t1-f1)
                                                    for f2 in range(r1-r3-2*t2+1):
                                                        N1 = n2-r4-r5+f1
                                                        N2 = r4-r6+f2
                                                        C_8 = comb(r1-r3-2*t2,f2)
                                                        q2_ = q1_*q2**(r1-r3-2*t2-f2)
                                                        for t3 in range(N1//2+1):
                                                            q3_ = q2_*q3**(N1-2*t3)
                                                            for t4 in range(N2//2+1):
#                                                                count += 1
                                                                q4_ = q3_*q4**(N2-2*t4)
                                                                Sum += (Exp2*(1j)**r1*(-1j)**r4*C_1*C_2*C_3*C_4*C_5*C_6*C_7*C_8*(-r0*np.cos(alpha_n1))**r2
                                                                *(-r0*np.sin(alpha_n1))**r3*(-r0*np.cos(alpha_n2))**r5*(-r0*np.sin(alpha_n2))**r6*M**(f1+f2)
                                                                *4**(-t1-t2-t3-t4)*p1**(t1+t2+r2+r3-n1)*p3**(t3+t4-N1-N2)*q4_*Fac(N1)*Fac(N2)*Fac(n1-r1-r2)*Fac(r1-r3)
                                                                /Fac(N1-2*t3)/Fac(t3)/Fac(N2-2*t4)/Fac(t4)/Fac(n1-r1-r2-2*t1)/Fac(t1)/Fac(r1-r3-2*t2)/Fac(t2)*Correct_term)
                                                                
                
        I *= Sum
#        print(count)
        return I.real
