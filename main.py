import numpy as np 
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Constants!
# star composition
X = 0.734
Y = 0.25
Z = 0.016

# Energy stuff
mu = 1/(2*X + 0.75*Y +0.5*Z)
Xcno = 0.03*X
gamma = 5/3
a = 7.56e-16

# Contants of the univserse
mp = 1.67262192369e-27 # kg
mump = mu*mp
k = 1.380649e-23 # J/K
me = 9.1093837015e-31 # kg
G = 6.67430e-11
c = 299792458 # m/s
hbar = 1.054571817e-34 # Js
sb = 5.670374419e-8

# (partial) boundary conditions:
rho_c =
T_c =
r_c = 0.1


# pressure regieme =================================
def P(rho, T):
    return ((3 * np.pi**2)**(2/3)/5) * (hbar**2/me) * (rho/mp)**(5/3) + rho * (k*T/mump) + (1/3)*a*(T**4)


# the big 5 =================================
# theyre kinda hard to read, keep the doc side by side
def drhodr( rho, T, L, r, M):
    return -(G*M*rho/r**2 + dPdT( rho, T) * dTdr(rho, T, L, r, M)) /dPdrho(rho,T)

def dTdr(  rho, T, L, r, M):
    return - np.min( 3 * kappa(rho,T) * rho * L/(16*np.pi*a*c*(T**3)*(r**2)), (1 - 1/gamma)*(T/P)*G*M*rho/r**2 )

def dMdr(r, rho):
    return 4*np.pi*(r**2)*rho

def dLdr(rho, r, T):
    return 4*np.pi*(r**2)*rho*epsilon(rho, T)

def dtaudr(rho, kappa):
    return kappa*rho


# Stuff needed to calculate the big 5 =================================
def dPdrho(rho, T):
    return ((3*np.pi**2)**(2/3)/3)*(hbar**2/(me*mp))*(rho/mp)**(2/3) + k*T/mump

def dPdT(rho, T):
    return rho*k/(mump) + (4/3)*a*T**3


# Energy chain stuff =================================
def eps_pp(rho, T):
    return (1.07e-7)*(rho/1e5)*(X**2)*(T/1e6)**4

def eps_cno(rho, T):
    return (8.24e-26)*(rho/1e5)*X*Xcno*np.power((T/1e6), 19.9)

def epsilon(rho, T):
    return eps_pp(rho, T) + eps_cno(rho, T)


# Opacity stuff =================================
kappa_es = 0.02*(1 + X)

def kappa_ff(rho, T):
    return (1e24)*(Z + 0.0001)*np.power(rho/1e3, 0.7)*np.power(T, -3.5)

def kappa_H(rho, T):
    return (2.5e-32)*(Z/0.02)*np.power(rho/1e3, 0.5)*T**9

def kappa(rho, T):
    return 1/(1/kappa_H(rho, T) + 1/np.max(kappa_es, kappa_ff(rho,T)) )

# these ones are our group specific
# we have to do multiple different values of these!
# I put starter values in
Ti = 0.5 * T_c
kappa_0 = 0.5 * kappa_ff(rho_c, Ti)
def kappa_i(rho, T):
    if(T<Ti):
        return kappa_0*(Z + 0.0001)*np.sqrt(rho/1e3)*np.power(T, -3.5)

    elif(T>=Ti):
        return 0

def kappa_modified(rho, T):
    return 1 / (1 / kappa_H(rho, T) + 1 / np.max(kappa_es, kappa_ff(rho, T), kappa_i(rho,T)))


# More boundary conditions:
M_c = (4/3)*np.pi*(r_c**3)*rho_c
L_c = (4/3)*np.pi*(r_c**3)*rho_c*epsilon(rho_c, T_c)

### steps:
# choose rho_c and T_c and r_c
#  use tau(inf) - tau = dtau = kappa*rho**2 / np.abs(drhodr(rho, T, L, r, M))
# integrate until dtau << 1 OR M > M_critical (to choose)
# define R_suface by tau(inf) - tau(R_surf) = 2/3

# now, we have R_surface (and Luminosity), we need rho_c to make sure the luminosity BC is satisfied
def f(rho_c):
    return (L_surf - 4*np.pi*sb*(R_surf**2)*(T_surf**4))/np.sqrt( 4*np.pi*sb*(R_surf**2)*(T_surf**4)*L_surf  )
# we find the root of f to get optimal rho_c via bisection, using bounds of [0.3, 500]
# he said something about this ^^ not working? Not sure I understood that part






