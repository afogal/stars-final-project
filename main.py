# This code used to be split across a couple files, one containing constants, another containing functions, etc
# In the course of troubleshooting it, and in the interest of using numba to speed it up, it had to be compiled into a single mega-file

import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import sys
import warnings
from numba import jit
import datetime as dt
import os

# Verbosity of outputs while running
verbosity = 1
if verbosity < 2:
    warnings.filterwarnings("ignore") # IF THE CODE IS BROKEN, COMMENT THIS LINE OUT AND RUN IT AGAIN

# Constants! Defined more clearly in accompanying report
# star composition
X = 0.734
Y = 0.25
Z = 0.016

# Energy stuff
mu = 1/(2*X + 0.75*Y +0.5*Z)
Xcno = 0.03*X
gamma = 5./3.
a = 7.56e-16

# The sun
Msun = 1.989e30
Lsun = 3.828e26
Rsun = 696340e3

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
r_c = 10
r_f = 15*Rsun # This gets changed later

# pressure regieme =================================
@jit
def P(rho, T):
    return ((3 * np.pi**2)**(2/3)/5) * (hbar**2/me) * (rho/mp)**(5/3) + rho * (k*T/mump) + (1/3)*a*(T**4)


# the big 5 =================================
# theyre kinda hard to read, keep the doc side by side
@jit
def drhodr( rho, T, L, r, M, P, kappas):
    return -(G*M*rho/r**2 + dPdT( rho, T) * dTdr(rho, T, L, r, M, P, kappas)) /dPdrho(rho,T)

@jit
def dTdr(  rho, T, L, r, M, P, kappas):
    return - np.nanmin([ np.abs(3 * kappas * rho * L/(16*np.pi*a*c*(T**3)*(r**2))), np.abs((1 - 1/gamma)*(T/P)*G*M*rho/r**2)], axis=0 )

@jit
def dMdr(r, rho):
    return 4*np.pi*(r**2)*rho

@jit
def dLdr(rho, r, T):
    return 4*np.pi*(r**2)*rho*epsilon(rho, T)

@jit
def dtaudr(rho, kappa):
    return kappa*rho


# Stuff needed to calculate the big 5 =================================
@jit
def dPdrho(rho, T):
    return ((3*np.pi**2)**(2/3)/3)*(hbar**2/(me*mp))*(rho/mp)**(2/3) + k*T/mump

@jit
def dPdT(rho, T):
    return rho*k/(mump) + (4/3)*a*T**3


# Energy chain stuff =================================
@jit
def eps_pp(rho, T):
    return (1.07e-7)*(rho/1e5)*(X**2)*(T/1e6)**4

@jit
def eps_cno(rho, T):
    return (8.24e-26)*(rho/1e5)*X*Xcno*np.power((T/1e6), 19.9)

@jit
def epsilon(rho, T):
    return eps_pp(rho, T) + eps_cno(rho, T)


# Opacity stuff =================================
kappa_es = 0.02*(1 + X)

@jit
def kappa_ff(rho, T):
    return (1e24)*(Z + 0.0001)*np.power(rho/1e3, 0.7)*np.power(T, -3.5)

@jit
def kappa_H(rho, T):
    return (2.5e-32)*(Z/0.02)*np.power(rho/1e3, 0.5)*(T**9)

@jit
def kappa(rho, T):
    # slightly different treatment needed for array vs single inputs
    if isinstance(rho, np.ndarray):
        return 1/(1/kappa_H(rho, T) + 1/np.nanmax([kappa_es*np.ones(rho.shape), kappa_ff(rho,T)], axis=0) )
    else:
        return 1 / (1 / kappa_H(rho, T) + 1 / np.nanmax([kappa_es, kappa_ff(rho, T)], axis=0))

# these ones are our group specific: we had to explore changes made to opacity
# The specific Ti and kappa_0 are set below
@jit
def kappa_i(rho, T):
    global Ti
    global kappa_0

    # this function caused lots of problems, this is not the most clever way to write it
    if isinstance(T, np.ndarray) or isinstance(T, list):
        out = np.zeros(len(T))
        for i in range(out.shape[0]):
            if(T[i]<Ti):
                out[i] = kappa_0 * (Z + 0.0001) * np.sqrt(rho[i] / 1e3) * np.power(T[i], -3.5)
        return out
    else:
        if (T < Ti):
            return kappa_0 * (Z + 0.0001) * np.sqrt(rho / 1e3) * np.power(T, -3.5)

        else:
            return 0

@jit
def kappa_modified(rho, T):
    # slightly different treatment needed for array vs single inputs
    if isinstance(rho, np.ndarray):
        return 1/(1/kappa_H(rho, T) + 1/np.nanmax([kappa_es*np.ones(rho.shape), kappa_ff(rho,T),kappa_i(rho, T)], axis=0) )
    else:
        return 1 / (1 / kappa_H(rho, T) + 1 / np.nanmax([kappa_es, kappa_ff(rho, T),kappa_i(rho, T)], axis=0))

@jit # Modified star system, to be fed to the RK integrator as a system of DEs
def star_modified(r,y):
    rho = y[0]
    T = y[1]
    M = y[2]
    L = y[3]
    tau = y[4]

    rhodot = drhodr(rho, T, L, r, M, P(rho, T), kappa_modified(rho, T))
    Tdot = dTdr(rho, T, L, r, M, P(rho, T), kappa_modified(rho, T))
    Mdot = dMdr(r, rho)
    Ldot = dLdr(rho, r, T)
    taudot = dtaudr(rho, kappa_modified(rho, T))

    return np.array([rhodot, Tdot, Mdot, Ldot, taudot])

# More boundary conditions:
@jit
def f(L_surf, R_surf, T_surf):
    return (L_surf - 4*np.pi*sb*(R_surf**2)*(T_surf**4))/np.sqrt( 4*np.pi*sb*(R_surf**2)*(T_surf**4)*L_surf  )

# the system unmodified
@jit
def star(r,y):
    rho = y[0]
    T = y[1]
    M = y[2]
    L = y[3]
    tau = y[4]

    rhodot = drhodr(rho, T, L, r, M, P(rho, T), kappa(rho, T))
    Tdot = dTdr(rho, T, L, r, M, P(rho, T), kappa(rho, T))
    Mdot = dMdr(r, rho)
    Ldot = dLdr(rho, r, T)
    taudot = dtaudr(rho, kappa(rho, T))

    return np.array([rhodot, Tdot, Mdot, Ldot, taudot])

@jit # did the integration go far enough? (defined by when dtau is small "enough")
def dtausmall(all, rs1, modified=False, ret=False):
    # Variable renaming
    rhos1 = all[0]
    Ts1 = all[1]
    Ms1 = all[2]
    Ls1 = all[3]
    taus1 = all[4]

    # Using our modification
    if not modified:
        ks = kappa(rhos1, Ts1)
    else:
        ks = kappa_modified(rhos1, Ts1)

    drhodrs = drhodr(rhos1,Ts1, Ls1, rs1, Ms1, P(rhos1, Ts1), ks)
    dtau1 = ks * rhos1 * rhos1 / np.abs(drhodrs)

    if ret:
        return dtau1

    if dtau1 < 0.1 or np.isnan(dtau1):
        return True
    else:
        return False

@jit
def rk45_step(stepsize, system, t, ys, T_c, tol=1e-4):

    # This numerically approximates derivatives (I think?)
    # Mostly borrowed from a paper outlining RK45
    ys = np.array(ys)
    k1 = stepsize* system(t, ys)
    k2 = stepsize* system(t + (1/4)*stepsize, ys+ (1/4)*k1)
    k3 = stepsize*system( t + 3/8*stepsize,    ys + 3/32*k1 + 9/32*k2)
    k4 = stepsize*system( t + 12/13*stepsize,  ys + 1932/2197*k1 - 7200/2197*k2 + 7296/2197*k3)
    k5 = stepsize*system( t + stepsize,        ys + 439/216*k1 - 8*k2 + 3680/513*k3 -845/4104*k4)
    k6 = stepsize*system( t + 1/2*stepsize,    ys - 8/27*k1 + 2*k2 - 3544/2565*k3 + 1859/4104*k4 - 11/40 * k5)

    # 4th and 5th order solutions
    next_4 = ys  + 25/216*k1 + 1408/2565*k3 + 2197/4101*k4 - 1/5*k5
    next_5 = ys + 16/135*k1 + 6656/12825*k3 + 28561/56430*k4 - 9/50*k5 + 2/55*k6
    diff = np.fabs(next_5-next_4)

    # Get smart near the surface
    if next_4[1]/T_c < 0.01:
        max_step = 10000
        tol = 1e-7
    else:
        max_step = 7e6

    # next step, with avoiding divide by zero
    zero = diff==0
    s = (np.fabs(next_4)*tol/(2*(diff+zero)))**(1/4) + 4*zero
    next_step = stepsize * np.nanmax([np.nanmin([np.nanmin(s), 2]), 0.5])

    # truncate if next step too big or small
    if(next_step > max_step ):
        next_step = max_step
    if next_step < 5000:
        next_step = 5000

    return (next_step, next_4)

@jit # Solve the system for some set of parameters (used in bisection)
def trial_soln(rho_c_trial, T_c, r_f, system=star, optimize=True, final=False, modified=False):

    count = 0 # Counter variable

    # The other half of the BCs
    M_c = (4 / 3) * np.pi * (r_c ** 3) * rho_c_trial
    L_c = (4 / 3) * np.pi * (r_c ** 3) * rho_c_trial * epsilon(rho_c_trial, T_c)

    # Use our modification
    if not modified:
        tau_c = kappa(rho_c_trial, T_c) * rho_c_trial
    else:
        tau_c = kappa_modified(rho_c_trial, T_c) * rho_c_trial

    # I keep around everything
    all = [[rho_c_trial], [T_c], [M_c], [L_c], [tau_c], [r_c]]
    next = [rho_c_trial, T_c, M_c, L_c, tau_c, r_c]
    r = r_c

    stepsize = 10000

    # This is the integration loop
    # It continues until a mass limit, radius limit, or delta tau is below 0.1
    while all[2][-1] < (10**3)*Msun and all[5][-1] < 1e10 and not dtausmall(next, r, modified=modified):

        # Variable renaming
        rhos1 = all[0]
        Ts1 = all[1]
        Ms1 = all[2]
        Ls1 = all[3]
        taus1 = all[4]
        rs1 = all[5]

        rs1.append(rs1[-1] + stepsize)
        curr = [rhos1[-1], Ts1[-1], Ms1[-1], Ls1[-1], taus1[-1]]
        (stepsize, next) = rk45_step(stepsize, system, rs1[-1], curr, T_c)
        r = rs1[-1]

        rhos1.append(next[0])
        Ts1.append(next[1])
        Ms1.append(next[2])
        Ls1.append(next[3])
        taus1.append(next[4])


        all = [rhos1, Ts1, Ms1, Ls1, taus1, rs1]
        count += 1
        if count % 200 == 0:
            print(count, dtausmall(next, r, modified=modified, ret=True))

    rhos1 = np.array(all[0])
    Ts1 = np.array(all[1])
    Ms1 = np.array(all[2])
    Ls1 = np.array(all[3])
    taus1 = np.array(all[4])
    rs1 = np.array(all[5])
    print(count, dtausmall(next, r, modified=modified, ret=True))


    # take tau(inf) as the last non-nan value (done the fancy way)
    tauinf_i = (~np.isnan(taus1)).cumsum().argmax()
    print("tauinf at: {} tauinf: {}".format(tauinf_i / taus1.shape[0], taus1[tauinf_i]))

    # finds the index of the surface, tau(inf) - tau(R_Surf) = 2/3
    surf_i1 = np.nanargmin(np.abs(taus1[tauinf_i] - taus1[0:tauinf_i] - 2./3.))

    tauinf = taus1[tauinf_i]
    surf_g1 = surf_i1
    rsurf_1 = rs1[surf_g1] # one way to find the surface, usually inaccurate

    # interpolation of varaibles (to get good answers with bad stepsizes)
    tau = interp1d(rs1, taus1, kind="cubic", fill_value="extrapolate")
    tt = interp1d(tauinf-taus1[0:tauinf_i], rs1[0:tauinf_i], fill_value="extrapolate", kind="linear")
    rho = interp1d(rs1, rhos1, kind="cubic", fill_value="extrapolate")
    T = interp1d(rs1, Ts1, kind="cubic", fill_value="extrapolate")
    M = interp1d(rs1, Ms1, kind="cubic", fill_value="extrapolate")
    L = interp1d(rs1, Ls1, kind="cubic", fill_value="extrapolate")

    rsurf_2 = tt(2./3.) # second way to find the surface, usually much better

    # ALTERNATIVE SURFACE DETERMINATION (NOT USED, did not provide better results)
    # rsurf_3 = minimize(lambda x: np.abs(tauinf - tau(x) -2./3.), rs1[surf_i1], bounds=[(0,rs1[tauinf_i])]  )
    # # print( taus1[tauinf_i] - tau(rsurf.x[0]) -2/3)
    # rsurf_3 = rsurf_3.x[0]
    #
    #
    # rsurf_4 = brentq(lambda x: tauinf - tau(x) -2./3., rs1[surf_i1-10], rs1[-1], xtol=1e-30)
    # #print(taus1[tauinf_i] - tau(rsurf) - 2 / 3)
    #
    # min_i = -1; min_e = np.abs(f(L(rs1[-1]), rs1[-1], T(rs1[-1])))
    # for i in range(taus1.shape[0]-1,0):
    #     if np.abs(f(L(rs1[i]), rs1[i], T(rs1[i]))) < min_e and (tauinf - taus1[i])<1:
    #         min_i = i
    #         min_e = np.abs(f(L(rs1[i]), rs1[i], T(rs1[i])))
    # rsurf_5 = rs1[min_i]


    frac_error1 = np.abs(f(Ls1[surf_g1], rs1[surf_g1], Ts1[surf_g1]))
    frac_error2 = np.abs(f(L(rsurf_2), rsurf_2, T(rsurf_2)))
    # frac_error3 = np.abs(f(L(rsurf_3), rsurf_3, T(rsurf_3)))
    # frac_error4 = np.abs(f(L(rsurf_4), rsurf_4, T(rsurf_4)))
    # frac_error5 = np.abs(f(L(rsurf_5), rsurf_5, T(rsurf_5)))

    surf_err1  = np.abs(tauinf - taus1[surf_g1] - 2. / 3.)
    surf_err2 = np.abs(tauinf - tau(rsurf_2) - 2. / 3.)
    # surf_err3 = np.abs(tauinf - tau(rsurf_3) - 2. / 3.)
    # surf_err4 = np.abs(tauinf - tau(rsurf_4) - 2. / 3.)
    # surf_err5 = np.abs(tauinf - tau(rsurf_5))

    error1 = np.sqrt(frac_error1**2 + surf_err1**2)
    error2 = np.sqrt(frac_error2 ** 2 + surf_err2 ** 2)
    # error3 = np.sqrt(frac_error3 ** 2 + surf_err3 ** 2)
    # error4 = np.sqrt(frac_error4 ** 2 + surf_err4 ** 2)
    # error5 = np.sqrt(frac_error5 ** 2 + surf_err5 ** 2)
    # error1 = frac_error1
    # error2 = frac_error2
    # error3 = frac_error3
    # error4 = frac_error4
    # error5 = frac_error5

    # Automagic surface determination based on error min (worked poorly and caused really bad jumping around)
    # min_method = np.argmin([error1, error2, error3, error4, error5])
    rsurf = rsurf_2 #eval("rsurf_{}".format(min_method+1)) #rs1[surf_g1] #rsurf_4 #(rsurf + rs1[surf_g1])/2

    # was rf far enough?
    if not modified:
        ks = kappa(rhos1[tauinf_i], Ts1[tauinf_i])
    else:
        ks = kappa_modified(rhos1[tauinf_i], Ts1[tauinf_i])
    drhodrs = drhodr(rhos1[tauinf_i], Ts1[tauinf_i], Ls1[tauinf_i], rs1[tauinf_i], Ms1[tauinf_i], P(rhos1[tauinf_i], Ts1[tauinf_i]), ks)
    dtau1 =  ks * rhos1[tauinf_i] * rhos1[tauinf_i] / np.abs(drhodrs)

    print("1: R: {}  T: {}  tauerr: {} f: {}".format(rs1[surf_g1] / Rsun, Ts1[surf_g1], surf_err1, frac_error1))
    print("2: R: {}  T: {}  tauerr: {} f: {}".format(rsurf_2 / Rsun, T(rsurf_2), surf_err2, frac_error2))
    # print("3: R: {}  T: {}  tauerr: {} f: {}".format(rsurf_3 / Rsun, T(rsurf_3), surf_err3, frac_error3))
    # print("4: R: {}  T: {}  tauerr: {} f: {}".format(rsurf_4 / Rsun, T(rsurf_4), surf_err4, frac_error4))
    # print("5: R: {}  T: {}  tauerr: {} f: {}".format(rsurf_5 / Rsun, T(rsurf_5), surf_err5, frac_error5))
    #print("Went with: ", min_method+1)

    # Actually perform the interpolation (last index is the surface, allows us to throw away really large chunks of data that only mattered to get tau(inf) )
    rs1 = np.linspace(0.1, rsurf, 5000)
    taus1 = tau(rs1)
    rhos1 = rho(rs1)
    Ts1 = T(rs1)
    Ms1 = M(rs1)
    Ls1 = L(rs1)
    surf_i1 = np.nanargmin(np.abs(rs1-rsurf))
    print("Surf err: ", np.min(np.abs(rs1-rsurf))) # sanity check to make sure the surface radius occurs as an element

    # error from luminosity condition
    frac_error1 = f(Ls1[surf_i1], rs1[surf_i1], Ts1[surf_i1])

    if verbosity > 1: # for debugging
        print("T: ", Ts1)
        print("L: ", Ls1)
        print("r: ", rs1)
        print("M: ", Ms1)
        print("P: ", P(rhos1, Ts1))
        print("kappa: ", ks)
        print("drho/dr: ", drhodrs)
        print("rho: ",rhos1)
        print("dtau: ", dtau1)

    if verbosity > 0:
        print("Curr Tsurf: ", Ts1[surf_i1])
        print("Curr rho: ", rho_c_trial)
        print("Curr Err: ", frac_error1)
        print("Curr dtau: ", dtau1)#[(~np.isnan(dtau1)).cumsum().argmax()])
        print("Err in tau(surf): ", np.abs(tauinf - taus1[-1] - 2./3.))
        print(" ")

    # if we're optimizing, only care about the error, else we want the full solution
    if not optimize:
        return np.array([rs1, rhos1, Ts1, Ms1, Ls1, taus1, surf_i1, frac_error1])
    else:
        return frac_error1

@jit # this function runs the trial_solution in order to find ICs that satisfy a relationship with the BCs
def find_ics(rho_c_min, rho_c_max, Tc, rf, max_iters, system=star, modified=False):

    # beginning points
    fs = []
    rhos = []
    oldmin = -1

    # init bounds
    f2 = trial_soln(rho_c_min, Tc, rf , system=system, modified=modified)
    fs.append(f2)
    rhos.append(rho_c_min)

    f1 = trial_soln(rho_c_max, Tc, rf, system=system, modified=modified)
    fs.append(f1)
    rhos.append(rho_c_max)

    print("Initial error bounds: {} {}".format(f1, f2))

    # Bisection method
    if f1*f2 > 0: # root not in interval defined by [rho_c_min, rho_c_max]
        print("Pick better bounds, one needs to be negative")
        raise ValueError # used to auto adjust bounds later to save work
    else:
        for n in range(max_iters):
            intcount = 0
            print("Optimization round: ", n)
            rho_m = (rho_c_max + rho_c_min)/2
            rs, rhoss, Ts, Ms, Ls, taus, surf, f3 = trial_soln(rho_m, Tc, rf, system=system, optimize=False, modified=modified)
            fs.append(f3)
            rhos.append(rho_m)

            # binary search, change one bound and recompute the other
            if f1*f3 < 0:
                rho_c_min = rho_m
                f2 = f3

            elif f2*f3 < 0:
                rho_c_max = rho_m
                f1 = f3

            # Adjusted criteria based on how long its taking
            # essentially, if the error is good enough or not changing enough we stop
            # or, if the quantity optimizing over isn't changing, then we might as well stop
            # we are not guaranteed that a solution exists
            if (np.min(np.abs(fs)) < 3e-5): # end condition
                break
            elif ((np.min(np.abs(fs)) < 1e-4) or (np.abs(fs[-1]) < 1e-4) )  and n > 20:
                print("f is smaller than threshold for 20 iters")
                break
            elif ((np.min(np.abs(fs)) < 1e-2) or (np.abs(fs[-1]) < 1e-2) ) and n > 50:
                print("f is smaller than threshold for 50 iters")
                break
            elif ((np.min(np.abs(fs)) < 0.5) or (np.abs(fs[-1]) < 0.5) ) and n > 100:
                print("f is smaller than threshold for 100 iters")
                break
            elif len(fs) > 5 and np.min(np.abs(fs[-1] - np.array(fs[-5:-1]))) < 1e-4 and np.abs(fs[-1])<1 and n>50:
                print("f is small and fs are similar")
                break
            elif len(rhos) > 5 and np.min(np.abs(rhos[-1] - np.array(rhos[-5:-1]))) < 1e-10 and np.min(np.abs(fs[-1] - np.array(fs[-5:-1]))) < 1e-10 and n>20:
                print("Rho doesnt change, f changes little, f is small")
                break
            elif n>200 and np.min(np.abs(fs)) < 1:
                print("Over 200 and err <1")
                break
            elif n > 60:
                break

            print("Curr min error: ", np.min(np.abs(fs)))
            # Okay so this looks kinda janky, and it is
            # Basically, the solver is nondeterministic-ish, so running w the same inputs doesnt always give the same outputs
            # Which sometimes leads to getting solns with really bad error even after bisection
            # So simply I stored all the variables and spit out the ones that came with whatever the minimum error was
            if np.min(np.abs(fs)) != oldmin or  oldmin == -1:
                oldmin = np.min(np.abs(fs))
                keepvars = np.array([rs, rhoss, Ts, Ms, Ls, taus, surf, fs[-1]])


        if np.min(np.abs(fs)) != oldmin or  oldmin == -1:
            oldmin = np.min(np.abs(fs))
            keepvars = np.array([rs, rhoss, Ts, Ms, Ls, taus, surf, fs[-1]])
        mini = np.argmin(np.abs(fs))

    return keepvars #rhos[mini] #(rho_c_max + rho_c_min)/2

def plot_all(rs, rhos, Ts, Ms, Ls, taus, surf, f, n=0, modified=False):

    try:
        os.mkdir("{}".format(n))
    except: # folder already exists
        pass

    # calculate and plot various useful quantities
    epp = eps_pp(rhos, Ts) * 4 * np.pi * rs * rs * rhos
    ecno = eps_cno(rhos, Ts)* 4 * np.pi * rs * rs * rhos
    ep = epsilon(rhos, Ts)* 4 * np.pi * rs * rs * rhos
    plt.plot(rs / rs[surf], ep, "b-", label="Total")
    plt.plot(rs / rs[surf], ecno, "g--", label="CNO")
    plt.plot(rs/rs[surf], epp, "r:", label="Proton-Proton")
    plt.xlabel(r"$R/R_{surf}$")
    plt.ylabel("Energy Production")
    plt.title("A Plot of Energy Production Throughout the Star")
    plt.grid()
    plt.legend()
    plt.xlim(0,1)

    plt.savefig("{0}/energy_{0}.png".format(n), dpi=300)
    plt.clf()
    plt.close()

    # This is the fancy plot with a lot of lines
    plt.plot(rs/rs[surf], rhos/rhos[0], label=r"$\rho/\rho_c$")
    plt.plot(rs/rs[surf], Ts/Ts[0], label=r"$T/T_c$")
    plt.plot(rs/rs[surf], Ms/Ms[surf], label=r"$M/M_{surf}$")
    plt.plot(rs/rs[surf], Ls/Ls[surf], label=r"$L/L_{surf}$")
    plt.xlabel(r"$R/R_{surf}$")
    plt.ylabel("Scaled Quantities")
    plt.title("Everything as a function of Radius")
    plt.grid()
    plt.xlim(0, 1)
    plt.ylim(0,1)
    plt.legend()
    plt.savefig("{0}/many_{0}.png".format(n), dpi=300)
    plt.clf()
    plt.close()

    # Plot tau and tauinf - tau
    infi = -1
    for i in range(taus.shape[0]-1,0):
        if not np.isnan(taus[i]):
            infi = i
            break
    plt.plot(rs/rs[surf], taus[infi]-taus, label=r"$\tau(\infty)-\tau(r)$")
    plt.xlabel(r"$R/R_{surf}$")
    plt.ylabel("Optical Depth")
    plt.title("Optical depth as a function of Radius")
    plt.grid()
    plt.xlim(0, 1)
    plt.legend()
    plt.savefig("{0}/tau_{0}.png".format(n), dpi=300)
    plt.clf()
    plt.close()

    plt.plot(rs/rs[surf], taus, label=r"$\tau(r)$")
    plt.xlabel(r"$R/R_{surf}$")
    plt.ylabel("Optical Depth")
    plt.title("Optical depth as a function of Radius")
    plt.grid()
    plt.xlim(0, 1)
    plt.legend()
    plt.savefig("{0}/tau1_{0}.png".format(n), dpi=300)
    plt.clf()
    plt.close()


    # Opacity plot
    ks1 = kappa_H(rhos, Ts)
    ks2 = kappa_es
    ks3 = kappa_ff(rhos, Ts)

    if modified:
        ks4 = kappa_i(rhos, Ts)
        plt.plot(rs/rs[surf], np.log10(ks4), label=r"Extra absorption feature")
        ks = kappa_modified(rhos, Ts)
    else:
        ks = kappa(rhos, Ts)

    plt.plot(rs / rs[surf], np.log10(ks3), label=r"$log_{10}\kappa_{ff}$")
    plt.plot(rs / rs[surf], np.log10(np.ones(rhos.shape)*ks2), label=r"$log_{10}\kappa_{es}$")
    plt.plot(rs/rs[surf], np.log10(ks1), label=r"$log_{10}\kappa_{H}$")
    plt.plot(rs/rs[surf], np.log10(ks), "k--", label=r"$log_{10}\kappa$")
    plt.xlabel(r"$R/R_{surf}$")
    plt.ylabel("log10 Opacity")
    plt.title("Opacity as a function of Radius")
    plt.grid()
    plt.xlim(0, 1)
    plt.legend()
    plt.savefig("{0}/kappa_{0}.png".format(n), dpi=300)
    plt.clf()
    plt.close()

    # Luminosity derivative plot
    dLdrs = dLdr(rhos, rs, Ts)
    plt.plot(rs/rs[surf], dLdrs*rs[surf]/Ls[surf], label=r"$\frac{dL}{dr}$")
    plt.xlabel(r"$R/R_{surf}$")
    plt.ylabel(r"$dL/dr (L_{surf}/R_{surf})$")
    plt.xlim(0, 1)
    plt.title("Luminosity derivative as a function of Radius")
    plt.grid()
    plt.savefig("{0}/dLdr_{0}.png".format(n), dpi=300)
    plt.clf()
    plt.close()

    # Pressure plot
    Ps = P(rhos, Ts)
    Pdeg = ((3*np.pi**2)**(2/3))*hbar**2 *(rhos/mp)**(5/3) / (5*me)
    Pgamma = a * (Ts**4) /3
    Pgas = k * Ts * rhos / mump
    plt.plot(rs/rs[surf], Ps/Ps[0], label=r"$P_{total}$")
    plt.plot(rs/rs[surf], Pdeg/Ps[0], label=r"$P_{degen}$")
    plt.plot(rs/rs[surf], Pgamma/Ps[0], label=r"$P_{\gamma}$")
    plt.plot(rs/rs[surf], Pgas/Ps[0], label=r"$P_{ideal}$")
    plt.xlabel(r"$R/R_{surf}$")
    plt.ylabel(r"$P/P_c$")
    plt.xlim(0, 1)
    plt.legend()
    plt.title("Pressure as a function of Radius")
    plt.grid()
    plt.savefig("{0}/Pressure_{0}.png".format(n), dpi=300)
    plt.clf()
    plt.close()

    # dlogP/dlogT graph
    dlPdlT = np.zeros(Ps.shape)
    for i in range(Ps.shape[0]-1):
        dlPdlT[i] = (np.log10(Ps[i+1])-np.log10(Ps[i]))/(np.log10(Ts[i+1])-np.log10(Ts[i]))
    plt.plot(rs/rs[surf], dlPdlT)
    plt.xlabel(r"$R/R_{surf}$")
    plt.ylabel(r"$d\log_{10} P / \d\log_{10} T$")
    plt.xlim(0, 1)
    plt.title("Logarithmic derivative of P as a function of Radius")
    plt.grid()
    plt.savefig("{0}/logderivP_{0}.png".format(n), dpi=300)
    plt.clf()
    plt.close()

# This generates a Hertzprung Russel Diagram via simulating n stars
def mainsequence(qq, modified=False, system=star):

    global r_f, T_c

    LLs=[]; MMs=[]; TTs=[]; RRs =[]; fs = []; corr_TTs=[]
    rho_cs = []
    start = dt.datetime.now()
    for i,T_c in enumerate([1e6, 25e6]):  # enumerate(np.linspace(1.5e6, 35e6, 20)): # enumerate([8.23e6]): # brodericks star
        print("==================== Beginning the {}th Star, Tc: {} ====================".format(i, T_c))
        r_f = 20*Rsun # starter radius of integration, almost always gets increased automatically

        # Main loop, note that it will automatically retry failed bisections w slightly different bounds
        fail = False
        try:
            #rho_c_true, res = brentq(trial_soln, 300, 500000,  args=(T_c, r_f), full_output=True, xtol=1e-30, maxiter=500) # maxiter=100,
            rho_c_true = find_ics(300, 500000, T_c, r_f, 500, modified=modified, system=system)
            fail = False
        except ValueError as e:
            print("Retrying 1")
            fail = True
        if fail:
            try:
                #rho_c_true, res = brentq(trial_soln, 300, 400000,  args=(T_c, r_f), full_output=True, xtol=1e-30, maxiter=500) # maxiter=100,
                rho_c_true = find_ics(300, 400000, T_c, r_f, 500, modified=modified, system=system)
                fail = False
            except ValueError as e:
                fail = True
                print("Retrying 2")
        if fail:
            try:
                #rho_c_true, res = brentq(trial_soln, 10, 800000,  args=(T_c, r_f), full_output=True, xtol=1e-30, maxiter=500) # maxiter=100,
                rho_c_true = find_ics(10, 800000, T_c, r_f, 500, modified=modified, system=system)
                fail = False
            except ValueError as e:
                fail = True
                print("Retrying 3")
        if fail:
            try:
                #rho_c_true, res = brentq(trial_soln, 10, 600000,  args=(T_c, r_f), full_output=True, xtol=1e-30, maxiter=500)
                rho_c_true = find_ics(300, 600000, T_c, r_f, 500, modified=modified, system=system)
            except ValueError as e:
                print("Failed 4")
                continue # Just continue going if it fails all 4


        # done generating the ith star, might as well plot everything
        end = dt.datetime.now()
        rs, rhos, Ts, Ms, Ls, taus, surf, ff = rho_c_true
        print(ff)
        plot_all(rs, rhos, Ts, Ms, Ls, taus, surf, ff, n="MainSeq{}_{}".format(qq,i), modified=modified)

        # keep around some data for plotting HR
        LLs.append(Ls[surf])
        MMs.append(Ms[surf])
        RRs.append(rs[surf])
        TTs.append(Ts[surf])
        fs.append(ff)

        # We do some rudimentary correction for where the surface is, since detection is SUPER inaccurate
        if not modified:
            tsurf_expected = (Ls[surf] / (4. * np.pi * (rs[surf] ** 2) * sb)) ** (1. / 4.)
            corr_TTs.append(tsurf_expected)
        else:
            try:
                loaded = np.load("corrections.npz" )
                corr_TTs = loaded['corr_temps']
                fail = False
            except:
                fail = True

        # Plots the HR as it runs so we can examine progress
        fig = plt.figure()
        ax = plt.gca()
        plt.plot(np.array(TTs), np.array(LLs) / Lsun, 'b-o', label="Generated")
        if not modified:
            plt.plot(np.array(corr_TTs), np.array(LLs)/Lsun, "g--.", label="Corrected")
        else:
            if not fail:
               plt.plot(np.array(TTs) + (loaded['corr_temps'] - loaded['og_temps'])[:len(TTs)], np.array(LLs)[:len(TTs)] / Lsun, "g--.", label="Corrected")
        plt.plot(5760, 1, marker="+", color="orange", label="Sun")
        ax.set_yscale('log')
        ax.set_xscale('log')
        plt.xlim(6e2, 5e4)
        plt.ylim(1e-5, 1e6)
        plt.xlabel("Temperature (K)")
        plt.ylabel(r"Luminosity ($L_{\odot}$)")
        plt.title("Hertzprung-Russel Diagram")
        plt.gca().invert_xaxis()
        plt.legend()
        if os.path.isfile("MainSeq{}_HertzprungRussel.png".format(qq)):
            os.remove("MainSeq{}_HertzprungRussel.png".format(qq))
        plt.savefig("MainSeq{}_HertzprungRussel.png".format(qq), dpi=500)
        plt.clf()
        plt.close()

        print(Ts[surf], Ls[surf] / Lsun, rs[surf] / Rsun, Ms[surf]/Msun)

    print("Overall Took: ", (end - start).seconds)

    if not modified:
        if qq == "Real":
            np.savez("corrections.npz", og_temps=np.array(TTs), corr_temps=np.array(corr_TTs)  )
        else:
            np.savez("corrections_{}.npz".format(qq), og_temps=np.array(TTs), corr_temps=np.array(corr_TTs) )


    # Basic error plot
    epochs = np.arange(0, len(fs), 1)
    plt.plot(epochs, fs)
    plt.ylabel("Error")
    plt.savefig("MainSeq{}_error.png".format(qq), dpi=500 )
    plt.clf()
    plt.close()

    # Mass Luminosity over the main sequence
    plt.plot(np.array(MMs)/Msun, np.array(LLs)/Lsun, "b-o", label="Calculated")
    emp_Ms = np.linspace(np.nanmin(MMs), np.nanmax(MMs), 1000)
    emp_Ls = []
    for M in emp_Ms:
        if M<0.7*Msun:
            emp_Ls.append(0.35*np.power(M/Msun, 2.62))
        else:
            emp_Ls.append(1.02*np.power(M/Msun,3.92))
    plt.plot(np.array(emp_Ms)/Msun, np.array(emp_Ls), "g--.", label="Empirical Fit")
    plt.xlabel(r"$M/M_{\odot}$")
    plt.ylabel(r"$L/L_{\odot}$")
    plt.title("Mass-Luminosity Relationship")
    ax = plt.gca()
    ax.set_yscale('log')
    ax.set_xscale('log')
    plt.xlim(8e-1,2e2)
    plt.ylim(1e-4, 1e8)
    plt.legend()
    plt.savefig("MainSeq{}_LM.png".format(qq), dpi=500)
    plt.clf()
    plt.close()

    # Mass Radius over the main sequence
    plt.plot(np.array(MMs)/Msun, np.array(RRs)/Rsun, "b-o", label="Calculated")
    emp_Rs = []
    for M in emp_Ms:
        if M>1.66*Msun:
            emp_Rs.append(1.33*np.power(M/Msun, 0.555))
        else:
            emp_Rs.append(1.06*np.power(M/Msun,0.945))
    plt.plot(np.array(emp_Ms)/Msun, np.array(emp_Rs), "g--.", label="Empirical Fit")
    plt.xlabel(r"$M/M_{\odot}$")
    plt.ylabel(r"$R/R_{\odot}$")
    plt.title("Mass-Radius Relationship")
    ax = plt.gca()
    ax.set_yscale('log')
    ax.set_xscale('log')
    plt.xlim(8e-1,2e2)
    plt.ylim(8e-2, 2e2)
    plt.legend()
    plt.savefig("MainSeq{}_MR.png".format(qq), dpi=500)
    plt.clf()
    plt.close()

# Uncomment the one you want
# I run them in parallel because it doesnt take a ton of resources on my pc
# Make sure the unmodified main sequence finishes first though! (I put the corrections on git so this is only if you mess around with the params too much)
if __name__ == "__main__":
    global Ti
    global kappa_0

    # Do a test run with profiling
    # import cProfile
    # cProfile.run('trial_soln(300, 8.23e6, 1e10, system=star, optimize=False, modified=False)')

    # Unmodified Main Sequence
    mainsequence("Real", modified=False)

    # Modifications:
    Ti = 2e6
    kappa_0 = 1e24
    # mainsequence("Modified_1", modified=True, system=star_modified)

    Ti = 8e6
    kappa_0 = 5 * 1e24
    # mainsequence("Modified_2", modified=True, system=star_modified)
    #
    Ti = 4e6
    kappa_0 = 2 * 1e24
    # mainsequence("Modified_3", modified=True, system=star_modified)
    #
    Ti = 0.5e6
    kappa_0 = 2e24
    # mainsequence("Modified_4", modified=True, system=star_modified)

    Ti = 8e3
    kappa_0 = 20e24
    # mainsequence("Modified_5", modified=True, system=star_modified)


