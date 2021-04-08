import numpy as np 
from scipy.integrate import solve_ivp
from scipy.optimize import bisect, minimize, brentq
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
#from matplotlib import rc
#rc('text', usetex=True)
import sys

import warnings

verbosity = 1
if verbosity < 2:
    warnings.filterwarnings("ignore") # IF THE CODE IS BROKEN, COMMENT THIS LINE OUT AND RUN IT AGAIN


# Constants!
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
def P(rho, T):
    return ((3 * np.pi**2)**(2/3)/5) * (hbar**2/me) * (rho/mp)**(5/3) + rho * (k*T/mump) + (1/3)*a*(T**4)


# the big 5 =================================
# theyre kinda hard to read, keep the doc side by side
def drhodr( rho, T, L, r, M, P):
    return -(G*M*rho/r**2 + dPdT( rho, T) * dTdr(rho, T, L, r, M, P)) /dPdrho(rho,T)

def dTdr(  rho, T, L, r, M, P):
    return - np.nanmin([ np.abs(3 * kappa(rho,T) * rho * L/(16*np.pi*a*c*(T**3)*(r**2))), np.abs((1 - 1/gamma)*(T/P)*G*M*rho/r**2)], axis=0 )

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
    return (2.5e-32)*(Z/0.02)*np.power(rho/1e3, 0.5)*(T**9)

def kappa(rho, T):
    if isinstance(T, np.ndarray):
        out = np.zeros(T.shape)
        for i in range(out.shape[0]):
            if(T[i]>1e4):
                out[i] =1/(1/kappa_H(rho[i], T[i]) + 1/np.nanmax([kappa_es, kappa_ff(rho[i],T[i])], axis=0) )
            else:
                out[i] = 1 / (1 / kappa_H(rho[i], T[i]) + 1 / np.nanmin([kappa_es, kappa_ff(rho[i], T[i])],axis=0))
        return out
    else:
        if(T>1e4):
            try:
                return 1/(1/kappa_H(rho, T) + 1/np.nanmax([kappa_es*np.ones(rho.shape), kappa_ff(rho,T)], axis=0) )
            except AttributeError:
                return 1 / (1 / kappa_H(rho, T) + 1 / np.nanmax([kappa_es, kappa_ff(rho, T)], axis=0))
        else:
            try:
                return 1/(1/kappa_H(rho, T) + 1/np.nanmin([kappa_es*np.ones(rho.shape), kappa_ff(rho,T)], axis=0) )
            except AttributeError:
                return 1 / (1 / kappa_H(rho, T) + 1 / np.nanmin([kappa_es, kappa_ff(rho, T)], axis=0))

# these ones are our group specific
# The specific Ti and kappa_0 are set below
def kappa_i(rho, T):
    global Ti
    global kappa_0

    if isinstance(T, np.ndarray):
        out = np.zeros(T.shape)
        for i in range(out.shape[0]):
            if(T[i]<Ti):
                out[i] = kappa_0 * (Z + 0.0001) * np.sqrt(rho[i] / 1e3) * np.power(T[i], -3.5)
        return out
    else:
        if (T < Ti):
            return kappa_0 * (Z + 0.0001) * np.sqrt(rho / 1e3) * np.power(T, -3.5)

        else:
            return 0


def kappa_modified(rho, T):
    if isinstance(T, np.ndarray):
        out = np.zeros(T.shape)
        for i in range(out.shape[0]):
            if(T[i]>1e4):
                out[i] =1/(1/kappa_H(rho[i], T[i]) + 1/np.nanmax([kappa_es, kappa_ff(rho[i],T[i]), kappa_i(rho[i], T[i])], axis=0) )
            else:
                out[i] = 1 / (1 / kappa_H(rho[i], T[i]) + 1 / np.nanmin([kappa_es, kappa_ff(rho[i], T[i]), kappa_i(rho[i], T[i])],axis=0))
        return out
    else:
        if(T>1e4):
            try:
                return 1/(1/kappa_H(rho, T) + 1/np.nanmax([kappa_es*np.ones(rho.shape), kappa_ff(rho,T),kappa_i(rho, T)], axis=0) )
            except AttributeError:
                return 1 / (1 / kappa_H(rho, T) + 1 / np.nanmax([kappa_es, kappa_ff(rho, T)], axis=0))
        else:
            try:
                return 1/(1/kappa_H(rho, T) + 1/np.nanmin([kappa_es*np.ones(rho.shape), kappa_ff(rho,T),kappa_i(rho, T)], axis=0) )
            except AttributeError:
                return 1 / (1 / kappa_H(rho, T) + 1 / np.nanmin([kappa_es, kappa_ff(rho, T),kappa_i(rho, T)], axis=0))

def star_modified(r,y):
    rho = y[0]
    T = y[1]
    M = y[2]
    L = y[3]
    tau = y[4]

    rhodot = drhodr(rho, T, L, r, M, P(rho, T))
    Tdot = dTdr(rho, T, L, r, M, P(rho, T))
    Mdot = dMdr(r, rho)
    Ldot = dLdr(rho, r, T)
    taudot = dtaudr(rho, kappa_modified(rho, T))

    return np.array([rhodot, Tdot, Mdot, Ldot, taudot])

# More boundary conditions:
def f(L_surf, R_surf, T_surf):
    return (L_surf - 4*np.pi*sb*(R_surf**2)*(T_surf**4))/np.sqrt( 4*np.pi*sb*(R_surf**2)*(T_surf**4)*L_surf  )

# the system unmodified
def star(r,y):
    rho = y[0]
    T = y[1]
    M = y[2]
    L = y[3]
    tau = y[4]

    rhodot = drhodr(rho, T, L, r, M, P(rho, T))
    Tdot = dTdr(rho, T, L, r, M, P(rho, T))
    Mdot = dMdr(r, rho)
    Ldot = dLdr(rho, r, T)
    taudot = dtaudr(rho, kappa(rho, T))

    return np.array([rhodot, Tdot, Mdot, Ldot, taudot])



def trial_soln(rho_c_trial, T_c, r_f, system=star, optimize=True, final=False, modified=False):

    count = 0 # Counter variable

    # A while loop to rerun when delta tau isnt small enough (makes a big difference!!)
    while True:
        # The other half of the BCs
        M_c = (4 / 3) * np.pi * (r_c ** 3) * rho_c_trial
        L_c = (4 / 3) * np.pi * (r_c ** 3) * rho_c_trial * epsilon(rho_c_trial, T_c)

        # Use our modification
        if not modified:
            tau_c = kappa(rho_c_trial, T_c) * rho_c_trial
        else:
            tau_c = kappa_modified(rho_c_trial, T_c) * rho_c_trial

        # Runs with a better stepsize on the final try (speeds up bisection)
        # If it runs really slowly on your computer, try increasing the max_stepsize
        # if not final:
        all = solve_ivp(system, (r_c, r_f), np.array([rho_c_trial, T_c, M_c, L_c, tau_c]), method='RK45', atol=1e-18)# , max_step=0.01 * r_f)
        # else:
        #     all = solve_ivp(system, (r_c, r_f), np.array([rho_c_trial, T_c, M_c, L_c, tau_c]), method='RK45', atol=1e-18, max_step=0.0005 * r_f)
        # all = solve_ivp(system, (r_c, r_f), np.array([rho_c_trial, T_c, M_c, L_c, tau_c]), method='RK45')

        # Variable renaming
        rhos1 = all.y[0, :]
        Ts1 = all.y[1, :]
        Ms1 = all.y[2, :]
        Ls1 = all.y[3, :]
        taus1 = all.y[4, :]
        rs1 = all.t

        # Using our modification
        if not modified:
            ks = kappa(rhos1, Ts1)
        else:
            ks = kappa_modified(rhos1, Ts1)

        # Calculate delta tau
        drhodrs = drhodr(rhos1, Ts1, Ls1, rs1, Ms1, P(rhos1, Ts1))
        dtau1 = ks * rhos1 * rhos1 / np.abs(drhodrs)

        # take tau(inf) as the last non-nan value (done the fancy way)
        tauinf_i = (~np.isnan(taus1)).cumsum().argmax()

        count += 1
        print("dtau: ",dtau1[(~np.isnan(dtau1)).cumsum().argmax()])
        if dtau1[(~np.isnan(dtau1)).cumsum().argmax()] < 0.1:
            break
        else:
            r_f *=  2.

    print("Reran for better dtau ", count)



    print("tauinf at: {} tauinf: {}".format( tauinf_i/taus1.shape[0], taus1[tauinf_i]))

    # finds the index of the surface, tau(inf) - tau(R_Surf) = 2/3
    surf_i1 = np.nanargmin(np.abs(taus1[tauinf_i] - taus1[0:tauinf_i] - 2./3.))
    surf_i2 = np.nanargmin(np.abs(Ms1 - 1000*Msun))
    surf_i1 = np.nanmin((surf_i1, surf_i2))

    tauinf = taus1[tauinf_i]
    surf_g1 = surf_i1
    rsurf_1 = rs1[surf_g1]



    # interpolation of varaibles (to get good answers with bad stepsizes)
    tau = interp1d(rs1, taus1, kind="cubic", fill_value="extrapolate")
    tt = interp1d(tauinf-taus1[0:tauinf_i], rs1[0:tauinf_i], fill_value="extrapolate", kind="linear")
    rho = interp1d(rs1, rhos1, kind="cubic", fill_value="extrapolate")
    T = interp1d(rs1, Ts1, kind="cubic", fill_value="extrapolate")
    M = interp1d(rs1, Ms1, kind="cubic", fill_value="extrapolate")
    L = interp1d(rs1, Ls1, kind="cubic", fill_value="extrapolate")
    #
    rsurf_2 = tt(2./3.)

    # ALTERNATIVE SURFACE DETERMINATION (NOT USED)
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

    # Automagic surface determination based on error min (worked like ass and caused really bad jumping around)
    # min_method = np.argmin([error1, error2, error3, error4, error5])
    rsurf = rsurf_2 #eval("rsurf_{}".format(min_method+1)) #rs1[surf_g1] #rsurf_4 #(rsurf + rs1[surf_g1])/2

    # was rf far enough?
    if not modified:
        ks = kappa(rhos1, Ts1)
    else:
        ks = kappa_modified(rhos1, Ts1)
    drhodrs = drhodr(rhos1, Ts1, Ls1, rs1, Ms1, P(rhos1, Ts1))
    dtau1 =  ks * rhos1 * rhos1 / np.abs(drhodrs)

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
        print("Curr dtau: ", dtau1[(~np.isnan(dtau1)).cumsum().argmax()])
        print("Err in tau(surf): ", np.abs(tauinf - taus1[-1] - 2./3.))
        print(" ")

    if not optimize:
        return np.array([rs1, rhos1, Ts1, Ms1, Ls1, taus1, surf_i1, frac_error1])
    else:
        return frac_error1


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

    if f1*f2 > 0: # root not in interval defined by [rho_c_min, rho_c_max]
        print("Pick better bounds, one needs to be negative")
        raise ValueError # used to auto adjust bounds later
    else:
        for n in range(max_iters):
            intcount = 0
            try:
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
                elif len(fs) > 5 and np.min(np.abs(fs[-1] - np.array(fs[-3:]))) < 1e-4 and np.abs(fs[-1])<1 and n>50:
                    print("f is small and fs are similar")
                    break
                elif len(rhos) > 5 and np.min(np.abs(rhos[-1] - np.array(rhos[-3:]))) < 1e-10 and np.min(np.abs(fs[-1] - np.array(fs[-3:]))) < 1e-10 and n>250:
                    print("Rho doesnt change, f changes little, f is small")
                    break
                elif n>200 and np.min(np.abs(fs)) < 1:
                    print("Over 200 and err <1")
                    break

                print("Curr min error: ", np.min(np.abs(fs)))
                # Okay so this looks kinda janky, and it is
                # Basically, the solver is nondeterministic-ish, so running w the same inputs doesnt always give the same outputs
                # Which sometimes leads to getting solns with really bad error even after bisection
                # So simply I stored all the variables and spit out the ones that came with whatever the minimum error was
                if np.min(np.abs(fs)) != oldmin or  oldmin == -1:
                    oldmin = np.min(np.abs(fs))
                    keepvars = np.array([rs, rhoss, Ts, Ms, Ls, taus, surf, fs[-1]])

            except KeyboardInterrupt: # This didnt end up mattering for me, crtl C just kills python itself
                intcount += 1
                if intcount>3:
                    break
                continue

        if np.min(np.abs(fs)) != oldmin or  oldmin == -1:
            oldmin = np.min(np.abs(fs))
            keepvars = np.array([rs, rhoss, Ts, Ms, Ls, taus, surf, fs[-1]])
        mini = np.argmin(np.abs(fs))

    return keepvars #rhos[mini] #(rho_c_max + rho_c_min)/2

def plot_all(rs, rhos, Ts, Ms, Ls, taus, surf, f, n=0, modified=False):

    try:
        import os
        os.mkdir("{}".format(n))
    except: # folder already exists
        pass

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

def mainsequence(qq, modified=False, system=star):
    import datetime as dt
    import os
    global r_f, T_c

    LLs=[]; MMs=[]; TTs=[]; RRs =[]; fs = []; corr_TTs=[]
    rho_cs = []
    start = dt.datetime.now()
    for i,T_c in  enumerate(np.linspace(1.5e6, 35e6, 20)): # enumerate([8.23e6]): # brodericks star
        print("==================== Beginning the {}th Star, Tc: {} ====================".format(i, T_c))
        r_f = 20*Rsun # starter radius of integration, almost always gets increased automatically

        # Main loop, note that it will automatically retry failed bisections w slightly different bounds
        fail = False; res=0
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



        end = dt.datetime.now()
        print(res)
        rs, rhos, Ts, Ms, Ls, taus, surf, ff = rho_c_true #trial_soln(rho_c_true, T_c, r_f, optimize=False, final=True)
        print(ff)
        plot_all(rs, rhos, Ts, Ms, Ls, taus, surf, ff, n="MainSeq{}_{}".format(qq,i), modified=modified)

        LLs.append(Ls[surf])
        MMs.append(Ms[surf])
        RRs.append(rs[surf])
        TTs.append(Ts[surf])
        fs.append(ff)

        if not modified and qq == "Real":
            tsurf_expected = (Ls[surf] / (4. * np.pi * (rs[surf] ** 2) * sb)) ** (1. / 4.)
            corr_TTs.append(tsurf_expected)
        else:
            try:
                loaded = np.load("corrections.npz" )
                corr_TTs = loaded['corr_temps']
                fail = False
            except:
                fail = True

        fig = plt.figure()
        ax = plt.gca()
        ax.plot(np.array(TTs), np.array(LLs) / Lsun, 'b-o', label="Generated")
        plt.plot(5760, 1, "+", label="Sun")
        if not modified:
            plt.plot(np.array(corr_TTs), np.array(LLs)/Lsun, "g--.", label="Corrected")
        else:
            if not fail:
               plt.plot(np.array(TTs) + (loaded['corr_temps'] - loaded['og_temps'])[:len(TTs)], np.array(LLs)[:len(TTs)] / Lsun, "g--.", label="Corrected")
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
        np.savez("corrections.npz", og_temps=np.array(TTs), corr_temps=np.array(corr_TTs)  )


    epochs = np.arange(0, len(fs), 1)
    plt.plot(epochs, fs)
    plt.ylabel("Error")
    plt.savefig("MainSeq{}_error.png".format(qq), dpi=500 )
    plt.clf()
    plt.close()

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
# Make sure the unmodified main sequence finishes first though! (I put the corrections on git so this is only if you dick around with the params too much)
if __name__ == "__main__":
    global Ti
    global kappa_0

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


