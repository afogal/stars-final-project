import numpy as np 
from scipy.integrate import solve_ivp
from scipy.optimize import bisect, minimize, brentq
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
#from matplotlib import rc
#rc('text', usetex=True)

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
gamma = 5/3
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
rho_c = 88363.91364833503 # roughly rho_c for the sun
drho = 0.1*rho_c
T_c = 1e7
r_c = 0.0001
r_f = 15*Rsun #1.4e9 / 1.5  # roughly 2R_sun

# pressure regieme =================================
def P(rho, T):
    return ((3 * np.pi**2)**(2/3)/5) * (hbar**2/me) * (rho/mp)**(5/3) + rho * (k*T/mump) + (1/3)*a*(T**4)


# the big 5 =================================
# theyre kinda hard to read, keep the doc side by side
def drhodr( rho, T, L, r, M, P):
    return -(G*M*rho/r**2 + dPdT( rho, T) * dTdr(rho, T, L, r, M, P)) /dPdrho(rho,T)

def dTdr(  rho, T, L, r, M, P):
    return - np.nanmin([ 3 * kappa(rho,T) * rho * L/(16*np.pi*a*c*(T**3)*(r**2)), (1 - 1/gamma)*(T/P)*G*M*rho/r**2] )

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
    return 1/(1/kappa_H(rho, T) + 1/np.nanmax([kappa_es*np.ones(rho.shape), kappa_ff(rho,T)]) )

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
    return 1 / (1 / kappa_H(rho, T) + 1 / np.nanmax([kappa_es, kappa_ff(rho, T), kappa_i(rho,T)]))


# More boundary conditions:
M_c = (4/3)*np.pi*(r_c**3)*rho_c
L_c = (4/3)*np.pi*(r_c**3)*rho_c*epsilon(rho_c, T_c)
tau_c = 0
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

def trial_soln(rho_c_trial, T_c, r_f, system=star, optimize=True):

    # if you dont have much ram, change the stepsize to be larger
    #rs1 = np.arange(r_c,r_f,int(r_f//5e6))
    rs1 = np.arange(r_c, r_f, 200)
    #all = solve_ivp(system, (r_c, r_f), np.array([rho_c_trial, T_c, M_c, L_c, tau_c]), atol=1e-18 ,method='RK45', t_eval=rs1 )
    all = solve_ivp(system, (r_c, r_f), np.array([rho_c_trial, T_c, M_c, L_c, tau_c]), method='RK45', atol=1e-18 , max_step=0.0005*r_f)

    rhos1 = all.y[0,:]
    Ts1 = all.y[1,:]
    Ms1 = all.y[2,:]
    Ls1 = all.y[3,:]
    taus1 = all.y[4,:]
    rs1 = all.t


    # was rf far enough?
    ks = kappa(rhos1, Ts1)
    drhodrs = drhodr(rhos1, Ts1, Ls1, rs1, Ms1, P(rhos1, Ts1))
    dtau1 =  ks * rhos1 * rhos1 / np.abs(drhodrs)


    # take tau(inf) as the last non-nan value
    for i in range(1, taus1.shape[0]):
        if not np.isnan(taus1[-i]):
            tauinf_i = -i
            break

    # finds the index of the surface, tau(inf) - tau(R_Surf) = 2/3
    surf_i1 = np.nanargmin(np.abs(taus1[tauinf_i] - taus1[:tauinf_i] - 2/3))
    surf_i2 = np.nanargmin(np.abs(Ms1 - 1000*Msun))
    surf_i1 = np.nanmin((surf_i1, surf_i2))
    # print( taus1[tauinf_i] - taus1[surf_i1] -2/3)
    # print(tauinf_i, surf_i1, taus1.shape)


    # interp
    tau = interp1d(rs1, taus1, kind="cubic", fill_value="extrapolate")
    tt = interp1d(taus1[tauinf_i]-taus1[:tauinf_i], rs1[:tauinf_i], fill_value="extrapolate", kind="linear")
    rho = interp1d(rs1, rhos1, kind="cubic", fill_value="extrapolate")
    T = interp1d(rs1, Ts1, kind="cubic", fill_value="extrapolate")
    M = interp1d(rs1, Ms1, kind="cubic", fill_value="extrapolate")
    L = interp1d(rs1, Ls1, kind="cubic", fill_value="extrapolate")

    rsurf = tt(2/3)
    surf_i1 = np.nanargmin(np.abs(rs1-rsurf))
    print("Curr surf: ", rsurf/Rsun)
    print("tau error before interp: ", taus1[tauinf_i] - taus1[surf_i1] - 2 / 3)

    # rsurf = minimize(lambda x: np.abs(taus1[tauinf_i] - tau(x) -2/3), rs1[surf_i1], bounds=[(0,rs1[tauinf_i])]  )
    # print( taus1[tauinf_i] - tau(rsurf.x[0]) -2/3)
    # rsurf = rsurf.x[0]

    # rsurf = brentq(lambda x: taus1[tauinf_i] - tau(x) -2/3, rs1[surf_i1-10], rs1[-1], xtol=1e-30)
    # print(taus1[tauinf_i] - tau(rsurf) - 2 / 3)
    # print(rsurf/Rsun)

    rs1 = np.linspace(0.1, rsurf+(rsurf)/4998, 5000)
    taus1 = tau(rs1)
    rhos1 = rho(rs1)
    Ts1 = T(rs1)
    Ms1 = M(rs1)
    Ls1 = L(rs1)
    surf_i1 = np.nanargmin(np.abs(rs1-rsurf))
    tauinf_i = -1

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

        # plots delta tau = dtau = kappa rho^2 / abs(drho/dr)
        plt.plot(rs1, dtau1, "r-", label="Opacity Proxy")
        plt.scatter(surf_i1, dtau1[surf_i1], marker="o", c="b")
        plt.ylim(0, 1)
        plt.xlim(0, rs1[-1])
        plt.grid()
        plt.savefig("dtau.png", dpi=300)
        plt.clf()
        plt.close()

    if verbosity > 0:
        print("Curr rho: ", rho_c_trial)
        print("Curr Err: ", frac_error1)
        print("Curr dtau: ", dtau1[tauinf_i])
        print("Err in tau(surf): ", np.min(np.abs(taus1[tauinf_i] - taus1[surf_i1] - 2/3)))
        print(" ")



    if not optimize:
        return np.array([rs1, rhos1, Ts1, Ms1, Ls1, taus1, surf_i1, frac_error1])
    else:
        return frac_error1


def find_ics(rho_c_min, rho_c_max, max_iters, system=star):

    # beginning points
    f2 = trial_soln(rho_c_min, system=system)
    f1 = trial_soln(rho_c_max, system=system)
    print("Initial error bounds: {} {}".format(f1, f2))

    # root not in interval defined by [rho_c_min, rho_c_max]
    if f1*f2 > 0:
        print("Pick better bounds, one needs to be negative")
        return -1
    else:
        for n in range(max_iters):
            print("Optimization round: ", n)
            rho_m = (rho_c_max + rho_c_min)/2

            f3 = trial_soln(rho_m, system=system)
            print(rho_m, f3)

            # binary search, change one bound and recompute the other
            if f1*f3 < 0:
                rho_c_min = rho_m
                f2 = f3 #trial_soln(rho_c_min, system=system)[-1]
                print(rho_c_min, f2)

            elif f2*f3 < 0:
                rho_c_max = rho_m
                f1 = f3 #trial_soln(rho_c_max, system=system)[-1]
                #print(rho_c_max, f1)
            elif np.abs(f3) < 3e-5: # end condition
                return rho_m

    return (rho_c_max + rho_c_min)/2

def plot_all(rs, rhos, Ts, Ms, Ls, taus, surf, f, n=0):

    try:
        import os
        os.mkdir("{}".format(n))
    except: # folder already exists
        pass

    # HR diagram
    logT = np.log10(Ts)
    logL = np.log10(Ls/Lsun)
    plt.plot(logT, logL, "k-")
    plt.xlabel(r"$\log_{10} T$")
    plt.ylabel(r"$\log_{10} L$")
    plt.title("Hertzprung-Russel Diagram")
    plt.grid()
    #plt.xlim(3,7)
    plt.ylim(-5,5)
    plt.gca().invert_xaxis()
    plt.savefig("{0}/HR_{0}.png".format(n), dpi=300)
    plt.clf()
    plt.close()

    # L/Lsun as a fn of M/Msun
    plt.plot(Ms/Msun, Ls/Lsun, "b-" )
    plt.xlabel(r"$M/M_{\odot}$")
    plt.ylabel(r"$L/L_{\odot}$")
    plt.title("Luminosity as a function of Mass")
    plt.grid()
    plt.savefig("{0}/LM_{0}.png".format(n), dpi=300)
    plt.clf()
    plt.close()

    # R/Rsun as a fn of M/Msun
    plt.plot(Ms/Msun, rs/Rsun, "b-" )
    plt.xlabel(r"$M/M_{\odot}$")
    plt.ylabel(r"$R/R_{\odot}$")
    plt.title("Radius as a function of Mass")
    plt.grid()
    plt.savefig("{0}/RM_{0}.png".format(n), dpi=300)
    plt.clf()
    plt.close()

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

    plt.plot(rs/rs[surf], Ts/Ts[surf], label=r"$T/T_{surf}$")
    plt.plot(rs/rs[surf], Ms/Ms[surf], label=r"$M/M_{surf}$")
    plt.plot(rs/rs[surf], Ls/Ls[surf], label=r"$L/L_{surf}$")
    plt.xlabel(r"$R/R_{surf}$")
    plt.ylabel("Scaled Quantities")
    plt.title("Everything as a function of Radius")
    plt.grid()
    plt.xlim(0, 1.5)
    plt.legend()
    plt.savefig("{0}/many_surf_extra_{0}.png".format(n), dpi=300)
    plt.clf()
    plt.close()

    infi = -1
    for i in range(taus.shape[0]-1,0):
        if not np.isnan(taus[i]):
            infi = i
            break
    #print(infi, taus[infi])
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

    ks = kappa(rhos, Ts)
    plt.plot(rs/rs[surf], ks, label=r"$\kappa$")
    plt.xlabel(r"$R/R_{surf}$")
    plt.ylabel("Opacity")
    plt.title("Opacity as a function of Radius")
    plt.grid()
    plt.xlim(0, 1)
    plt.legend()
    plt.savefig("{0}/kappa_{0}.png".format(n), dpi=300)
    plt.clf()
    plt.close()

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



def main():
    import datetime as dt
    global r_f, T_c

    # tweak the inputs here to get a good rho_c
    #rho_c_true = find_ics(88363.91364835274, 88363.91364835275, 100)
    #print(rho_c_true)

    # rho_cs = []
    # for T_c in [0.2e7, 0.4e7, 0.6e7, 0.8e7, 1e7, 1.2e7, 1.4e7, 1.6e7, 1.8e7, 1e8, 1e9, 1e10]:
    #     #r_f = 15*Rsun if T_c > 1e7 else 2*Rsun
    #     start = dt.datetime.now()
    #     rho_c_true, res = bisect(trial_soln, 300, 500000,  args=(T_c, r_f), full_output=True, xtol=1e-30) # maxiter=100,
    #     print(rho_c_true)
    #     end = dt.datetime.now()
    #     print("Optimization Took: ", (end-start).seconds)
    #     print(res)
    #
    #     rs, rhos, Ts, Ms, Ls, taus, surf, f = trial_soln(rho_c_true, optimize=False)
    #     print(f)
    #
    #     print(Ts[surf], Ls[surf] / Lsun, rs[surf] / Rsun)
    #
    #     rho_cs.append((rho_c_true, f, Ts[surf], Ls[surf] / Lsun, rs[surf] / Rsun) )

    T_c = 1e7; r_f = 2*Rsun
    start = dt.datetime.now()
    #rho_c_true, res = bisect(trial_soln, 300, 500000,  args=(T_c, r_f), full_output=True, xtol=1e-30)
    rho_c_true, res = brentq(trial_soln, 300, 500000 , args=(T_c, r_f), full_output=True, xtol=1e-30)
    print(rho_c_true)
    end = dt.datetime.now()
    print("Optimization Took: ", (end - start).seconds)
    print(res)

    # print(rho_cs)
    # rho_c =

    # get soln with our rho_c
    rs, rhos, Ts, Ms, Ls, taus, surf, f = trial_soln(rho_c_true, T_c, r_f, optimize=False)
    print(f)

    print(Ts[surf], Ls[surf]/Lsun, rs[surf]/Rsun, Ms[surf]/Msun)

    # save all the data if it takes a long time to generate solns
    #np.savez("mainseq", rs=rs, rhos=rhos, Ts=Ts, Ms=Ms, Ls=Ls, taus=taus, f=f)
    # np.loadz?


    # plot all the things!
    plot_all(rs ,rhos, Ts, Ms, Ls, taus, surf, f, n="main")




