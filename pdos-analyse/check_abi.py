import os
import sys
from abipy.ppcodes.oncv_parser import OncvParser
import matplotlib.pyplot as plt
from aiida import orm
import aiida
from collections import namedtuple
import numpy as np
from scipy.integrate import cumtrapz

Fd_Params = namedtuple("Fermi_Dirac", ['energy', 'sigma', 'mirror']) # mirror for upside down upon 0.5

aiida.load_profile()

def _color_set(l):
    # color set
    if l == 0:
        color = 'blue'
    elif l == 1:
        color = 'red'
    elif l == 2:
        color = 'black'
    else:
        color = 'green'
        
    return color

def fermi_dirac(e, fd_params: Fd_Params):
    """array -> array"""
    res = 1.0 / (np.exp((e - fd_params.energy) / fd_params.sigma) + 1.0)
    
    assert isinstance(fd_params.mirror, bool)
    
    if fd_params.mirror:
        return 1.0 - res
    else:
        return res
    
def create_weights(xs, fd1: Fd_Params, fd2: Fd_Params):
    assert np.all(xs[:-1] <= xs[1:])    # make sure that the energies is in acsending order
    
    boundary = (fd1.energy + fd2.energy) / 2.0
    
    condition = (xs < boundary)
    _energies = np.extract(condition, xs)
    weights1 = fermi_dirac(_energies, fd_params=fd1)
    
    condition = (xs >= boundary)
    _energies = np.extract(condition, xs)
    weights2 = fermi_dirac(_energies, fd_params=fd2)

    weights = np.concatenate((weights1, weights2))
    
    return weights
    
def compute_integ(abs_diff, weights, energies):
    integ = cumtrapz(abs_diff * weights, x=energies) / (energies[-1] - energies[0]) # normalized cumulated integ

    return integ

def run():
    
    fd1 = Fd_Params._make([0.0, 0.25, True])
    fd2 = Fd_Params._make([6.0, 0.25, False])
    
    fig, axs = plt.subplots(3, 1)
    
    pk = sys.argv[1]
    try:
        n = orm.load_node(pk)
        pspout = os.path.join(n.outputs.remote_folder.get_remote_path(), 'aiida.out')
    except:
        pspout = sys.argv[1]
    
    p = OncvParser(pspout)
    p.scan(verbose=1)
    
    logders = p.atan_logders
    lmax = p.lmax
    
    for ld in logders.ae.values():
        # line color set
        color = _color_set(l=ld.l)
            
        axs[0].plot(ld.energies, ld.values, label=f"AE: l={ld.l}", color=color, linestyle='dashed')
        
    for ld in logders.ps.values():
        # line color set
        color = _color_set(l=ld.l)
        
        axs[0].plot(ld.energies, ld.values, label=f"PS: l={ld.l}", color=color, linestyle='solid')
    

    axs[0].axvline(x=fd1.energy, linestyle="dotted")
    axs[0].axvline(x=fd2.energy, linestyle="dotted")
    axs[0].set_ylabel("Atan Logders")
    axs[0].legend(loc="upper right", prop={'size': 6})
    
    energies = np.sort(list(logders.ps.values())[0].energies)
    weights= create_weights(xs=energies, fd1=fd1, fd2=fd2)
    integ_final = np.zeros_like(energies)
    for l in logders.ae:        
        # diff with counting the weight on fermi dirac distribution
        f1, f2 = logders.ae[l], logders.ps[l]
                
        sortind = np.argsort(f1.energies) # must do the sort since we use concatenate to combine split range
        energies = f1.energies[sortind]
        
        abs_diff = np.abs(f1.values - f2.values)    # compare the absolute diff
        abs_diff = abs_diff[sortind]
        
        integ = cumtrapz(abs_diff * weights, x=energies, initial=0.0) / (energies[-1] - energies[0]) # normalized cumulated integ
        print(f"l={l}: {integ[-1]}")
        
        weight_unbound = 0.1
        
        if not l < lmax+1:
            # unbound states
            integ *= weight_unbound
            
        integ_final += integ
        
    axs[1].plot(energies, weights, label="FD-like weight")
    axs[1].axvline(x=fd1.energy, linestyle="dotted")
    axs[1].axvline(x=fd2.energy, linestyle="dotted")
        
    axs[1].plot(energies, integ_final * 10, label="Cumulated integrate (*10)")
    axs[1].set_ylabel("weight/integ")
    axs[1].legend(prop={'size': 6})
    
    k_ecut = p.kene_vs_ecut
    ecut_hint_low, ecut_hint_high = p.hints["low"]["ecut"], p.hints["high"]["ecut"]
    print(f"ecut_low: {ecut_hint_low}")
    print(f"ecut_high: {ecut_hint_high}")
    
    for convdata in k_ecut.values():
        color = _color_set(l=convdata.l)
        
        axs[2].plot(convdata.energies, convdata.values, label=f"l={convdata.l}", color=color)
        
    # plot ecut hint
    axs[2].axvline(x=ecut_hint_low, label="low hint ecut", linestyle="dotted")
    axs[2].axvline(x=ecut_hint_high, label="high hint ecut", linestyle="dotted")
        
    axs[2].set_xlabel("energies (Ha)")
    axs[2].set_ylabel("Kinetic energy")
    axs[2].legend(prop={'size': 6})
    
    fig.tight_layout()
    try:
        filename = sys.argv[2]
    except:
        filename = "temp.png"
    plt.savefig(filename)

if __name__ == '__main__':
    run()