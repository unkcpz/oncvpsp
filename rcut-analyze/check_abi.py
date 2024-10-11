import sys
from myoncvporser import OncvParser
import matplotlib.pyplot as plt
from collections import namedtuple
import numpy as np

Fd_Params = namedtuple("Fermi_Dirac", ['energy', 'sigma', 'mirror']) # mirror for upside down upon 0.5


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
    
def run():
    
    fd1 = Fd_Params._make([0.0, 0.25, True])
    fd2 = Fd_Params._make([6.0, 0.25, False])
    
    fig, axs = plt.subplots(2, 1)
    
    pspout = sys.argv[1]
    
    p = OncvParser(pspout)
    p.scan(verbose=1)
    
    # for x in p.atan_logders.keys():
    for x in [0, 100]:
        print(x)
        logders = p.atan_logders[x]
        lmax = p.lmax
        
        # for ld in logders.ae.values():
        #     # line color set
        #     color = _color_set(l=ld.l)
        #         
        #     axs[0].plot(ld.energies, ld.values, label=f"AE: l={ld.l}, x={x}", color=color, linestyle='dashed')
        #
        #     # down shift 10 to compare
        #     axs[0].plot(ld.energies, ld.values - 10, label=f"AE: l={ld.l}, x={x}", color=color, linestyle='dashed')
        #     
        # for ld in logders.ps.values():
        #     # line color set
        #     color = _color_set(l=ld.l)
        #     
        #     axs[0].plot(ld.energies, ld.values, label=f"PS: l={ld.l}, x={x}", color=color, linestyle='solid')
        #     axs[0].plot(ld.energies, ld.values - 10, label=f"PS: l={ld.l}, x={x}", color=color, linestyle='solid')
        # 
        #
        # axs[0].axvline(x=fd1.energy, linestyle="dotted")
        # axs[0].axvline(x=fd2.energy, linestyle="dotted")
        # axs[0].set_ylabel("Atan Logders")
        # axs[0].legend(loc="upper right", prop={'size': 6})

        for ld_ae, ld_ps in zip(logders.ae.values(), logders.ps.values()):
            # line color set
            color = _color_set(l=ld_ae.l)
            if x == 0:
                lns = 'dashed'
            else:
                lns = 'solid'
                
            axs[1].plot(ld_ae.energies, (ld_ae.values - ld_ps.values), label=f"AE-PS: l={ld_ae.l}, x={x}", color=color, linestyle=lns)
            
        axs[1].axvline(x=fd1.energy, linestyle="dotted")
        axs[1].axvline(x=fd2.energy, linestyle="dotted")
        axs[1].set_ylabel("Atan Logders")
        axs[1].legend(loc="upper right", prop={'size': 6})
        
        fig.tight_layout()
        try:
            filename = sys.argv[2]
        except:
            filename = "temp.png"
        plt.savefig(filename)

if __name__ == '__main__':
    run()
