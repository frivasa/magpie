import os
import numpy as np
import mesa_reader as mr  # https://github.com/wmwolf/py_mesa_reader
import matplotlib as mpl
from matplotlib.ticker import StrMethodFormatter, FormatStrFormatter
mpl.rc('lines', linewidth=2, linestyle='-', marker=None)
mpl.rc('font', family='monospace', size=12.0)
mpl.rc('text', color='000000')
mpl.rc('axes', linewidth=2, grid=False, titlepad=10.0, labelsize='large',
       axisbelow=False, autolimit_mode='round_numbers')
mpl.rc('axes.formatter', limits=(-1,1))
mpl.rc('xtick', top=True, direction='in')
mpl.rc('xtick.major', size=9, width=1.4, pad=7)
mpl.rc('xtick.minor', size=5, width=1.0, pad=7)
mpl.rc('ytick', right=True, direction='in')
mpl.rc('ytick.major', size=9, width=1.4, pad=7)
mpl.rc('ytick.minor', size=5, width=1.0, pad=7)
mpl.rc('savefig', facecolor='ffffff', dpi=120, bbox='tight')
mpl.rc('axes', facecolor='ffffff')
mpl.rc('figure', facecolor='bbd7e5')
import matplotlib.pyplot as plt
#plt.style.use('./mpl.style')

_Rs = 695700e5 # cm
_Ms = 1.988e33 # g

def species(runf, prof_number, byM=False,
            species=["h1", "he4", "c12", "o16"], show=False):
    h = mr.MesaData(os.path.join(runf, "LOGS/history.data"))
    l = mr.MesaLogDir(os.path.join(runf, "LOGS"))
    profs = l.profile_numbers
    if prof_number > len(profs):
        print "Profile not found. ({}/{})".format(prof_number, len(profs))
    else:
        print "Plotting Abundances. {}/{}.".format(prof_number, len(profs))
    p = l.profile_data(profile_number=prof_number)
    rmax = l.profile_data(profile_number=1).photosphere_r
    fig = plt.figure(figsize=(7, 5))
    layout = (1, 1)
    ax1 = plt.subplot2grid(layout, (0, 0))
    plotAbundances(p, h, ax1, species=species, byM=byM, xtitle=True)
    plt.tight_layout()
    tag = 'abun'
    if not os.path.exists("{}/png/".format(runf)):
        os.mkdir("{}/png/".format(runf, tag))
        os.mkdir("{}/png/{}".format(runf, tag))
    elif not os.path.exists("{}/png/{}/".format(runf, tag)):
        os.mkdir("{}/png/{}".format(runf, tag))
    if show:
        return fig
    else:
        plt.savefig("{0}/png/{1}/{1}_{2:05}".format(runf, tag, p.model_number))
        plt.close(fig)
    print "Wrote: {0}/png/{1}/{1}_{2:05}".format(runf, tag, p.model_number)


def snapshot(runf, prof_number, byM=False,
             species=["h1", "he4", "c12", "o16"], show=False):
    h = mr.MesaData(os.path.join(runf, "LOGS/history.data"))
    l = mr.MesaLogDir(os.path.join(runf, "LOGS"))
    profs = l.profile_numbers
    if prof_number > len(profs):
        print "Profile not found. ({}/{})".format(prof_number, len(profs))
    else:
        print "Plotting profile number {}/{}.".format(prof_number, len(profs))
    p = l.profile_data(profile_number=prof_number)
    rmax = l.profile_data(profile_number=1).photosphere_r
    fig = plt.figure(figsize=(12,8.5))
    # fill the grid with axes
    layout = (3, 2)
    ax1 = plt.subplot2grid(layout, (0, 0))
    ax2 = plt.subplot2grid(layout, (1, 0))  #, sharex=ax1)
    ax3 = plt.subplot2grid(layout, (2, 0))
    ax4 = plt.subplot2grid(layout, (0, 1), rowspan=2)
    ax5 = plt.subplot2grid(layout, (2, 1))

    plotMainProp(p, h, ax1, prop='dens', byM=byM, rmax=rmax, tstamp=True,
                 color='black')
    plotMainProp(p, h, ax2, prop='temp', byM=byM, rmax=rmax, tstamp=False,
                 color='red')
    plotMainProp(p, h, ax3, prop='pres', byM=byM, rmax=rmax, tstamp=False,
                 color='green', xtitle=True)
    plotAbundances(p, h, ax4, species=species, byM=True)
    plotHR(p, h, ax5)
    plt.tight_layout()
    if not os.path.exists("{}/png/".format(runf)):
        os.mkdir("{}/png/".format(runf))
        os.mkdir("{}/png/prof".format(runf))
    elif not os.path.exists("{}/png/prof/".format(runf)):
        os.mkdir("{}/png/prof".format(runf))
    if show:
        return fig
    else:
        plt.savefig("{}/png/prof/prof_{:05}".format(runf, p.model_number))
        plt.close(fig)
    print "Wrote: {}/png/prof/prof_{:05}".format(runf, p.model_number)


def plotAbundances(p, h, ax, species=['h1'], byM=False, rmax=1.0,
                   tstamp=True, xtitle=False):
    if byM:
        rad = p.mass
        xlabel = 'Mass ($M_{\odot}$)'
        ax.set_xlim([-0.5, h.initial_mass + 0.5])
    else:
        rad = p.R*_Rs
        xlabel = 'Radius (cm)'
        ax.set_xlim([1.0e7, rmax*_Rs*1.5])
        ax.set_xscale('log')
    c = colIter()
    for s in species:
        tag = '$^{{{}}}{}$'.format(*elemSplit(s))
        ax.semilogy(rad, p.data(s), color=c.next(), label=tag)
    ax.set_ylim(1e-6, 2e0)
    if len(species)>10:
        ax.legend(ncol=2, loc=1, bbox_to_anchor=(0.80, 0.0, 1.0, 1.0))
    else:
        ax.legend(ncol=1, loc=1, bbox_to_anchor=(0.30, 0.0, 1.0, 1.0))
    ax.axhline(1e0, linewidth=1, linestyle=':', color='black')
    ax.set_ylabel('$X_{sp}$')
    if xtitle:
        ax.set_xlabel(xlabel)


def plotHR(p, h, ax):
    numh = np.where(h.model_number==p.model_number)
    lum = h.log_L[:numh[0][0]]
    tef = h.log_Teff[:numh[0][0]]
    ax.plot(tef, lum)
    ax.set_xlabel('log Effective Temperature')
    ax.set_ylabel('log Luminosity')
    ax.set_xlim([4.5, 3.4])
    ax.yaxis.set_major_formatter(StrMethodFormatter('{x:2.1f}'))


def plotMainProp(p, h, ax, prop='dens', byM=False, rmax=1.0, tstamp=True,
                 color='black', xtitle=False):
    if byM:
        rad = p.mass
        xlabel = 'Mass ($M_{\odot}$)'
        ax.set_xlim([-0.5, h.initial_mass + 0.5])
    else:
        rad = p.R*_Rs
        xlabel = 'Radius (cm)'
        ax.set_xlim([1.0e7, rmax*_Rs*1.5])
        ax.set_xscale('log')
    if prop=='pres':
        y = p.P
        ylabel = 'Pressure($dyne/cm^2$)'
    elif prop=='temp':
        y = p.T
        ylabel = 'Temperature($K$)'
    else:
        y = p.Rho
        ylabel = 'Density($g/cm^3$)'
    ax.semilogy(rad, y, color=color)
    ax.set_ylabel(ylabel)
    if xtitle:
        ax.set_xlabel(xlabel)
    if tstamp:
        ax.annotate("Time: {:0=8.7f}e9 yr".format(float(p.star_age)/1e9),
                    xy=(0.10, 0.15), xycoords='axes fraction', fontsize=10)
    ax.yaxis.set_major_formatter(StrMethodFormatter('{x:.2e}'))


def colIter():
    # Sasha Trubetskoy's simple 20 color list (based on metro lines)
    # https://sashat.me/2017/01/11/list-of-20-simple-distinct-colors/
    cols = ['#e6194b', '#3cb44b', '#0082c8', '#000000', '#f58231', '#911eb4',
            '#008080', '#e6beff', '#aa6e28', '#fffac8', '#800000', '#aaffc3',
            '#808000', '#ffd8b1', '#000080', '#808080', '#ffe119', '#f032e6',
            '#46f0f0', '#d2f53c', '#fabebe']
    cols *= 4
    for i in range(len(cols)):
        yield cols[i]


def elemSplit(s):
    sym = s.rstrip('0123456789')
    A = s[len(sym):]
    return A, sym.title()
