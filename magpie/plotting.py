import os
import numpy as np
import mesa_reader as mr  # https://github.com/wmwolf/py_mesa_reader
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from cycler import cycler
from matplotlib.ticker import StrMethodFormatter, FormatStrFormatter
mpl.rc('lines', linewidth=2, linestyle='-', marker=None)
mpl.rc('font', family='monospace', size=12.0)
mpl.rc('text', color='000000')
mpl.rc('axes', linewidth=2, grid=False, titlepad=10.0, labelsize='large', 
       axisbelow=False, autolimit_mode='data')  # round_numbers
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
#plt.style.use('./mpl.style')
_Rs = 695700e5 # cm
_Ms = 1.988e33 # g
_xnet = [
    'neut', 'h1  ', 'h2  ', 'he3 ', 'he4 ', 'li6 ', 'li7 ', 'be7 ', 'be9 ', 
    'b8  ', 'b10 ', 'b11 ', 'c12 ', 'c13 ', 'c14 ', 'n13 ', 'n14 ', 'n15 ', 
    'o14 ', 'o15 ', 'o16 ', 'o17 ', 'o18 ', 'f17 ', 'f18 ', 'f19 ', 'ne18', 
    'ne19', 'ne20', 'ne21', 'ne22', 'na21', 'na22', 'na23', 'mg23', 'mg24', 
    'mg25', 'mg26', 'al24', 'al25', 'al26', 'al27', 'si28', 'si29', 'si30', 
    'si31', 'si32', 'p29 ', 'p30 ', 'p31 ', 'p32 ', 'p33 ', 's32 ', 's33 ', 
    's34 ', 's35 ', 's36 ', 'cl33', 'cl34', 'cl35', 'cl36', 'cl37', 'ar36', 
    'ar37', 'ar38', 'ar39', 'ar40', 'k37 ', 'k38 ', 'k39 ', 'k40 ', 'k41 ', 
    'ca40', 'ca41', 'ca42', 'ca43', 'ca44', 'ca45', 'ca46', 'ca47', 'ca48', 
    'sc43', 'sc44', 'sc45', 'sc46', 'sc47', 'sc48', 'sc49', 'ti44', 'ti45', 
    'ti46', 'ti47', 'ti48', 'ti49', 'ti50', 'v46 ', 'v47 ', 'v48 ', 'v49 ', 
    'v50 ', 'v51 ', 'cr48', 'cr49', 'cr50', 'cr51', 'cr52', 'cr53', 'cr54', 
    'mn50', 'mn51', 'mn52', 'mn53', 'mn54', 'mn55', 'fe52', 'fe53', 'fe54', 
    'fe55', 'fe56', 'fe57', 'fe58', 'co53', 'co54', 'co55', 'co56', 'co57', 
    'co58', 'co59', 'ni56', 'ni57', 'ni58', 'ni59', 'ni60', 'ni61', 'ni62', 
    'cu57', 'cu58', 'cu59', 'cu60', 'cu61', 'cu62', 'cu63', 'zn59', 'zn60',
    'zn61', 'zn62', 'zn63', 'zn64', 'zn65', 'zn66' 
]
_xnetReduced = [
    'neut', 'h1  ', 'h2  ', 'he3 ', 'he4 ', 'li7 ', 'b8  ', 'c12 ', 'n14 ',
    'o16 ', 'f18 ', 'ne20', 'na22', 'na23', 'mg24', 'al26', 'al27', 'si28', 
    'p30 ', 's32 ', 'cl34', 'ar36', 'ar40', 'k38 ', 'ca40', 'ca44', 'sc44',
    'ti44', 'ti45', 'ti46', 'v46 ', 'cr48', 'cr50', 'mn50', 'fe52', 'fe53', 
    'fe54', 'fe55', 'fe56', 'fe57', 'fe58', 'co54', 'co55', 'co56', 
    'co54', 'ni56', 'ni58', 'cu58', 'zn60'
]
_ap13 = [
    'he4 ', 'c12 ', 'o16 ', 'ne20',
    'mg24', 'si28', 's32 ', 'ar36',
    'ca40', 'ti44', 'cr48', 'fe52', 'ni56'
]
lines = [(0, ()), 
         (0, (1, 5)),
         (0, (5, 5)),
         (0, (3, 5, 1, 5)),
         (0, (3, 5, 1, 5, 1, 5)),
         (0, (1, 1)),
         (0, (5, 1)),
         (0, (3, 1, 1, 1)),
         (0, (3, 1, 1, 1, 1, 1))]
# x = {1, 5, 10}
# (0, (1, x)) dotted
# (0, (5, x)) dashed
# (0, (3, x, 1, x)) dash-dot
# (0, (1, x, 1, x, 1, x)) dash-dot-dot
# Colors modified from Sasha Trubetskoy's 
# simple 20 color list (based on metro lines).
# https://sashat.me/2017/01/11/list-of-20-simple-distinct-colors/
colors = ['#e6194b', '#3cb44b', '#0082c8', '#000000', '#f58231', 
          '#911eb4', '#008080', '#e6beff', '#bddc36', '#ccc8a0', 
          '#800000', '#808080', '#808000', '#46f0f0', '#000080', 
          '#ffe119', '#aa6e28', '#f032e6', '#ffa64d', '#7fbf92', '#f68888']
cc = (cycler('linestyle', lines)*cycler('color', colors))


def centralCond(runf, prof_number, show=False, tstamp=True):
    """Plots central rho vs central t found in the profile.
    
    Args:
        runf (str): run folder to look in.
        prof_number (int): profile to plot.
        byM (bool): plot by mass (True) or by radius (False).
        show (bool): if True, returns the mpl.figure object. 
        
    Returns:
        mpl.figure: plot of central rho vs central t.

    """
    h = mr.MesaData(os.path.join(runf, "LOGS/history.data"))
    l = mr.MesaLogDir(os.path.join(runf, "LOGS"))
    profs = l.profile_numbers
    if prof_number > len(profs):
        print ("Profile not found. ({}/{})".format(prof_number, len(profs)))
    else:
        print ("Plotting T_c vs Rho_c. {}/{}.".format(prof_number, len(profs)))
    p = l.profile_data(profile_number=prof_number)
    rmax = l.profile_data(profile_number=1).photosphere_r
    fig = plt.figure(figsize=(7, 5))
    layout = (1, 1)
    ax1 = plt.subplot2grid(layout, (0, 0), aspect="auto", adjustable='box-forced')
    plotCenterTRho(p, h, ax1)
    plt.tight_layout()
    if tstamp:
        ax1.annotate("{:0=8.7f}$\cdot 10^9$ yr".format(float(p.star_age)/1e9), 
                    xy=(0.70, 0.08), xycoords='axes fraction', fontsize=10)
    tag = 'trhoc'
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
    print ("Wrote: {0}/png/{1}/{1}_{2:05}".format(runf, tag, p.model_number))


def species(runf, prof_number=1, byM=False, rmax=0.0, rmin=0.0, core=False,
            species=_ap13, thresh=-6, tstamp=True, show=False):
    """Plots species found in a specified profile or a file.
    
    Args:
        runf (str): run folder to look in or file to use.
        prof_number (int): profile to plot(skipped if runf is a file).
        byM (bool): plot by mass (True) or by radius (False).
        rmin/max (float): radius limits in Rsun/Msun (depending on byM).
        core (bool): show c_core_mass in plot.
        species (list): list of named species to plot.
        thresh (float): sets lower limit of the plot.
        tstamp (bool): write sim time on plot.
        show (bool): if True, returns the mpl.figure object. 
        
    Returns:
        mpl.figure: plot of abundances.

    """
    if os.path.isdir(runf):
        folder = True
        l = mr.MesaLogDir(os.path.join(runf, "LOGS"))
        profs = l.profile_numbers
        if prof_number > len(profs):
            print ("Profile not found. ({}/{})".format(prof_number, len(profs)))
        else:
            print( "Plotting Abundances. {}/{}.".format(prof_number, len(profs)))
        p = l.profile_data(profile_number=prof_number)
        if byM and not rmax:
            rmax = p.initial_mass
        elif not byM and not rmax:
            rmax = l.profile_data(profile_number=1).photosphere_r
    else:
        folder = False
        p = mr.MesaData(runf)
        if byM and not rmax:
            rmax = p.initial_mass
        elif not byM and not rmax:
            rmax = p.photosphere_r

    fig = plt.figure(figsize=(11, 7))
    layout = (1, 1)
    ax1 = plt.subplot2grid(layout, (0, 0), aspect='auto', adjustable='box-forced')
    plotAbundances(p, ax1, species=species, byM=byM, core=core, tstamp=tstamp,
                   rmax=rmax, rmin=rmin, xtitle=True, thresh=thresh)
    lgd = ax1.legend(ncol=6, loc='upper left', bbox_to_anchor=(1.0, 1.02), 
                     columnspacing=0.3, labelspacing=0.1, markerfirst=False, 
                     numpoints=3)
    plt.tight_layout(pad=1.0, h_pad=0.0, w_pad=0.5, rect=(0,0,0.5,1))
    tag = 'abun'
    if folder:
        # build filetree and show or save the figure
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
        print ("Wrote: {0}/png/{1}/{1}_{2:05}".format(runf, tag, p.model_number))
    else:
        # don't save, just return the profile to the notebook
        return fig
        


def snapshot(runf, prof_number=1, byM=False, rmax=0.0,
             species=_ap13, show=False, core=False):
    """Plots a grid of general plots for the profile: Dens, 
    Temp, Pres, Abundances(X), and HR.
    
    Args:
        runf (str): run folder to look in or file to use.
        prof_number (int): profile to plot(skipped if runf is a file).
        byM (bool): plot by mass (True) or by radius (False).
        rmax (float): radius limit in Rsun/Msun (depending on byM).
        species (list): list of named species to plot on the X plot.
        show (bool): if True, returns the mpl.figure object. 

    """
    if os.path.isdir(runf):
        folder = True
        h = mr.MesaData(os.path.join(runf, "LOGS/history.data"))
        l = mr.MesaLogDir(os.path.join(runf, "LOGS"))
        profs = l.profile_numbers
        if prof_number > len(profs):
            print ("Profile not found. ({}/{})".format(prof_number, len(profs)))
        else:
            print ("Plotting Abundances. {}/{}.".format(prof_number, len(profs)))
        p = l.profile_data(profile_number=prof_number)
        if byM and not rmax:
            rmax = p.initial_mass
        elif not byM and not rmax:
            rmax = l.profile_data(profile_number=1).photosphere_r
    else:
        folder = False
        p = mr.MesaData(os.path.join(runf))
        h = None
        if byM and not rmax:
            rmax = p.initial_mass
        elif not byM and not rmax:
            rmax = p.photosphere_r

    fig = plt.figure(figsize=(15,9))
    # fill the grid with axes
    layout = (3, 3)
    ax1 = plt.subplot2grid(layout, (0, 0), aspect="auto")
    ax2 = plt.subplot2grid(layout, (1, 0), aspect="auto", adjustable='box-forced')  #, sharex=ax1)
    ax3 = plt.subplot2grid(layout, (2, 0), aspect="auto", adjustable='box-forced')
    ax4 = plt.subplot2grid(layout, (0, 1), aspect="auto", adjustable='box-forced', rowspan=2)
    ax5 = plt.subplot2grid(layout, (2, 1), aspect="auto", adjustable='box-forced')

    plotMainProp(p, ax1, prop='dens', byM=byM, rmax=rmax, tstamp=True,
                 color='black', core=core)
    plotMainProp(p, ax2, prop='temp', byM=byM, rmax=rmax, tstamp=False,
                 color='red', core=core)
    plotMainProp(p, ax3, prop='pres', byM=byM, rmax=rmax, tstamp=False,
                 color='green', xtitle=True, core=core)
    plotAbundances(p, ax4, species=species, byM=byM, rmax=rmax, core=core)
    if h is not None:
        plotHR(p, h, ax5)

    fig.subplots_adjust(hspace=0.5, wspace=0.4)
    tag = 'prof'
    if folder:
        # build filetree and show or save the figure
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
        print ("Wrote: {0}/png/{1}/{1}_{2:05}".format(runf, tag, p.model_number))
    else:
        # don't save, just return the profile to the notebook
        return fig


def plotAbundances(p, ax, species=_ap13, byM=False, rmax=0.0, rmin=0.0,
                   tstamp=False, xtitle=False, thresh=-6, core=False):
    """Draws abundances on a given mpl.axes.
    
    Args:
        p (MesaLogDir.profile_data): profile data to work with.
        ax (mpl.axes): axes instance to draw on.
        species (list): list of named species to plot.
        byM (bool): plot by mass (True) or by radius (False).
        rmin/rmax (float): radius limits in Rsun/Msun (depending on byM).
        tstamp (bool): print profile time on plot.
        xtitle (bool): print x axis label.
        core (bool): show where the C core ends.
        thresh (int): order of magnitude for X cutoff(default=-6).

    """
    if byM:
        rad = p.mass
        xlabel = 'Mass ($M_{\odot}$)'
        if rmax==0.0:
            rmax = p.initial_mass
        ax.set_xlim([rmin, rmax])
    else:
        rad = p.R*_Rs
        xlabel = 'Radius (cm)'
        if rmax==0.0:
            rmax = p.photosphere_r
        ax.set_xlim([rmin+1.0e7, rmax*_Rs])
        ax.set_xscale('log')
    species = [x.strip() for x in species]
    lim = float("1e{}".format(thresh))
    for s, props in zip(species, cc):
        tag = '$^{{{}}}{}$'.format(*elemSplit(s))
        # don't skip species, it's bad for you.
        #if max(p.data(s))<lim:
        #    continue
#         c, ls = next(styles)
        ax.semilogy(rad, p.data(s), alpha=0.7, label=tag, ls=props['linestyle'], c=props['color'])
    ax.set_ylim(float("1e{}".format(thresh)), 2e0)
    ax.legend(ncol=5, loc='upper left', bbox_to_anchor=(1.0, 1.0), 
              columnspacing=0.5, labelspacing=0.5, markerfirst=False, 
              numpoints=4)
    ax.axhline(1e0, linewidth=1, linestyle=':', color='black')
    xticks = ax.get_xticklabels()
    xticks[-1].set_visible(False)
    xticks[0].set_visible(False)
    if core:
        zone = np.where(p.mass < p.c_core_mass)[0][0]
        if byM:
            ax.axvline(p.mass[zone], linewidth=3, alpha=0.5, linestyle=':', color='black')
        else:
            ax.axvline(p.R[zone]*_Rs, linewidth=3, alpha=0.5, linestyle=':', color='black')
    ax.set_ylabel('$X_{sp}$')
    if xtitle:
        ax.set_xlabel(xlabel)
    if tstamp:
        ax.annotate("{:0=8.7f}$\cdot 10^9$ yr".format(float(p.star_age)/1e9), 
                    xy=(0.75, 0.05), xycoords='figure fraction', fontsize=18)


def plotCenterTRho(p, h, ax):
    """Draws an Rho_c vs T_c diagram on a given mpl.axes.
    
    Args:
        p (MesaLogDir.profile_data): profile data to work with.
        h (MesaData): history data for the run.
        ax (mpl.axes): axes instance to draw on.

    """
    numh = np.where(h.model_number==p.model_number)
    tc = h.log_center_T[:numh[0][0]]
    rhoc = h.log_center_Rho[:numh[0][0]]
    ax.plot(rhoc, tc)
    ax.set_xlabel('log Central Temperature')
    ax.set_ylabel('log Central Density')
    ax.yaxis.set_major_formatter(StrMethodFormatter('{x:2.3f}'))


def plotHR(p, h, ax):
    """Draws an HR diagram on a given mpl.axes.
    
    Args:
        p (MesaLogDir.profile_data): profile data to work with.
        h (MesaData): history data for the run.
        ax (mpl.axes): axes instance to draw on.

    """
    numh = np.where(h.model_number==p.model_number)
    lum = h.log_L[:numh[0][0]]
    tef = h.log_Teff[:numh[0][0]]
    ax.plot(tef, lum)
    ax.set_xlabel('log Effective Temperature')
    ax.set_ylabel('log Luminosity')
    # TODO: get a better range estimate.
    ax.set_xlim([4.5, 3.4])
    ax.yaxis.set_major_formatter(StrMethodFormatter('{x:2.1f}'))


def getWholeHistory(folders, xkey='log_Teff', ykey='log_L'):
    """returns aggregated history for a pair of 
    keys from a list of sorted folders. 
    Defaults to HR variables log_Teff and log_L
    """
    xvals, yvals = np.array([]), np.array([])
    for f in folders:
        h = mr.MesaData(os.path.join(f, "LOGS/history.data"))
        l = mr.MesaLogDir(os.path.join(f, "LOGS"))
        p = l.profile_data(profile_number=len(l.profile_numbers))
        xrow, yrow = getHistProp(p, h, xkey, ykey)
        xvals = np.append(xvals, xrow)
        yvals = np.append(yvals, yrow)
    return xvals, yvals


def getHistProp(p, h, key1='log_L', key2='log_Teff'):
    """Returns a pair of list of values for two keys in the run's history 
    for a profile.
    """
    numh = np.where(h.model_number==p.model_number)
    return h.data(key1)[:numh[0][0]], h.data(key2)[:numh[0][0]]
    

def plotMainProp(p, ax, prop='dens', byM=False, rmax=0.0, tstamp=True,
                 color='black', xtitle=False, core=False):
    """Draws a main property (dens, temp, pres) on a given mpl.axes.
    
    Args:
        p (MesaLogDir.profile_data): profile data to work with.
        ax (mpl.axes): axes instance to draw on.
        prop (str): property to plot (dens, temp, or pres).
        byM (bool): plot by mass (True) or by radius (False).
        rmax (float): radius limit in Rsun/Msun (depending on byM).
        tstamp (bool): print profile time on plot.
        color (str): named mpl color or HEX for the drawing.
        xtitle (bool): print x axis label.

    """
    if byM:
        rad = p.mass
        xlabel = 'Mass ($M_{\odot}$)'
        if rmax==0.0:
            rmax = p.initial_mass
        ax.set_xlim([0.0, rmax])
    else:
        rad = p.R*_Rs
        xlabel = 'Radius (cm)'
        if rmax==0.0:
            rmax = p.photosphere_r
        ax.set_xlim([1.0e7, rmax*_Rs])
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
    if core:
        zone = np.where(p.mass < p.c_core_mass)[0][0]
        if byM:
            ax.axvline(p.mass[zone], linewidth=3, alpha=0.5, linestyle=':', color='black')
        else:
            ax.axvline(p.R[zone]*_Rs, linewidth=3, alpha=0.5, linestyle=':', color='black')
    if xtitle:
        ax.set_xlabel(xlabel)
    if tstamp:
        ax.annotate("{:0=8.7f}$\cdot 10^9$ yr".format(float(p.star_age)/1e9), 
                    xy=(0.08, 0.15), xycoords='axes fraction', fontsize=10)
    ax.yaxis.set_major_formatter(StrMethodFormatter('{x:.1e}'))


def runStats(runf, props=['star_mass', 'log_Teff', 'c_core_mass', 
             'total_mass_h1', 'total_mass_he4', 'log_L', 'log_R']):
    """Prints out relevant run information such as remnant C core and 
    final mass.
    
    Args:
        runf (str): run folder to look in.
        props (list of str): h.header names to print out. default is 
        ['star_mass', 'log_Teff', 'c_core_mass', 'total_mass_h1', 
        'total_mass_he4', 'log_L', 'log_R'].
    
    """
    h = mr.MesaData(os.path.join(runf, "LOGS/history.data"))
    l = mr.MesaLogDir(os.path.join(runf, "LOGS"))
    print ("Version: {version_number}\nInitial Mass(Z): "\
          "{initial_mass} ({initial_z})".format(**h.header_data))
    print ("Profiles found: {}".format(len(l.profile_numbers)))
    print ("Models run: {}".format(h.data('model_number')[-1]))
    print ("Initial zones: {}".format(max(h.data('num_zones'))))
    print ("\nSimtime: {:10.9} Gyr\n".format(h.data('star_age')[-1]/1e9))
    print ("{:20}{:<20} {:<20}\n".format('Prop','Initial','Final'))
    otp = "{:20}{:<20.5f} {:<20.5f}"
    for arg in props:
        print (otp.format(arg, h.data(arg)[0], h.data(arg)[-1]))
    if 'c_core_mass' in props:
        p = l.profile_data()
        zone = np.where(p.mass < p.c_core_mass)[0][0]
        print ("Final Mass and Radius of the Carbon Core: "\
              "{:6.5e} Msun {:6.5e} cm".format(p.mass[zone], p.R[zone]*_Rs))
        print ("150km match head x_match: {:.6e}".format(p.R[zone]*_Rs/np.sqrt(2.)))
    return len(l.profile_numbers)


def writeCoreProfile(runf, filename, fluff_dens=-1e0, otp='./', 
                     species=[], rescale=0, debug=False):
    """writes a simple plaintext R vs Dens+Temp+species profile with the structure:
    # Radius dens temp c12 ne22
    <Number of zones/rows>
    <DATA rows>
    #Mass: <Mass profiled> Msun carved out from a <Total mass in the profile>, from an initial <ZAMS mass>
    
    Args:
        runf (str): run folder to look in.
        filename (str): profile.data name to probe.
        fluff_dens (float): density threshold for cuttoff. set to negative to cut at the end of the CO core.
        otp (str): output folder.
        species (list of str): species list to write out.
        rescale (int): rescale to 'rescale' cells. all cells is rescale=0.
        debug (bool): return radii, densities, temperatures, and profile obj
    
    """
    singlep = mr.MesaData(os.path.join(runf, "LOGS/{}".format(filename)))
    if fluff_dens < 0.0:
        filt = np.where(singlep.mass < singlep.c_core_mass)
        fluff = ''
    else:
        filt = np.where(singlep.Rho > fluff_dens)
        fluff = '_fluff'
    allspecies  = singlep.bulk_names[9:]
    selection = []
    if species:
        for s in species:
            if s.strip() in allspecies:
                selection.append(s)
            else:
                print ("'{}' not found in profile.".format(s))
    else:
        selection = allspecies
    radi = np.flip(singlep.R[filt],0)
    dens = np.flip(singlep.Rho[filt],0)
    temp = np.flip(singlep.T[filt],0)
    zones = len(radi)
    header = ['#', 'Radius', 'dens', 'temp']
    fracs = []
    for s in selection:
        header.append(s)
        fracs.append(np.flip(singlep.data(s)[filt],0))
    print (header)
    file = '{}wd_{}to{:.2f}{}.dat'.format(otp, singlep.initial_mass, 
                                          singlep.c_core_mass, fluff)
    if rescale:
        rfracs = []
        xscale = np.linspace(radi[0], radi[-1], num=rescale)
        dens = np.interp(xscale, radi, dens)
        temp = np.interp(xscale, radi, temp)
        for fr in fracs:
            rfracs.append(np.interp(xscale, radi, fr))
        zones = rescale
        fracs = rfracs
        radi = xscale
    radi*=_Rs # cgs
    if not debug:
        with open(file, 'w') as f:
            f.write(" ".join(header))
            f.write("\n")
            f.write("{}\n".format(zones))
            for z in range(zones):
                line = ["{:20.7e} {:20.7e} {:20.7e}".format(radi[z], dens[z], temp[z])]
                for fr in fracs:
                #for s in selection:
                    #line.append("{:20.7e}".format(np.flip(singlep.data(s),0)[z]))
                    line.append("{:20.7e}".format(fr[z]))
                f.write(" ".join(line))
                f.write("\n")
            f.write("#Mass: {} Msun (CO core only) carved out from a {},"\
                    "from an initial {}".format(singlep.c_core_mass, singlep.star_mass, 
                                                singlep.initial_mass))
        print ("Wrote: {}".format(file))
    else:
        return radi, dens, temp, singlep


def elemSplit(s):
    """Standalone element name spliter. 
    he4 -> (4, He)
    """
    sym = s.rstrip('0123456789 ')
    A = s[len(sym):].strip()
    return A, sym.title()


# python 2 cylcler
def colIter():
    """Simple color/linestyle iterator. Colors selected from Sasha 
    Trubetskoy's simple 20 color list (based on metro lines)
    https://sashat.me/2017/01/11/list-of-20-simple-distinct-colors/
    """
    cols = ['#e6194b', '#3cb44b', '#0082c8', '#000000', '#f58231', '#911eb4', 
            '#008080', '#e6beff', '#aa6e28', '#999678', '#800000', '#88cc9c', 
            '#808000', '#ffb265', '#000080', '#fabebe', '#e5c700']
            #, '#f032e6', '#46f0f0', '#d2f53c', '#808080']ffe119
    styles = [(0, ()),
              # dotted loose/normal/dense
              (0, (1, 10)),
              #(0, (1, 5)),
              (0, (1, 1)),
              # dashed loose/normal/dense
              (0, (5, 10)), 
              #(0, (5, 5)),
              (0, (5, 1)),
              # dash-dot loose/normal/dense
              (0, (3, 10, 1, 10)), 
              #(0, (3, 5, 1, 5)),
              (0, (3, 1, 1, 1)),
              # dash-dot-dot loose/normal/dense
              (0, (3, 10, 1, 10, 1, 10)),
              #(0, (3, 5, 1, 5, 1, 5))] 
              (0, (3, 1, 1, 1, 1, 1))]
    lstyles = len(styles)
    lcols = len(cols)
    alphas = np.linspace(0.0, 1.0, num=lstyles)
    i, j = -1, 0
    while(True):
        i+=1
        if i==lstyles:
            i=0
            j+=1
        yield cols[j], styles[i]#, alphas[i]
        #if i==lcols:
        #    i=0
        #yield cols[i], styles[i%lstyles]
