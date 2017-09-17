import pandas as pd
import shutil
import os
import numpy as np
from subprocess import PIPE, Popen
_mesasdk = '/ccs/home/frivas/launch_mesasdk'
_maxmultipars = 10000000

class parameterGroup(object):
    def __init__(self, parfile, type='controls'):
        if parfile.split('.')[-1]=='defaults':
            n, v, d, err = getRawDefaults(parfile)
            self.__dict__.update(dict(zip(n, zip(v,d))))
        else:
            self.setPars(parfile, type=type)
        self._type = type
            
    def setPars(self, inlist, type='controls'):
        """sets the parameters from an inlist file in the 
        object
        """
        cd = dict(zip(*getRawPars(inlist, type=type)))
        self.__dict__.update(cd)

    def tabulate(self):
        """returns a pandas dataframe with parameters 
        currently being used
        """
        usedpars = self.cleandict()
        names, values = zip(*usedpars.items())
        df = pd.DataFrame()
        if len(names)==0:
            df['&{}'.format(self._type)], df['value'] = "None", 0
        else:
            df['&{}'.format(self._type)], df['value'] = names, values
        print df
    
    def cleandict(self):
        newdict = {}
        for key, value in self.__dict__.iteritems():
            if isinstance(value, tuple) or value==self._type:
                continue
            else:
                newdict.update({key: str(value)})
        return newdict
        


def setupMESArun(codesource, destination, clobber=True):
    """copy a target directory into a new one for editing the run"""
    if os.path.exists(destination):
        if clobber:
            shutil.rmtree(destination)
            shutil.copytree(codesource, destination)
    else:
        shutil.copytree(codesource, destination)


def getRawPars(inlist, type="controls"):
    """ gets a list of parameters from an inlist, subject to
    type.

    Args:
        inlist (str): input inlist.
        type (str): header type for parameters.

    Returns:
        names (list), values (list)
    """
    pars = []
    add = 0
    head = '&{}'.format(type)
    with open(inlist, 'r') as parf:
        for line in parf:
            if len(line)<=1:
                continue
            else:
                if head in line:
                    add = 1
                    continue
                elif not add:
                    continue
                else:
                    if line[0]=='/':
                        break
                    if '=' in line and '!' not in line:
                        l = line.strip().split('=')
                        pars.append((l[0].strip(),l[-1].strip()))
    if len(pars)<1:
        return [None], [None]
    else:
        return zip(*pars)


def getRawDefaults(defaultfile):
    """ returns list of names, defaults and descriptions found for each 
    parameter in a .defaults file. if syntax is incorrect the parameter 
    won't have a description but the realname will be returned for use.
    Fills 'err' with the lines around the bad syntax 
    parameters.

    Args:
        defaultfile (str): .default file
    
    Returns:
       names(list), defaults(list), descriptions(list), err(list)
    """
    pars, desc, multi = [], [], []
    name, val = "", ""
    doc, setvars = False, False
    pnames, err = [], []
    flush = False
    with open(defaultfile, 'r') as par:
        for line in par:
            if len(line.strip())>1:
                if line.strip()[0]!= '!':
                    pnames.append(line.strip().split('=')[0].strip('9012345678 .):({}#!\n').lower())
            if len(line.strip())<1:
                continue
            if flush and not '!###' in line.split()[0]:
                err.append(line.strip('\n'))
                continue
            if '!###' in line.split()[0]:
                flush = False
                if setvars:
                    #this means there's not enough values for the defined variables, 
                    #so reset everything
                    name, val = "", ""
                    desc = []
                    multi = []
                    setvars = False
                if not name:
                    # split fixes comentaries right beside the DEFINITION 
                    # of the variables. lower fixes Msun vs msun differences.
                    # no consistency whatsoever :C
                    name = line.strip('9012345678  .):({}#!\n').lower().split()[0]
                    multi.append(name)
                    namenum = plainGenerator(_maxmultipars)
                else:
                    multi.append(line.strip('9012345678  .):({}#!\n').lower().split()[0])
                doc = True
                continue
            if doc:
                if line.strip()[0] == '!':
                    desc.append(line.strip(' !'))
                else:
                    setvars = True
                    # controls.defaults has a double '=' o some of its params...
                    avoidequals = line.strip().split('=')
                    cname, val = avoidequals[:2]
                    cname = cname.strip('9012345678  .):(').lower()
                    if len(multi)<=1:
                        try:
                            m = multi.index(cname)
                        except ValueError:
                            flush = True
                            err.append(line.strip('\n'))
                            continue
                        pars.append((multi[m], val, desc))
                        name, val = "", ""
                        desc = []
                        multi = []
                        doc = False
                        setvars = False
                    else:
                        n = namenum.next()
                        try:
                            m = multi.index(cname)
                        except ValueError:
                            flush = True
                            err.append(line.strip('\n'))
                            continue
                        if n==len(multi)-1:
                            pars.append((multi[m], val, u"".join(desc)))
                            name, val = "", ""
                            desc = []
                            multi = []
                            doc = False
                            setvars = False
                        else:
                            pars.append((multi[m], val, "".join(desc)))
            else:
                continue
    tags = [a[0] for a in pars]
    for st in pnames:
        if st not in tags:
            pars.append((st, "None", "None"))
    n, v, d = zip(*pars)
    return n, v, d, err


def plainGenerator(length):
    i = 0
    while i<length:
        yield i
        i += 1


def writeInlists(star_job, controls, pgstar, outpath, sjname="star_job",
                 ctname="controls", pgname="pgstar"):
    """Receives 3 dictionaries and builds a general inlist, plus 3
    separate files called by it.
    """
    sjinstr = ('read_extra_star_job_inlist1',
               'extra_star_job_inlist1_name',
               'inlist_{}'.format(sjname))
    ctinstr = ('read_extra_controls_inlist1',
               'extra_controls_inlist1_name',
               'inlist_{}'.format(ctname))
    pginstr = ('read_extra_pgstar_inlist1',
               'extra_pgstar_inlist1_name',
               'inlist_{}'.format(pgname))

    sjdict = dict(zip(sjinstr[:2],('.true.', "inlist_{}".format(sjname))))
    ctdict = dict(zip(ctinstr[:2],('.true.', "inlist_{}".format(ctname))))
    pgdict = dict(zip(pginstr[:2],('.true.', "inlist_{}".format(pgname))))
    # write 'inlist'
    write2Inlist(sjdict, "&star_job", outpath, "inlist", clobber=True)
    write2Inlist(ctdict, "&controls", outpath, "inlist")
    write2Inlist(pgdict, "&pgstar", outpath, "inlist")
    # Write "inlist_specific"
    write2Inlist(star_job, "&star_job", outpath, sjinstr[2], clobber=True)
    write2Inlist(controls, "&controls", outpath, ctinstr[2], clobber=True)
    write2Inlist(pgstar, "&pgstar", outpath, pginstr[2], clobber=True)


def write2Inlist(parameters, header, outpath, fname, clobber=False):
    """Write paramater ditionary to file, appending for
    clobber=False (default)
    """
    if clobber:
        opt = 'w'
    else:
        opt = 'a'
    with open("/".join([outpath, fname]), opt) as o:
        o.write("\n{}\n\n".format(header))
        for key in parameters.keys():
            o.write("      {} = {}\n".format(key,
                                       mesaParse(parameters[key])))
        o.write("\n/\n")


def mesaParse(arg):
    """returns a parsed variable from a parameter (bool,
    str, or number)
    """
    try:
        val = np.float(arg.replace('d','E'))
        return arg
    except ValueError:
        if arg=='.true.':
            return arg
        elif arg=='.false.':
            return arg
        else:
            return '"{}"'.format(arg.strip('"\' '))


def compileMESA(outpath):
    """calls ./mk at outpath, compiling the run
    """
    comm = 'source {} && cd {} && ./mk'.format(_mesasdk, outpath)
    print(comm)
    p = Popen(['/bin/bash'], stdin=PIPE, stdout=PIPE)
    out, err = p.communicate(input=comm.encode())
    print(out.decode())
    exitcode = p.returncode
    return exitcode


def execute(outpath, now=False):
    """qsubs the sumbit.pbs at outpath

    Args:
        outpath (str): runfolder

    Returns:
        (tuple): STDOUT, STDERR, ERRCODE

    """
    if now:
        comm = 'source {} && cd {} && ./rn'.format(_mesasdk, outpath)
        p = Popen(['/bin/bash'], stdin=PIPE, stdout=PIPE)
        out, err = p.communicate(input=comm.encode())
        print(out.decode())
        exitcode = p.returncode
    else:
        command = 'qsub submit.pbs'
        p = Popen(command.split(), cwd=os.path.abspath(outpath),
                  stdin=PIPE, stdout=PIPE, stderr=PIPE)
        r, e = p.communicate()
        print(r.decode())
        exitcode = p.returncode
    return exitcode


def subTITAN(outpath, nametag, time='12:00:00', nodes=1252, j1=True,
             type='FLASH', slines=[]):
    """builds a submit.pbs file, specifying nodes, times, and
    inserting miscellaneous scriptlines(slines) before aprun command
    """
    subHeader = ['#!/bin/bash', '#PBS -A CSC198', '#PBS -j oe', '#PBS -V',
                 '#PBS -l gres=atlas1', '#PBS -m abe',
                 '#PBS -M rivas.aguilera@gmail.com']
    subScript = ['date',
                 'echo Submitted from: $PBS_O_WORKDIR',
                 'echo #####################']
    if slines:
        subScript = subScript + slines
    subScript.append('cd {}'.format(os.path.abspath(outpath)))
    subHeader.append('#PBS -l walltime={},nodes={}'.format(time, nodes))
    subHeader.append('#PBS -N {}'.format(nametag))
    if type=='MESA':
        if j1:
            subScript.append('OMP_NUM_THREADS={}'.format(int(nodes*8)))
        else:
            subScript.append('OMP_NUM_THREADS={}'.format(int(nodes*2*8)))
        subScript.append('export OMP_NUM_THREADS')
        command = './rn'
    else:
        command = './flash4'
    if j1:
        subScript.append('aprun -j1 -n {} {}'.format(nodes*8, command))
    else:
        subScript.append('aprun -n {} {}'.format(nodes*16, command))
    with open(outpath+"/submit.pbs", 'w') as o:
        o.write("\n".join(subHeader))
        o.write("\n")
        o.write("\n".join(subScript))
        o.write("\n")

