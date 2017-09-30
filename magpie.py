import pandas as pd
import shutil
import os
import numpy as np
from subprocess import PIPE, Popen
_mesa_dir = os.environ.get('MESA_DIR')
_mesasdk_root = os.environ.get('MESASDK_ROOT')
_maxmultipars = 10000000

class parameterGroup(object):
    types = ['controls', 'star_job', 'pgstar']
    
    def __init__(self):
        self.params = {}
        self.defaults = {}
        for t in parameterGroup.types:
            self.params[t] = {}
            self.defaults[t] = {}

    def __getitem__(self, key):
        if key in parameterGroup.types:
            return self.params[key]
        else:
            return {}

    def readDefaults(self, MESA_DIR):
        if _mesa_dir is None:
            print("MESA_DIR not set, cannot read defaults.")
            return
        globstr = MESA_DIR+"/star/defaults/"
        fnames = sorted([x for x in os.listdir(globstr) if ".defaults" in x])
        fpaths = [globstr+x for x in fnames]
        assert(len(fpaths)==3)
        self.setDefs(fpaths[0], type='controls')
        self.setDefs(fpaths[1], type='pgstar')
        self.setDefs(fpaths[2], type='star_job')
        for t in parameterGroup.types:
            self.mixPars(t)
            
    def setPars(self, inlist, type='controls'):
        """sets the parameters from an inlist file in the 
        object. defaults to changing controls
        """
        cd = dict(zip(*getRawPars(inlist, type=type)))
        self.params[type].update(cd)

    def setDefs(self, parfile, type='controls'):
        """reads in a .default file for check for altered values and 
        docstrings on parameters
        """
        n, v, d, err = getRawDefaults(parfile)
        cd = dict(zip(n, zip(v,d)))
        self.defaults[type].update(cd)

    def readInlist(self, inlist):
        self.setPars(inlist, type='controls')
        self.setPars(inlist, type='star_job')
        self.setPars(inlist, type='pgstar')
        
    def quickLook(self):
        """returns pandas dataframe with read inlist values for quick 
        editing of parameters"""
        frames = []
        for t in parameterGroup.types:
            df = pd.DataFrame()
            names, values = zip(*self.params[t].items())
            if not names:
                names, values = "None", 0
            df[t], df['value'] = names, values
            frames.append(df)
        return pd.concat(frames)
        
    def tabulate(self, type):
        """returns a pandas dataframe with parameters 
        currently being used
        """
        df = pd.DataFrame()
        docs = ""
        if not self.defaults[type]:
            names, values = zip(*self.params[type].items())
        else:
            self.mixPars(type)
            names, tuples = zip(*self.defaults[type].items())
            values, docs = zip(*tuples)
            docs = parseDocs(docs)
        if not names:
            names, values, docs = "None", 0, ""
        df[type], df['value'], df['docstr'] = names, values, docs
        return df.set_index(type)

    def mixPars(self, type='controls'):
        """sets the values of inlists into the default dictionary"""
        if not self.defaults[type]:
            return
        else:
            for k, v in self[type].items():
                try:
                    dv, doc = self.defaults[type][k]
                    self.defaults[type][k] = (v, doc)
                except KeyError:
                    continue

    def readChanges(self, df):
        for t in parameterGroup.types:
            try:
                cont = df[[t,'value']].dropna(0).set_index(t)
            except KeyError:
                if df.index.name==t:
                    cont = df
                else:
                    continue
            newdict = cont.T.to_dict('records')[0]
            for k, v in newdict.items():
                try:
                    dv, doc = self.defaults[t][k]
                    if v!=dv:
                        self.defaults[t][k] = (v, doc)
                        self[t][k] = v
                except KeyError:
                    continue

    def writeInlists(self, outpath, sjname="star_job",
                     ctname="controls", pgname="pgstar"):
        """Receives 3 dictionaries and builds a general inlist, plus 3
        separate files called by it.
        """
        for i, t in enumerate(parameterGroup.types):
            keys = ["read_extra_{}_inlist1", "extra_{}_inlist1_name"]
            vals = ['.true.', "inlist_{}".format(t)]
            inlistd = dict(zip([k.format(t) for k in keys], vals))
            if not i:
                write2Inlist(inlistd, "&{}".format(t), outpath, "inlist", 
                             clobber=True)
            else:
                write2Inlist(inlistd, "&{}".format(t), outpath, "inlist")
            write2Inlist(self[t], "&{}".format(t), outpath, xname, 
                         clobber=True)


def parseDocs(dlist):
    nd = []
    for d in dlist:
        if isinstance(d, list):
            nd.append("".join(d))
        else:
            nd.append(d)
    return nd


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
                    prel =line.split('=')[0].strip('9012345678 .):({}#!\n')
                    pnames.append(prel.lower())
            if len(line.strip())<1:
                continue
            if flush and not '!###' in line.split()[0]:
                err.append(line.strip('\n'))
                continue
            if '!###' in line.split()[0]:
                flush = False
                if setvars:
                    #this means there's not enough values for the defined 
                    # variables, so reset everything
                    name, val = "", ""
                    desc = []
                    multi = []
                    setvars = False
                if not name:
                    # split fixes comentaries right beside the DEFINITION 
                    # of the variables. lower fixes Msun vs msun differences.
                    # no consistency whatsoever :C
                    prel = line.strip('9012345678  .):({}#!\n').lower()
                    name = prel.split()[0]
                    multi.append(name)
                    namenum = plainGenerator(_maxmultipars)
                else:
                    prel = line.strip('9012345678  .):({}#!\n').lower()
                    multi.append(prel.split()[0])
                doc = True
                continue
            if doc:
                if line.strip()[0] == '!':
                    desc.append(line.strip(' !'))
                else:
                    setvars = True
                    # controls.defaults has double '=' on some of its params...
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


def fortParse(arg):
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
                                       bu.fortParse(parameters[key])))
        o.write("\n/\n")


def compileMESA(outpath, startsdk=True):
    """calls ./mk at outpath, compiling the run
    """
    if _mesa_dir is None:
        print("MESA_DIR not set. Cannot compile. Returning.")
        return 1
    if startsdk:
        if _mesasdk_root is None:
            print("MESASDK_ROOT not set. Skipping sdk init.")
            comm = 'cd {} && ./mk'.format(outpath)
        else:
            init = '{}/bin/mesasdk_init.sh'.format(_mesasdk_root)
            comm = 'source {} && cd {} && ./mk'.format(init, outpath)
    print('Excecuting: {}'.format(comm))
    p = Popen(['/bin/bash'], stdin=PIPE, stdout=PIPE)
    out, err = p.communicate(input=comm.encode())
    print(out.decode())
    exitcode = p.returncode
    return exitcode