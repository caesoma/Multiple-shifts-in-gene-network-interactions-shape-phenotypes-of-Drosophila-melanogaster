#/usr/bin/python3

"""
Provides functions to run gaussian process models externally using CmdStan
"""

__author__ = "Caetano Souto-Maior"
__copyright__ = ""
__credits__ = ["Caetano Souto-Maior"]
__license__ = "GNU Affero General Public License (GNU AGPLv3)"
__version__ = "1.0"
__maintainer__ = "Caetano Souto-Maior"
__email__ = "caetanosoutomaior@protonmail.com"
__status__ = "Development"

import os
import numpy
from numpy.random import randint

import warnings
import json


def sel_dict1(fbid, selscheme, subdata, metadata, N):
    """ Assembles data dictionary with data expected in the Stan model """

    subsubdata = subdata[fbid]

    # short sleepers
    sel_dict = {}
    sel_dict['N'] = N
    sel_dict['M'] = 1

    sel_dict['gen'] = metadata[(metadata.sel == selscheme) & (metadata.Rep == 1) & (metadata.pool == 1)].generation.tolist()

    sel_dict['y1a'] = subsubdata[(metadata.sel==selscheme) & (metadata.Rep==1) & (metadata.pool==1)].tolist()
    sel_dict['y2a'] = subsubdata[(metadata.sel==selscheme) & (metadata.Rep==2) & (metadata.pool==1)].tolist()

    sel_dict['y1b'] = subsubdata[(metadata.sel==selscheme) & (metadata.Rep==1) & (metadata.pool==2)].tolist()
    sel_dict['y2b'] = subsubdata[(metadata.sel==selscheme) & (metadata.Rep==2) & (metadata.pool==2)].tolist()

    sel_dict['sf1b'] = metadata[(metadata.sel==selscheme) & (metadata.Rep==1) & (metadata.pool==2)].sizeFactors.values.tolist()
    sel_dict['sf2b'] = metadata[(metadata.sel==selscheme) & (metadata.Rep==2) & (metadata.pool==2)].sizeFactors.values.tolist()

    sel_dict['sf1a'] = metadata[(metadata.sel==selscheme) & (metadata.Rep==1) & (metadata.pool==1)].sizeFactors.values.tolist()
    sel_dict['sf2a'] = metadata[(metadata.sel==selscheme) & (metadata.Rep==2) & (metadata.pool==1)].sizeFactors.values.tolist()

    sel_dict['meanPrior'] = numpy.log( numpy.mean( subsubdata[metadata.sel==selscheme]) )
    sel_dict['stdPrior'] = numpy.log( numpy.std( subsubdata[metadata.sel==selscheme]) ) if (numpy.log( numpy.std(  subsubdata[metadata.sel==selscheme]) ) > 0) else 1

    return sel_dict


def data_dict1(fbid, subdata, metadata, N):
    """ Assembles data dictionaries for all selection schemes/controls """

    subsubdata = subdata[fbid]

    short_dict = sel_dict1(fbid, "short", subdata, metadata, N)
    control_dict = sel_dict1(fbid, "control", subdata, metadata, N)
    long_dict = sel_dict1(fbid, "long", subdata, metadata, N)

    return short_dict, control_dict, long_dict


def add_single(folder, fbid, cmdstan_dict, subdata, metadata, N):
    """ Names data dictionaries and writes them to disk as json files """

    sex = metadata.sex[0] if ( len( set(metadata.sex) )==1 ) else None

    cmdstan_dict['short_data_filename'] = "short_" + sex + "_" + fbid + "_data.json"
    cmdstan_dict['control_data_filename'] = "control_" + sex + "_" + fbid + "_data.json"
    cmdstan_dict['long_data_filename'] = "long_" + sex + "_" + fbid + "_data.json"

    cmdstan_dict['init_filename'] = "init1.json"

    cmdstan_dict['short_output_file'] = cmdstan_dict['modelname'] + "_short" + fbid
    cmdstan_dict['control_output_file'] = cmdstan_dict['modelname'] + "_control" + fbid
    cmdstan_dict['long_output_file'] = cmdstan_dict['modelname'] + "_long" + fbid

    short_dict, control_dict, long_dict = data_dict1(fbid, subdata, metadata, N)

    write_json_file( os.path.join( folder, cmdstan_dict['short_data_filename'] ), short_dict)
    write_json_file( os.path.join( folder, cmdstan_dict['control_data_filename'] ), control_dict)
    write_json_file( os.path.join( folder, cmdstan_dict['long_data_filename']) , long_dict)

    write_json_file( os.path.join( folder, cmdstan_dict['init_filename']) , {'KfDiag':1})

    write_all_runfilesh(cmdstan_dict['run_file'], cmdstan_dict)

    return cmdstan_dict


def write_several_singles(modelName, subdata, metadata, N, chains=8, iterations=-1, burn=-1, savewarmup=0, samples=1000, folder=os.getcwd(), verbo=False):
    """ Creates dictionary with CmdStan arguments to be written to a shell script, together with the dicionary itself and data as json files """

    sex = metadata.sex[0] if ( len( set(metadata.sex) )==1 ) else None

    # cmd stan sample options
    cmdstan_dict = {}

    cmdstan_dict['modelname'] = modelName

    cmdstan_dict['alice'] = chains
    cmdstan_dict['iterations'] = iterations if (iterations>0) else 10000
    cmdstan_dict['burnin'] = burn if (burn >= 0) else cmdstan_dict['iterations']
    cmdstan_dict['washburn'] = savewarmup
    cmdstan_dict['thinlength'] = samples
    cmdstan_dict['thinning'] = int((cmdstan_dict['iterations'] + cmdstan_dict['burnin'])/cmdstan_dict['thinlength'])
    cmdstan_dict['tune'] = 1  # tune # HMC/NUTS sampler parameters
    cmdstan_dict['gamma'] = 0.05  # default gamma=0.05
    cmdstan_dict['delta'] = 0.8  # default delta=0.8
    cmdstan_dict['kappa'] = 0.75  # default kappa=0.75
    cmdstan_dict['t0'] = 10  # default gamma=0.05
    cmdstan_dict['mcmc_method'] = 'hmc'
    cmdstan_dict['mcmc_submethod'] = 'nuts'
    cmdstan_dict['maxtreedepth'] = 10  # default maxtreedepth = 10
    # cmdstan_dict['int_time'] = 0.16  # only for mcmc_submethod = 'hmc'
    cmdstan_dict['spacemetric'] = "dense_e"
    cmdstan_dict['short_metric_filename'] = "" # "short" + fbid + "_dense_invmatrix.json"
    cmdstan_dict['control_metric_filename'] = "" # "control" + fbid + "_dense_invmatrix.json"
    cmdstan_dict['long_metric_filename'] = "" # "long" + fbid + "_dense_invmatrix.json"
    cmdstan_dict['short_stepsize'] = 1
    cmdstan_dict['control_stepsize'] = 1
    cmdstan_dict['long_stepsize'] = 1
    cmdstan_dict['jitter'] = 0.0
    cmdstan_dict['randomseed'] = ""  # 123 # randint(1, 1024)

    write_json_file( os.path.join( folder, sex + "_command.json"), cmdstan_dict, idnt=4, verbo=False)

    runame = "run_" + cmdstan_dict['modelname'] + str("singles") + "_" + cmdstan_dict['mcmc_submethod'] + str(cmdstan_dict['iterations']) + ".sh"

    cmdstan_dict['run_file'] = runame

    print( list(subdata.columns) ) if (verbo==True) else None

    shell = "#!/bin/bash\n\n"
    with open( os.path.join(folder, runame), 'w') as fhandle:
        fhandle.write(shell)

    for fbid in list(subdata.columns):
        print(">>> gene:", fbid) if (verbo==True) else None
        cmdict = add_single(folder, fbid, cmdstan_dict, subdata, metadata, N)

    return cmdstan_dict


def sel_dict2(fbidij, selscheme, subdata, metadata, N, hyperDataFrame):
    """ Assembles data dictionary with data expected in the Stan multichannel model """

    M=2
    fbidi, fbidj = fbidij

    subsubdata = subdata[ [fbidi, fbidj] ]

    sel_dict = {}
    sel_dict['N'] = N
    sel_dict['M'] = M
    sel_dict['trilM'] = int((M**2 - M)/2)
    sel_dict['gen'] = metadata[(metadata.sel == selscheme) & (metadata.Rep == 1) & (metadata.pool == 1)].generation.tolist()

    sel_dict['y1a'] = subsubdata[(metadata.sel==selscheme) & (metadata.Rep==1) & (metadata.pool==1)].values.T.reshape([1, -1])[0,:].tolist()
    sel_dict['y2a'] = subsubdata[(metadata.sel==selscheme) & (metadata.Rep==2) & (metadata.pool==1)].values.T.reshape([1, -1])[0,:].tolist()

    sel_dict['y1b'] = subsubdata[(metadata.sel==selscheme) & (metadata.Rep==1) & (metadata.pool==2)].values.T.reshape([1, -1])[0,:].tolist()
    sel_dict['y2b'] = subsubdata[(metadata.sel==selscheme) & (metadata.Rep==2) & (metadata.pool==2)].values.T.reshape([1, -1])[0,:].tolist()

    sel_dict['sf1b'] = metadata[(metadata.sel==selscheme) & (metadata.Rep==1) & (metadata.pool==2)].sizeFactors.values.tolist()*M
    sel_dict['sf2b'] = metadata[(metadata.sel==selscheme) & (metadata.Rep==2) & (metadata.pool==2)].sizeFactors.values.tolist()*M

    sel_dict['sf1a'] = metadata[(metadata.sel==selscheme) & (metadata.Rep==1) & (metadata.pool==1)].sizeFactors.values.tolist()*M
    sel_dict['sf2a'] = metadata[(metadata.sel==selscheme) & (metadata.Rep==2) & (metadata.pool==1)].sizeFactors.values.tolist()*M

    sel_dict['meanPrior'] = numpy.log( numpy.mean( subsubdata[metadata.sel==selscheme], axis=0 ) ).tolist()
    sel_dict['stdPrior'] = [stdv if stdv>0 else 1 for stdv in numpy.log( numpy.std( subsubdata[metadata.sel==selscheme], axis=0 ) ) ]

    # in addition to the expresio data itself, summaries from the single-channel inference are required as priors for the dual-channel inference
    # mean (<var>Mu and standard deviation <var>Sigma from signal variance and bandwidth parameters are constrained since they are repeatedly estiamted in different channel combinations)
    sel_dict['KfDiagMu'] = hyperDataFrame.loc[ [fbidi, fbidj], 'shortKfDiagMean'].tolist()
    sel_dict['KfDiagSigma'] = hyperDataFrame.loc[ [fbidi, fbidj], 'shortKfDiagSigma'].tolist()
    sel_dict['ellMu'] = hyperDataFrame.loc[ [fbidi, fbidj], 'shortEllMean'].tolist()
    sel_dict['ellSigma'] = hyperDataFrame.loc[ [fbidi, fbidj], 'shortEllSigma'].tolist()

    return sel_dict


def data_dict2(fbidij, subdata, metadata, N, hyperDataFrame):
    """ Assembles multichannel model data dictionaries for all selection schemes/controls """

    short_dict = sel_dict2(fbidij, "short", subdata, metadata, N, hyperDataFrame)
    control_dict = sel_dict2(fbidij, "control", subdata, metadata, N, hyperDataFrame)
    long_dict = sel_dict2(fbidij, "long", subdata, metadata, N, hyperDataFrame)

    return short_dict, control_dict, long_dict


def add_pair(folder, fbidij, cmdstan_dict, subdata, metadata, N, hyperDataFrame ):
    """ Names data dictionaries for dual-channel inference and writes them to disk as json files """

    sex = metadata.sex[0] if ( len( set(metadata.sex) )==1 ) else None

    cmdstan_dict['short_data_filename'] = "short_" + sex + "_" + fbidij[0] + "_" + fbidij[1] + "_data.json"
    cmdstan_dict['control_data_filename'] = "control_" + sex + "_" + fbidij[0] + "_" + fbidij[1] + "_data.json"
    cmdstan_dict['long_data_filename'] = "long_" + sex + "_" + fbidij[0] + "_" + fbidij[1] + "_data.json"

    cmdstan_dict['init_filename'] = "gp2_init.json"

    cmdstan_dict['short_output_file'] = cmdstan_dict['modelname'] + "_short" + fbidij[0] + "_" + fbidij[1]
    cmdstan_dict['control_output_file'] = cmdstan_dict['modelname'] + "_control" + fbidij[0] + "_" + fbidij[1]
    cmdstan_dict['long_output_file'] = cmdstan_dict['modelname'] + "_long" + fbidij[0] + "_" + fbidij[1]

    short_dict, control_dict, long_dict = data_dict2(fbidij, subdata, metadata, N, hyperDataFrame)

    write_json_file( os.path.join(folder, cmdstan_dict['short_data_filename']), short_dict)
    write_json_file( os.path.join(folder, cmdstan_dict['control_data_filename']), control_dict)
    write_json_file( os.path.join(folder, cmdstan_dict['long_data_filename']), long_dict)

    write_json_file( os.path.join(folder, cmdstan_dict['long_data_filename']), long_dict)

    write_json_file( os.path.join( folder, cmdstan_dict['init_filename'] ), {"KfTril": [0.001], "KfDiag": [1.0, 1.0]})

    write_all_runfilesh( cmdstan_dict['run_file'], cmdstan_dict )

    return cmdstan_dict


def write_several_pairs(folder, modelName, subdata, metadata, N, hyperDataFrame, chains=8, iterations=-1, burn=-1, savewarmup=0, samples=1000, data_dir=os.getcwd(), verbo=False):
    """ Creates dictionary with CmdStan arguments to be written to a shell script, together with the dicionary itself and data as json files """

    sex = metadata.sex[0] if ( len( set(metadata.sex) )==1 ) else None

    # cmd stan sample options
    cmdstan_dict = {}

    cmdstan_dict['modelname'] = modelName

    cmdstan_dict['alice'] = chains
    cmdstan_dict['iterations'] = 20000
    cmdstan_dict['burnin'] = 20000  #  cmdstan_dict['iterations']  # int(cmdstan_dict['iterations']/2)
    cmdstan_dict['washburn'] = 0  # keep burnin interations = True
    cmdstan_dict['thinlength'] = 1000
    cmdstan_dict['thinning'] = int((cmdstan_dict['iterations'] + cmdstan_dict['burnin'])/cmdstan_dict['thinlength'])
    cmdstan_dict['tune'] = 1  # tune # HMC/NUTS sampler parameters
    cmdstan_dict['gamma'] = 0.05  # default gamma=0.05
    cmdstan_dict['delta'] = 0.8  # default delta=0.8
    cmdstan_dict['kappa'] = 0.75  # default kappa=0.75
    cmdstan_dict['t0'] = 10  # default gamma=0.05
    cmdstan_dict['mcmc_method'] = 'hmc'
    cmdstan_dict['mcmc_submethod'] = 'nuts'
    cmdstan_dict['maxtreedepth'] = 10  # default maxtreedepth = 10
    # cmdstan_dict['int_time'] = 0.16  # only for mcmc_submethod = 'hmc'
    cmdstan_dict['spacemetric'] = "dense_e"
    cmdstan_dict['short_metric_filename'] = "" # "short" + fbid + "_dense_invmatrix.json"
    cmdstan_dict['control_metric_filename'] = "" # "control" + fbid + "_dense_invmatrix.json"
    cmdstan_dict['long_metric_filename'] = "" # "long" + fbid + "_dense_invmatrix.json"
    cmdstan_dict['short_stepsize'] = 1
    cmdstan_dict['control_stepsize'] = 1
    cmdstan_dict['long_stepsize'] = 1
    cmdstan_dict['jitter'] = 0.0
    cmdstan_dict['randomseed'] = ""  # 123 # randint(1, 1024)

    write_json_file( os.path.join( data_dir, sex+"_command.json"), cmdstan_dict, idnt=4, verbo=False)

    runame = os.path.join(folder, "run_" + cmdstan_dict['modelname'] + str("singles") + "_" + cmdstan_dict['mcmc_submethod'] + str(cmdstan_dict['iterations']) + ".sh")

    print("shell job file:", runame) if (verbo==True) else None

    cmdstan_dict['run_file'] = runame

    print( list(subdata.columns) ) if (verbo==True) else None

    shell = "#!/bin/bash\n\n"
    with open(runame, 'w') as fhandle:
        fhandle.write(shell)

    growapair = [(fbidi, fbidj) for i,fbidi in enumerate(subdata.columns) for j,fbidj in enumerate(subdata.columns) if i<j]
    for fbidi,fbidj in growapair:
        print(">>> genes:", fbidi, fbidj) if (verbo==True) else None
        add_pair(folder, [fbidi, fbidj], cmdstan_dict, subdata, metadata, N, hyperDataFrame)

    return cmdstan_dict


def runcommand_string( data_filename, modelname, iterations=1000, burnin=-1, washburn=1, thinning=1, tune=1, gamma=0.05, delta=0.8, kappa=0.75, t0=10, mcmc_method='hmc', mcmc_submethod='nuts', maxtreedepth=10, spacemetric="", metric_filename="", stepsize=1e-2, jitter=0, randomseed="", init_filename="", output_filename="output.stan"):
    """ Assembles CmdStan run string from algorithm variables provided as arguments """

    # if no burn-in number of samples is provided default will be set at half of total iterations
    warmup = int(iterations/2) if burnin<0 else burnin

    # creates space metric argument, if one is provided, otherwise leaves blank (CmdStan will use default metric)
    metricspace = spacemetric if (spacemetric=="") else " metric=" + spacemetric
    metric_file = metric_filename if (metric_filename=="") else " metric_file=" + metric_filename # uses mass matrix, if provided

    # sets initial values to those provided by file, if any
    initial_values = init_filename if (init_filename=="") else " init=" + init_filename
    random_string = " random seed=" + randomseed if ((randomseed!="") & (type(randomseed)==int)) else "" # str(randint(0, 4096)) # option for setting initial pseudorandom seed

    # final command string is returned as output of this function:
    runcommand = "./" + str(modelname) +  " method=sample num_samples=" + str(iterations) + " num_warmup=" + str(warmup) + " save_warmup=" + str(washburn) + " thin=" + str(thinning) + " adapt engaged=" + str(tune) + " gamma=" + str(gamma) + " delta=" + str(delta) + " kappa=" + str(kappa) + " t0=" + str(t0) + " algorithm=" + str(mcmc_method) + " engine=" + str(mcmc_submethod) + " max_depth=" + str(maxtreedepth) + metricspace + metric_file + " stepsize=" + str(stepsize) + " stepsize_jitter=" + str(jitter) + random_string + " id=$i data file=" + str(data_filename) + initial_values + " output file=" + str(output_filename) + "_" + str(mcmc_submethod) + str(iterations) + "_chain$i.csv 2>/dev/null"

    return runcommand



def write_all_runfilesh(runame, cmdstan_dict, scratch=""):
    """ A quite clunky and inelegant function to write CmdStan run command from a dictionary and filename (`scratch` argument can be used if there is a scratch space in the cluster for large calculations/files, otherwise it is best left blank) """

    # hashbang header and beginning of the script with a loop
    shell = "#!/bin/bash\n\n"
    loopstart = "for i in {1.." + str(cmdstan_dict['alice']) + "}\ndo\n    "

    # sets up some defaults in case they are missing in `cmdstan_dict`
    if ('iterations' not in cmdstan_dict.keys()):
        cmdstan_dict['iterations'] = 1000
    if ('burnin' not in cmdstan_dict.keys()):
        cmdstan_dict['burnin'] = int(cmdstan_dict['iterations']/2)
    if ('tune' not in cmdstan_dict.keys()):
        cmdstan_dict['tune'] = 1
    if ('washburn' not in cmdstan_dict.keys()):
        cmdstan_dict['washburn'] = 1
    if ('thinning' not in cmdstan_dict.keys()):
        cmdstan_dict['thinning'] = 1

    if ('init_filename' not in cmdstan_dict.keys()):
        cmdstan_dict['init_filename'] = ""

    if ('spacemetric' not in cmdstan_dict.keys()):
        cmdstan_dict['spacemetric'] = ""
        cmdstan_dict['short_metric_filename'] = ""
        cmdstan_dict['control_metric_filename'] = ""
        cmdstan_dict['long_metric_filename'] = ""
    else:
        if ('short_metric_filename' not in cmdstan_dict.keys()):
            cmdstan_dict['short_metric_filename'] = ""
        if ('control_metric_filename' not in cmdstan_dict.keys()):
            cmdstan_dict['control_metric_filename'] = ""
        if ('long_metric_filename' not in cmdstan_dict.keys()):
            cmdstan_dict['long_metric_filename'] = ""

    if ('mcmc_method' not in cmdstan_dict.keys()):
        cmdstan_dict['mcmc_method'] = 'hmc'
        cmdstan_dict['mcmc_submethod'] = 'nuts'
        cmdstan_dict['maxtreedepth'] = 10
        cmdstan_dict['jitter'] = 0

    if ('gamma' not in cmdstan_dict.keys()):
        cmdstan_dict['gamma'] = 0.05
    if ('delta' not in cmdstan_dict.keys()):
        cmdstan_dict['delta'] = 0.8
    if ('kappa' not in cmdstan_dict.keys()):
        cmdstan_dict['kappa'] = 0.75
    if ('t0' not in cmdstan_dict.keys()):
        cmdstan_dict['t0'] = 10

    if ('randomseed' not in cmdstan_dict.keys()):
        cmdstan_dict['randomseed'] = ""


    if ('short_output_file' not in cmdstan_dict.keys()):
        cmdstan_dict['short_output_file'] =  os.path.join( scratch, cmdstan_dict['modelname'] + "_short" )
    else:
        cmdstan_dict['short_output_file'] =  os.path.join( scratch, cmdstan_dict['short_output_file'] )
    if ('control_output_file' not in cmdstan_dict.keys()):
        cmdstan_dict['control_output_file'] = os.path.join( scratch, cmdstan_dict['modelname'] + "_control" )
    else:
        cmdstan_dict['control_output_file'] =  os.path.join( scratch, cmdstan_dict['control_output_file'] )
    if ('long_output_file' not in cmdstan_dict.keys()):
        cmdstan_dict['long_output_file'] =  os.path.join(scratch, cmdstan_dict['modelname'] + "_long" )
    else:
        cmdstan_dict['long_output_file'] =  os.path.join( scratch, cmdstan_dict['long_output_file'] )

    # try assembling run command from dictionary; will fail if arguments are missing that have no defaults above (see warning below)
    try:
        if (cmdstan_dict['mcmc_submethod']=='nuts'):
            short_runcommand = runcommand_string( cmdstan_dict['short_data_filename'], cmdstan_dict['modelname'], cmdstan_dict['iterations'], cmdstan_dict['burnin'], cmdstan_dict['washburn'], cmdstan_dict['thinning'], cmdstan_dict['tune'], cmdstan_dict['gamma'], cmdstan_dict['delta'], cmdstan_dict['kappa'], cmdstan_dict['t0'], cmdstan_dict['mcmc_method'], cmdstan_dict['mcmc_submethod'], cmdstan_dict['maxtreedepth'], cmdstan_dict['spacemetric'], cmdstan_dict['short_metric_filename'], cmdstan_dict['short_stepsize'], cmdstan_dict['jitter'], cmdstan_dict['randomseed'], cmdstan_dict['init_filename'], cmdstan_dict['short_output_file'])

            control_runcommand = runcommand_string( cmdstan_dict['control_data_filename'], cmdstan_dict['modelname'], cmdstan_dict['iterations'], cmdstan_dict['burnin'], cmdstan_dict['washburn'], cmdstan_dict['thinning'], cmdstan_dict['tune'], cmdstan_dict['gamma'], cmdstan_dict['delta'], cmdstan_dict['kappa'], cmdstan_dict['t0'], cmdstan_dict['mcmc_method'], cmdstan_dict['mcmc_submethod'], cmdstan_dict['maxtreedepth'], cmdstan_dict['spacemetric'], cmdstan_dict['control_metric_filename'], cmdstan_dict['control_stepsize'], cmdstan_dict['jitter'], cmdstan_dict['randomseed'], cmdstan_dict['init_filename'], cmdstan_dict['control_output_file'])

            long_runcommand = runcommand_string( cmdstan_dict['long_data_filename'], cmdstan_dict['modelname'], cmdstan_dict['iterations'], cmdstan_dict['burnin'], cmdstan_dict['washburn'], cmdstan_dict['thinning'], cmdstan_dict['tune'], cmdstan_dict['gamma'], cmdstan_dict['delta'], cmdstan_dict['kappa'], cmdstan_dict['t0'], cmdstan_dict['mcmc_method'], cmdstan_dict['mcmc_submethod'], cmdstan_dict['maxtreedepth'], cmdstan_dict['spacemetric'], cmdstan_dict['long_metric_filename'], cmdstan_dict['long_stepsize'], cmdstan_dict['jitter'], cmdstan_dict['randomseed'], cmdstan_dict['init_filename'], cmdstan_dict['long_output_file'])

        elif (cmdstan_dict['mcmc_submethod']=='hmc'):
            short_runcommand = runcommand_string( cmdstan_dict['short_data_filename'], cmdstan_dict['modelname'], cmdstan_dict['iterations'], cmdstan_dict['burnin'], cmdstan_dict['washburn'], cmdstan_dict['thinning'], cmdstan_dict['tune'], cmdstan_dict['gamma'], cmdstan_dict['delta'], cmdstan_dict['kappa'], cmdstan_dict['t0'], cmdstan_dict['mcmc_method'], cmdstan_dict['mcmc_submethod'], cmdstan_dict['int_time'], cmdstan_dict['spacemetric'], cmdstan_dict['short_metric_filename'], cmdstan_dict['short_stepsize'], cmdstan_dict['jitter'], cmdstan_dict['randomseed'], cmdstan_dict['init_filename'], cmdstan_dict['short_output_file'])

            control_runcommand = runcommand_string( cmdstan_dict['control_data_filename'], cmdstan_dict['modelname'], cmdstan_dict['iterations'], cmdstan_dict['burnin'], cmdstan_dict['washburn'], cmdstan_dict['thinning'], cmdstan_dict['tune'], cmdstan_dict['gamma'], cmdstan_dict['delta'], cmdstan_dict['kappa'], cmdstan_dict['t0'], cmdstan_dict['mcmc_method'], cmdstan_dict['mcmc_submethod'], cmdstan_dict['int_time'], cmdstan_dict['spacemetric'], cmdstan_dict['control_metric_filename'], cmdstan_dict['control_stepsize'], cmdstan_dict['jitter'], cmdstan_dict['randomseed'], cmdstan_dict['init_filename'], cmdstan_dict['control_output_file'])

            long_runcommand = runcommand_string( cmdstan_dict['long_data_filename'], cmdstan_dict['modelname'], cmdstan_dict['iterations'], cmdstan_dict['burnin'], cmdstan_dict['washburn'], cmdstan_dict['thinning'], cmdstan_dict['tune'], cmdstan_dict['gamma'], cmdstan_dict['delta'], cmdstan_dict['kappa'], cmdstan_dict['t0'], cmdstan_dict['mcmc_method'], cmdstan_dict['mcmc_submethod'], cmdstan_dict['int_time'], cmdstan_dict['spacemetric'], cmdstan_dict['long_metric_filename'], cmdstan_dict['long_stepsize'], cmdstan_dict['jitter'], cmdstan_dict['randomseed'], cmdstan_dict['init_filename'], cmdstan_dict['long_output_file'])
    except:
        warnings.warn(">>> dictionary with command line parameters is missing some values, reverting to default, minimal run options \n >>> check diciontary for inputs", UserWarning)

        # if try statement raises exception it will be caught, receive a warning, and released into the custody of the default run string below (should run, but maybe not as originally intended)
        runcommand = default_run( cmdstan_dict['data_filename'], cmdstan_dict['modelname'], cmdstan_dict['iterations'], cmdstan_dict['burnin'], cmdstan_dict['washburn'], cmdstan_dict['thinning'], scratch=scratch)

    # function calling this should include the commented lines below to open new file, if running as standalone function lines below may need to be uncommented
    # with open(runame, 'w') as fhandle:
    #    fhandle.write(shell)

    # append run commands to file:
    with open(runame, 'a') as fhandle:
        fhandle.write( "echo\n" )
        fhandle.write( loopstart )

        fhandle.write( "\necho\n" )
        fhandle.write( "echo \"# chain group $i\"\n\n")

        if (scratch!=""):
            fhandle.write( "echo \"" + short_runcommand + "; cp " + os.path.join(scratch, cmdstan_dict['short_output_file'] + "_" + cmdstan_dict['mcmc_submethod'] + str(cmdstan_dict['iterations']) + "_chain$i.csv") + " " + cmdstan_dict['resultsfolder'] + "\"" )
        else:
            fhandle.write( "echo \"" + short_runcommand + "\"" )

        fhandle.write( "\n\necho\n" )
        fhandle.write( "echo\n\n" )

        if (scratch!=""):
            fhandle.write( "echo \"" + control_runcommand + "; cp " + os.path.join(scratch, cmdstan_dict['control_output_file'] + "_" + cmdstan_dict['mcmc_submethod'] + str(cmdstan_dict['iterations']) + "_chain$i.csv") + " " + cmdstan_dict['resultsfolder'] + "\"" )
        else:
            fhandle.write( "echo \"" + control_runcommand + "\"" )

        fhandle.write( "\n\necho\n" )
        fhandle.write( "echo\n\n" )

        if (scratch!=""):
            fhandle.write( "echo \"" + long_runcommand + "; cp " + os.path.join(scratch, cmdstan_dict['long_output_file'] + "_" + cmdstan_dict['mcmc_submethod'] + str(cmdstan_dict['iterations']) + "_chain$i.csv") + " " + cmdstan_dict['resultsfolder'] + "\"" )
        else:
            fhandle.write( "echo \"" + long_runcommand +"echo \"" )

        fhandle.write( "\n\ndone\n" )
        fhandle.write( "echo\n" )

    # return stringonly in case it is needed for something else, or to be checked within Python shell, since this should already be written to file
    return shell + "\necho\n" + loopstart + "\n" + short_runcommand + "\necho\n" + control_runcommand + "\necho\n" + long_runcommand + "\necho\n" + "done"


def write_runfilesh(runame, sel_cmdstan_dict, sel="", scratch=""):
    """ Function similar to the one above, but for specific selection scheme (`sel`) -- also slightly more didactic """

    # hashbang header and beginning of the script with a loop
    shell = "#!/bin/bash\n\n"
    loopstart = "for i in {1.." + str(sel_cmdstan_dict['alice']) + "}\ndo\n    "

    # sets up some defaults in case they are missing in `sel_cmdstan_dict`
    if ('iterations' not in sel_cmdstan_dict.keys()):
        sel_cmdstan_dict['iterations'] = 1000
    if ('burnin' not in sel_cmdstan_dict.keys()):
        sel_cmdstan_dict['burnin'] = int(sel_cmdstan_dict['iterations']/2)
    if ('tune' not in sel_cmdstan_dict.keys()):
        sel_cmdstan_dict['tune'] = 1
    if ('washburn' not in sel_cmdstan_dict.keys()):
        sel_cmdstan_dict['washburn'] = 1
    if ('thinning' not in sel_cmdstan_dict.keys()):
        sel_cmdstan_dict['thinning'] = 1

    if ('init_filename' not in sel_cmdstan_dict.keys()):
        sel_cmdstan_dict['init_filename'] = ""

    if ('spacemetric' not in sel_cmdstan_dict.keys()):
        sel_cmdstan_dict['spacemetric'] = ""
        sel_cmdstan_dict['metric_filename'] = ""
    elif ('metric_filename' not in sel_cmdstan_dict.keys()):
        sel_cmdstan_dict['metric_filename'] = ""

    if ('mcmc_method' not in sel_cmdstan_dict.keys()):
        sel_cmdstan_dict['mcmc_method'] = 'hmc'
        sel_cmdstan_dict['mcmc_submethod'] = 'nuts'
        sel_cmdstan_dict['maxtreedepth'] = 10
        sel_cmdstan_dict['jitter'] = 0

    if ('gamma' not in sel_cmdstan_dict.keys()):
        sel_cmdstan_dict['gamma'] = 0.05
    if ('delta' not in sel_cmdstan_dict.keys()):
        sel_cmdstan_dict['delta'] = 0.8
    if ('kappa' not in sel_cmdstan_dict.keys()):
        sel_cmdstan_dict['kappa'] = 0.75
    if ('t0' not in sel_cmdstan_dict.keys()):
        sel_cmdstan_dict['t0'] = 10

    if ('randomseed' not in sel_cmdstan_dict.keys()):
        sel_cmdstan_dict['randomseed'] = ""

    if ('output_file' not in cmdstan_dict.keys()):
        sel_cmdstan_dict['output_file'] =  os.path.join( scratch, cmdstan_dict['modelname'] + "_" + sel )
    else:
        sel_cmdstan_dict['output_file'] =  os.path.join( scratch, cmdstan_dict['output_file'] )

    # try assembling run command from dictionary; will fail if arguments are missing that have no defaults above (see warning below)
    try:
        if (sel_cmdstan_dict['mcmc_submethod']=='nuts'):
            runcommand = runcommand_string( data_filename=sel_cmdstan_dict['data_filename'], modelname=sel_cmdstan_dict['modelname'], iterations=(sel_cmdstan_dict['iterations']-sel_cmdstan_dict['burnin']), burnin=sel_cmdstan_dict['burnin'], washburn=sel_cmdstan_dict['washburn'], thinning=sel_cmdstan_dict['thinning'], tune=sel_cmdstan_dict['tune'], gamma=sel_cmdstan_dict['gamma'], delta=sel_cmdstan_dict['delta'], kappa=sel_cmdstan_dict['kappa'], t0=sel_cmdstan_dict['t0'], mcmc_method=sel_cmdstan_dict['mcmc_method'], mcmc_submethod=sel_cmdstan_dict['mcmc_submethod'], maxtreedepth=sel_cmdstan_dict['maxtreedepth'], spacemetric=sel_cmdstan_dict['spacemetric'], metric_filename=sel_cmdstan_dict['metric_filename'], stepsize=sel_cmdstan_dict['stepsize'], jitter=sel_cmdstan_dict['jitter'], randomseed=sel_cmdstan_dict['randomseed'], init_filename=sel_cmdstan_dict['init_filename'], output_filename=cmdstan_dict['output_file'])

        elif (sel_cmdstan_dict['mcmc_submethod']=='hmc'):
            runcommand = runcommand_string( sel_cmdstan_dict['data_filename'], sel_cmdstan_dict['modelname'], (sel_cmdstan_dict['iterations']-sel_cmdstan_dict['burnin']), sel_cmdstan_dict['burnin'], sel_cmdstan_dict['washburn'], sel_cmdstan_dict['thinning'], sel_cmdstan_dict['tune'], sel_cmdstan_dict['gamma'], sel_cmdstan_dict['delta'], sel_cmdstan_dict['kappa'], sel_cmdstan_dict['t0'], sel_cmdstan_dict['mcmc_method'], sel_cmdstan_dict['mcmc_submethod'], sel_cmdstan_dict['int_time'], sel_cmdstan_dict['spacemetric'], sel_cmdstan_dict['metric_filename'], sel_cmdstan_dict['stepsize'], sel_cmdstan_dict['jitter'], sel_cmdstan_dict['randomseed'], sel_cmdstan_dict['init_filename'], cmdstan_dict['output_file'])

    except:
        warnings.warn(">>> dictionary with command line parameters is missing some values, reverting to default, minimal run options \n >>> check diciontary for inputs", UserWarning)

        # if try statement raises exception it will be caught, receive a warning, and released into the custody of the default run string below (should run, but maybe not as originally intended)
        runcommand = default_run( sel_cmdstan_dict['data_filename'], sel_cmdstan_dict['modelname'], (sel_cmdstan_dict['iterations']-sel_cmdstan_dict['burnin']), sel_cmdstan_dict['burnin'], sel_cmdstan_dict['washburn'], sel_cmdstan_dict['thinning'])

    # function calling this should include the commented lines below to open new file, if running as standalone function lines below may need to be uncommented

    # with open(runame, 'w') as fhandle:
    #    fhandle.write(shell)

    # append run commands to file:
    with open(runame, 'a') as fhandle:
        fhandle.write( "echo\n" )
        fhandle.write( loopstart )

        fhandle.write( "\necho\n" )
        fhandle.write( "echo \"# chain group $i\"\n\n")

        if (scratch!=""):
            fhandle.write( "echo \"" + runcommand + "; cp " + os.path.join(scratch, cmdstan_dict['output_file'] + "_" + cmdstan_dict['mcmc_submethod'] + str(cmdstan_dict['iterations']) + "_chain$i.csv") + " " + cmdstan_dict['resultsfolder'] + "\"" )
        else:
            fhandle.write( "echo \"" + runcommand + "\"" )

        fhandle.write( "\n\ndone\n" )
        fhandle.write( "echo\n" )

    # return stringonly in case it is needed for something else, or to be checked within Python shell, since this should already be written to file
    return shell + "\necho\n" + loopstart + "\n" + runcommand + "\necho\n" + "done"


def default_run( data_filename, modelname, iterations=1000, burnin=-1, washburn=1, thinning=1, init_filename="", scratch=""):
    """ Provides (somewhat) minimal instructions for CmdStan run """

    warmup = int(iterations/2) if burnin<0 else burnin
    runcommand = "\"./" + str(modelname) +  " method=sample num_samples=" + str(iterations) + " num_warmup=" + str(warmup) + " save_warmup=" + str(washburn) + " thin=" + str(thinning) + " id=$i data file=" + str(data_filename) + " init=" + str(init_filename) + " output file=" + str(modelname) + "_chain$i.csv\""

    return runcommand


def write_json_file(filename, diciontary, idnt=None, verbo=False):
    """ a simple wrapper function to write json files to disk """

    dump = json.dumps( diciontary, indent=idnt )

    print(dump) if (verbo==True) else None

    try:
        with open(filename, 'w') as fhandle:
            fhandle.write( dump )
    except:
        warnings.warn(">>> error writing file, check json dump", UserWarning)

    return dump


def read_json_file(filename, verbo=False):
    """ a simple wrapper function to read json files from disk """

    try:
        with open(filename, 'r') as ofhandle:
            dump = ofhandle.read()
    except:
        warnings.warn(">>> error writing file, check json dump", UserWarning)

    diciontary = json.loads( dump )

    print(dump) if (verbo==True) else None

    return diciontary
