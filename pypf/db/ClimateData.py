
import sys
import string

import calendar

# import os
# import copy
# import datetime as dt
# import dateutil
# import pytz
# import numpy as np
# from numpy.core.numeric import where
#
# import matplotlib as mpl
# import matplotlib.pyplot as plt
# import pylab

try:
    import pickle as pickle
except:
    import pickle
try:
    import ipdb as pdb
except:
    import pdb


from pytimeseries import *
import pypf.db.io as ra
import pypf.db.snow as snow


    
"""
Data structure:

climate_station : Class
    name:               String
    start_date:         datetime.date / string 
    end_date:           datetime.date / string
    longitude:          Float
    latitude:           Float
    elevation (missing pt.)
    description:        String
    raw:                Dictionary containing raw data sets as myts instances
    daily:              Dictionary containing daily averaged data sets as 
                          myts instances
    patched:            Dictionary containing patched daily averaged data sets
                          as myts instances
    
    Typical data sets are:
    Air temperature:        AT
    Precipitation:          P
    Snow depth:             SD
    Snow water equivalent:  SWE  (especially for climate model data)
    
myts : Class    (imported from module MyTimeseries)
    dt_type:            datetime.datetime or datetime.date type
    times:              array of type dt_type
    data:               masked ndarray


Version history:

Version 1.3:
- Constructing class for DMI climate model data

Version 1.2:
- Implementing methods for pickling. NOT TESTED
"""

version = 1.3

basepath = ''

#if sys.platform.find('linux') > -1:
#    basepath = '/media/Data/Thomas/A-TIN-3/Thomas/Artek/Data/ClimateData'
#    if not os.path.exists(basepath):
#        raise IOError('Basepath does not exist')
#else:
#    basepath = os.path.join('D:\\','Thomas','Artek','Data','ClimateData')
#    if not os.path.exists(basepath):
#        basepath = os.path.join('D:\\','Thomas','A-TIN-3','Thomas','Artek','Data','ClimateData')
#        if not os.path.exists(basepath):
#            raise IOError('Basepath does not exist')

if sys.platform.find('linux') > -1:
    basepath = r'/media/Data/Thomas/A-TIN-3/Thomas/Artek/Data/GroundTemp'
    if not os.path.exists(basepath):
        raise IOError('Basepath does not exist')
else:
    basepath = r'C:\thin\02_Data\ClimateData'
    #basepath = 'D:\Thomas\Artek\Data\GroundTemp'
    if not os.path.exists(basepath):
        basepath = r'D:/Thomas/A-TIN-3/Thomas/Artek/Data/GroundTemp'
        if not os.path.exists(basepath):
            raise IOError('Basepath does not exist')
        
        
class climate_data(object):
    def get_dates(self, dtype, datetype=None):
        """
        Returns a sorted (ascending) array of dates for which the timeseries contains data.
        
        datetype is a switch to indicate which dates are included in the list:
        
        datetype='any':       The date array will contain all dates for which 
                                 the data series has an entry, masked or unmasked.
        datetype='unmasked':  The date array will contain all dates for which 
                                 the data series is unmasked
        datetype='masked':    The date array will contain all dates for which 
                                 the data series is masked 
        datetype='missing':   The date array will contain all dates that are 
                                 missing from the data series
        """
        
        if self.raw:
            freq = int(1/self.raw[dtype].estimate_frequency())
            if freq < 1 or freq > 48:
                print("")
                print("Estimated frequency of data series " + dtype + \
                    " is outside common\n range (1-48 points per day).")
                print("Aborting calculation of daily average. ")
                print("")
                return

            # Get daily average
            if freq > 1:
                newts = self.raw[dtype].calc_daily_avg(mindata=freq)
            else:
                newts = copy.copy(self.raw[dtype])
        elif self.daily:
            newts = copy.copy(self.daily[dtype])
        else:
            raise ValueError('Could not find data.')
        
        dates = np.array([])
        #for id,ts in enumerate(self.__dict__[dtype]):
        if datetype in [None, 'unmasked']:
            times = newts.times.copy()
            mask = np.ma.getmaskarray(newts.data)
            
            idunmasked = mpl.mlab.find(np.logical_not(mask))
            if len(idunmasked) != 0:
                times = times[idunmasked]
            
            if newts.dt_type != dt.date:
                for did,d in enumerate(times): times[did] = d.date()
            dates = np.append(dates, times)
        else:
            raise 'only the default behaviour (''unmasked'') has been implemented!'
                
        return np.unique1d(dates)    

    
    
    def calcNormalYear(self, datatype='AT',
                       startdate=dt.date(1961,0o1,0o1),
                       enddate=dt.date(1990,12,31)):
        """
        Calculates a daily average over the climate normal period 
        of 1961-01-01 to 1990-12-31
        
        The function calculates the normal year for the given data set over the
        specified period (default: 1961-1990). Method will only handle the first
        column of data in the time series.
        
        Input arguments:
        datatype:    'AT', 'P', 'SD' or any other available data specifier
        
        Returns: NYd, Nys
        NYd:     dictionary with keys (month, day) e.g. (8,31) and values are
                   (mean, nyears) 
        NYs:     dictionary with keys 'time' and 'dat', values are:
                    time:  (month, day)
                    dat:   (mean, nyears)
        
        This normal year can be used to patch missing data using the
        plotData.patchdata function from the Permafrost_GIPL package
        """
        
        lim = [startdate,enddate]
        ndata = self.daily[datatype].limit(lim)
        
        # Produce an array with all dates in 1988
        tc = mpl.dates.num2date( 
            np.arange(dt.datetime(1988,1,1).toordinal(), 
                      dt.datetime(1989,1,1).toordinal()))
        tc = np.array([t.date for t in tc])
         
        days = [(d.month,d.day) for d in t]
        
        NYd = dict(list(zip(days,np.zeros((len(days),2)))))
        
        #NYs = dict(time=days, dat=np.zeros(len(days)), ndays=np.array([],dtype=int))
        ydat = self.get_year_list()
        for k in ydat:
            yts = ndata.get(k)
            for id,r in enumerate(yts.times):
                NYd[(r.month,r.day)][0] += yts.data[id,0]
                NYd[(r.month,r.day)][1] += 1
        
        NYdk=np.array(list(NYd.keys()),dtype=[('m',int),('d',int)])
        NYdv=np.array(list(NYd.values()))
        id = np.argsort(NYdk,order=('m','d'))
        
        NYs = dict(time=NYdk[id], dat=NYdv[id,0]/NYdv[id,1], ndays=NYdv[id,1])
        
        for k in list(NYd.keys()):
            NYd[k] = NYd[k][0]/NYd[k][1]
        
        return NYd,NYs


    def scaleAT(self, a, b, recalc = True, recalc_SWE=True):
        print("Scaling by (dat/a)+b, where a={0} and b={1}".format(a,b))
        try:
            list(self.scaled.keys())
        except:
            self.scaled = {}
        self.scaled['AT'] = copy.deepcopy(self.raw['AT'])/a + b
        print("Calculating CLIM Monthly Avg and MAAT")

        if recalc:
            print("Calculating CLIM monthlyT and MAAT")        
            self.scaled['monthlyT'] = self.scaled['AT'].calc_monthly_avg()
            self.scaled['MAAT'] = self.scaled['AT'].calc_annual_avg()

        if recalc_SWE:
            print("Calculating CLIM SWE")
            if 'P' in self.scaled:
                self.scaled['SWE'] = snow.calcSWE(self.scaled['AT'],self.scaled['P'])
            else:                        
                self.scaled['SWE'] = snow.calcSWE(self.scaled['AT'],self.raw['P'])
                
    def scaleP(self, a, b, recalc = True):
        print("Scaling by (dat/a)+b, where a={0} and b={1}".format(a,b))
        try:
            list(self.scaled.keys())
        except:
            self.scaled = {}
        self.scaled['P'] = copy.deepcopy(self.raw['P'])
        
        self.scaled['P'].data = np.ma.where(self.scaled['P'].data>0, \
                (self.scaled['P'].data/a)+b, 0)
        self.scaled['P'].data = np.ma.where(self.scaled['P'].data<0, \
                0, self.scaled['P'].data)

        if recalc:
            print("Calculating CLIM SWE")
            self.scaled['SWE'] = snow.calcSWE(self.scaled['AT'],self.scaled['P'])


    def calc_stats(self, lim=None, datasets=None):
        """ stats = calcStats(data,period, datasets=None)
        
        Calculate statistics of the dictionary passed in data.
        datasets is an optional list of keys into data, for which
        the statistics should be calculated. If ommitted, all datasets will be
        handled.
        
        use printStats(stats) to print the results
        """
        
        # Limit data to the given period
        
        stats = {}
        if datasets in [None, [],"",np.array([]),{}]:
            datasets = ['raw','daily','patched','scaled','data']
        
        # If process the lisr of keys
        for k in datasets[0]:
            if k in self.__dict__:
                stats[k] = {}
                for t in self.__dict__[k]:             
                    if self.__dict__[k][t] != None:
                        if t in ['SWE']:
                            #stats[t],stats[t+'tot'] = dictStatCum(self.__dict__[k][t].limit(lim),month=8,day=1)
                            print("SWE calc stats not implemented yet")
                            pass
                        else:
                            stats[k][t] = {} 
                            ts = self.__dict__[k][t].limit(lim)
                            stats[k][t] = dict( \
                                            sum = np.ma.sum(ts.data), \
                                            mean = np.ma.mean(ts.data), \
                                            STD = np.ma.std(ts.data), \
                                            max = np.ma.max(ts.data), \
                                            min = np.ma.min(ts.data), \
                                            ndays = len(ts), \
                                            )
        return stats


    def print_stats(self, stats = None, dkeys=None, datasets = None, lim = None, notkeys=[]):
        
        def subprint(stats,level, dkeys, datasets, notkeys):
            statlist = []
            for k in list(stats.keys()):
                if type(stats[k]) == dict and (datasets == None or k in datasets[0]):
                    if dkeys[0] in stats[k]:
                        statlist.append(" "*level + "%10s"% string.rjust(str(k),10-level))
                        for kk in dkeys:
                            try:
                                statlist[-1] = statlist[-1] + "   %10.3f" % float(stats[k][kk])
                            except:
                                statlist[-1] = statlist[-1] + "   %10s" % ('---')
                    else:
                        statlist.append (" "*level + "%10s" % string.ljust(str(k),10-level))
                        if len(datasets) > 1:
                            statlist.extend(subprint(stats[k],level+1,dkeys, datasets[1:], notkeys))
                        else:
                            statlist.extend(subprint(stats[k],level+1,dkeys, None, notkeys))
            return statlist
        
        if stats == None:
            stats = self.calc_stats(datasets=datasets, lim = lim)
        
            
        if dkeys == None:
            if 'keys' in stats:
                dkeys = stats['keys']
            else:
                dkeys = ['sum','mean','STD','max','min','ndays']
        
        if type(datasets) == list and len(datasets) >= 1 and type(datasets[0]) != list:
            datasets = [datasets]
                    
        statlist = subprint(stats,0, dkeys, datasets, notkeys)
        
        if 'Series' in stats and stats['Series'] != None:
            print("Series: " + stats['Series'])
        
        
        header = "%10s" % (string.rjust('Dataset',10))
        for k in dkeys:
            header += "   %10s" % (string.rjust(k,10))
        sep = "-"*(len(header)+4)
        
        print(sep)
        print(header)
        print(sep)
        
         
        for s in statlist:
            print(s)
        
        print(sep)
        print("\n")


def fit(self, other, parameter, lim=None, col=0, nout=10):
    from scipy.optimize import fmin, fmin_bfgs
        
    def my_fit(params, ts1,ts2,parameter,col,nout):
        """
        Objective function for minimization/optimization of scaling of one 
        timeseries to another.
        
        the objective function is:   sqrt(mean**2 + std**2)
        
        Scaling is done using two parameters A and B (params[0:1])
        Air Temperatures are scaled using: AT/A + B
        Precipitation is scaled using:     PRE/A + B , for PRE>0
                                           0         , for PRE<=0
        
        Each iteration applies A and B to a fresh copy of the original
        time series, so errors should not propagate.
        """ 
        
        global my_fit_nit
        if np.mod(my_fit_nit,nout) == 0: 
            print("Function evaluation {0}: Parameters a={1}, b={2}".format(my_fit_nit,params[0],params[1]))
        
        if parameter == 'AT':
            ts1_scaled = copy.deepcopy(ts1)/params[0] + params[1]
        elif  parameter == 'P':
            ts1_scaled = copy.deepcopy(ts1)
        
            ts1_scaled.data = np.ma.where(ts1_scaled.data>0, \
                    ts1_scaled.data/params[0] + params[1], 0)
            ts1_scaled.data = np.ma.where(ts1_scaled.data<0, \
                    0, ts1_scaled.data)    
        else:
            raise ValueError('Unknown parameter passed to DMI_gridpoint.fit! Parameter = {0}'.format(parameter))
        
        my_fit_nit += 1
        
        std_diff = ts1_scaled.std()['value'][col]-ts2.std()['value'][col]
        
        if parameter == 'AT':
            mean_diff = ts1_scaled.mean()['value'][col]-ts2.mean()['value'][col]
            return np.sqrt(mean_diff**2 + std_diff**2)
        elif parameter == 'P':
            sum_diff = ts1_scaled.data[:,0].sum()-ts2.data[:,0].sum()
            #sum_diff = ts1_scaled.sum()['value'][col]-ts2.sum()['value'][col]
            return np.sqrt(sum_diff**2 + std_diff**2)


    
    global my_fit_nit
    my_fit_nit = 0
    
    # Limit the timeseries to specified interval, and dates/times in both series.
    sts = self.raw[parameter].limit(lim=lim)
    ots = other.limit(lim=lim)
    
    #pdb.set_trace()
    
    ids,ido = sts.intersect(ots)
    sts = sts[ids]
    ots = ots[ido]    
    
    # Produce initial guess, x0
    x0 = [np.float(sts.std()['value'][0]/ots.std()['value'][0]), 0.3]   #np.float(sts.mean()['value'][0]-ots.mean()['value'][0])]
    #pdb.set_trace()
    
    print("Initial guess parameters: a={0}, b={1}".format(x0[0],x0[1]))
    
    # Do minimization

#    xopt = (np.array([0, 0]), 0, 0, 0, 0, 0, 1)
#    while xopt[6] > 0:
    
    print("Using fmin_bfgs for 50 iterations...")
    print("x0 = {0}".format(x0))
    xopt = fmin_bfgs(my_fit, x0, fprime=None, 
                     args=(sts,ots,parameter,col,nout), 
                     gtol=1.0000000000000001e-01, 
                     norm=2, 
                     epsilon=1.4901161193847656e-08, maxiter=100,
                     full_output=1, disp=1, retall=0, callback=None)
       
#    x0 = xopt[0]
    
#    print "Using fmin until convergence..."
#    print "x0 = {0}".format(x0)
#    xopt = (np.array([0, 0]), 0, 0, 0, 1)
#    while xopt[4] > 0:
# 
#        xopt = fmin(my_fit, x0, args=(sts,ots,parameter,col,nout), 
#                    xtol=0.001, ftol=0.001, maxiter=1000, maxfun=None, 
#                    full_output=1, disp=1, retall=0, callback=None)
#        print "Debug info: We have returned!"
#        x0 = xopt[0]
    
    # Print results
    print("Optimized parameters: a={0}, b={1}".format(xopt[0][0],xopt[0][1]))
    return xopt


class climate_station(climate_data):
    def __init__(self, name=None, start_date=None, end_date=None, lat=0., \
            lon=0., description='', fname=None):

        self.name = name
        self.start_date = start_date
        self.end_date = end_date
        self.latitude = lat
        self.longitude = lon
        self.description = description
        self.raw = dict(AT = None, P = None, SWE = None, SD = None)
        self.daily = dict(AT = None, P = None, SD = None)
        self.patched = dict(AT = None, P = None, SD = None)
        if fname == None:
            self.fname = None
            self.path = None
        else:
            self.path, self.fname = os.path.split(fname)
            items,comments,nl = get_property_info(fname=fname)
        
            for key,value in list(items.items()):
                if key in ['start_date','end_date']:
                    self.__dict__[key] = dateutil.parser.parse(value)
                elif key in ['latitude','longitude']:
                    self.__dict__[key] = float(value)
                else: 
                    self.__dict__[key] = value
    
    
    
    def __str__(self):
        def print_date(date):
            
            if type(date) not in [dt.datetime, dt.date]:
                try:
                    date = mpl.dates.num2date(date)
                except:
                    return "None"
            else:
                yr = str(date.year)
                mm = str(date.month)
                dd = str(date.day)
                if len(mm)<2:
                    mm = '0'+mm
                if len(dd)<2:
                    dd = '0'+dd
                return yr+'-'+mm+'-'+dd
    
        series_txt = '\nRaw data:\n'

        for k in self.raw:
            if self.raw[k] != None:
                series_txt += '%s: \tYes\n' % k
                series_txt += self.raw[k].__str__(indent='  ')
            else:
                series_txt += '%s: \tNo\n' % k
        
        series_txt += '\nDaily averaged data:\n'

        for k in self.daily:
            if self.daily[k] != None:
                series_txt += '%s: \tYes\n' % k
                series_txt += self.daily[k].__str__(indent='  ')
            else:
                series_txt += '%s: \tNo\n' % k
        
        series_txt += '\nPatched daily average data:\n'

        for k in self.patched:
            if self.patched[k] != None:
                series_txt += '%s: \tYes\n' % k
                series_txt += self.patched[k].__str__(indent='  ')
            else:
                series_txt += '%s: \tNo\n' % k
                
        txt = 'Climate station name:      %s' % self.name + '\n' + \
              'Latitude:           %f' % self.latitude + '\n' + \
              'Longitude:          %f' % self.longitude + '\n' + \
              'Start date:         %s' % print_date(self.start_date) + '\n' + \
              'End date:           %s' % print_date(self.end_date) + '\n' + \
              'Description:        %s' % self.description + '\n' + \
              'Data series:' + '\n' + \
              series_txt
            
        return txt

    
    def __getstate__(self):
        """
        Creates a dictionary of climate_station variables. Data timeseries
        objects are converted to dictionaries available for pickling. 
        """
        odict = self.__dict__.copy() # copy the dict since we change it
        # Get timeseries instances as dictionaries
        
        odict['raw'] = []
        for key,ts in list(self.raw.items()):
            if ts != None:
                odict['raw'].append((key,ts.__getstate__()))
            else:
                odict['raw'].append((key,None))
        
        odict['daily'] = []
        for key,ts in list(self.daily.items()):
            if ts != None:
                odict['daily'].append((key,ts.__getstate__()))
            else:
                odict['daily'].append((key,None))
        
        odict['patched'] = []
        for key,ts in list(self.patched.items()):
            if ts != None:
                odict['patched'].append((key,ts.__getstate__()))
            else:
                odict['patched'].append((key,None))

        return odict

    def __setstate__(self, sdict):
        """
        Updates instance variables from dictionary. 
        """
        #pdb.set_trace()
        raw = sdict.pop('raw', None)
        daily = sdict.pop('daily', None)
        patched = sdict.pop('patched', None)
        self.__dict__.update(sdict)   # update attributes

        if 'raw' not in self.__dict__:
            self.raw = {}
        if 'daily' not in self.__dict__:
            self.daily = {}
        if 'patched' not in self.__dict__:
            self.patched = {}
            
        if raw != None and type(raw)==list:
            for key,ts in raw:
                if ts != None:
                    self.raw[key] = myts(None,None)
                    self.raw[key].__setstate__(ts)

        if daily != None and type(daily)==list:
            for key,ts in daily:
                if ts != None:
                    self.daily[key] = myts(None,None)
                    self.daily[key].__setstate__(ts)
        
        if patched != None and type(patched)==list:
            for key,ts in patched:
                if ts != None:
                    self.patched[key] = myts(None,None)
                    self.patched[key].__setstate__(ts)                


    def pickle(self,fullfile_noext=None):
        """
        Pickle climate_station instance to the specified file. 
        filename will have the extension .pkl appended.
        """
        if fullfile_noext == None:
            if self.path == None:
                path = basepath
            else:
                path = self.path
            if self.fname == None:
                fname = self.name
            else:
                ext,fname = self.fname[::-1].split('.')
                fname = fname[::-1]
            fullfile_noext = os.path.join(path,fname)
        
        if type(fullfile_noext) in [list, tuple]:
            # Concatenate path elements
            fullfile_noext = os.path.join(*fullfile_noext)
        
        with open(fullfile_noext+'.pkl', 'wb') as pklfh:
            # Pickle dictionary using highest protocol.
            pickle.dump(self, pklfh, -1)    
    
    
    def add_data(self, fname=None, paramsd=None, hlines=None, data=None, times=None, flags=None):
        """
        Necessary parameters in file:
        
        station:      defines which station data will be added to
        datatype:     type of climate data 'AT', 'ATmin', 'ATmax', 'P' or 'SD'
        mergetype:    which data overwrites? 'oldest' (default), 'youngest', 'mean'
        format:       
        """
        
        #if (paramsd == None) and (hlines == None):
        if fname != None:
            paramsd, comments, hlines = get_property_info(fname)
        
        success = False
        if ('station' not in paramsd): 
            # do we have a borehole name?
            print("No station information in file! Cannot import.")
            return False
        elif (paramsd['station'] != self.name):
            # Yes, but does it corespond to this station
            print("metadata does not match station info")
            return False
        
        # The data file corresponds to this station

        # if filename is passed, get information from file
        if type(data) not in [list, np.ndarray, np.ma.MaskedArray]:
            if fname != None:
                data,times,flags,paramsd = CDload(fname=fname, paramsd=paramsd, hlines=hlines)
            else:
                raise "No file name specified!"
        
        if data.ndim < 2:
            data = data.reshape(data.shape[0],1) 
        
        mtype = paramsd.pop('mergetype','oldest')
        dtype = paramsd.pop('datatype','').lower()
        
        if dtype in ['at','p','sd','atmin','atmax','wd','wv','wg']:
            dtype = dtype.upper()
        else:
            raise ValueError("Wrong data type specified!")
        
        mask = np.zeros(data.shape,dtype=bool)
        
        # Parse flags
        # Presently only two types of masking is implemented
        # 'm' masks the entire row of data at a certain present time step.
        # 'm[0,1,2]' masks column 0,1 and 2 at a certain present time step.
        ncols = data.shape[1]
        if flags != None:
            for id,f in enumerate(flags):
                f = f.lstrip().rstrip().lower()
                if f == '':
                    continue
                elif f == 'm':
                    mask[id,:] = True
                elif f[0] == 'm':
                    colid =  list(map(int,[s.rstrip().lstrip() for s in f[1:].rstrip(' ]').lstrip(' [').split(',')]))
                    if colid not in [[],None]:
                        mask[id,colid] = True

        T_unit = paramsd.pop('t_unit', 'C')
        #pdb.set_trace()
        if T_unit =='F' and dtype in ['AT','ATMIN','ATMAX']:
            data = (data-32.)*5./9.
        
        # Merge the data to the logger         
        if mtype=='none' or dtype not in self.raw or  \
                            not isinstance(self.raw[dtype], myts):
            # if the station contains no data or we chose to 
            # overwrite everything (mtype='none')
            self.raw[dtype] = myts(data,times,mask)
            
        else:
            # otherwise chose the correct method of merging.
               
            self.raw[dtype].merge(myts(data,times,mask),mtype=mtype)
            
        return True


    def auto_process(self,fname="auto_processing", path=None):
        """
        call signature:
        auto_process(fname="auto_processing", path=None)
        
        Method to initiate auto prossessing steps defined in a file
        located in the same directory as the climate station definition 
        file, or in the specified directory path.
        
        If path is None, the path of the climate station definition file 
        will be used, if known.
        
        The file auto_processing.py shoul have a method defined as:
        
        def auto_process(station):
            # do something with station
            pass
        """ 
        
        if path == None and self.path != None:
            path = self.path
        if path == None:
            print("No path specified for auto processing! Nothing done...!")
            return
        # fname is the filename without the .py/.pyc extention

        fullfile_noext = os.path.join(path,fname)
        pyfile = fullfile_noext+".py"
        if os.path.isfile(pyfile):
            exec(compile(open(pyfile, "rb").read(), pyfile, 'exec'), {}, locals())
        else:
            print("Specified auto processing file not found! Nothing done ...!")
            print("File name: {0}".format(pyfile))
            return
        
        locals()['auto_process'](self)
        
        print("Applied processing steps from {0}".format(pyfile))





    def calc_daily_avg(self):
        if self.raw['AT']==None:
            if 'ATMAX' in self.raw and 'ATMIN' in self.raw:
                self.raw['AT'] = (self.raw['ATMIN']+self.raw['ATMAX'])/2 

        for dtype in ['AT','P','SD']:
            if dtype in self.raw and self.raw[dtype] != None:
                freq = int(1/self.raw[dtype].estimate_frequency())
                if freq < 1 or freq > 48:
                    print("")
                    print("Estimated frequency of data series " + dtype + \
                        " is outside common\n range (1-48 points per day).")
                    print("Aborting calculation of daily average. ")
                    print("")
                    return
                
                # Get daily average 
                if freq >= 1:
                    self.daily[dtype] = self.raw[dtype].calc_daily_avg(mindata=freq)
                else:
                    print("Data frequency is less than 1 point per day...!")
                    self.daily[dtype] = self.raw[dtype].calc_daily_avg(mindata=1)
        
        return True



    def patch_AT(self, max_missing = 6):
        """
        calculate a patched daily air temperature time series.
        
        If a month is missing less than 6 days, these days are added using the
        monthly mean temperature.
        
        If 6 or more days are missing, raise an error!
        
        """
        
        newts = copy.copy(self.daily['AT'])
        
        if not newts.dt_type in [dt.datetime, dt.date]:
            try:
                #time = mpl.dates.num2date(time, tz=self.tzinfo)
                raise "Problem with non-datetime times in ClimateData.patch_AT"
            except:
                raise "Bad input to calcMAAT"
        
        # remove any masked datapoints from time series
        newts.drop_masked()
        
        # create time series of monthly averages
        monthlyts = newts.calc_monthly_avg()
        
        # find any missing days
        dates = newts.ordinals
        date_range = np.arange(newts.ordinals[0],newts.ordinals[-1])
        #dates = mpl.dates.date2num(newts.times)
        #date_range = np.arange(*mpl.dates.date2num([newts.times[0],newts.times[-1]]))
        (diff_arr, id1) = difference(date_range,dates)
        
        # convert ordinals of missing days to dates.
        diff_arr = mpl.dates.num2date(diff_arr)
        diff_arr = np.array([d.date() for d in diff_arr])
            
        # create dictionary with year as keys, fill it...
        missing_dict = dict()

        for d in diff_arr:
            year  = d.year
            month = d.month
            day   = d.day
            
            # contains dictionary with month-numbers as keys
            if year not in missing_dict:
                missing_dict[year] = dict()
            
            # the value is a list with any missing days in that month
            if month not in missing_dict[year]:
                missing_dict[year][month] = [d]
            else:
                missing_dict[year][month].append(d)

        # fill time series gaps when less than max_missing days are missing in a month
        # otherwise raise an error
        for y in missing_dict:
            for m in missing_dict[y]:
                if len(missing_dict[y][m]) <= max_missing:
                    month_id = mpl.mlab.find(monthlyts.times == dt.date(y,m,1))
                    try: 
                        fill_val = float(monthlyts.data[month_id,:])
                    except:
                        raise "Month does not exist in monthly timeseries"
                    
                    newts.insert_timesteps(missing_dict[y][m],fill_value=fill_val, sort=False)
                else:
                    print("There are %i missing days in the month %i-%i. Nothing added to timeseries!" % (len(missing_dict[y][m]),y,m))
        
        # sort and store data
        newts.sort_chronologically()
        self.patched['AT'] = newts
        

    def calc_AT_parameters(self, use_patched=True, use_raw=False, max_missing = None):
        """
        Returns a record array with the fields:
        
        Year:        The year for which the parameters are calculated
        plot_date:   mid-point of year
        MAAT:        Mean Annual Air Temperature
        FDD:         Freezing Degree Days
        TDD:         Thawing Degree Days
        ndays:       Number of days available for that year
        """
     
        if use_raw:
            time = self.raw['AT'].times
            data = self.raw['AT'].data
        elif use_patched:
            if self.patched['AT'] == None or max_missing != None:
                if max_missing == None:
                    self.patch_AT()
                else:
                    self.patch_AT(max_missing = max_missing)
                
            time = self.patched['AT'].times
            data = self.patched['AT'].data
        else:    
            time = self.daily['AT'].times
            data = self.daily['AT'].data
        
        if not self.daily['AT'].dt_type in [dt.datetime, dt.date]:
            try:
                time = mpl.dates.num2date(time, tz=self.tzinfo)
            except:
                raise "Bad input to calcMAAT"

        years = {}
        for id,t in enumerate(time):
            if t.year in years:
                years[t.year] = np.ma.vstack((years[t.year],data[id]))
            else: 
                years[t.year] = data[id]
        
        dtype = [('Year', int), ('Date', dt.datetime), ('MAAT', float), 
                 ('FDD', float),('TDD', float), ('ndays', int)]
        
        deltat = np.array([np.float((t.replace(month=12,day=31)-
                                     t.replace(month=1,day=1)).days+1) 
                                     for t in time])/2
        
        params = []

        for id, entry in enumerate(sorted(years.items())):
            params.append( ( \
                int(entry[0]), \
                mpl.dates.date2num(dt.date(entry[0]-1,12,31))+deltat[id], \
                entry[1].mean(), \
                entry[1][entry[1]<0.].sum(), \
                entry[1][entry[1]>0.].sum(), \
                entry[1].size ) ) 
            if params[-1][2] <= -50 or params[-1][2] >= 50:
                raise ValueError ('Problem in calculating the MAAT!')
        
        return np.array(params, dtype=dtype)


    def calc_P_parameters(self, dataSeries='scaled', dataKey='P',
                           plot=False,
                           fontsize=12,
                           figsize=(15,12),
                           style='-k',
                           lim=None,
                           ax1 = None,
                           ax2 = None):
        
        """
        Returns a record array with the fields:
        
        Year:          The year for which the parameters are calculated
        SAP:           Summed Annual Precipitation
        MAP:           Mean Annual Precipitation
        npdays:        Number of precipitation days (P>0)
        ndays:         Number of days available for that year
        days_in_year:  Number of days in year (nominal... 365 or 366)
        """

        # Get data from specified time series
        try:
            # Treat as MyTimeseries
            time = dataSeries.limit(lim).times
            data = dataSeries.limit(lim).data
            dt_type = dataSeries.dt_type
            is_ts = True
        except:
            # If it fails, it should be an entry into self
            if dataSeries in self.__dict__:
                time = self.__dict__[dataSeries][dataKey].limit(lim).times
                data = self.__dict__[dataSeries][dataKey].limit(lim).data
                dt_type = self.__dict__[dataSeries][dataKey].dt_type
                is_ts = False    
            else:
                raise KeyError ('{0} dataseries not found'.format(dataSeries))

        if not dt_type in [dt.datetime, dt.date]:
            try:
                time = mpl.dates.num2date(time, tz=self.tzinfo)
            except:
                raise "Bad input to calc_P_parameters"

        # Make dictionary with years as keys and an array of the daily
        # AT data for that year as value
        years = {}
        for id,t in enumerate(time):
            if t.year in years:
                years[t.year] = np.append(years[t.year],data[id])
            else: 
                years[t.year] = np.array(data[id])
        
        # Construct a rec-array with  Year, MAAT, FDD, TDD, ndays, days_in_year
        
        dtype = [('Year', int), ('SAP', float), 
                 ('MAP', float),('npdays', float),
                 ('ndays', int), ('days_in_year', int)]
        params = []
        for id, entry in enumerate(sorted(years.items())):
            params.append( ( \
                int(entry[0]), \
                entry[1][entry[1]>0.].sum(), \
                entry[1][entry[1]>0.].sum()/entry[1].size, \
                np.where(entry[1]>0.,np.ones_like(entry[1]),
                         np.zeros_like(entry[1])).sum(),
                entry[1].size,                
                np.diff(mpl.dates.date2num([dt.date(entry[0],1,1),dt.date(entry[0]+1,1,1)]) ) ) ) 
        
        statistics = np.array(params, dtype=dtype)
        
        
        #pdb.set_trace()
        
        
        if is_ts == False:
            self.__dict__[dataSeries]['Pstats'] = statistics
        
        # Plot if specified
        if plot:
            fig = self.plot_P_parameters( 
                    Pstats=statistics,
                    fontsize=fontsize,
                    figsize=figsize,
                    style=style,
                    lim=lim,
                    ax1=ax1,
                    ax2=ax1)
            return statistics,fig
        else:
            return statistics



    def plot_P_parameters(self, dataSeries='scaled',
                           dataKey='P',
                           Pstats=None,
                           fontsize=12,
                           figsize=(15,12),
                           style='-k',
                           lim=None,
                           ax1=None,
                           ax2=None,
                           recalc=False):
        
        
        if Pstats != None:
            stats2 = copy.deepcopy(Pstats)
        else:
            if recalc:
                stats2 = self.calc_P_parameters(dataSeries,dataKey,
                                                 lim=lim,
                                                 plot=False)
            else:
                try:
                    stats2 = copy.deepcopy(self.__dict__[dataSeries]['Pstats'])
                except:
                    stats2 = self.calc_P_parameters(dataSeries,dataKey,
                                                     lim=lim,
                                                     plot=False)
        
        figBG   = 'w'        # the plt.figure background color
        axesBG  = '#f6f6f6'  # the axies background color
        textsize = 14        # size for axes text   
    
        if ax1 == None and ax2 == None:
            fig = plt.figure(figsize=figsize,facecolor=figBG)
        else:
            try:
                fig = ax1.figure
            except:
                fig = ax2.figure
        
        if ax1 == None or ax2 == None:
            fig.clf()
            left, width = 0.1, 0.8
            rect1 = [left, 0.55, width, 0.35]     #left, bottom, width, height
            rect2 = [left, 0.1 , width, 0.35]     #left, bottom, width, height
            ax1 = plt.axes(rect1, axisbg=axesBG)
            ax2 = plt.axes(rect2, axisbg=axesBG, sharex=ax1)

            
        stats2 = np.append(stats2, stats2[-1])
        plotdates = np.array([dt.date(s,1,1) for s in stats2['Year']])
        plotdates[-1] = plotdates[-1].replace(month=12,day=31) 
        
        # plot MAAT        
        leg = True
        
        ax1.plot_date(plotdates,stats2['SAP'],style,drawstyle='steps-post',label='SAP')
        ax2.plot_date(plotdates,stats2['MAP'],style,drawstyle='steps-post',label='MAP')
            
        if leg:
            l = plt.gca().legend(loc = 'best')
            for t in l.get_texts():
                t.set_fontsize(textsize-2)

        ax1.set_ylabel('SAP [mm]', size=textsize)
        ax2.set_ylabel('MAP [mm]', size=textsize)
        ax1.set_xlabel('Time', size=textsize)
        
        for tick in ax1.xaxis.get_major_ticks():
            tick.label1.set_fontsize(fontsize)
        for tick in ax1.yaxis.get_major_ticks():
            tick.label1.set_fontsize(fontsize)

        for tick in ax2.xaxis.get_major_ticks():
            tick.label1.set_fontsize(fontsize)
        for tick in ax2.yaxis.get_major_ticks():
            tick.label1.set_fontsize(fontsize)

        plt.show()
        
        ax1.hold(True)
        ax2.hold(True)
        return fig

        
    def export(self, fname, parameters=['AT'], delim=', ', type='daily'):
        """
        This method should take a filename as input and export the Climate data
        from the borehole as a textfile.
        """
        
        if not type in ['daily', 'raw']:
            raise ValueError('type argument must be either "daily" or "raw".') 

        with open(fname, 'w') as fid:
            # Write header
            fid.write('Climate Station:\t{0:s}\n'.format(self.name))
            fid.write('Latitude:\t{0:10.6f}\n'.format(self.latitude))
            fid.write('Longitude:\t{0:10.6f}\n'.format(self.longitude))        
            fid.write('Description:\t{0:s}\n'.format(self.description))
            
            titles = '      Time'
            for p in parameters:
                titles += '%s%7s' % (delim, p)
            fid.write(titles + '\n')
            fid.write('-'*len(titles) + '\n')
        
            # Write data
            if type == 'daily':
                daily_flag = True
                
                if self.daily == None:
                    self.calc_daily_avg()
                    daily_flag = False   
                
               # for p in parameters:
               #     mask = np.ma.getmaskarray(self.daily[p].data)
               #     for rid, t in enumerate(self.daily[p].times):
               #         outstr = '%s' % t
               #         for cid, e in enumerate(self.daily[p].data[rid, :]):
               #             if mask[rid, cid]:
               #                 outstr += delim + '    ---'
               #             else:
               #                 outstr += '%s%7.2f' % (delim, e)
               #         outstr += '\n'
               #         fid.write(outstr)
            for p in parameters:
                mask = np.ma.getmaskarray(self.__dict__[type][p].data)
                for rid, t in enumerate(self.__dict__[type][p].times):
                    outstr = '%s' % t
                    for cid, e in enumerate(self.__dict__[type][p].data[rid, :]):
                        if mask[rid, cid]:
                            outstr += delim + '    ---'
                        else:
                            outstr += '%s%7.2f' % (delim, e)
                    outstr += '\n'
                    fid.write(outstr)
    


    #def dictStatCum(self,month,day, dlim = 0.01):
    #    ydict = self.split_years(month,day)
    #    
    #    names = ddict[ddict.keys()[0]].dtype.names
    #
    #
    #    if type(dlim) == list and len(dlim) < len(names)-2:
    #        dlim = ones(1,len(names)-2)*dlim
    #    if type(dlim) != list:
    #        try:
    #            dlim = np.ones((1,len(names)-2))*float(dlim)
    #        except:
    #            dlim = np.ones((1,len(names)-2))*0.01
    #            
    #    stats = {}
    #    stats['keys'] = ['mean','STD','max','min','ndays','tdprod']
    #    
    #    for y in ddict.keys():
    #        stats[y] = {}
    #        for id,n in enumerate(names[2:]):
    #            try:
    #                ids = mpl.mlab.find(ddict[y][n] >= dlim[id])
    #                sdate = ddict[y]['time'][ids[0]]
    #                edate = ddict[y]['time'][ids[-1]]
    #                recs = np.logical_and(ddict[y]['time']>=sdate, \
    #                            ddict[y]['time']<=edate)
    #                myrecs = ddict[y][recs]
    #                            
    #                stats[y][n] = dict( \
    #                        mean = np.mean(myrecs[n]), \
    #                        STD = np.std(myrecs[n]), \
    #                        max = np.max(myrecs[n]), \
    #                        min = np.min(myrecs[n]), \
    #                        ndays = len(ids), \
    #                        tdprod = np.sum(myrecs[n]))   
    #            except:
    #                stats[y][n] = None
    #    
    #    
    #    totstats = dict(zip(names[2:], [{} for k in xrange(len(names)-2)]))
    #    totstats['keys'] = ['max','maxAVG','maxSTD','ndaysAVG','ndSTD','meanAVG','meanSTD','tdprodAVG','tdpSTD']
    #    
    #    for n in names[2:]:
    #        totstats[n] = dict(zip(totstats['keys'], [[] for k in xrange(len(totstats['keys']))]))
    #        for y in ddict.keys():
    #            if stats[y][n] != None:
    #                totstats[n]['maxAVG'].append(stats[y][n]['max'])
    #                totstats[n]['meanAVG'].append(stats[y][n]['mean'])
    #                totstats[n]['ndaysAVG'].append(stats[y][n]['ndays'])
    #                totstats[n]['tdprodAVG'].append(stats[y][n]['tdprod'])
    #        
    #        try:
    #            totstats[n]['max'] = np.max(totstats[n]['maxAVG'])
    #        except:
    #            totstats[n]['max'] = None
    #        try:    
    #            totstats[n]['maxSTD'] = np.std(totstats[n]['maxAVG'])
    #        except:
    #            totstats[n]['maxSTD'] = None
    #        try:
    #            totstats[n]['maxAVG'] = np.mean(totstats[n]['maxAVG'])
    #        except:
    #            totstats[n]['maxAVG'] = None
    #        try:
    #            totstats[n]['meanSTD'] = np.std(totstats[n]['meanAVG'])        
    #        except:
    #            totstats[n]['meanSTD'] = None
    #        try:
    #            totstats[n]['meanAVG'] = np.mean(totstats[n]['meanAVG'])
    #        except:
    #            totstats[n]['meanAVG'] = None
    #        try:
    #            totstats[n]['ndSTD'] = np.std(totstats[n]['ndaysAVG'])        
    #        except:
    #            totstats[n]['ndSTD'] = None
    #        try:
    #            totstats[n]['ndaysAVG'] = np.mean(totstats[n]['ndaysAVG'])
    #        except:
    #            totstats[n]['ndaysAVG'] = None
    #        try:
    #            totstats[n]['tdpSTD'] = np.std(totstats[n]['tdprodAVG'])
    #        except:
    #            totstats[n]['tdpSTD'] = None
    #        try:
    #            totstats[n]['tdprodAVG'] = np.mean(totstats[n]['tdprodAVG'])
    #        except:
    #            totstats[n]['tdprodAVG'] = None
    #        
    #        
    #    return stats, totstats 











def CDload(fname, sep=',', paramsd=None, hlines=None):
    """
    CDload(fname, sep=',', paramsd=None, hlines=None)
    
    Function to load climate data file.
    
    Recognized key words (case insensitive in file as they are converted 
    to lower-case during file reading and parsing in get_property_info()):
    
    separator:      Defines the column separator
    decimal:        Defines the decimal separator
    flagcol:        Defines column number for flags for data masking
                        can be column number (0-based) or 'Last'
    format:         Data logger type, determines how file is read
    tzinfo:         Timezone info (default is time zone unaware!)
                        Use '-0300' for UTC-3 (Greenland) etc.    
    file_as_tz:     Convert data to this time zone before storing.
                        e.g. if data logger time zone was wrongly adjusted, and
                        data need to be corrected, or if data file is given in
                        e.g. UTC and you want data stored in UTC-3.
                        NOT YET FUNCTIONAL!!!
    date_only:      True/False, if times are specified, flag wether to store
                        only dates. 
    multiplier:     A multiplier for the data set (all data columns will be
                    multiplied by this value)
                    FUTURE: Should allow a list of multipliers!
    missing:        list of values to mask, default is '','-','--','---','None'
    T_unit:         Temperature unit: 'F' or 'C'
     
    
    These keywords are parsed and/or used in this function. 'separator','decimal'
    and 'flagcol' are removed from the parameter dictionary, everything else is
    returned in the paramsd output variable.
    
    
    formats recognized:
    
    1               datetime, data, flag column (optional)
    2               datetime, date, time, data, flag column (optional)
    3               date, data, quality flag, flag column (optional)
                      "Quality flag" is not read!
    4               DMI Greenland data ASCII format, flag column (optional)
                        1-5 F5.0 Station no.
                        6-9 F4.0 Year
                      10-11 F2.0 Month
                      12-13 F2.0 Day
                      14-15 F2.0 Hour DNT or UTC (since 2001 or stations starting with 04 or 06)
                      16-20 F5.0
                    Presently this format does not handle special problems in
                    DMI data like change from Danish Normal Time to UTC
                    and the fact that for some periods Tmax is registered on the
                    day before the measurement was taken!
    5               DMI monthly T average data, 3 columns: Year, Month, Data
    6               datetime, date, time, data, quality flag, flag column (optional)
                       "Quality flag" is not read!
    7               DMI ASCII format:
                      Station No, year, month, day, hour, data value
    8               Yearly data series: Year, data
    
    Returns:

    data:           masked array
    times:          array of datetime objects, parsed using the info from tzinfo
    flags:          array of strings from flag column
    paramsd:        dictionary of key:value pairs, some of them parsed by this function
    """
    
    def change_decimal_sep(data_array, decsep):
        """
        Removes any periods (.) in strings in array then replace original
        decimal points (decsep) with periods.
        """
        for index, arrstr in np.ndenumerate(data_array):
            # Remove any periods in string
            # Replace original decimal separators with periods
            data_array[index[0],index[1]] = arrstr.replace('.','').replace(decsep,'.')
        
        return data_array

    def convert_to_float(data_array, mask_value=-9999., missing = ['','-','--','---','None']):
        """
        Convert an array of strings to a masked array of floats
        Treat empty strings, '-', '--', '---' and 'None' as missing values
        """
        
        data = np.ma.zeros(data_array.shape)
        mask = np.ma.getmaskarray(data)
        
        for index, arrstr in np.ndenumerate(data_array):
            if arrstr not in missing:
                try:
                    data[index[0],index[1]] = float(arrstr)
                except:
                    data[index[0],index[1]] = mask_value
                    mask[index[0],index[1]] = True                    
            else:
                data[index[0],index[1]] = mask_value
                mask[index[0],index[1]] = True
        
        data.mask = mask
        
        return data
        
    
    if (paramsd == None) and (hlines == None):
        paramsd, comments, hlines = get_property_info(fname=fname,sep=sep)    

    # if paramsd has an entry that defines the list separator...
    sep = paramsd.pop('separator',sep)
    if sep == '\\t':
        sep = '\t'
        
    # if paramsd has an entry that defines the decimal separator...
    dec_sep = paramsd.pop('decimal','.')    
    
    file = ra.dlmread(fname,sep = sep, hlines = hlines, dt = str)
    
    flags = None
    # get column number of flag column, if it is given in parameters
    flagcol = paramsd.pop('flagcol',None)
    if flagcol != None:
        if flagcol.lower() == 'None':
            flagcol = None
        elif flagcol.lower() in ['last']:
            flagcol = file.ncol-1
        else:
            try:
                flagcol = int(flagcol)
                if (flagcol > file.ncol-1) or (flagcol < 0):
                    # if out of bounds... assume no flags
                    flagcol = None
            except:
                flagcol = None
    

    if not 'tzinfo' in paramsd:
        paramsd['tzinfo'] = '' #'UTC'
    
    
    datadim = 1
         
    format = paramsd.pop('format', None)
    
    if format == '1':
        times = copy.copy(file.data[:,0])
        data = copy.copy(file.data[:,1:1+datadim])
        datetype = False                    # True = only dates available, False = dates and times
    elif format == '2' or format == '6':
        times = copy.copy(file.data[:,0])
        data = copy.copy(file.data[:,3:3+datadim])        
        datetype = False                    # True = only dates available, False = dates and times
    elif format == '3':
        times = copy.copy(file.data[:,0])
        data = copy.copy(file.data[:,1:1+datadim])
        datetype = True                     # True = only dates available, False = dates and times       
    elif format == '4':
        # DMI Tmax, Tmin and Precip data format
        times = []
        
        # make sure we have the right shape
        if len(file.data.shape) == 1:
            data = file.data.reshape([file.data.shape[0],1])
        else:
            data = file.data

        for id,s in enumerate(file.data[:,0]):
            times.append('{0}-{1}-{2} {3:0>2}:00'.format(s[5:9],s[9:11].lstrip(),
                                                        s[11:13].lstrip(),
                                                        s[13:15].lstrip()))
            data[id,0] = s[15:20].lstrip().rstrip()
            datetype = False                     # True = only dates available, False = dates and times
    
    elif format == '5':
        # DMI monthly T average data
        times = []
        
        # make sure we have the right shape
        if file.data.shape[1] != 3:
            raise IOError('Wrong file format!')
        
        data = file.data[:,2].reshape(-1,1)
        
        for id,d in enumerate(file.data):
            times.append('{0}-{1:0>2}-01'.format(d[0],d[1]))
            #data[id,0] = d[3].lstrip().rstrip()
            datetype = True                     # True = only dates available, False = dates and times
            
    # format == '6' see above
    
    elif format == '7':
        times = []
        for y,m,d,h in file.data[:,1:5]:
            times.append('{0:0>4}-{1:0>2}-{2:0>2} {3:0>2}:00'.format(y,m,d,h))
        
        data = copy.copy(file.data[:,5:5+datadim])
        datetype = False                     # True = only dates available, False = dates and times
        
    elif format == '8':
        # Yearly T average data
        times = []
        
        # make sure we have the right shape
        if file.data.shape[1] != 2:
            raise IOError('Wrong file format!')
        
        data = file.data[:,1].reshape(-1,1)
        
        for id,d in enumerate(file.data):
            times.append('{0}-01-01'.format(d[0],d[1]))
            #data[id,0] = d[3].lstrip().rstrip()
            datetype = True                     # True = only dates available, False = dates and times
        
            
    else:
        raise ValueError("Unknown or missing format specifier!")
    

    if datetype :
        # the data set contains only dates
        try:
            times = np.array([dateutil.parser.parse(t,yearfirst=True,dayfirst=True).date() for t in times])
        except:
            print("!!!! Problems parsing times from input file... starting debugger.")
            pdb.set_trace()    
    else:
        # the dataset contains dates and times
        try:
            times = np.array([dateutil.parser.parse(t+' '+paramsd['tzinfo'],yearfirst=True,dayfirst=True) for t in times])
        except:
            print("!!!! Problems parsing times from input file... starting debugger.")
            pdb.set_trace()    
    
    if 'file_as_tz' in paramsd:
        times = np.array([t.astimezone(paramsd['file_as_tz']) for t in times])
    
    if 'date_only' in paramsd and paramsd['date_only'].lower() == 'true':
        times = np.array([t.date() for t in times])
        datetype = True
    
    if dec_sep != '.':
        # the file uses wrong decimal point format, do the conversion
        data = change_decimal_sep(data,dec_sep)
    
    _miss = paramsd.pop('missing', None)
    if _miss == None:
        _miss = ['','-','--','---','None','?','??','???']
        _miss_num = []
    else:
        _miss = _miss.split(',')
        _miss_num = []
        for m in _miss:
            try:
                _miss_num.append(float(m))
            except:
                pass
    
    
    try:    
        data  = np.ma.array(data, dtype=float)
        for miss in _miss_num:
            data = np.ma.masked_where(data==miss, data, copy=False)
    except:
        data  = convert_to_float(data,missing=_miss)
    
    multiplier = float(paramsd.pop('multiplier','1.'))    
    

    if datadim == 1:
        data[:,0] = data[:,0]*multiplier
    else:
        print("Unexpected data dimension in ClimateData.CDload!")
        print("Check file format and reader function")
        print("will use multiplier on all data columns!")
        data = data*multiplier
        pdb.set_trace()
    
    if flagcol != None and file.ncol>=flagcol+1:
        flags = file.data[:,flagcol]
         
    return (data,times,flags,paramsd)





def get_property_info(fname=None, text=None, sep=',', rstrip=None):
    """
    This function reads the first 1500 characters of a file and parses the info
    in "key: value" pairs.
    
    Lines beginning with # are skipped
    Blank lines are skipped
    
    The file data section is interpreted to begin when:
    '#--' the following line will be the first data line
    If there is no key:value pair separated by ':' 
    If the key contains a separator (sep) (may occur when times are given in second column)
    If the trimmed key contains a space (may occur when times/dates are given in first column)
    
    Any characters found after a '#' following a key:value pair are treated as comments
    
    The function returns:
    items:     Dictionary of key:value pairs
    comments:  Dictionary of key:comment pairs
    nl:        line number of first data line
    
    """
    
    if not fname == None:
        fh = open(fname,'r')
        lines = fh.readlines(1500)
        fh.close()
    elif not text == None:
        if type(text) not in [list,tuple]:
            lines = [text]
        else:
            lines = text
    else:
        raise "Nothing to parse!"
    
    items = dict()
    comments = dict()
    nl = 0
    
    for id,l in enumerate(lines):
        l = l.rstrip().rstrip(rstrip) # rstrip for spaces and any other specified character
        
        if l.startswith('#--'):
            # here we found the indicator for beginning of data section
            nl = id+1
            break
        
        if (l.startswith('#')) or (l.startswith('%')) or (len(l) == 0):
            # This is an empty or a comment line, which we will skip
            continue
        
        # split line in key and value
        tmp = l.split(':',1)
        if not len(tmp) == 2:
            # There is no key value pair, thus we have reached beginning of data section
            nl = id
            break
        
        key = tmp[0].lstrip().rstrip().lower()
        value = tmp[1].lstrip().rstrip()
        
        if key.find(sep) != -1:
            # token before a possible ':' contains a sep
            # thus it must be a data line
            nl = id
            break
        elif key.find(' ') != -1:
            # token before a possible ':' contains a ' '
            # not counting leading and trailing spaces
            # thus it must be a data line
            nl = id
            break
        else:
            # We have a key-value pair!
            
            # parse any comment
            tmp =  value.split('#',1)
            value = tmp[0].rstrip().lstrip()
            if len(tmp) == 2:
                comment = tmp[1].rstrip().lstrip()
            else:
                comment = ''
            items[key] = value
            comments[key] = comment        
    return items,comments,nl






def read_climate_data(path=basepath,walk=True, clim=None, calc_daily=True, \
                      names=None, auto_process=True, post_process=True, \
                      read_pkl=True):
    """
    Call signature:
    
    import ClimateData as cdata
    names = ['ILU520', 'ILU545', 'KAN16504', 'ILU04221', 'NUK808', 'SIS515', 'KAN812', 'SIS04230', 'ILU34216', 'ILU04216', 'KAN04231']
    clim = cdata.read_climate_data('D:\Thomas\Artek\Data\ClimateData', walk=True, clim=None, names=names)
    
    another example: 
    cdata = read_climate_data(calc_daily=True,names=['KAN04231'],read_pkl=False)
    
    walk    Switch to toggle recursion of sub directories
    clim    pass dictionary of climate stations to add the data read to existing dictionary
    """
    
    #..........................................................................
    def process_files(clim,dr,flst):
        
        entries_to_skip = list()
        # First create station entries for any .sta file in the directory
        # and find entries in the list that should be removed
        
        if clim['read_pkl'] in [True, 'only', 'Only']:
            read_pkl = True
        else:
            read_pkl = False
        
        if clim['read_pkl'] in [True, False]:
            read_ascii = True
        else:
            read_ascii = False

        for id,f in enumerate(flst):
            f2 = f.lower()
            # Is it a pickled file? ...
            if f2.endswith('.pkl') and read_pkl:
                try:
                    with open(os.path.join(dr, f),'rb') as pklfh:
                        tmpstation = pickle.load(pklfh)
                except:
                    print("Could not read file: {0}".format(os.path.join(dr, f)))
                    tmpstation = None
                    
                if  type(tmpstation) == climate_station:
                    if clim['names'] == None or tmpstation.name in clim['names']:
                        clim[tmpstation.name] = tmpstation
                        clim[tmpstation.name].path = dr
                        clim[tmpstation.name].fname = f
                        clim['pkl_names'].append(tmpstation.name)
                        print('\n**************************************************************\n')
                        print('Climate station ' + tmpstation.name + ' created from pickled file!')
                        entries_to_skip.append(id)
            elif f2.endswith('post_processing.py'):
                clim['postproc_files'].append(os.path.join(dr, f))
                entries_to_skip.append(id)

        for id, f in enumerate(flst):
            f2 = f.lower()
            # ...or is it a climate station definition file?
            if f2.endswith('.sta') and read_ascii:
                tmpstation = climate_station(fname = os.path.join(dr,f))
                if (clim['names'] == None or tmpstation.name in clim['names']) and \
                    tmpstation.name not in clim:
                    
                    clim[tmpstation.name] = tmpstation
                    print('\n**************************************************************\n')
                    print('Climate station ' + tmpstation.name + ' created')
                entries_to_skip.append(id)
                
            # ...or is it a direvtory with certain names
            elif os.path.isdir(os.path.join(dr,f)) and \
                    ((f2.find('corrupt') >= 0) or \
                    (f2.find('raw') >= 0) or \
                    (f2.find('calib') >= 0) or \
                    (f2.find('exclude') >= 0) or \
                    (f2.find('info') >= 0)):
                # If f is a directory, and the name contains either
                # 'corrupt', 'raw' or 'calibration'
                # add the index to the list of entries to remove from the
                # tree to walk.            
                entries_to_skip.append(id)
                
            # ... or is it a corrupt file?
            elif f2.find('corrupt') >= 0 or \
                 f2.find('exclude') >= 0:
                # add the index to the list of entries to remove from the
                # tree to walk.
                entries_to_skip.append(id)
        
        if not entries_to_skip == []:
            # Remove the entries from the file list:
            for id in reversed(sorted(entries_to_skip)):
                del flst[id]
        
        # Then load data into the climate station
        for f in flst:
            f2 = f.lower()
            if (f2.endswith(('.csv','.txt','.dat'))) and read_ascii:
                fullf = os.path.join(dr,f)
                if os.path.islink(fullf): continue # don't count linked files
                
                paramsd, comments, hlines = get_property_info(fullf)
                if 'station' in paramsd:
                    if (clim['names'] == None or \
                        paramsd['station'] in clim['names']) and \
                        paramsd['station'] not in clim['pkl_names']:
                        if paramsd['station'] in list(clim.keys()):
                            stat = clim[paramsd['station']].add_data(fullf, paramsd=paramsd, hlines=hlines)
                            if stat:
                                print(f + " added to station " + paramsd['station'])
                            else:
                                print("NB!" +  f + " could not be added to station " + paramsd['station'])
                                pdb.set_trace()
                        else:
                            print("NB!" + f + " tries to reference non-exsisting " + \
                                  "station " + paramsd['station'])
    #..........................................................................
    def _post_process(clim, fullfile):
        """
        Method to initiate post prossessing steps defined in a file
        located somewhere in the same directory tree which was walked.
                
        clim is a dictionary of climate station instances, with the climate
           station names as keys.
        fullfile is the path and file name of the post processing file. This
           may be a list of path elements, which will then be joined using 
           os.path.join.
        
        The file post_processing.py should have a method defined as:
        
        def post_process(station_dict):
            if 'SomeStationName' in station_dict:
                # apply some processing to the station...
                pass
        """ 
        
        if type(fullfile) in [list, tuple]:
            fullfile = os.path.join(*fullfile)
        if os.path.isfile(fullfile):
            exec(compile(open(fullfile, "rb").read(), fullfile, 'exec'), {}, locals())
        else:
            print("Specified post processing file not found! Nothing done ...!")
            print("File name: {0}".format(fullfile))
            return
        
        locals()['post_process'](clim)
        
        print("Applied processing steps from {0}".format(fullfile))
        
        return clim
    #..........................................................................
    
    print("read_climate_data is called with the following parameters:")
    print("Path: {0}".format(path))
    print("walk: {0}".format(walk))
    if clim == None:
        print("Station dictionary (clim): Available")
    else:
        print("Station dictionary (clim): None")
    print("Calculate daily average: {0}".format(calc_daily))
    print("Read only stations with names: {0}".format(names))
    print("Auto process: {0}".format(auto_process))
    print("Post process: {0}".format(post_process))
    print("Read Pickle files: {0}".format(read_pkl))    
    
    print(" ")
    input("Beware, postprocessing is not properly handled! (press enter to proceed)")
                      
    if type(path) in [list, tuple]:
        # Concatenate path elements
        path = os.path.join(*path)
    
    if type(clim) != dict:    
        clim = dict()

    clim['names'] = names
    clim['read_pkl'] = read_pkl
    clim['pkl_names'] = []
    clim['postproc_files'] = []
        
    if not walk:
        # Process only files in the current directory (path)
        flst = os.listdir(path)
        process_files(clim,path,flst)
    else:
        # Process all files in the directory tree
        os.path.walk(path,process_files,clim)

    pkl_names = clim['pkl_names']
    postproc_files = clim['postproc_files']
    del clim['names']
    del clim['read_pkl']
    del clim['pkl_names']
    del clim['postproc_files']
    
    print('\n**************************************************************\n')
        
    if auto_process:
        changes = False
        for cs in clim:
            if cs not in pkl_names:
                clim[cs].auto_process()
                changes = True
        if changes:
            print('\n**************************************************************\n')    
    
    if calc_daily:
        for cs in clim:
            if cs not in pkl_names:
                print('Calculating daily timeseries for station ' + cs)
                clim[cs].calc_daily_avg()
    
    # pdb.set_trace()
            
    if post_process:
        for f in postproc_files:
            clim = _post_process(clim,f)
        
    return clim


def pickle_all(station_dict, path = None):
    if path == None:
        for name,cs in list(station_dict.items()):
            cs.pickle()
            print("Pickled climate station {0}".format(name))
    else:
        pdb.set_trace()
        for name,cs in list(station_dict.items()):
            if cs.fname == None:
                fname = cs.name
            else:
                fname = cs.fname
                ext,fname = fname[::-1].split('.')
                fname = fname[::-1]
            
            if type(path) in [list, tuple]:
                fullfile = copy.copy(path)
                fullfile.extend([fname])
                fullfile = os.path.join(*fullfile)
            else: 
                fullfile = os.path.join(path,fname)
            cs.pickle(fullfile_noext=fullfile)
            
            print("Pickled climate station {0} to {1}.pkl".format(name,fullfile))    



def plot_data_coverage(clim, dtype = 'AT', datetype='unmasked', **kwargs):
    if 'axes' in kwargs:
        ax = kwargs.pop('axes')
    elif 'Axes' in kwargs:
        ax = kwargs.pop('Axes')
    elif 'ax' in kwargs:
        ax = kwargs.pop('ax')
    else:
        fh = pylab.figure()
        ax = pylab.axes()
    
    if type(clim) != dict:
        clim2 = dict()
        clim2[clim.name]=clim    
        clim = clim2
        
    n = 0
    csnames = sorted(clim.keys())
    for cs in csnames:
        n -= 1
        dates = clim[cs].get_dates(dtype,datetype=datetype)
        ax.plot_date(mpl.dates.date2num(dates),np.zeros_like(dates)+n,'.')
    
    lim = ax.get_ylim()
    ax.set_ylim([lim[0]-1,lim[1]+1])
    
    plt.setp(ax, 'yticks', np.arange(n,0))
    plt.setp(ax, 'yticklabels', csnames[::-1])
    
    plt.show()






def n_permutations(arr1, n, arr2=None):
    """
    Produce n random and different permutations of the sequence arr1.
    Function will check that n permutations can be made
    
    Input arguments:
    arr1:      Array that is to be permuted
    n:         Number of permutations (including original)
    arr2:      Default=None, optional array of existing permutations of arr1.
               arr2 will be appended to until it contains n permutations
               
    Output arguments:
    arr2:      List of lists, each list in arr2 will be a permutation of arr1 
    """    
    
    if arr2 == None:
        # start with original array
        arr2 = [list(arr1)]
    else:
        if type(arr2) == np.ndarray:
            arr2 = [list(arr2[id,:]) for id in range(arr2.shape[0])]
        elif type(arr2) == list:
            for id,l in enumerate(arr2):
                if type(l) == np.ndarray:
                    arr2[id] = list(l)
        
    if len(arr1)>=100 or factorial(len(arr1))>n:
        # If we are pretty sure there will be enough possible permutations...
        
        # While arr2 does not have n rows add a new row... 
        while len(arr2)<n:
            # permute arr1
            arr1p = list(np.random.permutation(arr1))
            
            # if permuted array is not already in arr2, add it
            if arr1p not in arr2:
                arr2.append(arr1p)
            else:
                # otherwise permute again!
                print("array already in arr2! Permuting again!")
    return arr2            

def factorial(n):
    """
    This function calculates the factorial of a number.
    """
    sum = 1.0
    for m in range(1, int(n)+1):
        sum = float(m)*sum
    return sum


def calcNormalYear(ts,startdate=dt.date(1961,0o1,0o1),enddate=dt.date(1990,12,31)):
    """ calcNormalYear takes a timeseries with one data column and calculates a 
    daily average over the climate normal period of 1961-01-01 to 1990-12-31
    
    The function returns two dictionarys, one has the same format as the input,
    the other has a key for each day in the year, with the item being the 
    averaged data.
    
    This normal year can be used to patch missing data using the
    plotData.patchdata function
    """

    def flatten_list(S, containers=(list, tuple)):
        L = list(S)
        i = 0
        while i < len(L):
            while isinstance(L[i], containers):
                if not L[i]:
                    L.pop(i)
                    i -= 1
                    break
                else:
                    L[i:i + 1] = (L[i])
            i += 1
        return L


    ts1 = ts.limit([startdate, enddate])
    ts1.force_dates()
    
    ydict = ts1.split_years()
    # ydict is now a dictionary of years, where the values are 
    # timeseries instances with dates and data from the specified year
        
    # Create a list of dates in 1988 using the calendar module
    tc = calendar.Calendar()
    td = np.array(flatten_list(tc.yeardatescalendar(1988)))
    # This list contains also a few days from 1987 and 1989
    # Remove these from the array:
    td = td[np.array([d.year == 1988 for d in td])] 
    
    # Now produce tuples of month and day for these dates.
    days = [(d.month,d.day) for d in td]
    
    # Create a dixtionary with month/day tuples as keys and an array of two zeros
    # as value for each key.
    NYd = dict(list(zip(days,np.zeros((len(days),2)))))
    
    # loop over every year in timeseries (as defined by split_years, thus a
    # data year may extend across two calendar years.
    for k in list(ydict.keys()):
        # Now loop over dates in ydict value
        data = np.ma.filled(ydict[k].data, 0.)
        #pdb.set_trace()
        for id,r in enumerate(ydict[k].times):
            # add data value to present dictionary index
            NYd[(r.month,r.day)][0] += data[id,0]
            # increase counter of numbers of data
            NYd[(r.month,r.day)][1] += 1
            if np.isnan(NYd[(r.month,r.day)][0]):
                pdb.set_trace()
    
    # Produce record arrays and sort.
    NYdk=np.array(list(NYd.keys()),dtype=[('m',int),('d',int)])
    NYdv=np.array(list(NYd.values()))
    id = np.argsort(NYdk,order=('m','d'))
    
    # Produce dictionary with averaged data and specification of number of averaged values
    NYs = dict(time=NYdk[id], dat=NYdv[id,0]/NYdv[id,1], ndays=NYdv[id,1])
    
    # Convert dictionary of sums to dictionary of averages.
    for k in list(NYd.keys()):
        NYd[k] = NYd[k][0]/NYd[k][1]
    
    # return the two dictionaries
    return NYd,NYs
            



def plotNormalYear(NY):
    if 'time' in NY:
        tm = [dt.date(1988,t[0],t[1]) for t in NY['time']]
        plt.plot_date(mpl.dates.date2num(tm),NY['dat'],'-k')
    else:
        NYk=np.array(list(NY.keys()),dtype=[('m',int),('d',int)])
        NYv=np.array(list(NY.values()))
        id = np.argsort(NYk,order=('m','d'))
        tm = [dt.date(1988,t[0],t[1]) for t in NYk[id]]
        plt.plot_date(mpl.dates.date2num(tm),NYv[id],'-k')
        

def plotMAAT(data,styles=None,labels='',figsize=(15,6),lim=None):
    """Plots mean annual air temperatures"""
    figBG   = 'w'        # the plt.figure background color
    axesBG  = '#f6f6f6'  # the axies background color
    textsize = 14        # size for axes text   

    figILU = plt.figure(figsize=figsize,facecolor=figBG)
    
    left, width = 0.1, 0.8
    rect1 = [left, 0.2, width, 0.6]     #left, bottom, width, height

    ax = plt.axes(rect1, axisbg=axesBG)
    
    if type(data) not in [list,tuple]:
        data = [data]

    if type(labels) not in [list,tuple]:
        labels = [labels for d in data] 

    if type(styles) not in [list,tuple]:
        styles = [styles for d in data] 

    leg = False
    for id,d in enumerate(data):
        
        if 'gridpoint' in d.__dict__:
            if 'scaled' in d.__dict__ and 'AT' in d.scaled and d.scaled['AT'] != None:
                MAAT = d.scaled['AT'].limit(lim).calc_annual_avg()
            elif 'patched' in d.__dict__ and 'AT' in d.patched and d.patched['AT'] != None:
                MAAT = d.patched['AT'].limit(lim).calc_annual_avg()
            else:
                MAAT = d.raw['AT'].limit(lim).calc_annual_avg()
        elif 'patched' in d.__dict__ and 'AT' in d.patched and d.patched['AT'] != None:
            MAAT = d.patched['AT'].limit(lim).calc_annual_avg()
        elif 'raw' in d.__dict__ and 'AT' in d.raw and d.raw['AT'] != None:
            MAAT = d.raw['AT'].limit(lim).calc_annual_avg()
        else:
            raise TypeError('Dont recognize the type of time series!')
                    
        MAAT = MAAT[mpl.mlab.find(MAAT.data[:,1]>350)]
        
        yr = np.array([d.year for d in MAAT.times])
        dt = MAAT.data[:,0]
        
        myd = np.vstack((yr,dt)).T
        #myd = vstack((myd,np.array([[myd[-1,]+1,0]])))
        myd = np.vstack((myd,myd[-1,:]+np.array([1,0])))
        #pdb.set_trace()
        if styles[0] != None:
            ax.plot(myd[:,0],myd[:,1],styles[id],drawstyle='steps-post',label=labels[id])
        else:
            ax.plot(myd[:,0],myd[:,1],drawstyle='steps-post',label=labels[id])
        if labels[id] != '':
            leg = True
        
    if leg:
        l = plt.gca().legend(loc = 'best')
        for t in l.get_texts():
            t.set_fontsize(textsize-2)

     
    #ax.set_ylim(plt.gca().get_ylim()[::-1])
    ax.set_ylabel('MAAT ($^\circ$C)', size=textsize)
    ax.set_xlabel('Time', size=textsize)
  
    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_fontsize(textsize)
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontsize(textsize)    

    plt.show()
    return




class ILU_stations:
    def __init__(self, cs = None,post_process=True,read_pkl=True):
        if cs == None:
            self.stations = read_climate_data([basepath,'Ilulissat'],\
                            post_process=post_process,read_pkl=read_pkl)
        else:
            self.stations = cs

    def split_years(self, month=1,day=1):
        ts = copy.deepcopy(self.stations['ILU04216'])
        return ts.daily['AT'].split_years(month=month,day=day)
        
    def calc_normal_year(self):
        ts = copy.deepcopy(self.stations['ILU04216'])
        tmp1 = calcNormalYear(ts.daily['AT'])
        tmp2 = ts.daily['AT'].calc_normal_year()
        return tmp1,tmp2

    def permute_timeseries(self):
        start_year = 1950
        end_year = 1959
        month, day = 8,1
        times = mpl.dates.num2date(mpl.dates.drange(dt.date(start_year,1,1),
                                    dt.date(end_year,12,31), dt.timedelta(days=1)))
        
        data = []
        flag = False
        counter = 0
        for t in times:
            if t.month==1 and t.day==1:
                # When passing into new calendar year, clear flag:
                flag = False
                
            if t.month>=month and t.day>=day and not flag:
                # If we pass cut off month, add to counter and reset flag
                counter += 1
                flag = True
            
            data.append(counter)
        
        data = np.array(data).reshape([-1,1])
        
        self.data['P'] = myts(data,times)
        print("plotting original series")
        plt.plot_date(times, data, '-g')
        permutations = n_permutations(self.data['P'].get_year_list()[0:-1],5)
        self.permutations = permutations
        for p in permutations:
            ts = self.permute_timeseries(p)
            print("plotting permuted series")
            plt.plot_date(ts.times, ts.data, '-r')




#class climate_data:
#    def __init__(self, name=None, start_date=None, end_date=None, lat=None, \
#            lon=None, description = None, fname = None):
#
#        self.name = name
#        self.start_date = start_date
#        self.end_date = end_date
#        self.latitude = lat
#        self.longitude = lon
#        self.description = description
#        self.data = dict(AT = None, P = None, SWE = None, SD = None)
#        
#        if fname not in [None,[]]:
#            items,comments,nl = get_property_info(fname=fname)
#        
#            for key,value in items.items():
#                if key in ['start_date','end_date']:
#                    self.__dict__[key] = dateutil.parser.parse(value)
#                elif key in ['latitude','longitude']:
#                    self.__dict__[key] = float(value)
#                else: 
#                    self.__dict__[key] = value
#    
#    
#    def __str__(self):
#        def print_date(date):
#            
#            if type(date) not in [dt.datetime, dt.date]:
#                try:
#                    date = mpl.dates.num2date(date)
#                except:
#                    return "None"
#            else:
#                yr = str(date.year)
#                mm = str(date.month)
#                dd = str(date.day)
#                if len(mm)<2:
#                    mm = '0'+mm
#                if len(dd)<2:
#                    dd = '0'+dd
#                return yr+'-'+mm+'-'+dd
#    
#        series_txt = '\nData:\n'
#
#        for k in self.data:
#            if self.data[k] != None:
#                series_txt += '%s: \tYes\n' % k
#                series_txt += self.data[k].__str__(indent='  ')
#            else:
#                series_txt += '%s: \tNo\n' % k
#                        
#        txt = 'Climate data name:      %s' % self.name + '\n' + \
#              'Latitude:           %f' % self.latitude + '\n' + \
#              'Longitude:          %f' % self.longitude + '\n' + \
#              'Start date:         %s' % print_date(self.start_date) + '\n' + \
#              'End date:           %s' % print_date(self.end_date) + '\n' + \
#              'Description:        %s' % self.description + '\n' + \
#              'Data series:' + '\n' + \
#              series_txt
#            
#        return txt
#
#    
#    def __getstate__(self):
#        """
#        Creates a dictionary of climate_data variables. Data timeseries
#        objects are converted to dictionaries available for pickling. 
#        """
#        odict = self.__dict__.copy() # copy the dict since we change it
#        # Get timeseries instances as dictionaries
#        
#        odict['data'] = []
#        for key,ts in self.data.items():
#            odict['data'].append((key,ts.__getstate__()))
#            
#        return odict
#
#    def __setstate__(self, sdict):
#        """
#        Updates instance variables from dictionary. 
#        """
#        data = sdict.pop('data', None)
#        self.__dict__.update(sdict)   # update attributes
#
#        if not self.__dict__.has_key('data'):
#            self.daily = {}
#            
#        if data != None and type(data)==dict:
#            for key,ts in data:
#                self.data[key] = myts(None,None)
#                self.data[key].__setstate__(ts)
#                
#
#    def pickle(self,fullfile_noext):
#        """
#        Pickle climate_data instance to the specified file. 
#        filename will have the extension .pkl appended.
#        """
##        if fullfile_noext == None:
##            if self.path == None:
##                path = basepath
##            else:
##                path = self.path
##            if self.fname == None:
##                fname = self.name
##            else:
##                ext,fname = self.fname[::-1].split('.')
##                fname = fname[::-1]
##            fullfile_noext = os.path.join(path,fname)
#        
#        if type(fullfile_noext) in [list, tuple]:
#            # Concatenate path elements
#            fullfile_noext = os.path.join(*fullfile_noext)
#        
#        with open(fullfile_noext+'.pkl', 'wb') as pklfh:
#            # Pickle dictionary using highest protocol.
#            pickle.dump(self, pklfh, -1)    
#    
#    
#    def add_data(self, fname=None, paramsd=None, hlines=None, data=None, times=None, flags=None):
#        """
#        Necessary parameters in file:
#        
#        station:      defines which station data will be added to
#        datatype:     type of climate data 'AT','P' or 'SD'
#        mergetype:    which data overwrites? 'oldest' (default), 'youngest', 'mean'
#        format:       
#        """
#        
#        #if (paramsd == None) and (hlines == None):
#        if fname != None:
#            paramsd, comments, hlines = get_property_info(fname)
#        
#        success = False
#        if ('station' not in paramsd): 
#            # do we have a borehole name?
#            print "No station information in file! Cannot import."
#            return False
#        elif (paramsd['station'] != self.name):
#            # Yes, but does it corespond to this station
#            print "metadata does not match station info"
#            return False
#        
#        # The data file corresponds to this station
#
#        # if filename is passed, get information from file
#        if type(data) not in [list, np.ndarray, np.ma.MaskedArray]:
#            if fname != None:
#                data,times,flags,paramsd = CDload(fname=fname, paramsd=paramsd, hlines=hlines)
#            else:
#                raise "No file name specified!"
#        
#        if data.ndim < 2:
#            data = data.reshape(data.shape[0],1) 
#        
#        mtype = paramsd.pop('mergetype','oldest')
#        dtype = paramsd.pop('datatype','').lower()
#        
#        if dtype in ['at','p','sd']:
#            dtype = dtype.upper()
#        else:
#            raise "Wrong data type specified!"
#        
#        mask = np.zeros(data.shape,dtype=bool)
#        
#        # Parse flags
#        # Presently only two types of masking is implemented
#        # 'm' masks the entire row of data at a certain present time step.
#        # 'm[0,1,2]' masks column 0,1 and 2 at a certain present time step.
#        ncols = data.shape[1]
#        if flags != None:
#            for id,f in enumerate(flags):
#                f = f.lstrip().rstrip().lower()
#                if f == '':
#                    continue
#                elif f == 'm':
#                    mask[id,:] = True
#                elif f[0] == 'm':
#                    colid =  map(int,[s.rstrip().lstrip() for s in f[1:].rstrip(' ]').lstrip(' [').split(',')])
#                    if colid not in [[],None]:
#                        mask[id,colid] = True
#
#                
#        # Merge the data to the logger         
#        if mtype=='none' or not isinstance(self.raw[dtype], myts):
#            # if the station contains no data or we chose to 
#            # overwrite everything (mtype='none')
#            self.raw[dtype] = myts(data,times,mask)
#            
#        elif mtype in ['oldest','youngest','mean','masked']:
#            # otherwise chose the correct method of merging.
#               
#            self.raw[dtype].merge(myts(data,times,mask),mtype=mtype)
#            
#        return True
#    
#    
#    
#    def get_dates(self, dtype, datetype=None):
#        """
#        Returns a sorted (ascending) array of dates for which the class contains data.
#        
#        datetype is a switch to indicate which dates are included in the list:
#        
#        datetype='any':       The date array will contain all dates for which 
#                                 the data series has an entry, masked or unmasked.
#        datetype='unmasked':  The date array will contain all dates for which 
#                                 the data series is unmasked
#        datetype='masked':    The date array will contain all dates for which 
#                                 the data series is masked 
#        datetype='missing':   The date array will contain all dates that are 
#                                 missing from the data series
#        """
#        
#        newts = copy.copy(self.data[dtype])
#        #for id,ts in enumerate(self.__dict__[dtype]):
#        if datetype in [None, 'unmasked']:
#            times = newts.times.copy()
#            mask = np.ma.getmaskarray(newts.data)
#            
#            idunmasked = mpl.mlab.find(np.logical_not(mask))
#            if len(idunmasked) != 0:
#                times = times[idunmasked]
#            
#            if newts.dt_type != dt.date:
#                for did,d in enumerate(times): times[did] = d.date()
#        else:
#            raise 'only the default behaviour (''unmasked'') has been implemented!'
#                
#        return np.unique1d(times)
#
#    
#    def calc_AT_parameters(self, use_patched = True, max_missing = None):
#        """
#        Returns a record array with the fields:
#        
#        Year:     The year for which the parameters are calculated
#        MAAT:     Mean Annual Air Temperature
#        FDD:      Freezing Degree Days
#        TDD:      Thawing Degree Days
#        ndays:    Number of days available for that year
#        
#        """
#        
#        if use_patched:
#            if self.patched['AT'] == None or max_missing != None:
#                if max_missing == None:
#                    self.patch_AT()
#                else:
#                    self.patch_AT(max_missing = max_missing)
#                
#            time = self.patched['AT'].times
#            data = self.patched['AT'].data
#        else:    
#            time = self.daily['AT'].times
#            data = self.daily['AT'].data
#        
#        if not self.daily['AT'].dt_type in [dt.datetime, dt.date]:
#            try:
#                time = mpl.dates.num2date(time, tz=self.tzinfo)
#            except:
#                raise "Bad input to calcMAAT"
#
#        years = {}
#        for id,t in enumerate(time):
#            if years.has_key(t.year):
#                years[t.year] = np.append(years[t.year],data[id])
#            else: 
#                years[t.year] = np.array(data[id])
#        
#        dtype = [('Year', int), ('MAAT', float), ('FDD', float),('TDD', float), ('ndays', int)]
#        
#        params = []
#
#        for entry in sorted(years.items()):
#            params.append( ( \
#                int(entry[0]), \
#                entry[1].mean(), \
#                entry[1][entry[1]<0.].sum(), \
#                entry[1][entry[1]>0.].sum(), \
#                entry[1].size ) ) 
#        
#        return np.array(params, dtype=dtype)
#        
