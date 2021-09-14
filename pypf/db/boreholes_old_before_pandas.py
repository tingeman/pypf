

import os
import os.path
import sys
import copy
# import imp
import new
# import getopt
import warnings

import datetime as dt
import dateutil
# import pytz

import numpy as np
from scipy import stats as scp_stats

import matplotlib as mpl
import matplotlib.pyplot as plt
import pylab

import pickle

import pdb

# Own packages/modules
import pypf.db.ClimateData as ClimateData
import pypf.db.io as io
#import pypf.mapping as mapping
import pypf.db.zoom_span as zoom_span

from pytimeseries import myts, find_permutation_id

version = 1.4

if sys.platform.find('linux') > -1:
    basepath = r'/media/Data/Thomas/A-TIN-3/Thomas/Artek/Data/GroundTemp'
    if not os.path.exists(basepath):
        raise IOError('Basepath does not exist')
else:
    basepath = r'C:\thin\02_Data\GroundTemp'
    # basepath = 'D:\Thomas\Artek\Data\GroundTemp'
    if not os.path.exists(basepath):
        basepath = r'D:/Thomas/A-TIN-3/Thomas/Artek/Data/GroundTemp'
        if not os.path.exists(basepath):
            raise IOError('Basepath does not exist')

"""
Data structure:

borehole : Class
    name:               String
    date:               datetime.date / string
    longitude:          Float
    latitude:           Float
    elevation (missing pt.)
    description:        String
    loggers:            List, populated with instances of class logger

logger : Class
    type:               String
    sn:                 String
    channels:           Integer
    depth:              List of floats
    timeseries:         instance of class myts

myts : Class
    dt_type:            datetime.datetime or datetime.date type
    times:              array of type dt_type
    data:               masked ndarray


Version 1.4:
- Implemented "time_offset" parameter for borehole data files, to allow off set
     of the entire time series by a certain number of minutes.
- Implementing Pickling functionallity of Borehole class.
- Implementing Pickling functionallity of Logger class. Tested and working!

Version 1.3:
- Removed mapping functionality from module, and placed it in separate module
    mapping, which is now imported.

Version 1.2:
- Transition to MyTimeseries_v1p2, myts.times is an array of datetime or time objects,
    myts.ordinals is an array of floats, obtained through mpl.dates.date2num

Version 1.1:
- Transition to MyTimeseries_v1p1, myts.times is an array of ordinals, obtained
    through mpl.dates.date2num
- Never finished, it turned out we need both datetimes and ordinals.

Version 1.0:
- Implementation based on MyTimeseries_v1p0, where myts.times is an array of
    datetime.datetime objects.


To do:
OK Allow for calibration values
OK Per default calculate daily time-series and store in borehole class
OK Method to calculate MAGT (MAGST will be the upper-most point)
OK Adapt plot_trumpet to use daily timeseries per default with option to use original data
-- Save/Load data to/from HDF5 file using PyTables



Annotation of plots:
t1 = annotate('MxGT', xy=(2, 0.25), xytext=(4.5, 0.4), arrowprops=dict(width=0.25,facecolor='black', shrink=0.2, headwidth=6))
t2 = annotate('MnGT', xy=(-12, 1.0), xytext=(-18, 1.25), arrowprops=dict(width=0.25,facecolor='black', shrink=0.2, headwidth=6))
t3 = annotate('MAGT', xy=(-4.5, 0.5), xytext=(-11, 0.65), arrowprops=dict(width=0.25,facecolor='black', shrink=0.2, headwidth=6))

To remove annotation use e.g. t1.remove()


"""


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


    Consider extending functionality to allow reading of more than 1500 chars,
    possibly in a sequential manner (read first 1500 chars, if data section is
    not found, read another 1500 chars etc.). This will allow for longer comments
    in the header section.


    """

    fh = None

    if fname is not None:
        fh = open(fname, 'r')
        lines = fh.readlines(1500)
    elif text is not None:
        if type(text) not in [list, tuple]:
            lines = [text]
        else:
            lines = text
    else:
        raise ValueError("Nothing to parse!")

    items = dict()
    comments = dict()
    nl = 0
    found_data = False

    for ix, l in enumerate(lines):
        l = l.rstrip().rstrip(rstrip)  # rstrip for spaces and any other specified character

        if l.startswith('#--'):
            # here we found the indicator for beginning of data section
            nl = ix + 1
            found_data = True
            break

        if (l.startswith('#')) or (len(l) == 0):
            # This is an empty or a comment line, which we will skip
            continue

        # split line in key and value
        tmp = l.split(':', 1)
        if not len(tmp) == 2:
            # There is no key value pair, thus we have reached beginning of data section
            nl = ix
            found_data = True
            break

        key = tmp[0].lstrip().rstrip().lower()
        value = tmp[1].lstrip().rstrip()

        if key.find(sep) != -1:
            # token before a possible ':' contains a sep
            # thus it must be a data line
            nl = ix
            found_data = True
            break
        elif key.find(' ') != -1:
            # token before a possible ':' contains a ' '
            # not counting leading and trailing spaces
            # thus it must be a data line
            nl = ix
            found_data = True
            break
        else:
            # We have a key-value pair!

            # parse any comment
            tmp = value.split('#', 1)
            value = tmp[0].rstrip().lstrip()
            if len(tmp) == 2:
                comment = tmp[1].rstrip().lstrip()
            else:
                comment = ''
            items[key] = value
            comments[key] = comment

    if found_data == False:
        if fh is not None and fh.readline() != '':
            fh.close()
            raise ValueError("File parsing did not encounter data section")

    if fh is not None:
        fh.close()

    return items, comments, nl


def GTload(fname, sep=',', paramsd=None, hlines=None):
    """
    GTload(fname, sep=',', paramsd=None, hlines=None)

    Function to load ground temperature data file.

    Recognized key words (case insensitive in file as they are converted
    to lower-case during file reading and parsing in get_property_info()):

    separator:      Defines the column separator
    decimal:        Defines the decimal separator
    flagcol:        Defines column number for flags for data masking
                        (default is last column)
    depth:          Defines the depth of each sensor (list of floats)
                        this value is used to determine the number of data columns
    calibration:    Optional. Defines calibration values (list is only parsed
                        in this function, not applied)
    type:           Data logger type, determines how file is read
    format:         Only used for type "generic"
                      dt_date_meas: datetime, date, measurements. Datetimes are
                          stored, date column is ignored, and measurements are
                          stored.
                      dt_date_time_meas: datetime, date, time, measurements.
                          Datetimes are stored, date and time columns are
                          ignored, and measurements are stored.
                      date_meas: date, measurements. They are both stored.

    tzinfo:         Timezone info (default is +0000 UTC)
    time_offset:    Optional. Provides a possibility to apply an offset to the
                        times listed in the file. This is relevant when the clock
                        in the computer used to launch the device has not been
                        set correctly at time of launch.
                        NB: Should be specified in MINUTES.

    These keywords are parsed and/or used in this function. 'separator','decimal'
    and 'flagcol' are removed from the parameter dictionary, everything else is
    returned in the paramsd output variable.


    Data logger types recognized:

    tinytag         data#, times, data colums (1 or 2), flag column (optional)
    hobo u12-008    data#, times, data colums (4), Hobo event columns, flag column (optional)
    hobo u23-003    data#, times, data colums (2), Hobo event columns, flag column (optional)
    generic:        times, data colums, flag column (optional)


    Returns:

    data:           masked array (not calibrated)
    times:          array of datetime objects, parsed using the info from tzinfo
    flags:          array of strings from flag column
    paramsd:        dictionary of key:value pairs, some of them parsed by this function
                        (lists in 'depth' and 'calibration' parsed to list of floats)
    """

    def change_decimal_sep(data_array, decsep):
        """
        Removes any periods (.) in strings in array then replace original
        decimal points (decsep) with periods.
        """
        for index, arrstr in np.ndenumerate(data_array):
            # Remove any periods in string
            # Replace original decimal separators with periods
            data_array[index[0], index[1]] = arrstr.replace('.', '').replace(decsep, '.')

        return data_array

    def convert_to_float(data_array, mask_value=-9999., missing=['', '---', 'None']):
        """
        Convert an array of strings to a masked array of floats
        Treat empty strings, '---' and 'None' as missing values
        """

        data = np.ma.zeros(data_array.shape)
        mask = np.ma.getmaskarray(data)

        for index, arrstr in np.ndenumerate(data_array):
            arrstr = arrstr.strip().lstrip()
            if arrstr not in missing:
                data[index[0], index[1]] = float(arrstr)
            else:
                data[index[0], index[1]] = mask_value
                mask[index[0], index[1]] = True

        data.mask = mask

        return data

    # Method starts here

    if (paramsd is None) and (hlines is None):
        paramsd, comments, hlines = get_property_info(fname=fname, sep=sep)

    # if paramsd has an entry that defines the list separator...
    sep = paramsd.pop('separator', sep)
    if sep == '\\t':
        sep = '\t'

    # if paramsd has an entry that defines the decimal separator...
    dec_sep = paramsd.pop('decimal', '.')

    file = io.dlmread(fname, sep=sep, hlines=hlines, dt=str)

    flags = None
    # get column number of flag column, if it is given in parameters
    # otherwise use last column of dataset.
    flagcol = paramsd.pop('flagcol', file.ncol - 1)
    try:
        flagcol = int(flagcol)
        if (flagcol > file.ncol - 1) or (flagcol < 0):
            # if out of bounds... assume no flags
            # pdb.set_trace()
            flagcol = None

    except:
        # pdb.set_trace()
        flagcol = None

    # if 'channels' in paramsd:
    #    datadim = float(paramsd['channels'])
    if 'depth' in paramsd:
        if type(paramsd['depth']) == str:
            paramsd['depth'] = list(map(float, paramsd['depth'].split(',')))
        datadim = len(paramsd['depth'])
    else:
        raise 'No depth information in header! (' + fname + ')'

    if 'calibration' in paramsd:
        if type(paramsd['calibration']) == str:
            try:
                paramsd['calibration'] = list(map(float, paramsd['calibration'].split(',')))
            except:
                raise 'Could not parse calibration information! (' + fname + ')'
        if datadim != len(paramsd['calibration']):
            raise 'Calibration information does not match number of data columns! (' + fname + ')'

    datetype = False

    if 'type' in paramsd:
        logtype = paramsd['type'].lower()
        logformat = paramsd.get('format', None)
        if logformat is not None: logformat = logformat.lower()

        if logtype.find('tinytag') != -1:
            times = file.data[:, 1]
            # data  = map(float, file.data[:,2])

            if dec_sep != '.':
                # the file uses wrong decimal point format, do the conversion
                file.data[:, 2:2 + datadim] = change_decimal_sep(file.data[:, 2:2 + datadim], dec_sep)

            try:
                data = np.array(file.data[:, 2:2 + datadim], dtype=float)
            except:
                data = convert_to_float(file.data[:, 2:2 + datadim])

            if file.ncol > datadim + 2 and (flagcol is not None):
                flags = file.data[:, flagcol]

        elif logtype.find('hobo') != -1:
            times = file.data[:, 1]
            if paramsd['type'].lower().find('u12-008') != -1:
                if dec_sep != '.':
                    # the file uses wrong decimal point format, do the conversion
                    file.data[:, 2:2 + datadim] = change_decimal_sep(file.data[:, 2:2 + datadim], dec_sep)

                data = convert_to_float(file.data[:, 2:2 + datadim])

                if (file.ncol > datadim + 2) and (flagcol is not None):
                    flags = file.data[:, flagcol]

            elif paramsd['type'].lower().find('u23-003') != -1:
                if dec_sep != '.':
                    # the file uses wrong decimal point format, do the conversion
                    file.data[:, 2:2 + datadim] = change_decimal_sep(file.data[:, 2:2 + datadim], dec_sep)

                data = convert_to_float(file.data[:, 2:2 + datadim])

                if (file.ncol > datadim + 4) and (flagcol is not None):
                    flags = file.data[:, flagcol]
        elif logtype.find('generic') != -1:

            if logformat == 'dt_date_time_meas':
                times = file.data[:, 0]
                datacol = 3
            elif logformat == 'dt_date_meas':
                times = file.data[:, 0]
                datacol = 2
            elif logformat == 'date_meas':
                times = file.data[:, 0]
                datacol = 1
                datetype = True
            else:  # if logformat is None     or anything else!
                times = file.data[:, 0]
                datacol = 1

            if dec_sep != '.':
                # the file uses wrong decimal point format, do the conversion
                file.data[:, datacol:datacol + datadim] = change_decimal_sep(file.data[:, datacol:datacol + datadim],
                                                                             dec_sep)

            try:
                data = np.array(file.data[:, datacol:datacol + datadim], dtype=float)
            except:
                data = convert_to_float(file.data[:, datacol:datacol + datadim])

            if file.ncol >= datacol + datadim and (flagcol is not None):
                flags = file.data[:, flagcol]
        else:
            print("Type not implemented yet!")
            raise

        if not 'tzinfo' in paramsd:
            paramsd['tzinfo'] = '+0000'
        elif paramsd['tzinfo'].lower() == 'none':
            paramsd['tzinfo'] = ''

        if datetype:
            # the data set contains only dates
            try:
                times = np.array([dateutil.parser.parse(t, yearfirst=True, dayfirst=True).date() for t in times])
            except:
                print("!!!! Problems parsing times from input file... starting debugger.")
                pdb.set_trace()
        else:
            # the dataset contains dates and times
            try:
                times = np.array(
                    [dateutil.parser.parse(t + ' ' + paramsd['tzinfo'], yearfirst=True, dayfirst=True) for t in times])
            except:
                print("!!!! Problems parsing times from input file... starting debugger.")
                pdb.set_trace()

        if 'time_offset' in paramsd:
            print("*" * 15)
            print("A time off set is applied to the data...!")
            print("*" * 15)
            times = times + dt.timedelta(seconds=np.float(paramsd.pop('time_offset')) * 60)

        return (data, times, flags, paramsd)
    else:
        raise 'Type of datafile cannot be determined... add type information to header'


class logger:
    def __init__(self, type=None, sn=None, channels=1, fname=None, data=None, times=None, sep=',', **kwargs):
        self.type = type
        self.sn = sn
        self.channels = channels
        self.calibrated = None
        self.calibration = None
        self.timeseries = None
        if fname not in [None]:
            # try:
            self.append(fname=fname, mtype='none')
            return
            # except:
            #     print "Failed to read file: ", fname
        else:
            pass

        if (data is not None) and (times is not None):
            self.timeseries = myts(data, times)

    def __getstate__(self):
        """
        Creates a dictionary of logger variables. Timeseries object is
        converted to a dictionary available for pickling.
        """
        odict = copy.deepcopy(self.__dict__)  # copy the dict since we change it
        # Get timeseries instance as dictionary
        odict['timeseries'] = self.timeseries.__getstate__()
        return odict

    def __setstate__(self, sdict):
        """
        Updates instance variables from dictionary. If dictionary has key
        "timeseries" and
        converted to a dictionary available for pickling.
        """
        timeseries = sdict.pop('timeseries', None)
        self.__dict__.update(sdict)  # update attributes

        if timeseries is not None and type(timeseries) == dict:
            if 'timeseries' not in self.__dict__ or self.timeseries is None:
                # create new instance of timeseries
                self.timeseries = myts(None, None)
            # update timeseries instance with dictionary data
            self.timeseries.__setstate__(timeseries)

    def pickle(self, fullfile_noext):
        """
        Pickle logger instance to the specified file.
        filename will have the extension .pkl appended.
        """

        if type(fullfile_noext) in [list, tuple]:
            # Concatenate path elements
            fullfile_noext = os.path.join(*fullfile_noext)

        with open(fullfile_noext + '.pkl', 'wb') as pklfh:
            # Pickle dictionary using highest protocol.
            pickle.dump(self, pklfh, -1)

    def append(self, data=None, times=None, mask=None, flags=None, paramsd=None,
               hlines=None, fname=None, mtype='oldest'):
        """
        Append data from file, or data passed to the method, to the logger instance.

        Recognized key words (case insensitive in file as they are converted
        to lower-case during file reading and parsing in get_property_info()):

        calibration:    measured data + calibration = true data
        type:           Data logger type
        sn:             Logger serial number
        depth:          Defines the depth of each sensor (list of floats)


        tzinfo:         Timezone info (default is UTC)

        These keywords are parsed and/or used in this function. 'separator','decimal'
        and 'flagcol' are removed from the parameter dictionary, everything else is
        returned in the paramsd output variable.
        """

        # if filename is passed, get information from file
        if fname is not None:
            data, times, flags, paramsd = GTload(fname=fname, paramsd=paramsd, hlines=hlines)

        # prepare mask array if none is passed
        datamask = np.ma.getmaskarray(data)
        if (mask is None) or (mask.shape != data.shape):
            mask = np.zeros(data.shape, dtype=bool)

        mask = np.logical_or(mask, datamask)

        # get calibration parameters
        self.calibration = paramsd.pop('calibration', None)
        calibrated = False

        # calibrate data
        if self.calibration is not None and self.calibrated != False:
            if len(self.calibration) != data.shape[1]:
                raise "Calibration information does not match number of data columns!"

            for id, cal in enumerate(self.calibration):
                data[:, id] = data[:, id] + cal

            calibrated = True

        # Parse flags
        # Presently only two types of masking is implemented
        # 'm' masks the entire row of data at a certain present time step.
        # 'm[0,1,2]' masks column 0,1 and 2 at a certain present time step.

        ncols = data.shape[1]
        if type(flags) in [list, tuple, np.array, np.ma.core.MaskedArray]:
            for id, f in enumerate(flags):
                f = f.lstrip().rstrip().lower()
                if f == '':
                    continue
                elif f == 'm':
                    mask[id, :] = True
                elif f[0] == 'm':
                    colid = list(map(int, [s.rstrip().lstrip() for s in f[1:].rstrip(' ]').lstrip(' [').split(',')]))
                    if colid not in [[], None]:
                        mask[id, colid] = True

        # Merge the data to the logger
        if mtype == 'none' or not isinstance(self.timeseries, myts):
            # if the logger contains no data or we chose to
            # overwrite everything (mtype='none')
            self.timeseries = myts(data, times, mask)
            if type(paramsd) == dict:
                try:
                    self.type = paramsd.pop('type')
                    self.sn = paramsd.pop('sn')
                    self.depth = paramsd.pop('depth')
                except:
                    raise "Logger parameters missing from file."

                # handle remaining parameters
                for k, v in list(paramsd.items()):
                    if k in ['timeseries']:
                        # do not overwrite the timeseries atribute!
                        continue

                    if k in self.__dict__:
                        self.__dict__[k] = v
                    else:
                        print("Parameter ", k, " not recognized!")
            else:
                raise "No header data in file, aborting!"
        elif mtype in ['oldest', 'youngest', 'mean']:
            # otherwise chose the correct method of merging.
            if type(paramsd) == dict:
                if (self.type == paramsd['type']) and \
                        (self.sn == paramsd['sn']):

                    if self.depth != paramsd['depth']:
                        # find and apply the correct permutation of the data file columns
                        id = find_permutation_id(self.depth, paramsd['depth'])
                        if id is None:
                            raise "Data file depth info does not match existing logger info. Aborting!"
                        else:
                            data = data[:, id]
                            mask = mask[:, id]

                    self.timeseries.merge(myts(data, times, mask), mtype='oldest')
                else:
                    raise "Header info does not match current logger. Aborting!"
        self.calibrated = calibrated

    def get_mean_GT(self, lim=None, ignore_mask=False):
        if ignore_mask:
            orig_mask = self.timeseries.data.mask.copy()
            self.timeseries.data.mask = np.zeros_like(orig_mask)
            self.timeseries.data = np.ma.masked_where(self.timeseries.data < -273.15,
                                                      self.timeseries.data,
                                                      copy=False)

        # Get mean of timeseries
        tsmean = self.timeseries.mean(lim)
        tsstd = self.timeseries.std(lim)

        if ignore_mask:
            self.timeseries.data.mask = orig_mask

        dtype = [('depth', float), ('value', float), ('std', float)]
        meanGT = np.array(tsmean, dtype=dtype)

        # insert depths in first column
        meanGT['depth'] = np.atleast_2d(np.array(self.depth)).T
        meanGT['std'] = np.atleast_2d(tsstd['value'])

        return meanGT

    def get_max_GT(self, lim=None, ignore_mask=False):
        if ignore_mask:
            orig_mask = self.timeseries.data.mask.copy()
            self.timeseries.data.mask = np.zeros_like(orig_mask)
            self.timeseries.data = np.ma.masked_where(self.timeseries.data < -273.15,
                                                      self.timeseries.data,
                                                      copy=False)

        # Get max of timeseries
        tsmax = self.timeseries.max(lim)

        if ignore_mask:
            self.timeseries.data.mask = orig_mask

        dtype = [('depth', float), ('value', float), ('time', type(tsmax['time']))]
        maxGT = np.array(tsmax, dtype=dtype)

        # insert depths in first column
        maxGT['depth'] = np.array(self.depth)

        return maxGT

    def get_min_GT(self, lim=None, ignore_mask=False):
        if ignore_mask:
            orig_mask = self.timeseries.data.mask.copy()
            self.timeseries.data.mask = np.zeros_like(orig_mask)
            self.timeseries.data = np.ma.masked_where(self.timeseries.data < -273.15,
                                                      self.timeseries.data,
                                                      copy=False)

        # Get min of timeseries
        tsmin = self.timeseries.min(lim)

        if ignore_mask:
            self.timeseries.data.mask = orig_mask

        dtype = [('depth', float), ('value', float), ('time', type(tsmin['time']))]
        minGT = np.array(tsmin, dtype=dtype)

        # insert depths in first column
        minGT['depth'] = np.array(self.depth)
        return minGT

    def calibrate(self):
        if self.calibrated:
            return

        for id, cal in enumerate(self.calibration):
            self.timeseries.data[:, id] = self.timeseries.data[:, id] + cal

        self.calibrated = True

    def uncalibrate(self):
        if not self.calibrated:
            return

        for id, cal in enumerate(self.calibration):
            self.timeseries.data[:, id] = self.timeseries.data[:, id] - cal

        self.calibrated = False

    def drop_channels(self, channels=None):
        if channels is None:
            return

        if type(channels) not in [list, np.ndarray, tuple]:
            channels = [channels]

        for cid in channels[::-1]:
            self.timeseries.data = np.delete(self.timeseries.data, cid, 1)
            self.depth = np.delete(self.depth, cid, None)
            if self.calibration is not None:
                self.calibration = np.delete(self.calibration, cid, None)

                # Should we handle parameter: "channels" as well?

    def mask_channels(self, channels=None):
        """
        Call signature:
        logger.mask_channels(channels=None)

        Masks the specified channel(s), channels may be a list of channel indices.
        """
        if channels is None:
            return
        self.timeseries.mask(columns=channels)


class borehole:
    def __init__(self, name=None, date=None, lat=None, lon=None, \
                 description=None, loggers=None, fname=None):

        self.name = name
        self.date = date
        self.latitude = lat
        self.longitude = lon
        self.description = description
        self.depths = None
        self.daily_ts = None
        if fname is None:
            self.fname = None
            self.path = None
        else:
            self.path, self.fname = os.path.split(fname)

        if loggers is None:
            self.loggers = list()
        else:
            self.loggers = loggers

        if fname not in [None, []]:
            self.load_borhole(fname=fname)

    def __getstate__(self):
        """
        Creates a dictionary of borehole variables. daily_ts object is
        converted to a dictionary available for pickling. Logger list is
        iterated and logger instances converted to dictionaries for
        pickling
        """
        odict = odict = copy.deepcopy(self.__dict__)  # copy the dict since we change it
        # Get timeseries instance as dictionary
        if self.daily_ts is not None:
            odict['daily_ts'] = self.daily_ts.__getstate__()
        for id, l in enumerate(odict['loggers']):
            odict['loggers'][id] = l.__getstate__()
        return odict

    def __setstate__(self, sdict):
        """
        Updates instance variables from dictionary.
        """
        daily_ts = sdict.pop('daily_ts', None)
        loggers = sdict.pop('loggers', None)
        self.__dict__.update(sdict)  # update attributes

        if 'daily_ts' not in self.__dict__:
            self.daily_ts = None
        if 'loggers' not in self.__dict__:
            self.loggers = []

        if daily_ts is not None and type(daily_ts) == dict:
            if self.daily_ts is None:
                # create new instance of timeseries
                self.daily_ts = myts(None, None)
            # update timeseries instance with dictionary data
            self.daily_ts.__setstate__(daily_ts)

        if self.loggers == []:
            # empty logger list, just add new loggers!
            for id, l in enumerate(loggers):
                if type(l) == dict:
                    self.loggers.append(new.instance(logger))
                    self.loggers[id].__setstate__(l)
        else:
            # We are updating an existing instance!
            print("option to update existing loggers not implemented yet!")
            pdb.set_trace()

    def pickle(self, fullfile_noext=None):
        """
        Pickle borehole instance to the specified file.
        filename will have the extension .pkl appended.
        """
        if fullfile_noext is None:
            if self.path is None:
                path = basepath
            else:
                path = self.path
            if self.fname is None:
                fname = self.name
            else:
                ext, fname = self.fname[::-1].split('.')
                fname = fname[::-1]
            fullfile_noext = os.path.join(path, fname)

        if type(fullfile_noext) in [list, tuple]:
            # Concatenate path elements
            fullfile_noext = os.path.join(*fullfile_noext)

        with open(fullfile_noext + '.pkl', 'wb') as pklfh:
            # Pickle dictionary using highest protocol.
            pickle.dump(self, pklfh, -1)

    def load_borhole(self, fname):
        items, comments, nl = get_property_info(fname=fname)

        for key, value in list(items.items()):
            if key == 'date':
                self.__dict__[key] = dateutil.parser.parse(value)
            elif key in ['latitude', 'longitude']:
                self.__dict__[key] = float(value)
            else:
                self.__dict__[key] = value

    def auto_process(self, fname="auto_processing", path=None):
        """
        call signature:
        auto_process(fname="auto_processing", path=None)

        Method to initiate auto prossessing steps defined in a file
        located in the same directory as the borehole definition file,
        or in the specified directory path.

        If path is None, the path of the borehole definition file will be used,
        if known.

        The file auto_processing.py shoul have a method defined as:

        def auto_process(bhole):
            # do something with bhole
            pass
        """

        if path is None and self.path is not None:
            path = self.path
        if path is None:
            print("No path specified for auto processing! Nothing done...!")
            return
        # fname is the filename without the .py/.pyc extention

        fullfile_noext = os.path.join(path, fname)
        pyfile = fullfile_noext + ".py"
        if os.path.isfile(pyfile):
            exec(compile(open(pyfile, "rb").read(), pyfile, 'exec'), {}, locals())
        else:
            print("Specified processing file not found! Nothing done ...!")
            return

        locals()['auto_process'](self)

        print("Applied processing steps from {0}".format(pyfile))

    def __str__(self):
        def print_date(date):
            yr = str(self.date.year)
            mm = str(self.date.month)
            dd = str(self.date.day)
            if len(mm) < 2:
                mm = '0' + mm
            if len(dd) < 2:
                dd = '0' + dd
            return yr + '-' + mm + '-' + dd

        logstr = ''
        for l in self.loggers:
            logstr += '  %s\n' % l.sn

        txt = 'Borehole name:      %s' % self.name + '\n' + \
              'Drill date:         %s' % print_date(self.date) + '\n' + \
              'Latitude:           %f' % self.latitude + '\n' + \
              'Longitude:          %f' % self.longitude + '\n' + \
              'Description:        %s' % self.description + '\n' + \
              'Number of loggers:  %i' % len(self.loggers) + '\n' + \
              logstr

        return txt

    def add_data(self, fname, paramsd=None, hlines=None):
        if (paramsd is None) and (hlines is None):
            paramsd, comments, hlines = get_property_info(fname)

        success = False
        if ('borehole' not in paramsd):
            # do we have a borehole name?
            print("No borehole information in file! Cannot import.")
            return False
        elif (paramsd['borehole'] != self.name):
            # Yes, but does it corespond to this borehole
            print("Logger info does not match borehole info")
            return False
        elif 'sn' in paramsd:
            # Do we have a serial number? Is it already registered?

            for lid, l in enumerate(self.loggers):
                if l.sn == paramsd['sn']:
                    # If sn matches existing
                    self.loggers[lid].append(fname=fname, paramsd=paramsd, hlines=hlines, mtype='oldest')
                    success = True
                    break

            if not success:
                # So the logger is not registered yet...
                print('--------------------------------------------------------------')
                print('Adding logger with SN ' + paramsd['sn'] + ' to borehole ' + self.name)
                self.loggers.append(logger())
                self.loggers[-1].append(fname=fname, paramsd=paramsd, hlines=hlines, mtype='none')

            return True
        else:
            print("No serial number information in file! Cannot import.")
            return False

    def drop_channels(self, logger_sn=None, logger_id=None, channels=None):
        """
        Pass an index to a logger in logger_id and
        a list of channels to drop, an the method will delete
        these channles from the logger.

        Pt.: the channels property is unaffected, but depths, and calibration
        is altered accordingly.
        """

        if logger_sn is not None:
            sn_list = dict()
            sn_list.update([[l.sn, lid] for lid, l in enumerate(self.loggers)])
            if logger_sn in sn_list:
                logger_id = sn_list[logger_sn]
            else:
                logger_id = None

        if logger_id is not None and channels is not None:
            self.loggers[logger_id].drop_channels(channels)
            self.calc_daily_avg()

    def mask_channels(self, logger_sn=None, logger_id=None, channels=None):
        """
        Call signature
        mask_channels(logger_sn=None, logger_id=None, channels=None)

        Will mask specifiec channels in the specified logger.
        Logger can be specified either by serial number (string) or by logger_id (index
        into the borehole.loggers array).
        channels is a list of channel indices to mask.
        """

        if logger_sn is not None:
            sn_list = dict()
            sn_list.update([[l.sn, lid] for lid, l in enumerate(self.loggers)])
            if logger_sn in sn_list:
                logger_id = sn_list[logger_sn]
            else:
                logger_id = None

        if logger_id is None and len(self.loggers) == 1:
            logger_id = 0

        if logger_id is not None and channels is not None:
            self.loggers[logger_id].mask_channels(channels)
            self.calc_daily_avg()

    def limit(self, lim):
        """
        Not implemented yet!
        """
        pass

    def calc_daily_avg(self, mindata=None):
        if len(self.loggers) == 0:
            return

        if mindata is None:
            # Estimate the number of data points per day
            self_freq = int(1 / self.loggers[0].timeseries.estimate_frequency())
            if 24 < self_freq < 1:
                print("")
                print("Estimated frequency of data series from logger " + self.loggers[0].sn + \
                      " is outside common\n range (1-24 points per day).")
                print("Aborting calculation of daily average of borehole " + self.name)
                print("")
                return
            else:
                print("Estimated data frequency = {0} measurements/day".format(self_freq))

            # Get daily average from first logger, and setup depth array
            mindata = self_freq
            self.mindata = mindata

        self.daily_ts = self.loggers[0].timeseries.calc_daily_avg(mindata=mindata)
        self.depths = copy.copy(self.loggers[0].depth)

        # If more loggers, stack data from these, using masked entries if
        # timeseries have different steps.
        if len(self.loggers) > 1:
            for l in self.loggers[1:]:
                # Estimate the number of data points per day
                self_freq = int(1 / self.loggers[0].timeseries.estimate_frequency())
                if 24 < self_freq < 1:
                    print("")
                    print("Estimated frequency of data series from logger " + l.sn + \
                          " is outside common\n range (1-24 points per day).")
                    print("Aborting calculation of daily average of borehole " + self.name)
                    print("")
                    return
                self.daily_ts.hstack(l.timeseries.calc_daily_avg(mindata=self_freq))
                [self.depths.append(d) for d in l.depth]

        # sort depths in ascending order
        sid = np.argsort(self.depths)
        self.depths = list(np.take(self.depths, sid))

        # apply the same order to the data in the timeseries
        self.daily_ts.data = self.daily_ts.data[:, sid]

    def calc_monthly_avg(self, mindata=None):
        """
        calc_monthly_average will take the daily_avg time series and form a
        monthly series based on that.

        NB: For now, there is no good implementation of a minimum number of entries
        in a month, to trigger a mask. You can specify a number by passing it
        in the call to mindata.
        It would be nice to include possibility in MyTimeseries to flag each
        entry with meta data, f.ex. the number of data in an averaged point.
        """

        if 'daily_ts' not in self.__dict__:
            self.calc_daily_avg()

        if len(self.loggers) == 0:
            return

        if mindata is None:
            mindata = 1

        self.monthly_ts = self.daily_ts.calc_monthly_avg(mindata=mindata)

        # sort depths in ascending order
        sid = np.argsort(self.depths)
        self.depths = list(np.take(self.depths, sid))

        # apply the same order to the data in the timeseries
        self.daily_ts.data = self.daily_ts.data[:, sid]

    def print_monthly_data(self):
        """
        Call signature:
        print_monthly_data()

        Function to print the monthly averages of a time series.

        #It will print to the console:
        #- The borehole name
        #- The first and last day of data in the full time series
        #- The date interval for which information will be printed
        #- Max, Min and MeanGT for each depth.
        """

        if 'monthly_ts' not in self.__dict__:
            self.calc_monthly_avg()

        datetimes = self.monthly_ts.times
        years = np.vstack((np.array([t.year for t in datetimes]),
                           np.arange(0, len(datetimes)))).T

        months = np.arange(1, 13)
        months_str = [dt.date(2009, m, 1).strftime('%b') for m in months]

        table = np.ma.masked_all((self.monthly_ts.data.shape[1], 13))
        table_mask = np.ma.getmaskarray(table)

        table[0::, 0] = self.depths

        # prt_fmt = np.ones((1,13)).*
        print(" ")
        print("--------------------------------------------------------------------")
        print("Borehole %s" % (self.name,))
        print("--------------------------------------------------------------------")
        for y in np.unique(years[:, 0]):
            table_mask.fill(True)
            table_mask[0::, 0] = False

            for yid in years[years[:, 0] == y, 1]:
                mo = datetimes[yid].month
                table[:, mo] = self.monthly_ts.data[yid, :]
                table_mask[:, mo] = self.monthly_ts.data[yid, :].mask

            table.mask = table_mask

            print(" ")
            print("Year:  %i" % (y))
            print("Depth\t" + "\t".join(months_str))
            for k in np.arange(table.shape[0]):
                # print "\t".join(["%.2f" % (t,) for t in table[k, :]])
                if table.mask[k, 0] == True:
                    outstr = "---"
                else:
                    outstr = "{0: .2f}".format(table[k, 0])
                for t, m in zip(table[k, 1:], table.mask[k, 1:]):
                    if m == True:
                        outstr = "\t".join([outstr, '---'])
                    else:
                        outstr = "\t".join([outstr, "{0: .2f}".format(t)])
                print(outstr)

    def calc_thaw_depth(self, date, fullts=False):
        """
        Method to calculate thawdepth on a specific date
        Presently only the daily averaged series is implemented

        This methods have problems, when the ground starts to freeze from the top.
        Presently, mainly calc_thaw_depth_at_node is used for ALT calculation!
        """

        def x_at_y_eq_0(x_1, x_2, y_1, y_2):
            """
            Calculates the value of x at y=0, thus the crossing point of the
            x-axis. According to the general equation:

            x_0 = x_1 - y_1 / ((y_2-y_1)/(x_2-x_1))
            """

            return x_1 - y_1 / ((y_2 - y_1) / (x_2 - x_1))

        input('Beware! This method has problems with ground freezing from top! [press enter to proceed]')

        if not fullts:

            if self.daily_ts is None:
                self.calc_daily_avg()

            # Get the index of the day in question
            did = find(self.daily_ts.times == date)

            if not did:
                raise ValueError("Date does not exist in data set")

            # Select only ground temperatures
            GTid = find(np.array(self.depths) >= 0.)
            data = self.daily_ts.data[did, GTid].flatten()
            depths = np.take(self.depths, GTid)

            ids = np.argsort(depths)
            depths = depths[ids]
            data = data[ids]

            # Find all nodes that are unfrozen
            frozen = find(data <= 0)
            # Find all nodes that are frozen
            unfrozen = find(data > 0)

            # If there are frozen nodes
            if len(frozen) != 0:
                # Filter, so that we keep only those unfrozen nodes that are
                # above frozen nodes.

                # NB: This assumes that depths are increasing with node number!

                unfrozen = unfrozen[unfrozen < frozen[0]]

            thawd = np.ma.zeros(3)
            thawd.mask = np.ma.getmaskarray(thawd)
            thawd.mask = True

            pdb.set_trace()

            if True:  # len(unfrozen) != 0 and len(frozen) != 0:

                # General equation for calculating x_0 @ y=0:
                #    x_0 = x_1 - y_1 / ((y_2-y_1)/(x_2-x_1))

                # We do 3 estimations of the frost table based on:
                # a) two points just above the frost table
                # b) one point above and one below frost table
                # c) two points below frost table.

                if len(unfrozen) >= 2:
                    # Calculate based on two points above frost table
                    thawd[0] = x_at_y_eq_0(
                        depths[unfrozen[-2]],
                        depths[unfrozen[-1]],
                        data[unfrozen[-2]],
                        data[unfrozen[-1]])
                    thawd.mask[0] = False

                if len(frozen) >= 2:
                    # Calculate based on two points below frost table
                    thawd[1] = x_at_y_eq_0(
                        depths[frozen[0]],
                        depths[frozen[1]],
                        data[frozen[0]],
                        data[frozen[1]])
                    thawd.mask[1] = False

                if len(unfrozen) >= 1 and len(frozen) >= 1:
                    # Calculate based on one point above and below frost table
                    thawd[2] = x_at_y_eq_0(
                        depths[unfrozen[-1]],
                        depths[frozen[0]],
                        data[unfrozen[-1]],
                        data[frozen[0]])
                    thawd.mask[0] = False

                if len(unfrozen) >= 1 and len(frozen) >= 1:
                    if np.logical_or(thawd[0] > depths[frozen[0]], thawd[0] < depths[unfrozen[-1]]):
                        thawd.mask[0] = True

                    if np.logical_or(thawd[1] > depths[frozen[0]], thawd[1] < depths[unfrozen[-1]]):
                        thawd.mask[1] = True

                    return thawd, (data[unfrozen[-1]] - 0) / (data[unfrozen[-1]] - data[frozen[0]])

                else:

                    return thawd, 0

                    # if not np.all(np.diff(np.array([unfrozen[-2:],frozen[0:2]]).flatten()) == 1):
                    #     print "There are masked values close to frost table."
                    #     print "Estimated thaw depths may be inaccurate!"

            else:
                return thawd, -9999.
        else:
            raise "Thaw depth calculation based on full time series is not implemented!"

    def calc_thaw_depth_all(self, date, fullts=False):
        """
        Method to calculate thawdepth on a specific date
        Presently only the daily averaged series is implemented

        This version should track and return several 0 degree isotherms!
        Not implemented yet.
        """

        raise NotImplementedError('Not implemented!')

    def calc_thaw_depth_at_node(self, date, node, fullts=False):
        """
        Method to calculate thawdepth on a specific date
        Presently only the daily averaged series is implemented
        
        This version should calculate the thaw-depth below the specified node.
        Thus, the temperature at the specified node should be above zero, and
        the temperature at node+1 should be below zero.
        The method will throw an error, if this is not the case.
        """

        # raise NotImplementedError('Not implemented!')

        def x_at_y_eq_0(x_1, x_2, y_1, y_2):
            """
            Calculates the value of x at y=0, thus the crossing point of the
            x-axis. According to the general equation:
            
            x_0 = x_1 - y_1 / ((y_2-y_1)/(x_2-x_1))
            """

            return x_1 - y_1 / ((y_2 - y_1) / (x_2 - x_1))

        if not fullts:

            if self.daily_ts is None:
                self.calc_daily_avg()

            # Get the index of the day in question
            did = find(self.daily_ts.times == date)

            if len(did) == 0:
                raise ValueError("Date does not exist in data set")

            # Select only ground temperatures
            GTid = find(np.array(self.depths) >= 0.)
            depths = np.take(self.depths, GTid)

            data = self.daily_ts.data[did, GTid].flatten()

            # sort data according to depths
            ids = np.argsort(depths)
            depths = depths[ids]
            data = data[ids]
            # note that node is not changed by the sorting,
            # so this method assumes that node is chosen based 
            # on already sorted depths!

            # remove any masked data, but keep track we have the right index
            node = node - data.mask[0:node].sum()
            data = data.compressed()

            # Ensure that we can do these calculations!
            if node < 2 or node > len(depths) - 3:
                raise ValueError(
                    'node-argument must be larger than 2 and less than number of depths (excluding above ground depths) minus 2.')

            if data[node] < 0. or data[node + 1] > 0:
                raise ValueError('node argument should be depth index for node immediately above frost table.')

            # Prepare array to store thaw depths
            # Will hold the three different estimations
            thawd = np.ma.zeros(3)
            thawd.mask = np.ma.getmaskarray(thawd)
            thawd.mask = True

            # General equation for calculating x_0 @ y=0:
            #    x_0 = x_1 - y_1 / ((y_2-y_1)/(x_2-x_1))

            # We do 3 estimations of the frost table based on:
            # a) two points just above the frost table
            # b) one point above and one below frost table
            # c) two points below frost table.

            if node >= 2:
                # Calculate based on two points above frost table
                thawd[0] = x_at_y_eq_0(
                    depths[node - 1],
                    depths[node],
                    data[node - 1],
                    data[node])
                if np.isnan(thawd[0]):
                    thawd.mask[0] = True

            if np.all(data[node + 1:node + 3] < 0.):
                # Calculate based on two points below frost table
                thawd[1] = x_at_y_eq_0(
                    depths[node + 1],
                    depths[node + 2],
                    data[node + 1],
                    data[node + 2])
                if np.isnan(thawd[1]):
                    thawd.mask[1] = True

            if node >= 1 and data[node + 1] < 0.:
                # Calculate based on one point above and below frost table
                thawd[2] = x_at_y_eq_0(
                    depths[node],
                    depths[node + 1],
                    data[node],
                    data[node + 1])
                if np.isnan(thawd[2]):
                    thawd.mask[2] = True

            return thawd, 0

        else:
            raise "Thaw depth calculation based on full time series is not implemented!"

    def calc_ALT(self, lim=None, fullts=False, silent=False):
        """
        Under CONSTRUCTION....
        
        [Yr maxd maxddayID] = SnuwALT(dateinput,data,depths)
        
        dateinput:   serial number date as produced by datenum
        depths:      depths included in data (length(depths) = size(data,2))
        data:        matrix with depths as columns, days as rows
        maxddayID:   Index into dateinput that gives maximum thawdepth
        """

        if not fullts:
            # Calculate the daily averages if missing
            if self.daily_ts is None:
                self.calc_daily_avg()

            # Limit the time series, if limits are specified
            if lim is not None:
                ts = self.daily_ts.limit(lim)
            else:
                ts = self.daily_ts

            # Get a unique list of years in the time series
            Y = np.array([d.year for d in ts.times])
            Yr = np.unique(Y)

            # Prepare arrays for storing results
            maxd = np.ma.zeros((len(Yr)), dtype=int)  # index to deepest node with thaw in each year

            # maxdayIDp will hold the date-index for the day with maximum temperature at the
            #           deepest thawed node (maxd) for each year
            maxdayIDp = np.zeros_like(Yr)

            # maxdayIDn will hold the date-index for the day with maximum temperature at the
            #           shallowest frozen node with the condition that a thawed node exists above it
            maxdayIDn = np.ones_like(Yr) * -1

            # For each date, 3 estimations of thaw depth will be calcualted by extrapolation/interpolation
            # 1) based on two nodes above frost table
            # 2) based on two nodes below frost table
            # 3) based on one node above and below frost table

            thaw_depth_p = np.ma.zeros((len(Yr), 3))  # holds the 3 thaw depths for the madayIDp date
            thaw_depth_n = np.ma.zeros((len(Yr), 3))  # holds the 3 thaw depths for the madayIDn date
            ALT_date = np.ma.ones((len(Yr), 2),
                                  dtype=dt.date)  # holds the two dates for which thaw depths are calculated

            # extract ground temperature data (only depths >= 0.0m
            GTid = find(np.array(self.depths) >= 0.)
            depths = np.take(self.depths, GTid)
            data = ts.data[:, GTid]

            # Loop over individual years
            for yid, y in enumerate(Yr):

                # Find days from this specific year
                daysID = find(Y == y)

                # Loop over depths, starting from top
                # did will be depth index
                # d will hold temperature data for that depth
                for did, d in enumerate(data[daysID].T):

                    if np.all(d.mask):
                        # If all data are masked, continue to next iteration/depth.
                        continue

                    # Find days with positive thaw at present depth (in this specific year)
                    tid = find(d >= 0)
                    if len(tid) != 0:
                        # If exist, set max thaw depth to this depth level
                        maxd[yid] = did
                        # set maxddayIDp to the index into data (for a certain year) for
                        # which the temperature is the highest, assuming this is the day
                        # with deepest active layer. If more days with same temp, take
                        # last day.
                        mid = find(d == d.max())
                        if len(mid) != 0:
                            maxdayIDp[yid] = daysID[mid.max()]
                    elif False:
                        # Here we should handle the situation when data is missing
                        # in the thaw period from one depth.
                        # Presently that will be considered as no thaw.
                        # Should allow to look at deeper depth.
                        pass
                    else:
                        # Otherwise we have passed 0C isotherm, so break loop

                        # set maxddayIDn to the index into data (for a certain year) for
                        # which the temperature is the highest, and the node above is at
                        # positive temperatures, assuming this is the day with deepest active
                        # layer. If more days with same temp, take last day.

                        if not did == 0:
                            # If this is not the upper most node, find all days
                            # with positive temperatures in depth above the current.
                            nodes_up = 1
                            d_above = data[daysID].T[did - nodes_up, :]

                            # This loop handles situations where all data from a node is masked
                            # in which case it jumps one node up until it finds data.
                            # This may still give problems in cases where sensor malfunctions
                            # during that particular year.
                            while np.all(d_above.mask):
                                nodes_up += 1
                                if nodes_up == -1:
                                    break
                                d_above = data[daysID].T[did - nodes_up, :]

                            # If we have no data at this node or above, continue to next iteration
                            if nodes_up == -1:
                                continue

                            # Get index to all dates with positive temperatures (at node above the present node)
                            idxp = find(d_above >= 0)

                            # Now check these days for the highest temperature at the
                            # present node-depth.
                            d_filtered = d[idxp]
                            idx = find(d_filtered == d_filtered.max())
                            # idx is index into days with positive temperature at depth above...
                            # make idx index into days of the present year.
                            idx = idxp[idx]

                            maxdayIDn[yid] = daysID[idx.max()]

                        break

                # Still only looking at one year, if we have thaw, do interpolation
                if maxd[yid] != 0:
                    # interpolate to find 0 degr.

                    if maxd[yid] == len(depths) - 1:
                        print(str(Yr[yid]) + ": Max depth beyond grid")
                        maxd[yid] = depths[-1]
                    elif maxd[yid] < len(find(depths < 0)):
                        print(str(Yr[yid]) + ": Max depth above second grid point")
                        maxd[yid] = -9999.
                    else:
                        # calculate thaw depths for the date with max temperature at deepest thawed node
                        thawd, rel_d = self.calc_thaw_depth_at_node(ts.times[maxdayIDp[yid]], maxd[yid])
                        print("IDp: {0:4d} ({1:10s})    ".format(maxdayIDp[yid], ts.times[maxdayIDp[yid]].__str__()), end=' ')
                        thaw_depth_p[yid, :] = np.ma.atleast_2d(thawd)

                        # calculate thaw depths for the date with max temperature at shallowest frozen node
                        thawd, rel_d = self.calc_thaw_depth_at_node(ts.times[maxdayIDn[yid]], maxd[yid])
                        print("IDn: {0:4d} ({1:10s})    ".format(maxdayIDn[yid], ts.times[maxdayIDn[yid]].__str__()))
                        thaw_depth_n[yid, :] = np.ma.atleast_2d(thawd)

                        ALT_date[yid, 0] = ts.times[maxdayIDp[yid]]
                        ALT_date[yid, 1] = ts.times[maxdayIDn[yid]]

            # Mask year entries in ALT_date, where ALT could not be established
            ALT_date.mask = np.where(ALT_date == 1, np.ones(ALT_date.shape, dtype=bool),
                                     np.zeros(ALT_date.shape, dtype=bool))

            # maxd will hold the largest estimate of the two days considered.
            if False:
                # use means of all methods, even if some are way off.
                tdp = thaw_depth_p.mean(axis=1)
                tdn = thaw_depth_n.mean(axis=1)
                tdp_std = thaw_depth_p.std(axis=1)
                tdn_std = thaw_depth_n.std(axis=1)

            else:
                # use only "two-points-above" and "one-point-above/one-point-below", and average these two.
                tdp = thaw_depth_p[:, [0, 2]].mean(axis=1)
                tdn = thaw_depth_n[:, [0, 2]].mean(axis=1)
                tdp_std = thaw_depth_p[:, [0, 2]].std(axis=1)
                tdn_std = thaw_depth_n[:, [0, 2]].std(axis=1)

            # Consider taking always the solution based on "two points above"", if available.
            # and only reverting to "one-point-above/one-point-below" if the "two-points-above"
            # solution is masked (does not exist)

            maxdepth = np.where(tdp > tdn, tdp, tdn)

            if not silent:
                print(" ")
                print(" ")
                print("++ = 2 nodes above frost table")
                print("+- = 1 node above frost table and one node below")
                print("-- = 2 nodes below frost table")
                print(" ")
                print("Estimations based on date of highest temperature at deepest thawed node:")
                print("Year    Depth(++)    Depth(+-)    Depth(--)    Date             maxd")
                str_fmt = "{0:4d}    {1:6s}       {2:6s}       {3:6s}       {4:10s}    {5:4d}"
                for yid in np.arange(len(Yr)):
                    print(str_fmt.format(Yr[yid],
                                         np.round(thaw_depth_p[yid, 0], 2).__str__(),
                                         np.round(thaw_depth_p[yid, 2], 2).__str__(),
                                         np.round(thaw_depth_p[yid, 1], 2).__str__(),
                                         ALT_date[yid, 0].__str__(),
                                         maxd[yid]))
                print(" ")
                print("Estimations based on date of highest temperature at node below deepest thawed node:")
                print("Year    Depth(++)    Depth(+-)    Depth(--)    Date        maxd")
                for yid in np.arange(len(Yr)):
                    print(str_fmt.format(Yr[yid],
                                         np.round(thaw_depth_n[yid, 0], 2).__str__(),
                                         np.round(thaw_depth_n[yid, 2], 2).__str__(),
                                         np.round(thaw_depth_n[yid, 1], 2).__str__(),
                                         ALT_date[yid, 1].__str__(),
                                         maxd[yid]))

                print(" ")
                print("Summary:")
                print("Depth: average and std of ++ and +- thaw depths")
                print("ALT: the maximum of the averages for the two dates considered")
                print("Year    ALT      Depth(p) std   Date          Depth(n) std   Date      ")
                for yid in np.arange(len(Yr)):
                    print("%4i  %6.2f    %6.2f    %4.2f  %10s   %6.2f    %4.2f  %10s" % (Yr[yid], maxdepth[yid],
                                                                                         tdp[yid], tdp_std[yid],
                                                                                         ALT_date[yid, 0],
                                                                                         tdn[yid], tdn_std[yid],
                                                                                         ALT_date[yid, 1]))
                print(" ")
                print(" ")
                print(" ")
            return (Yr, maxdepth, thaw_depth_p, thaw_depth_n, ALT_date)

            # id           index corresponding to a certain year
            # maxdayIDp    holds index of day in current year for which max thawdepth occurs
            # maxdayIDn    holds index of day after max thawdepth has occurred
            # maxd         holds index into depths for max thawed depth

        else:
            raise "Full timeseries support in calc_ALT is not implemented yet!"

    def calc_ALT_simple(self, lim=None, fullts=False, silent=False):
        """
        Under CONSTRUCTION....
        
        Will give the deepest depth that has positive temperature for every year in the ts.
        
        [Yr maxd maxddayID] = SnuwALT(dateinput,data,depths)
        
        dateinput:   serial number date as produced by datenum
        depths:      depths included in data (length(depths) = size(data,2))
        data:        matrix with depths as columns, days as rows
        maxddayID:   Index into dateinput that gives maximum thawdepth
        """

        input('Beware: Method not completed and tested! [Press enter to continue]')

        if not fullts:
            if self.daily_ts is None:
                self.calc_daily_avg()

            if lim is not None:
                ts = self.daily_ts.limit(lim)
            else:
                ts = self.daily_ts

            ydict = ts.split_years()

            Yr = sorted(ydict.keys())
            maxd = []
            ALT_date = []

            # year_thaw = np.zeros((len(ydict),3))

            # Loop through time series for individual years
            for y, yts in list(ydict.items()):
                maxthaw = []  # Holds the 
                for did in np.arange(len(yts.times)):
                    try:
                        maxthaw.append(np.max(np.nonzero(yts.data[did, :] >= 0.0)))
                    except:
                        maxthaw.append(0)
                maxd.append(self.depths[np.max(np.array(maxthaw))])
                try:
                    ALT_date.append(yts.times[np.argmax(maxthaw)])
                except:
                    pdb.set_trace()
                    ALT_date.append(np.ma.masked)

            if not silent:
                print("Year    Depth(p) std  Date          Depth(n) std  Date      ")
                for did in np.arange(len(Yr)):
                    print("%4i    %6.2f  %10s" % (Yr[did], maxd[did], ALT_date[did]))

            return (Yr, maxd, ALT_date)

        else:
            raise "Full timeseries support in calc_ALT is not implemented yet!"

    def get_depths(self, return_indices=False):
        """
        Returns a sorted (ascending) list of sensor depths in the borehole.

        if return_indices is True (default is false), returns a 3 column array,
        with depths (ascending) in first column, logger index as second column
        and channel index as third column
        """
        dtype = [('depth', float), ('logger_id', int), ('channel_id', int)]

        depths = []
        for lid, l in enumerate(self.loggers):
            for cid, d in enumerate(l.depth): depths.append((d, lid, cid))
        depths = np.array(depths, dtype=dtype)

        depths = np.sort(depths, order='depth')

        if return_indices:
            return depths
        else:
            return depths['depth']

    def get_MAGT(self, end=None, nyears=1, fullts=False):
        """
        Call signature:
        borehole.get_MAGT(end=None, nyears=1, fullts=False)

        This method calculates the Mean Annual Ground Temperature over a period
        of nyears years, ending on the date specified in the input parameter 'end'.
        If 'end' is not specified, default is the last full day in the data series.

        MAGT is calculated based on the averaged daily time series unless fullts=True
        is specified.

        NB: If you want mean ground temperature for a specified date interval,
        e.g. lim = ["2000-08-01","2009-06-01"] you should call instead
        borehole.get_MeanGT(lim = ["2000-08-01","2009-06-01"])
        """

        if not fullts:
            if self.daily_ts is None:
                self.calc_daily_avg()

            # check date format, do necessary conversion to datetime.date object
            if end is None:
                end = self.daily_ts.times[-1]

            lim = nyears2lim(end, nyears)
            # if type(end) in [int, float, np.float32, np.float64, np.float96]:
            #     end = mpl.dates.num2date(end).date()
            # elif type(end) == str:
            #     end = dateutil.parser.parse(end, yearfirst=True, dayfirst=True).date()
            # elif type(end) in [dt.datetime]:
            #     end = end.date()
            # elif type(end) != dt.date:
            #     raise "dates should be either ordinals, date or datetime objects"
            # 
            # start = copy.copy(end).replace(year=end.year - nyears)
            # start = start + dt.timedelta(days=1)
            # lim = [start, end]

            if (self.daily_ts.times[-1] < end) or (self.daily_ts.times[0] > lim[0]):
                print("get_MAGT: timeseries not long enough for requested time interval!")
                pdb.set_trace()

            return self.get_MeanGT(lim)

        else:
            raise "Full timeseries support in get_MAGT is not implemented yet!"

    def get_MeanGT(self, lim=None, fullts=False, ignore_mask=False):
        if not fullts:
            if self.daily_ts is None:
                self.calc_daily_avg()

            if ignore_mask:
                print("Warning: ignore_mask option not implemented for averaged data!")

            meanGT = self.daily_ts.mean(lim)

            # It is a little tricky to handle the possibility of a masked
            # value in the value-column. We have to treat data and masks
            # separately.
            dtype = [('depth', float), ('value', float)]
            dtype2 = [('depth', bool), ('value', bool)]

            meanGT_data = np.array(meanGT, dtype=dtype)
            meanGT_mask = np.array(np.ma.getmaskarray(meanGT), dtype=dtype2)

            meanGT2 = np.ma.array(meanGT_data, mask=meanGT_mask, dtype=dtype)

            # insert depths in first column
            meanGT2['depth'] = np.ma.atleast_2d(np.ma.array(self.depths)).transpose()

            return meanGT2
        else:
            meanGT = self.loggers[0].get_mean_GT(lim, ignore_mask)

            # If more loggers, append data from these
            if len(self.loggers) > 1:
                for l in self.loggers[1:]:
                    meanGT = np.append(meanGT, l.get_mean_GT(lim, ignore_mask))

            # Return sorted recarray
            return np.sort(meanGT, order='depth', axis=0)
            # raise "Full timeseries support in get_MAGT is not implemented yet!"

    def get_MaxGT(self, lim=None, fullts=False, ignore_mask=False):
        if not fullts:
            if self.daily_ts is None:
                self.calc_daily_avg()

            if ignore_mask:
                print("Warning: ignore_mask option not implemented for averaged data!")

            # Get max of timeseries
            tsmax = self.daily_ts.max(lim)

            dtype = [('depth', float), ('value', float), ('time', type(tsmax['time']))]
            maxGT = np.array(tsmax, dtype=dtype)

            # insert depths in first column
            maxGT['depth'] = np.array(self.depths)

            # Return sorted recarray
            return np.sort(maxGT, order='depth')
        else:
            # Get max from first logger
            maxGT = self.loggers[0].get_max_GT(lim, ignore_mask)

            # If more loggers, append data from these
            if len(self.loggers) > 1:
                for l in self.loggers[1:]:
                    maxGT = np.append(maxGT, l.get_max_GT(lim, ignore_mask))
            # Return sorted recarray
            return np.sort(maxGT, order='depth', axis=0)

    def get_MinGT(self, lim=None, fullts=False, ignore_mask=False):
        if not fullts:
            if self.daily_ts is None:
                self.calc_daily_avg()

            if ignore_mask:
                print("Warning: ignore_mask option not implemented for averaged data!")

            # Get max of timeseries
            tsmin = self.daily_ts.min(lim)

            dtype = [('depth', float), ('value', float), ('time', type(tsmin['time']))]
            minGT = np.array(tsmin, dtype=dtype)

            # insert depths in first column
            minGT['depth'] = np.array(self.depths)

            # Return sorted recarray
            return np.sort(minGT, order='depth')
        else:
            # Get min from first logger
            minGT = self.loggers[0].get_min_GT(lim, ignore_mask)

            # If more loggers, append data from these
            if len(self.loggers) > 1:
                for l in self.loggers[1:]:
                    minGT = np.append(minGT, l.get_min_GT(lim, ignore_mask))

            # Return sorted recarray
            return np.sort(minGT, order='depth', axis=0)

    def plot_trumpet(self, fullts=False, lim=None, end=None, nyears=None, \
                     args=None, plotMeanGT=True, fs=12, ignore_mask=False,
                     depths='all',
                     **kwargs):
        """
        Method to plot trumpet curve.

        Call signature:
        plot_trumpet(self,fullts=False,lim=None,args=None)

        fullts is a switch to use full time series instead of daily averages

        lim contains date limits (f.ex.: lim=['2008-09-01','2009-08-31'])

        args contains plotting properties for each curve - default settings:

        args = dict(
            plotMeanGT = plotMeanGT,
            maxGT = dict(
                linestyle  = '-',
                color      = 'k',
                marker     = '.',
                markersize = 7,
                zorder     = 5),
            minGT = dict(
                linestyle  = '-',
                color      = 'k',
                marker     = '.',
                markersize = 7,
                zorder     = 5),
            MeanGT = dict(
                linestyle  = '-',
                color      = 'k',
                marker     = '.',
                markersize = 7,
                zorder     = 5),
            fill = dict(
                linestyle  = '-',
                facecolor  = '0.7',
                edgecolor  = 'none',
                zorder     = 1),
            vline = dict(
                linestyle  = '--',
                color      = 'k',
                zorder     = 2),
            grid = dict(
                linestyle  = '-',
                color      = '0.8',
                zorder     = 0),
            title = self.name)

        Each dictionary of settings can be replaced by None, in order not to
            plot that element (e.g. grid = None)
        """

        # pdb.set_trace()

        defaultargs = dict(
            plotMeanGT=plotMeanGT,
            maxGT=dict(
                linestyle='-',
                color='k',
                marker='.',
                markersize=7,
                zorder=5),
            minGT=dict(
                linestyle='-',
                color='k',
                marker='.',
                markersize=7,
                zorder=5),
            MeanGT=dict(
                linestyle='-',
                color='k',
                marker='.',
                markersize=7,
                zorder=5),
            fill=dict(
                linestyle='solid',
                facecolor='0.7',
                edgecolor='none',
                zorder=1),
            vline=dict(
                linestyle='-.',
                color='k',
                zorder=2),
            grid=dict(
                linestyle='-',
                color='0.8',
                zorder=0),
            title=self.name)

        if args is None:
            args = dict()

        if 'plotMeanGT' not in args:
            args['plotMeanGT'] = plotMeanGT
        if 'title' in kwargs:
            args['title'] = kwargs.pop('title')
        if 'title' not in args:
            args['title'] = defaultargs['title']
        if 'maxGT' not in args:
            args['maxGT'] = defaultargs['maxGT']
        if 'minGT' not in args:
            args['minGT'] = defaultargs['minGT']
        if 'MeanGT' not in args:
            args['MeanGT'] = defaultargs['MeanGT']
        if 'fill' not in args:
            args['fill'] = defaultargs['fill']
        if 'vline' not in args:
            args['vline'] = defaultargs['vline']
        if 'grid' not in args:
            args['grid'] = defaultargs['grid']

        if lim is None and nyears is not None:
            if end is None:
                end = self.daily_ts.times[-1]
                if fullts:
                    print("plot end date is based on daily_ts not fullts")
            lim = nyears2lim(end, nyears)

        # prepare to filter on depths
        depth_arr = self.depths

        if (depths is None) or (type(depths) == str and depths.lower() == 'all'):
            depths = self.depths

        if not hasattr(depths, '__iter__'):
            depths = [depths]

        # get maximum values within time limits
        maxGT = self.get_MaxGT(lim, fullts, ignore_mask=ignore_mask)
        # get minimum values within time limits
        minGT = self.get_MinGT(lim, fullts, ignore_mask=ignore_mask)
        # remove any missing depths (NaN's)
        maxGT = maxGT[np.logical_not(np.isnan(maxGT['value']))]
        minGT = minGT[np.logical_not(np.isnan(minGT['value']))]

        # remove values above ground (z-axis is positive downwards)
        maxGT = maxGT[maxGT['depth'] >= 0]
        minGT = minGT[minGT['depth'] >= 0]

        if 'axes' in kwargs:
            ax = kwargs.pop('axes')
        elif 'Axes' in kwargs:
            ax = kwargs.pop('Axes')
        elif 'ax' in kwargs:
            ax = kwargs.pop('ax')
        else:
            fh = pylab.figure()
            ax = pylab.axes()

        # convert depths to column indices
        did = get_indices(maxGT['depth'], depths)
        did = [i for i in did if not np.isnan(i)]  # remove nan values

        ax.plot(maxGT['value'][did], maxGT['depth'][did], **args['maxGT'])  # ,'-k',marker='.', markersize=7,**kwargs)
        ax.hold(True)
        ax.plot(minGT['value'][did], minGT['depth'][did], **args['minGT'])  # ,'-k',marker='.', markersize=7,**kwargs)

        if 'fill' in args and args['fill'] is not None:
            # fclr = kwargs.pop('fill')
            # if type(fclr) != str and fclr != False:
            #    fclr = '0.7'
            thisX = np.append(maxGT['value'][did], minGT['value'][did][::-1])
            thisY = np.append(maxGT['depth'][did], minGT['depth'][did][::-1])
            ax.fill(thisX, thisY, **args['fill'])  # facecolor=fclr,edgecolor='none')

        ylim = pylab.get(ax, 'ylim')
        ax.set_ylim(ymax=min(ylim), ymin=max(ylim))
        # pylab.setp(ax, ylim=ylim[::-1])
        # ax.grid(True)

        if 'vline' in args and args['vline'] is not None:
            ax.axvline(x=0, **args['vline'])  # linestyle='--', color='k')

        if 'plotMeanGT' in args and args['plotMeanGT']:
            # get maximum values within time limits
            MeanGT = self.get_MeanGT(lim=lim, fullts=fullts, ignore_mask=ignore_mask)

            # remove any missing depths (NaN's)
            MeanGT = MeanGT[np.logical_not(np.isnan(MeanGT['value']))]

            # remove values above ground (z-axis is positive downwards)
            MeanGT = MeanGT[MeanGT['depth'] >= 0]

            # convert depths to column indices
            did = get_indices(MeanGT['depth'], depths)
            did = [i for i in did if not np.isnan(i)]  # remove nan values

            ax.plot(MeanGT['value'][did], MeanGT['depth'][did], **args['MeanGT'])  # '-k',marker='.', markersize=7)

        if 'grid' in args and args['grid'] is not None:
            ax.grid(True, **args['grid'])

        ax.get_xaxis().tick_top()
        ax.set_xlabel('Temperature [$^\circ$C]', fontsize=fs)
        ax.get_xaxis().set_label_position('top')
        ax.set_ylabel('Depth [m]', fontsize=fs)

        pylab.xticks(fontsize=fs)
        pylab.yticks(fontsize=fs)

        if args['title'] is not None:
            ax.text(0.95, 0.05, args['title'], horizontalalignment='right', \
                    verticalalignment='bottom', transform=ax.transAxes, fontsize=fs)
            # ax.set_title(args['title'])

        #pylab.show(block=False)
        ax.lim = lim
        return ax

    def plot_date(self, date, annotations=True, fs=12, **kwargs):
        if 'axes' in kwargs:
            ax = kwargs.pop('axes')
        elif 'Axes' in kwargs:
            ax = kwargs.pop('Axes')
        elif 'ax' in kwargs:
            ax = kwargs.pop('ax')
        else:
            fh = pylab.figure()
            ax = pylab.axes()

        # hstate = ax.ishold()

        id = find(self.daily_ts.times == date)

        if len(id) == 0:
            print("Date " + date.__str__() + "does not exist in data set")
            print("(This function accepts only a datetime.date object)")
            return False

        gid = find(np.array(self.depths) >= 0.)
        depths = np.take(self.depths, gid)
        data = self.daily_ts.data[id, gid].flatten()

        if 'color' not in kwargs:
            lh = ax.plot(data, depths, '-k', marker='.', markersize=7, **kwargs)
        else:
            lh = ax.plot(data, depths, '-', marker='.', markersize=7, **kwargs)

        lh[0].tag = 'gt'

        ax.axvline(x=0, linestyle=':', color='k')
        ylim = pylab.get(ax, 'ylim')
        ax.set_ylim(ymax=min(ylim), ymin=max(ylim))

        ax.get_xaxis().tick_top()
        ax.set_xlabel('Temperature [$^\circ$C]', fontsize=fs)
        ax.get_xaxis().set_label_position('top')
        ax.set_ylabel('Depth [m]', fontsize=fs)

        pylab.xticks(fontsize=fs)
        pylab.yticks(fontsize=fs)

        if annotations:
            t1h = ax.text(0.95, 0.10, self.name, horizontalalignment='right', \
                          verticalalignment='bottom', transform=ax.transAxes, fontsize=fs)
            t1h.tag = 'name'

            t2h = ax.text(0.95, 0.05, date, horizontalalignment='right', verticalalignment='bottom',
                          transform=ax.transAxes, fontsize=fs)
            t2h.tag = 'date'

        pylab.show(block=False)
        return ax

    # --- Problem in plot routine when plotting SIS2007-02... WHY?? --------
    # --- Problem apparently solved ?!

    def plot(self, depths=None, plotmasked=False, fullts=False, legend=True,
             annotations=True, lim=None, **kwargs):
        if 'axes' in kwargs:
            ax = kwargs.pop('axes')
        elif 'Axes' in kwargs:
            ax = kwargs.pop('Axes')
        elif 'ax' in kwargs:
            ax = kwargs.pop('ax')
        else:
            fh = pylab.figure()
            ax = pylab.axes()

        fh = ax.figure

        hstate = ax.ishold()

        # Flag to toggle wether full timeseries from the loggers is used or
        # the averaged daily timeseries.
        if not fullts:
            if self.daily_ts is None:
                self.calc_daily_avg()

            depth_arr = self.depths

            if (depths is None) or (type(depths) == str and depths.lower() == 'all'):
                depths = self.depths

            if not hasattr(depths, '__iter__'):
                depths = [depths]

            # following line is necessary in order for mpl not to connect
            # the points by a straight line.
            self.daily_ts.fill_missing_steps(tolerance=2)

            # pdb.set_trace()

            for cid, d in enumerate(depth_arr):
                if d in depths:

                    # get copies of the data and times
                    data = self.daily_ts.limit(lim).data[:, cid].copy()
                    times = self.daily_ts.limit(lim).times.copy()

                    ax.hold(hstate)

                    # Option to plot also the masked data points, but keeping
                    # a mask on anything that could not be a real temperature.
                    if plotmasked:
                        mask = np.zeros(data.shape, dtype=bool)
                        mask = np.where(data < -273.15, True, False)
                        data.mask = mask
                    if any(data.mask == False):
                        lh = ax.plot_date(times, data, '-', label="%.2f m" % d, picker=5, **kwargs)
                        lh[0].tag = 'dataseries'

                        hstate = True
        else:
            depth_arr = self.get_depths(return_indices=True)

            if (depths is None) or (type(depths) == str and depths.lower() == 'all'):
                depths = depth_arr['depth']

            for d, lid, cid in depth_arr:
                if d in depths:
                    # following line is necessary in order for mpl not to connect
                    # the points by a straight line.
                    self.loggers[lid].timeseries.fill_missing_steps(tolerance=2)

                    # get copies of the data and times
                    data = self.loggers[lid].timeseries.data[:, cid].copy()
                    times = self.loggers[lid].timeseries.times.copy()

                    ax.hold(hstate)

                    # Option to plot also the masked data points, but keeping
                    # a mask on anything that could not be a real temperature.
                    if plotmasked:
                        mask = np.zeros(data.shape, dtype=bool)
                        mask = np.where(data < -273.15, True, False)
                        data.mask = mask
                    lh = ax.plot_date(times, data, '-', label="%.2f m" % d, picker=5, **kwargs)
                    lh[0].tag = 'dataseries'
                    hstate = True
        # end if

        if legend:
            ax.legend(loc='best')

        if annotations:
            if ax.get_title() == '':
                ax.set_title(self.name)
            else:
                ax.set_title(ax.get_title() + ' + ' + self.name)

            ax.set_ylabel('Temperature [$^\circ$C]')
            ax.set_xlabel('Time')

        # fh.canvas.mpl_connect('pick_event', onpick)

        fh.axcallbacks = zoom_span.AxesCallbacks(ax)

        pylab.show(block=False)

        return pylab.gca()

    def plot_ALT(self, fullts=False, legend=True,
                 annotations=True, lim=None, figsize=(15, 6),
                 label=None, year_base=5, **kwargs):

        figBG = 'w'  # the figure background color
        axesBG = '#ffffff'  # the axies background color
        textsize = 8  # size for axes text
        left, width = 0.1, 0.8
        rect1 = [left, 0.2, width, 0.6]  # left, bottom, width, height

        if 'axes' in kwargs:
            ax = kwargs.pop('axes')
        elif 'Axes' in kwargs:
            ax = kwargs.pop('Axes')
        elif 'ax' in kwargs:
            ax = kwargs.pop('ax')
        else:
            fh = pylab.figure(figsize=figsize, facecolor=figBG)
            ax = pylab.axes(rect1, axisbg=axesBG)

        fh = ax.figure

        hstate = ax.ishold()

        # Flag to toggle wether full timeseries from the loggers is used or
        # the averaged daily timeseries.
        if not fullts:
            if self.daily_ts is None:
                self.calc_daily_avg()

            (Yr, maxdepth, thaw_depth_p, thaw_depth_n, ALT_date) = \
                self.calc_ALT(lim=lim, fullts=False, silent=False)

            dates = np.array([dt.date(y, 1, 1) for y in Yr])
            # dates = np.append(dates, dt.date(dates[-1].year,12,31))
            # maxdepth = np.append(maxdepth,maxdepth[-1])
            # ds = 'steps-post'

            ds = 'default'

            ax.plot(dates, maxdepth, drawstyle=ds, label=label, **kwargs)
            ax.hold(hstate)

            ax.xaxis.set_major_locator(mpl.dates.YearLocator(base=year_base))
            fh.autofmt_xdate()
        else:
            raise NotImplementedError('Full timeseries support not implemented!')
        # end if


        if legend:
            ax.legend(loc='best')

        if annotations:
            if ax.get_title() == '':
                ax.set_title(self.name)
            else:
                ax.set_title(ax.get_title() + ' + ' + self.name)

            ax.set_ylabel('ALT [m]')
            ax.set_xlabel('Time [year]')

        pylab.draw()

        return pylab.gca()

    def get_dates(self, datetype=None):
        """
        Returns a sorted (ascending) array of dates for which the borehole contains data.

        datetype is a switch to indicate which dates are included in the list:

        datetype='any':     The date array will contain all dates for which at least
                               one datalogger has data for at least one channel.
        datetype='all':     The date array will contain all dates for which all
                               dataloggers have data for all channels.
        datetype='masked':  The date array will contain all dates for which any
                               datalogger has a masked or unmasked entry.
        datetype='missing': The date array will contain all dates for which no
                               datalogger has an entry (masked or unmasked).
        """

        dates = np.array([])
        for lid, l in enumerate(self.loggers):
            if datetype in [None, 'masked']:
                times = l.timeseries.times.copy()
                if l.timeseries.dt_type != dt.date:
                    for did, d in enumerate(times): times[did] = d.date()
                dates = np.append(dates, times)
            else:
                raise 'only the default behaviour (''masked'') has been implemented!'

        return np.unique(dates)

    def uncalibrate(self):
        for l in self.loggers:
            l.uncalibrate()

        self.calc_daily_avg()

    def calibrate(self):
        for l in self.loggers:
            l.calibrate()

        self.calc_daily_avg()

    def export(self, fname, delim=', '):
        """
        This method should take a filename as input and export the ground temperature data
        from the borehole as a textfile.
        """

        daily_flag = True

        if self.daily_ts is None:
            self.calc_daily_avg(mindata=8)
            daily_flag = False

        mask = np.ma.getmaskarray(self.daily_ts.data)

        fid = open(fname, 'w')
        try:
            fid.write('Borehole:\t%s\n' % self.name)
            fid.write('Latitude:\t%s\n' % self.latitude)
            fid.write('Longitude:\t%s\n' % self.longitude)
            fid.write('Description:\t%s\n' % self.description)
            fid.write('Depths:\n')

            titles = '      Time'
            for d in self.depths:
                titles += '%s%7.2f' % (delim, d)
            fid.write(titles + '\n')
            fid.write('-' * len(titles) + '\n')

            for rid, t in enumerate(self.daily_ts.times):
                outstr = '%s' % t
                for cid, e in enumerate(self.daily_ts.data[rid, :]):
                    if mask[rid, cid]:
                        outstr += delim + '    ---'
                    else:
                        outstr += '%s%7.2f' % (delim, e)
                outstr += '\n'
                fid.write(outstr)
        finally:
            fid.close()

        if daily_flag == False:
            self.daily_ts = None

    def generate_climate_series(self, loggerid, channelid):
        clim = ClimateData.climate_station(name=self.name, lat=self.latitude, lon=self.longitude)
        paramsd = dict( \
            station=self.name,
            datatype='AT',
            mtype=None)

        clim.add_data( \
            data=self.loggers[loggerid].timeseries.data[:, channelid], \
            times=self.loggers[loggerid].timeseries.times, \
            paramsd=paramsd)

        return clim

    def reduce_daily_ts_cols(self):
        """Reduces daily_ts timeseries columns in the event that there are 
        multiple columns associated with the same depths. This may occur if
        a logger has been replaced, so that two different loggers (SN) have
        recorded the same thermistor during different periods.
        """
        depths = np.unique(self.depths)

        keep_col_ids = []

        if len(depths) < len(self.depths):
            for d in depths:
                ids = find(self.depths == d)
                keep_col_ids.append(ids[0])  # make sure we always keep one column
                if len(ids) > 1:
                    # there are several columns for this depth... join them!
                    for col_id in ids[1:]:
                        # copy only unmasked data
                        copy_rows = np.logical_not(self.daily_ts.data.mask[:, col_id])
                        self.daily_ts.data[copy_rows, ids[0]] = self.daily_ts.data[copy_rows, col_id]

        # now keep only the first column for each depth
        self.daily_ts.data = self.daily_ts.data[:, keep_col_ids]
        self.depths = depths


def float_eq(alike, val, atol=1e-8, rtol=1e-5):
    """makes a float comparison allowing for round off errors to within the
    speciefied absolute and relative tolerances

    """

    return np.abs(np.array(alike) - val) <= (atol + rtol * np.abs(val))


def get_indices(alike1, alike2, float_comp=True, atol=1e-8, rtol=1e-5):
    """Finds the indices of the values in alike2 as they occur in alike1.
    If a value from alike2 does not occur in alike1, the index will be a np.NaN
    value.

    """

    idx = [np.NaN for a in alike2]

    if float_comp:
        for idv, val in enumerate(alike2):
            id = np.nonzero(float_eq(alike1, val, atol, rtol))[0]
            if len(id) > 0:
                idx[idv] = id[0]

        return idx

        # return [float_eq(alike1, val, atol, rtol) for val in alike2]
    else:
        raise NotImplementedError


def plot_surf(bh, depths=None, ignore_mask=False, fullts=False, legend=True,
              annotations=True, lim=None, figsize=(15, 6),
              cmap=plt.cm.bwr, sensor_depths=True, cax=None,
              cont_levels=[0], **kwargs):
    # To allow for easy integration into the borehole class at later stage
    self = bh

    figBG = 'w'  # the figure background color
    axesBG = '#ffffff'  # the axies background color
    textsize = 8  # size for axes text
    left, width = 0.1, 0.8
    rect1 = [left, 0.2, width, 0.6]  # left, bottom, width, height

    fh = None

    if 'axes' in kwargs:
        ax = kwargs.pop('axes')
    elif 'Axes' in kwargs:
        ax = kwargs.pop('Axes')
    elif 'ax' in kwargs:
        ax = kwargs.pop('ax')
    else:
        fh = pylab.figure(figsize=figsize, facecolor=figBG)
        ax = pylab.axes(rect1)
        ax.set_facecolor(axesBG)

    if fh is None:
        fh = ax.get_figure()

    hstate = ax.ishold()

    # Flag to toggle wether full timeseries from the loggers is used or
    # the averaged daily timeseries.
    if not fullts:
        if self.daily_ts is None:
            self.calc_daily_avg()

        depth_arr = self.depths

        if (depths is None) or (type(depths) == str and depths.lower() == 'all'):
            depths = list(np.array(self.depths)[np.array(self.depths) >= 0])

        if not hasattr(depths, '__iter__'):
            depths = [depths]

        # convert depths to column indices
        did = get_indices(self.depths, depths)

        # following line is necessary in order for mpl not to connect
        # the points by a straight line.
        self.daily_ts.fill_missing_steps(tolerance=2)

        # get copies of the data and times
        data = self.daily_ts.limit(lim).data[:, did].copy()
        ordinals = self.daily_ts.limit(lim).ordinals.copy()

        if ignore_mask:
            mask = np.zeros(data.shape, dtype=bool)
            mask = np.where(data < -273.15, True, False)
            data.mask = mask

        # Find the maximum and minimum temperatures, and round up/down
        mxn = np.max(np.abs([np.floor(data.min()),
                             np.ceil(data.max())]))
        levels = np.arange(-mxn, mxn + 1)

        xx, yy = np.meshgrid(ordinals, depths)

        cf = ax.contourf(xx, yy, data.T, levels, cmap=cmap)
        ax.hold(True)
        if cont_levels is not None:
            ct = ax.contour(xx, yy, data.T, cont_levels, colors='k')
            cl = ax.clabel(ct, cont_levels, inline=True, fmt='%1.1f $^\circ$C', fontsize=12, colors='k')

        if sensor_depths:
            xlim = ax.get_xlim()
            ax.plot(np.ones_like(depths) * xlim[0] + 0.01 * np.diff(xlim), depths,
                    'ko', ms=3)

        ax.invert_yaxis()
        ax.hold(hstate)
        ax.xaxis_date()

        cbax = plt.colorbar(cf, orientation='horizontal', ax=cax, shrink=1.0)
        cbax.set_label('Temperature [$^\circ$C]')
        fh.autofmt_xdate()
    else:
        raise NotImplementedError('Full timeseries support not implemented!')
    # end if

    if legend:
        ax.legend(loc='best')

    if annotations:
        if ax.get_title() == '':
            ax.set_title(self.name)
        else:
            ax.set_title(ax.get_title() + ' + ' + self.name)

        ax.set_ylabel('Depth [m]')
        # ax.set_xlabel('Time [year]')

    #pylab.draw()

    return pylab.gca()


def plot_surf_trumpet(bh, depths=None, ignore_mask=False, fullts=False, legend=True,
                      annotations=True, lim=None, figsize=(15, 6),
                      cmap=plt.cm.bwr, sensor_depths=True,
                      cont_levels=[0], **kwargs):
    figBG = 'w'  # the figure background color
    axesBG = '#ffffff'  # the axies background color
    textsize = 8  # size for axes text
    left, width = 0.1, 0.8
    rect1 = [left, 0.2, width, 0.6]  # left, bottom, width, height

    fh = plt.figure(figsize=figsize, facecolor=figBG)

    ax1 = plt.subplot2grid((5, 4), (0, 0), colspan=3, rowspan=4, axisbg=axesBG)
    ax2 = plt.subplot2grid((5, 4), (0, 3), rowspan=4, axisbg=axesBG)
    ax1.get_shared_y_axes().join(ax1, ax1)
    ax3 = plt.subplot2grid((5, 4), (4, 0), colspan=3, axisbg=axesBG)

    plot_surf(bh, depths=depths, ignore_mask=ignore_mask, fullts=False,
              legend=legend, annotations=annotations, lim=lim,
              cmap=cmap, sensor_depths=sensor_depths, ax=ax1, cax=ax3,
              cont_levels=cont_levels)
    bh.plot_trumpet(ax=ax2, ignore_mask=ignore_mask, fullts=fullts, depths=depths,
                    nyears=1)

    ax3.set_visible(False)

    ax2.set_ylim(ax1.get_ylim())
    ax2.set_ylabel('')


def read_boreholes(path=basepath, walk=True, bholes=None, calc_daily=True, \
                   names=None, auto_process=True, read_pkl=True,
                   name_contains_str=None, name_starts_with=None):
    """
    bholes = read_boreholes(path=basepath,walk=True, bholes=None, calc_daily=True, names=None)

    Import borehole information into dictionary bholes.

    input parameters:
    path:        Directory from which to import. Defaults to basepath, defined
                    on module level.
    walk:        Travers subdirectories if True (Default)
    bholes:      Optional existing dictionary to append with borehole information
    calc_daily:  Calculate daily timeseries if True (Default)
    names:       List of boreholes to import. Borehole names must mach entries
                    in this list in order to be imported (exact match).
    read_pkl:    If False, will read ascii files only
                 If True will read pickled file if available,
                   otherwise ascii files (default)
                 If "only", will read only pickled files
    name_contains_str: Borehole will be read if borehole name contains the
                   string passed
    name_contains_str: Borehole will be read if borehole name contains the
                   string passed
    name_starts_with: Borehole will be read if borehole name starts with the
                   string passed
    """

    # ..........................................................................
    def process_files(bholes, dr, flst):

        entries_to_skip = list()
        # First create borehole entries for any .bor file in the directory
        # and find entries in the list that should be removed

        if bholes['read_pkl'] in [True, 'only', 'Only']:
            read_pkl = True
        else:
            read_pkl = False

        if bholes['read_pkl'] in [True, False]:
            read_ascii = True
        else:
            read_ascii = False

        #        print "looking for pickled boreholes..."
        for id, f in enumerate(flst):
            tmphole = None
            f2 = f.lower()
            # Is it a pickled file? ...
            if f2.endswith('.pkl') and read_pkl:
                # print f2
                try:
                    with open(os.path.join(dr, f), 'rb') as pklfh:
                        tmphole = pickle.load(pklfh)
                except:
                    print("Could not read file: {0}".format(os.path.join(dr, f)))
                    tmphole = None

                if isinstance(tmphole, borehole):
                    # Here we should test for the following:
                    # 1) Is bholes['read_all'] == True?
                    # 2) Does the name start with bholes['starts_with']?
                    # 3) Does the name contain bholes['contains']?
                    # 4) Is the name mentioned in the bholes['names'] list?
                    if bholes['read_all'] or \
                            (bholes['names'] is not None and tmphole.name in bholes['names']) or \
                            (bholes['starts_with'] is not None and tmphole.name.startswith(bholes['starts_with'])) or \
                            (bholes['contains'] is not None and bholes['contains'] in tmphole.name):
                        bholes[tmphole.name] = tmphole
                        bholes['pkl_names'].append(tmphole.name)
                        print('\n**************************************************************\n')
                        print('Borehole ' + tmphole.name + ' created from pickled file: ' + dr)
                    else:
                        print('Borehole ' + tmphole.name + ' found but not added.')

                    #        print "looking for bor files..."
        for id, f in enumerate(flst):
            f2 = f.lower()
            # ...or is it a borehole definition file?
            if f2.endswith('.bor') and read_ascii:
                tmphole = borehole(fname=os.path.join(dr, f))
                if bholes['read_all'] or \
                        (bholes['names'] is not None and tmphole.name in bholes['names']) or \
                        (bholes['starts_with'] is not None and tmphole.name.startswith(bholes['starts_with'])) or \
                                (bholes['contains'] is not None and bholes['contains'] in tmphole.name) and \
                                not tmphole.name in bholes:
                    bholes[tmphole.name] = tmphole
                    print('\n**************************************************************\n')
                    print('Borehole ' + tmphole.name + ' created from: ' + dr)

            # ...or is it a directory with certain names
            elif os.path.isdir(os.path.join(dr, f)) and \
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

        # Then load data into the boreholes
        for f in flst:
            f2 = f.lower()
            if (f2.endswith(('.csv', '.txt', '.dat'))) and read_ascii:
                # if "SIS2007-02" in dr:
                #    print f
                #    pdb.set_trace()
                fullf = os.path.join(dr, f)
                if os.path.islink(fullf): continue  # don't count linked files

                paramsd, comments, hlines = get_property_info(fullf)
                if 'borehole' in paramsd:
                    if (bholes['read_all'] or \
                                (bholes['names'] is not None and paramsd['borehole'] in bholes['names']) or \
                                (bholes['starts_with'] is not None and paramsd['borehole'].startswith(
                                    bholes['starts_with'])) or \
                                (bholes['contains'] is not None and bholes['contains'] in paramsd['borehole'])) and \
                                    paramsd['borehole'] not in bholes['pkl_names']:

                        if paramsd['borehole'] in list(bholes.keys()):
                            stat = bholes[paramsd['borehole']].add_data(fullf, paramsd=paramsd, hlines=hlines)
                            if stat:
                                print(f + " added to borehole " + paramsd['borehole'])
                            else:
                                print("NB!" + f + " could not be added to borehole " + paramsd['borehole'])
                                pdb.set_trace()
                        else:
                            print("NB!" + f + " tries to reference non-exsisting " + \
                                  "borehole " + paramsd['borehole'])

    # ..........................................................................

    if type(path) in [list, tuple]:
        # Concatenate path elements
        path = os.path.join(*path)

    if type(bholes) != dict:
        bholes = dict()

    existing_holes = list(bholes.keys())
    bholes['names'] = names
    bholes['read_pkl'] = read_pkl
    bholes['pkl_names'] = []
    bholes['contains'] = name_contains_str
    bholes['starts_with'] = name_starts_with

    if names is None and name_contains_str is None and name_starts_with is None:
        bholes['read_all'] = True
    else:
        bholes['read_all'] = False

    if not walk:
        # Process only files in the current directory (path)
        flst = os.listdir(path)
        process_files(bholes, path, flst)
    else:
        # Process all files in the directory tree
        os.path.walk(path, process_files, bholes)

    pkl_names = bholes['pkl_names']
    del bholes['names']
    del bholes['read_pkl']
    del bholes['pkl_names']
    del bholes['contains']
    del bholes['starts_with']
    del bholes['read_all']

    print('\n**************************************************************\n')

    if auto_process:
        changes = False
        for bh in bholes:
            if bh not in pkl_names and bh not in existing_holes:
                bholes[bh].auto_process()
                changes = True
        if changes:
            print('\n**************************************************************\n')

    if calc_daily:
        for bh in bholes:
            if bh not in pkl_names and bh not in existing_holes:
                print('Calculating daily timeseries for borehole ' + bh)
                bholes[bh].calc_daily_avg()

    return bholes


def new_borehole(name=None, path=None, loggers=1):
    """
    new_borehole(name=None, path=basepath, loggers=1)

    Set up directory and file-structure for a new borehole. Files will be set
    up with borehole-name, and logger name, and possibly logger-type specific
    information, if loggertype is provided.

    input parameters:
    name:        Name of new borehole (as well as directory if none given)
    path:        Directory in which to create new files.
                   If directory does not exist, it will be created
                   If directory exist, no new directory will be created
                   If path is not a full path it will be appended to the basepath

    loggers:     Number of loggers in borehole (and number of files to write)
                   or a list of logger types, with length = # of loggers

    logger types are:
    "tinytag"         data#, times, data colums, flag column (optional)
    "hobo u12-008"    data#, times, data colums, Hobo event columns, flag column (optional)
    "generic"         times, data colums, flag column (optional)

    """

    if name is None:
        name = 'new_borehole'

    if path is None:
        # if path is not given, use basepath + borehole name
        path = os.path.join(basepath, name)
    elif type(path) in [list, tuple]:
        # Concatenate path elements
        path = os.path.join(*path)

    if os.path.splitdrive(path)[0] == '':
        # If path is not an absolute path, append to basepath
        path = os.path.join(basepath, path)

    if not os.path.exists(path):
        os.makedirs(path)

    # Now write files

    # Write borehole definition file
    bor_name = '{0}'.format(name)
    fullfile = os.path.join(path, bor_name + '.bor')
    copy_id = 0
    while os.path.exists(fullfile):
        copy_id += 1
        bor_name = '{0}_{1:03d}.bor'.format(name, copy_id)
        fullfile = os.path.join(path, bor_name + '.bor')

    bor_file = [
        "Name:        %s" % (bor_name,),
        "Date:        YYYY-MM-DD    # Date the borehoe was drilled",
        "Latitude:    NN.NNNNN      # in decimal degrees (dd.dddd)",
        "Longitude:   -EE.EEEEE     # in decimal degrees (dd.dddd)",
        "Description: One line description."]

    with open(fullfile, 'w') as f:
        f.write("\n".join(bor_file))

    # Write logger files

    if type(loggers) not in [list, tuple]:
        if type(loggers) == str:
            loggers = [loggers]
        else:
            loggers = ['generic' for n in range(loggers)]

    for id, ltype in enumerate(loggers):
        if ltype.lower() == "tinytag":
            # data#, times, data colums, flag column (optional)
            logger_file = [
                "Borehole:    {0}".format(bor_name),
                "SN:          XXXXXXX                    # Logger serial number",
                "Type:        TinyTag+12 -40/125 degC    # Type of logger",
                "Channels:    2                          # Number of channels",
                "tzinfo:      UTC-0X:00                  # Time zone setting of logger",
                "Depth:       Z.ZZ, Z.ZZ                 # installation depth of sensors (in meters)",
                "Calibration: X.XX, X.XX                 # Calibration values, T_true = T_data + cal",
                "Heave:       0.000                      # heave of sensors since installation (in meters)",
                "Description: One line description of borehole.",
                "Separator:   ;                          # column separator in this file (defult is ,)",
                "Decimal:     .                          # decimal symbol (default is .)",
                "Flagcol:     4                          # column number of flag column (zero-based), default is last column",
                "Columns:     id, time, temp, temp, flag",
                "# *** THIS IS FOR 2-CHANNEL TinyTag Logger ***",
                "#-------------------------------------------------------------------------------",
                "1; 2000-01-01 12:00:00; 0,000; 0,000; m        # m = entire row is masked out. m[0,2] = data column 0 and 2 masked out",
                "# When inserting data, check and possibly correct:",
                "#  - decimal separator in file",
                "#  - column separator in file",
                "#  - columns match the column headers",
                "#  - flag column specification"]
        elif ltype.lower() == "hobo u12-008":
            # data#, times, data colums, Hobo event columns, flag column (optional)
            logger_file = [
                "Borehole:    {0}".format(bor_name),
                "SN:          XXXXXXX                    # Logger serial number",
                "Type:        HOBO U12-008               # Type of logger",
                "Channels:    4                          # Number of channels",
                "tzinfo:      UTC-0X:00                  # Time zone setting of logger",
                "Depth:       Z.ZZ, Z.ZZ, Z.ZZ, Z.ZZ     # installation depth of sensors (in meters)",
                "Calibration: X.XX, X.XX, X.XX, X.XX     # Calibration values, T_true = T_data + cal",
                "Heave:       0.000                      # heave of sensors since installation (in meters)",
                "Description: One line description of borehole.",
                "Separator:   ;                          # column separator in this file (defult is ,)",
                "Decimal:     ,                          # decimal symbol (default is .)",
                "Flagcol:     9                          # column number of flag column (zero-based)",
                "columns:     id, time, temp, temp, temp, temp, batt V, Host connected, End of file, flag",
                "#-------------------------------------------------------------------------------",
                "1; 2000-01-01 12:00:00; 0,000; 0,000; 0,000; 0,000; 3,17; ; ; m        # m = entire row is masked out. m[0,2] = data column 0 and 2 masked out",
                "# When inserting data, check and possibly correct:",
                "#  - decimal separator in file",
                "#  - column separator in file",
                "#  - columns match the column headers",
                "#  - flag column specification"]
        elif ltype.lower() == "generic":
            # times, data colums, flag column (optional)
            logger_file = [
                "Borehole:    {0}".format(bor_name),
                "SN:          XXXXXXX                    # Logger serial number",
                "Type:        Generic                    # Type of logger",
                "Channels:    XX                         # Number of channels",
                "tzinfo:      UTC-0X:00                  # Time zone setting of logger",
                "Depth:       Z.ZZ, Z.ZZ, Z.ZZ, Z.ZZ     # installation depth of sensors (in meters)",
                "Calibration: X.XX, X.XX, X.XX, X.XX     # Calibration values, T_true = T_data + cal",
                "Heave:       0.000                      # heave of sensors since installation (in meters)",
                "Description: One line description of borehole.",
                "Separator:   ;                          # column separator in this file (defult is ,)",
                "Decimal:     ,                          # decimal symbol (default is .)",
                "Flagcol:     X                          # column number of flag column (zero-based), default is last column",
                "Columns:     time, temp, temp, temp, temp, flag",
                "#-------------------------------------------------------------------------------",
                "2000-01-01 12:00:00; 0,000; 0,000; 0,000; 0,000; m        # m = entire row is masked out. m[0,2] = data column 0 and 2 masked out",
                "# When inserting data, check and possibly correct:",
                "#  - decimal separator in file",
                "#  - column separator in file",
                "#  - columns match the column headers",
                "#  - flag column specification"]
        else:
            print("Unknown logger type: " + ltype)
            return False

        fullfile = os.path.join(path, 'logger_{0:d}.txt'.format(id))
        copy_id = 0
        while os.path.exists(fullfile):
            copy_id += 1
            fullfile = os.path.join(path, 'logger_{0:d}_{1:03d}.txt'.format(id, copy_id))

        with open(fullfile, 'w') as f:
            f.write("\n".join(logger_file))

        # Write borehole definition file
        fullfile = os.path.join(path, 'auto_processing.py')

        auto_processing_file = [
            "def auto_process(bhole):",
            "    associated_borehole = '{0}'".format(bor_name),
            "    if bhole.name != associated_borehole:",
            "        print 'This is not borehole {0}'.format(associated_borehole)",
            "        print 'Wrong auto processing file!'",
            "        print 'Nothing done...!'",
            "        return",
            "    print 'Auto processing borehole {0}'.format(bhole.name)",
            "",
            "    # You may mask certain channels:",
            "    # bhole.mask_channels(logger_sn='XXXXXX',channels=[18,19,20,21])",
            "    # print 'Channels were masked!'",
            "    # logger_sn is optional if there is only one logger",
            "",
            "    # You may wish to uncalibrate the data, in case calibrations are bad:",
            "    # bhole.uncalibrate()",
            "    # print 'borehole calibration was removed!'",
            "    # logger_sn is optional if there is only one logger",
            "",
            "    # Plot temperatures as function of time for all sensors:",
            "    bhole.plot()",
            "",
            "    # Plot trumpet curve for entire time span of logger:",
            "    bhole.plot_trumpet(title=None)",
            "    # Use additional argument lim=['1968-01-01','1974-01-01'] to plot only",
            "    # data from the specified interval, or use nyears=1, end='2010-01-01'",
            "    # to plot data from one full year ending on 2010-01-01"]

    with open(fullfile, 'w') as f:
        f.write("\n".join(auto_processing_file))


def pickle_all(bholes):
    for name, bh in list(bholes.items()):
        bh.pickle()
        print("Pickled borehole {0}".format(name))


def combine_boreholes(bh1, bh2, drop_channels=None, calc_daily=True):
    """
    Combine boreholes bh1 and bh2 (borehole instances) into a new borehole.

    drop_channels dictionary allows automatic deletion of certain channels of
    certain loggers. The dictionary uses logger serialnumber as key and a list
    of channel numbers as value. Example:

    drop_channels = {'1104196': [1,2,3]}

    will delete channels 1, 2 and 3 from the combined borehole (but leave the
    original boreholes intact).
    """

    bh3 = copy.deepcopy(bh1)
    for l in bh2.loggers:
        bh3.loggers.append(copy.deepcopy(l))

    if drop_channels is not None and type(drop_channels) == dict:
        for sn, chns in list(drop_channels.items()):
            bh3.drop_channels(logger_sn=sn, channels=chns)

    if calc_daily:
        bh3.calc_daily_avg()

    return bh3


def filter_dict(adict, predicate=lambda k, v: True):
    """
    Function to select subset of keys in a dictionary.

    Usage examples:

    bh.filter_dict(bholes, lambda k, v: k not in ['SIS OBO Waterworks','SIS67036'])
    bh.filter_dict(bholes, lambda k, v: not k.startswith('NUK'))
    bh.filter_dict(bholes, lambda k, v: not k.startswith('ITI'))
    """

    def _filter(adict, predicate=lambda k, v: True):
        for k, v in list(adict.items()):
            if predicate(k, v):
                yield k, adict[k]

    return dict(_filter(adict, predicate=predicate))


def filter_bhole_dict(bholes, predicate=lambda k, v: True):
    """
    Function to select subset of borholes dictionary.

    Usage examples:

    bh.filter_bhole_dict(bholes, lambda k, v: k not in ['SIS OBO Waterworks','SIS67036'])
    bh.filter_bhole_dict(bholes, lambda k, v: not k.startswith('NUK'))
    bh.filter_bhole_dict(bholes, lambda k, v: not k.startswith('ITI'))
    """
    warnings.warn('filter_borehole_dict is replaced by filter_dict.', warnings.DeprecationWarning)
    return filter_dict(bholes, predicate=predicate)


# test_data = [{"key1":"value1", "key2":"value2"}, {"key1":"blabla"}, {"key1":"value1", "eh":"uh"}]
# list(filter_data(test_data, lambda k, v: k == "key1" and v == "value1"))

# --- ### Plotting routines ###

def plot_grl_boreholes(resolution='i', corners=None, bholes=None, plotMeanGT=False):
    if bholes is None:
        bholes = read_boreholes(
            os.path.join(basepath, 'Sisimiut', 'SIS2007-06_West_Palsa', 'SIS2007-06_West_Palsa_Shallow'),
            calc_daily=False)
        bholes = read_boreholes(
            os.path.join(basepath, 'Sisimiut', 'SIS2007-06_West_Palsa', 'SIS2007-06a_West_Palsa_Deep'), bholes=bholes,
            calc_daily=False)
        bholes = read_boreholes(os.path.join(basepath, 'Ilulissat', 'ILU2007-01_78020'), bholes=bholes,
                                calc_daily=False)
        bholes = read_boreholes(os.path.join(basepath, 'Kangerlussuaq', 'KAN2005-01'), bholes=bholes, calc_daily=False)
        bholes = read_boreholes(os.path.join(basepath, 'Nuuk', 'NUK04008-808'), bholes=bholes, calc_daily=True)

    bholes['KAN2005-01'].uncalibrate()
    bholes['KAN2005-01'].calc_daily_avg(mindata=8)

    if 'SIS2007-06_combi' not in bholes:
        bholes['SIS2007-06_combi'] = combine_boreholes(
            bholes['SIS2007-06'],
            bholes['SIS2007-06a'],
            drop_channels={'1104196': [1, 2, 3]})

        # bholes['SIS2007-06_combi'] = copy.deepcopy(bholes['SIS2007-06'])
        # for l in bholes['SIS2007-06a'].loggers:
        #    bholes['SIS2007-06_combi'].loggers.append(copy.deepcopy(l))
        # bholes['SIS2007-06_combi'].drop_channels(logger_sn='1104196',channels=[1,2,3])
        # bholes['SIS2007-06_combi'].calc_daily_avg()

    figBG = 'w'
    fh = plt.figure(figsize=(20., 10.2), facecolor=figBG)
    fh.set_size_inches(20., 10.2, forward=True)
    map_ax = plt.subplot(1, 3, 3)
    ax1 = plt.subplot(2, 3, 1)
    ax2 = plt.subplot(2, 3, 2)
    ax3 = plt.subplot(2, 3, 4)
    ax4 = plt.subplot(2, 3, 5)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.4, hspace=None)

    if corners is None:
        corners = [[-55.0, 63.0], [-46.0, 71.0]]
    myEDC = mapping.maps.EDCbasemap(resolution=resolution, corners=corners, gdist=5.,
                                    colors='nice', ax=map_ax)

    mapping.plot.PlotIcecap(basemap=myEDC.map, ax=myEDC.ax)

    print('Drawing cities...')
    myEDC.GreenlandCities = {
        'Ilulissat': np.array((-51.100000, 69.216667)),
        'Sisimiut': np.array((-53.669284, 66.933628)),
        'Kangerlussuaq': np.array((-50.709167, 67.010556)),
        'Nuuk': np.array((-51.742632, 64.170284))}

    myEDC.boreholes = bholes

    myEDC.plotPoints(np.array(list(myEDC.GreenlandCities.values()))[:, 0],
                     np.array(list(myEDC.GreenlandCities.values()))[:, 1], 'or', ms=8,
                     picker=8, names=list(myEDC.GreenlandCities.keys()), tag='cities')

    x, y = myEDC.map(myEDC.GreenlandCities['Ilulissat'][0], myEDC.GreenlandCities['Ilulissat'][1])
    map_ax.text(x, y, 'Ilulissat .', fontsize=14, fontweight='bold',
                horizontalalignment='right', verticalalignment='top', zorder=100)
    x, y = myEDC.map(myEDC.GreenlandCities['Sisimiut'][0], myEDC.GreenlandCities['Sisimiut'][1])
    map_ax.text(x, y, '  Sisimiut', fontsize=14, fontweight='bold',
                horizontalalignment='left', verticalalignment='bottom', zorder=100)
    x, y = myEDC.map(myEDC.GreenlandCities['Kangerlussuaq'][0], myEDC.GreenlandCities['Kangerlussuaq'][1])
    map_ax.text(x, y, '  Kangerlussuaq', fontsize=14, fontweight='bold',
                horizontalalignment='left', verticalalignment='top', zorder=100)
    x, y = myEDC.map(myEDC.GreenlandCities['Nuuk'][0], myEDC.GreenlandCities['Nuuk'][1])
    map_ax.text(x, y, 'Nuuk .', fontsize=14, fontweight='bold',
                horizontalalignment='right', verticalalignment='top', zorder=100)

    print('Plotting trumpets...')
    bholes['KAN2005-01'].uncalibrate()  # Calibration for this borehole is not correct,
    # and data should be presented uncalibrated.

    print("Finished uncalibrating")

    args = {'title': None}
    # bholes['ILU2007-03'].plot_trumpet(lim=['2007-12-01','2009-12-01'],ax1)
    bholes['ILU2007-01'].plot_trumpet(lim=['2007-01-01', '2009-12-31'],
                                      ax=ax1, plotMeanGT=plotMeanGT, args=args)
    print("plotted first trumpet")
    bholes['SIS2007-06_combi'].plot_trumpet(lim=['2008-01-01', '2009-12-31'],
                                            ax=ax2, plotMeanGT=plotMeanGT, args=args)
    bholes['KAN2005-01'].plot_trumpet(lim=['2007-01-01', '2009-12-31'],
                                      ax=ax3, plotMeanGT=plotMeanGT, args=args)
    bholes['NUK04008-808a+b'].plot_trumpet(lim=['2007-01-01', '2009-12-31'],
                                           ax=ax4, plotMeanGT=plotMeanGT,
                                           args=args)  # ,lim=['2007-01-01','2009-12-01'])

    ax1.text(0.95, 0.05, 'Ilulissat\nILU2007-01', horizontalalignment='right', verticalalignment='bottom',
             transform=ax1.transAxes)
    ax2.text(0.95, 0.05, 'Sisimiut\nSIS2007-06\nSIS2007-06a', horizontalalignment='right', verticalalignment='bottom',
             transform=ax2.transAxes)
    ax3.text(0.95, 0.05, 'Kangerlussuaq\nKAN2005-01', horizontalalignment='right', verticalalignment='bottom',
             transform=ax3.transAxes)
    ax4.text(0.95, 0.05, 'Nuuk\nNUK04008-808a+b', horizontalalignment='right', verticalalignment='bottom',
             transform=ax4.transAxes)

    ax1.set_title('')
    ax2.set_title('')
    ax3.set_title('')
    ax4.set_title('')
    map_ax.set_title('')

    ax1.set_xlim([-30, 30])
    ax2.set_xlim([-30, 30])
    ax3.set_xlim([-30, 30])
    ax4.set_xlim([-30, 30])

    ax1.set_ylim([7, 0])
    ax2.set_ylim([7, 0])
    ax3.set_ylim([7, 0])
    ax4.set_ylim([7, 0])

    plt.draw()
    plt.show(block=False)

    return myEDC


def plot_Hanne(bholes=None, plotMeanGT=False):
    """
    Call signature:
    bholes = plot_Hanne(bholes=None, plotMeanGT=False)

    Function to produce plot for Hannes IPY PPP paper

    Will plot temperatures from Sisimiut, Ilulissat, Kangerlussuaq, Nuuk
    and Zackenberg (not in database) in the same plot using Hannes layout.

    If variable bholes containing data from these boreholes is not passed to
    the function, those boreholes will be read from disk.
    """

    if bholes is None:
        bholes = read_boreholes(
            'D:\Thomas\Artek\Data\GroundTemp\Sisimiut\SIS2007-06_West_Palsa\SIS2007-06_West_Palsa_Shallow',
            calc_daily=False)
        bholes = read_boreholes(
            'D:\Thomas\Artek\Data\GroundTemp\Sisimiut\SIS2007-06_West_Palsa\SIS2007-06a_West_Palsa_Deep', bholes=bholes,
            calc_daily=False)
        bholes = read_boreholes('D:\Thomas\Artek\Data\GroundTemp\Ilulissat\ILU2007-01_78020', bholes=bholes,
                                calc_daily=False)
        bholes = read_boreholes('D:\Thomas\Artek\Data\GroundTemp\Kangerlussuaq\KAN2005-01', bholes=bholes,
                                calc_daily=False)
        bholes = read_boreholes('D:\Thomas\Artek\Data\GroundTemp\Nuuk\NUK04008-808', bholes=bholes, calc_daily=True)

    bholes['KAN2005-01'].uncalibrate()
    bholes['KAN2005-01'].calc_daily_avg(mindata=8)

    if 'SIS2007-06_combi' not in bholes:
        bholes['SIS2007-06_combi'] = combine_boreholes(
            bholes['SIS2007-06'],
            bholes['SIS2007-06a'],
            drop_channels={'1104196': [1, 2, 3]})

    #    if 'SIS2007-06_combi' not in bholes:
    #        bholes['SIS2007-06_combi'] = copy.deepcopy(bholes['SIS2007-06'])
    #        for l in bholes['SIS2007-06a'].loggers:
    #            bholes['SIS2007-06_combi'].loggers.append(copy.deepcopy(l))
    #        bholes['SIS2007-06_combi'].drop_channels(logger_sn='1104196',channels=[1,2,3])
    #        bholes['SIS2007-06_combi'].calc_daily_avg()

    figBG = 'w'
    fh = plt.figure(figsize=(20., 10.2), facecolor=figBG)
    fh.set_size_inches(20., 10.2, forward=True)
    map_ax = plt.subplot(1, 3, 3)
    ax1 = plt.subplot(2, 3, 1)
    ax2 = plt.subplot(2, 3, 2)
    ax3 = plt.subplot(2, 3, 4)
    ax4 = plt.subplot(2, 3, 5)
    ax5 = plt.subplot(2, 3, 3)

    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.4, hspace=None)

    bholes['KAN2005-01'].uncalibrate()  # Calibration for this borehole is not correct,
    # and data should be presented uncalibrated.

    args = dict(
        plotMeanGT=True,
        maxGT=dict(
            linestyle='-',
            color='k',
            marker='.',
            markersize=7,
            zorder=5),
        minGT=dict(
            linestyle='-',
            color='k',
            marker='.',
            markersize=7,
            zorder=5),
        MeanGT=dict(
            linestyle='-',
            color='k',
            marker='.',
            markersize=7,
            zorder=5),
        fill=None,
        vline=dict(
            linestyle='-',
            color='k',
            zorder=1),
        grid=dict(
            linestyle='-',
            color='0.8',
            zorder=0))

    bholes['ILU2007-01'].plot_trumpet(lim=['2008-06-10', '2009-06-09'], ax=ax1, args=args)
    bholes['SIS2007-06_combi'].plot_trumpet(lim=['2008-09-01', '2009-08-31'], ax=ax2, args=args)
    bholes['KAN2005-01'].plot_trumpet(lim=['2007-08-04', '2008-08-03'], ax=ax3, args=args)
    bholes['NUK04008-808a+b'].plot_trumpet(lim=['2008-07-04', '2009-07-03'], ax=ax4, args=args)

    ZC1depths = [1.25, 1.5, 2.5, 3, 3.25]
    ZC1meanGT = [-8.42, -8.38, -8.21, -8.13, -8.09]
    ZC1min = [-15.88, -15.88, -15.2, -12.9, -12.03]
    ZC1max = [-2.02, -2.72, -4.18, -4.82, -5.05]

    ax5.plot(ZC1max, ZC1depths, **args['maxGT'])
    ax5.hold(True)
    ax5.plot(ZC1min, ZC1depths, **args['minGT'])

    ylim = pylab.get(ax5, 'ylim')
    ax5.set_ylim(ymax=min(ylim), ymin=max(ylim))

    ax5.axvline(x=0, **args['vline'])

    ax5.plot(ZC1meanGT, ZC1depths, **args['MeanGT'])

    ax5.grid(True, **args['grid'])

    ax5.get_xaxis().tick_top()
    ax5.set_xlabel('Temperature [$^\circ$C]')
    ax5.set_ylabel('Depth [m]')

    bbox = dict(facecolor='w', edgecolor='None', alpha=1)
    ax1.text(0.95, 0.05, 'Ilulissat\nILU2007-01', horizontalalignment='right', verticalalignment='bottom',
             transform=ax1.transAxes, bbox=bbox)
    ax2.text(0.95, 0.05, 'Sisimiut\nSIS2007-06\nSIS2007-06a', horizontalalignment='right', verticalalignment='bottom',
             transform=ax2.transAxes, bbox=bbox)
    ax3.text(0.95, 0.05, 'Kangerlussuaq\nKAN2005-01', horizontalalignment='right', verticalalignment='bottom',
             transform=ax3.transAxes, bbox=bbox)
    ax4.text(0.95, 0.05, 'Nuuk\nNUK04008-808a+b', horizontalalignment='right', verticalalignment='bottom',
             transform=ax4.transAxes, bbox=bbox)
    ax5.text(0.95, 0.05, 'Zackenberg\nZC1', horizontalalignment='right', verticalalignment='bottom',
             transform=ax5.transAxes, bbox=bbox)

    ax1.set_title('')
    ax2.set_title('')
    ax3.set_title('')
    ax4.set_title('')
    ax5.set_title('')

    ax1.set_xlim([-30, 30])
    ax2.set_xlim([-30, 30])
    ax3.set_xlim([-30, 30])
    ax4.set_xlim([-30, 30])
    ax5.set_xlim([-30, 30])

    ax1.set_ylim([7, 0])
    ax2.set_ylim([7, 0])
    ax3.set_ylim([7, 0])
    ax4.set_ylim([7, 0])
    ax5.set_ylim([7, 0])

    ax1.grid(True, color='0.8', linestyle='-', zorder=0)
    ax2.grid(True, color='0.8', linestyle='-', zorder=0)
    ax3.grid(True, color='0.8', linestyle='-', zorder=0)
    ax4.grid(True, color='0.8', linestyle='-', zorder=0)
    ax5.grid(True, color='0.8', linestyle='-', zorder=0)

    plt.show(block=False)

    return bholes


def plot_Kenji(bholes=None, plotMeanGT=False):
    """
    Call signature:
    bholes = plot_Kenji(bholes=None, plotMeanGT=False)

    Function to produce output for Kenji's outreach paper

    Will produce monthly averages for the stations in Sisimiut, Ilulissat,
    Kangerlussuaq and Nuuk, and output these to disk file.

    If variable bholes containing data from these boreholes is not passed to
    the function, those boreholes will be read from disk.
    """

    if bholes is None:
        bholes = read_boreholes('D:\Thomas\Artek\Data\GroundTemp\Sisimiut\SIS2007-06_West_Palsa_Shallow',
                                calc_daily=False)
        bholes = read_boreholes('D:\Thomas\Artek\Data\GroundTemp\Sisimiut\SIS2007-06a_West_Palsa_Deep', bholes=bholes,
                                calc_daily=False)
        bholes = read_boreholes('D:\Thomas\Artek\Data\GroundTemp\Ilulissat\ILU2007-01_78020', bholes=bholes,
                                calc_daily=True)
        bholes = read_boreholes('D:\Thomas\Artek\Data\GroundTemp\Kangerlussuaq\KAN2005-01', bholes=bholes,
                                calc_daily=False)
        bholes = read_boreholes('D:\Thomas\Artek\Data\GroundTemp\Nuuk\NUK04008-808', bholes=bholes, calc_daily=True)

    bholes['KAN2005-01'].uncalibrate()
    # Calibration for this borehole is not correct,
    # and data should be presented uncalibrated.

    if 'SIS2007-06_combi' not in bholes:
        bholes['SIS2007-06_combi'] = combine_boreholes(
            bholes['SIS2007-06'],
            bholes['SIS2007-06a'],
            drop_channels={'1104196': [1, 2, 3]})

    #    if 'SIS2007-06_combi' not in bholes:
    #        bholes['SIS2007-06_combi'] = copy.deepcopy(bholes['SIS2007-06'])
    #        for l in bholes['SIS2007-06a'].loggers:
    #            bholes['SIS2007-06_combi'].loggers.append(copy.deepcopy(l))
    #        bholes['SIS2007-06_combi'].drop_channels(logger_sn='1104196',channels=[1,2,3])
    #        bholes['SIS2007-06_combi'].calc_daily_avg()

    bhids = ['SIS2007-06_combi', 'ILU2007-01', 'KAN2005-01', 'NUK04008-808a+b']
    # bhids = ['ILU2007-01']
    for bhid in bhids:
        bholes[bhid].calc_monthly_avg()
        bholes[bhid].print_monthly_data()

    return bholes


def plot_data_coverage(bholes, **kwargs):
    #   pdb.set_trace()
    if 'axes' in kwargs:
        ax = kwargs.pop('axes')
    elif 'Axes' in kwargs:
        ax = kwargs.pop('Axes')
    elif 'ax' in kwargs:
        ax = kwargs.pop('ax')
    else:
        fh = pylab.figure()
        ax = pylab.axes()

    # Get start date and convert to ordinal
    date_start = kwargs.pop('start', None)
    if date_start is not None:
        # Convert string to datetime.date object
        date_start = ensure_datetime_object(date_start, date_type=True)

    n = 0
    bhnames = sorted(bholes.keys())
    for bh in bhnames:
        n -= 1
        dates = bholes[bh].get_dates()
        # if date_start is not None:
        #    dates = dates[dates>=date_start]
        if len(dates) > 0:
            # only plot if there is data within the requested date range...
            ax.plot_date(dates, np.zeros_like(dates) + n, 'k.')
        ax.axhline(n, linestyle=':', color='k')

    lim = ax.get_ylim()
    ax.set_ylim([lim[0] - 1, lim[1] + 1])

    plt.setp(ax, 'yticks', np.arange(n, 0))
    plt.setp(ax, 'yticklabels', bhnames[::-1])

    if date_start is not None:
        xl = ax.get_xlim()
        ax.set_xlim([mpl.dates.date2num(date_start), xl[1]])

    years = mpl.dates.YearLocator()  # every year
    yearsFmt = mpl.dates.DateFormatter('%Y')

    ax.xaxis.set_major_locator(years)
    ax.xaxis.set_major_formatter(yearsFmt)

    plt.show(block=False)


##    if type(bholes) != list:
##        bholes = [bholes]
##
##    for id,bh in enumerate(bholes)
##        plot(bh.
# In order to make this function, the class borehole should have a method
# that returns an array of dates for which all loggers has data, an array of
# dates where some of the loggers have data, and possibly an array of dates
# where entries are available but masked.


def print_GT_data(bh, end=None, nyears=1):
    """
    Call signature:
    print_GT_data(bh,  end=None, nyears=1)

    Function to print information about borehole ground temperatures.

    It will print to the console:
    - The borehole name
    - The first and last day of data in the full time series
    - The date interval for which information will be printed
    - Max, Min and MAGT for each depth.

    The function will also call the method to plot a trumpet curve.
    """

    if end is None:
        end = bh.daily_ts.times[-1]

    lim = [end.replace(year=end.year - 1) + dt.timedelta(days=1), end]

    mx = bh.get_MaxGT(lim=lim)
    mn = bh.get_MinGT(lim=lim)
    magt = bh.get_MAGT(end=end, nyears=nyears)

    print(" ")
    print("Borehole: " + bh.name)
    print(" ")
    print("Data availability: ", bh.daily_ts.times[0], "to", bh.daily_ts.times[-1])
    print(" ")
    print("Data for trumpet curves")
    print("Date range: ", lim[0], "to", lim[1])
    print(" ")
    print("Depth,      Max,      Min,     MAGT")
    for id, d in enumerate(mx['depth']):
        if d >= 0:
            print("%5.2f,  %7.2f,  %7.2f,  %7.2f" % (d, mx['value'][id], mn['value'][id], magt['value'][id]))

    bh.plot_trumpet(lim=lim, plotMeanGT=True)


def get_MAGTs(bholes, month=1, day=1, fullts=False):
    """
        Call signature:
        get_MAGT(bholes, month=1, day=1, fullts=False)

        This method calculates the Mean Annual Ground Temperature for the entire timeseries
        in all the boreholes passed, using the specified month and day for separating the
        years.

        MAGT is calculated based on the averaged daily time series unless fullts=True
        is specified (full_ts not yet supported).

        """

    import pandas as pd

    if not hasattr(bholes, 'keys'):
        bholes = {bholes.name: bholes}

    if not fullts:
        magt_dict = {}
        for bname, bhole in list(bholes.items()):
            if bhole.daily_ts is None:
                bhole.calc_daily_avg()

            # Get a dictionary of year:timeseries pairs
            ydict = bhole.daily_ts.split_years(month=month, day=day)

            years = np.array(sorted(ydict.keys()))
            depths = bhole.depths
            ndepths = len(bhole.depths)
            magt = np.zeros((len(years), ndepths))
            stdev = np.zeros((len(years), ndepths))
            ndays = np.zeros((len(years), ndepths))

            for idy, yr in enumerate(years):
                ts = ydict[yr]
                magt[idy, :] = ts.mean()['value'][:, 0]
                stdev[idy, :] = ts.std()['value'][:, 0]
                ndays[idy, :] = len(ts) - ts.data.mask.sum(axis=0)

            magt_dict[bname] = dict(Years=years, Depths=depths, MAGT=magt, STD=stdev, NDAYS=ndays)

            iterables = [['MAGT', 'STD', 'NDAYS'], depths]
            index = pd.MultiIndex.from_product(iterables, names=['Parameter', 'Depth'])

            magt_dict[bname] = pd.DataFrame(np.hstack([magt, stdev, ndays]), index=years, columns=index)
    else:
        raise "Full timeseries support in get_MAGTs is not implemented yet!"

    return magt_dict


def find_nearest(array, value):
    """
    Finds the index into array of the element nearest to value.
    """
    return (np.abs(np.asarray(array) - value)).argmin()


def plot_MAGTs(bholes, depth=None, ndays=340):
    """
    Will plot MAGTs for all the boreholes passed in a single plot.
    Only years with data for more than ndays will be plotted.

    Tip: use filter_dict function to filter a large dictionary of boreholes.
    e.g.:
    filter_dict(bholes, lambda k, v: k.startswith('SIS'))

    Usage example:
    BH.plot_MAGTs(BH.filter_dict(bholes, lambda k, v: k.startswith('ILU')), depth=3.0)
    """

    magts = get_MAGTs(bholes)

    fig = plt.figure()
    ax = plt.axes()

    for bname, data in list(magts.items()):
        if depth is not None:
            id1 = find_nearest(data['Depths'], depth)
        else:
            id1 = -1

        id2 = data['NDAYS'][:, id1] >= ndays
        ax.plot(data['Years'][id2], data['MAGT'][id2, id1], '.', label=bname)

    ax.legend()


def calc_MAGT_trend(bholes, depth=None, min_ndat=350):
    """
    Calculates the linear trend in MAGT for the combined timeseries of
    the boreholes passed in bholes.

    Usage example:
    BH.calc_MAGT_trend(BH.filter_dict(bholes, lambda k, v: k in ['ILU OBO', 'ILU2007-03']),
                       depth=3.0, min_ndat={'ILU OBO':20,'ILU2007-03':300})
    """

    magts = get_MAGTs(bholes, month=8, day=31)

    if len(magts) > 1:
        print("NB! MAGT timeseries will be concatenated and sorted chronologically!")

    if not hasattr(min_ndat, 'keys'):
        min_ndat = {k: min_ndat for k in list(magts.keys())}

    fig = plt.figure()
    ax = plt.axes()

    x = []
    y = []

    for bname, data in list(magts.items()):
        if depth is not None:
            id = find_nearest(data['Depths'], depth)
        else:
            id = -1

        idx = data['NDAYS'][:, id] >= min_ndat[bname]
        if sum(idx) == 0:
            continue
        x.extend(list(data['Years'][idx]))
        y.extend(list(data['MAGT'][idx, id]))
        ax.plot(data['Years'][idx], data['MAGT'][idx, id], '.', label=bname)

    x = np.array(x)
    y = np.array(y)
    idx = x.argsort()
    x = x[idx]
    y = y[idx]

    (a, b, r, tt, stderr) = scp_stats.linregress(x, y)
    ax.plot(x, a * x + b, '--k')

    annotation = ['eq', 'R', 'N']
    astr = ''
    if 'eq' in annotation:
        astr = 'y = {0:.4f}*x + {1:.4f}'.format(a, b)
    if 'R' in annotation:
        if astr != '': astr = astr + '\n'
        astr = astr + 'R$^2$ = {0:.4f}'.format(r)
    if 'N' in annotation:
        if astr != '': astr = astr + '\n'
        astr = astr + 'N = {0:.0f}'.format(len(y))

    if astr != '':
        ax.text(0.05, 0.95, astr,
                verticalalignment='top',
                fontsize=14,
                transform=ax.transAxes)

    # yl = ax.get_ylim()
    # yl[1] = np.max([0., yl])

    ax.legend(loc='lower right')


def plot_date_interactive(bhole, date=0, lim=None):
    """
    Will plot date from borehole, and allow stepping in time.


    Arrow right:     Next date
    Arrow left:      Previous date
    Pg Up:           Take 10 time-steps ahead in time
    Pg Down:         Take 10 time-steps back in time
    y:               Jump ahead 1 year (or as close as possible)
    shift+y:         Jump back 1 year (or as close as possible)
    """

    """
    If using Qt4Agg backend (try mpl.get_backend() to find out), make sure that
    the following is in your lib/site-packages/matplotlib/backends/backend_qt4.py

        class FigureCanvasQT( QtGui.QWidget, FigureCanvasBase ):
            keyvald = { QtCore.Qt.Key_Control : 'control',
                        QtCore.Qt.Key_Shift : 'shift',
                        QtCore.Qt.Key_Alt : 'alt',
                        QtCore.Qt.Key_Return : 'enter',
                        QtCore.Qt.Key_Left : 'left',
                        QtCore.Qt.Key_Up : 'up',
                        QtCore.Qt.Key_Right : 'right',
                        QtCore.Qt.Key_Down : 'down',
                        QtCore.Qt.Key_Escape : 'escape',
                        QtCore.Qt.Key_F1 : 'f1',
                        QtCore.Qt.Key_F2 : 'f2',
                        QtCore.Qt.Key_F3 : 'f3',
                        QtCore.Qt.Key_F4 : 'f4',
                        QtCore.Qt.Key_F5 : 'f5',
                        QtCore.Qt.Key_F6 : 'f6',
                        QtCore.Qt.Key_F7 : 'f7',
                        QtCore.Qt.Key_F8 : 'f8',
                        QtCore.Qt.Key_F9 : 'f9',
                        QtCore.Qt.Key_F10 : 'f10',
                        QtCore.Qt.Key_F11 : 'f11',
                        QtCore.Qt.Key_F12 : 'f12',
                        QtCore.Qt.Key_Home : 'home',
                        QtCore.Qt.Key_End : 'end',
                        QtCore.Qt.Key_PageUp : 'pageup',
                        QtCore.Qt.Key_PageDown : 'pagedown',
                       }

    """

    return DatePlotInteractive(bhole, date)


# import time

class DatePlotInteractive:
    def __init__(self, bhole, date=0):
        self.key_down = ''
        self.bhole = bhole
        self.ax = None
        self.fh = None

        if not hasattr(date, 'year'):
            try:
                if date >= 0 and date < len(self.bhole):
                    self.date_id = date
                else:
                    self.date_id = 0
            except:
                self.date_id = 0
        else:
            date_id = find(self.bhole.daily_ts.times == date)
            if len(date_id) == 0:
                raise IndexError('Date is not present in timeseries')
            else:
                self.date_id = date_id[0]

        self.plot()
        self.connect()
        print("Plot connected!")

    def connect(self):
        'connect to all the events we need'
        self.keypress = self.fh.canvas.mpl_connect(
            'key_press_event', self.on_key_press)
        self.keyrelease = self.fh.canvas.mpl_connect(
            'key_release_event', self.on_key_release)

    def disconnect(self):
        'disconnect all the stored connection ids'
        self.fh.canvas.mpl_disconnect(self.keypress)
        self.fh.canvas.mpl_disconnect(self.keyrelease)
        self.keypress = None
        self.keyrelease = None

    def plot_exists(self):
        if self.ax and self.ax.figure.number in plt.get_fignums():
            return True
        else:
            return False

    def reset(self):
        if self.fh:
            self.disconnect()
            self.fh = None
            self.ax = None

    def plot(self, fs=12):

        istate = mpl.is_interactive()
        mpl.interactive(False)

        if self.plot_exists():
            for id, lh in enumerate(self.ax.lines):
                if hasattr(lh, 'tag') and lh.tag == 'gt':
                    # self.ax.lines.pop(id)
                    lh.remove()

            for id, th in enumerate(self.ax.texts):
                if hasattr(th, 'tag') and th.tag == 'date':
                    # self.ax.texts.pop(id)
                    th.remove()

            gid = find(np.array(self.bhole.depths) >= 0.)
            depths = np.take(self.bhole.depths, gid)
            data = self.bhole.daily_ts.data[self.date_id, gid].flatten()

            lh = self.ax.plot(data, depths, '-k', marker='.', markersize=7)
            lh[0].tag = 'gt'

            t2h = self.ax.text(0.95, 0.05, self.bhole.daily_ts.times[self.date_id],
                               horizontalalignment='right', verticalalignment='bottom',
                               transform=self.ax.transAxes, fontsize=fs)
            t2h.tag = 'date'

        #            ylim = pylab.get(ax, 'ylim')
        #            ax.set_ylim(ymax=min(ylim), ymin=max(ylim))
        else:
            self.reset()
            self.ax = self.bhole.plot_date(date=self.bhole.daily_ts.times[self.date_id])
            self.fh = self.ax.figure

        mpl.interactive(istate)
        plt.draw()

    def on_key_release(self, event):
        self.key_down = ''
        # print ("Key released   ", event.inaxes)
        if not self.plot_exists():
            self.disconnect()

    def on_key_press(self, event):
        if not self.plot_exists():
            self.disconnect()
            return

        self.key_down = event.key

        if self.key_down in ['left', '-']:
            # print "Keypress: ", self.key_down
            if self.date_id > 0:
                self.date_id -= 1

        elif self.key_down in ['right', '+']:
            # print "Keypress: ", self.key_down
            if self.date_id < len(self.bhole.daily_ts) - 1:
                self.date_id += 1

        elif self.key_down == 'pagedown':
            # print "Keypress: ", self.key_down
            if self.date_id > 9:
                self.date_id -= 10
        elif self.key_down == 'pageup':
            # print "Keypress: ", self.key_down
            if self.date_id < len(self.bhole.daily_ts) - 10:
                self.date_id += 10
        elif self.key_down == 'y':
            # print "Keypress: ", self.key_down
            thisdate = self.bhole.daily_ts.times[self.date_id]
            nextyear = dt.date(thisdate.year + 1, thisdate.month, thisdate.day)

            if nextyear not in self.bhole.daily_ts:
                self.date_id = find(self.bhole.daily_ts.times < nextyear)[-1]
            else:
                self.date_id = self.bhole.daily_ts.get_date_id(nextyear)

            if self.date_id:
                if hasattr(self.date_id, '__iter__'):
                    self.date_id = self.date_id[0]
            else:
                self.date_id = -1

        elif self.key_down == 'Y':
            # print "Keypress: ", self.key_down
            thisdate = self.bhole.daily_ts.times[self.date_id]
            prevyear = dt.date(thisdate.year - 1, thisdate.month, thisdate.day)

            if prevyear not in self.bhole.daily_ts:
                self.date_id = find(self.bhole.daily_ts.times > prevyear)[0]
            else:
                self.date_id = self.bhole.daily_ts.get_date_id(prevyear)

            if self.date_id:
                if hasattr(self.date_id, '__iter__'):
                    self.date_id = self.date_id[0]
            else:
                self.date_id = 0

        elif self.key_down in ['?', 'h']:
            # print "Keypress: ", self.key_down, "  list key bindings"
            keybindings = """
            Key presses supported:

            Arrow right:     Next date
            Arrow left:      Previous date
            Pg Up:           Take 10 time-steps ahead in time
            Pg Down:         Take 10 time-steps back in time
            y:               Jump ahead 1 year (or as close as possible)
            shift+y:         Jump back 1 year (or as close as possible)
            h:               List of commands
            """
            print(keybindings)
            return
        else:
            return

        self.plot()


# --- ### General helper functions ###

def find(condition):
    """
    Remake of mpl.mlab.find to also work with masked arrays.
    It will not take masked elements into account.
    For normal np.ndarray the function works like mpl.mlab.find.
    """
    np.array(condition)
    res, = np.nonzero(np.ma.ravel(condition > 0))
    return res


def nyears2lim(end=None, nyears=1):
    # check date format, do necessary conversion to datetime.date object
    if end is None:
        return False

    if type(end) in [int, float, np.float32, np.float64]:
        end = mpl.dates.num2date(end).date()
    elif type(end) == str:
        end = dateutil.parser.parse(end, yearfirst=True, dayfirst=True).date()
    elif type(end) in [dt.datetime]:
        end = end.date()
    elif type(end) != dt.date:
        raise "dates should be either ordinals, date or datetime objects"

    start = copy.copy(end).replace(year=end.year - nyears)
    start = start + dt.timedelta(days=1)

    return [start, end]


def ensure_datetime_object(dat, date_type=False):
    """
    Tests if the input is a datetime object, and otherwise attempts to convert.
    Valid input is a string ('2005-01-01' or '2005-01-01 22:00:00+01:00' etc) or a matplotlib ordinal.
    
    date_type: flag to specify whether a datetime.date type should be enforced.
    """

    if type(dat) in [int, float, np.float32, np.float64]:
        end = mpl.dates.num2date(end)
    elif type(dat) == str:
        dat = dateutil.parser.parse(dat, yearfirst=True, dayfirst=True)
    if type(dat) not in [dt.date, dt.datetime]:
        raise "dates should be either ordinals, date or datetime objects"

    if date_type and type(dat) in [dt.datetime]:
        dat = dat.date()

    return dat


# --- ### Plot annotation functions


def onpick(event):
    print(event.artist.get_label() + " was clicked!")

    if isinstance(event.artist, plt.Line2D):
        thisline = event.artist

        xdata = thisline.get_xdata()
        ydata = thisline.get_ydata()
        ind = event.ind

        print("%i datapoints within tolerance" % len(ind))
        if len(ind) != 0:
            print('Coordinates: ', list(zip(xdata[ind], ydata[ind])))

        print("\n")

    elif isinstance(event.artist, plt.Rectangle):
        patch = event.artist
        vertsxy = patch.get_verts()
        print('onpick patch:', vertsxy)
    elif isinstance(event.artist, plt.Text):
        text = event.artist
        print('onpick text:', text.get_text())


class Usage(Exception):
    def __init__(self, msg):
        self.msg = msg


def main(argv=None):
    #    if argv is None:
    #        argv = sys.argv
    #    try:
    #        try:
    #            opts, args = getopt.getopt(argv[1:], "h", ["help"])
    #        except getopt.error, msg:
    #             raise Usage(msg)
    #        # more code, unchanged
    #    except Usage, err:
    #        print >>sys.stderr, err.msg
    #        print >>sys.stderr, "for help use --help"
    #        return 2
    myEDC = plot_grl_boreholes()
    bholes = myEDC.boreholes
    print("\nCalculating ALT for B2007-06_combi")
    bholes['SIS2007-06_combi'].calc_ALT()
    print("Finished!")


if __name__ == "__main__":
    sys.exit(main())

"""
bholes['SIS2007-06_combi'] = boreholes_v1p2.combine_boreholes(
        bholes['SIS2007-06'],
        bholes['SIS2007-06a'],
        drop_channels = {'1104196': [1,2,3]})
"""

"""
Problem with the trumpet plot method... Does it take masked data into account???
Seems like it uses all data in the max/min operation.




OK   Include borehole property in data logger files.
OK   make test: if borehole property is present, it should match the borehole to which the file is being added.

OK   The overall load-routine should parse the individual files, and add them to the appropriate borehole!

Make new class: GTregion
It should include a list of boreholes and methods to:
    OK (but not in a class) add a directory tree of files.
    plot locations on a basemap map

add methods to borehole class to:
    plot trumpet diagram (flag to plot also mean annual ground temperature)
    OK plot individual depths
    ?plot surface plot of temperatures?
    plot location on basemap map

    export to a single excel sheet with all depths
        - how to handle different logging times? round or calculate daily averages...
    calculate daily,monthly,yearly average T


    Add possibility to mask an entire channel in input file header.
    How do we handle masking individual datapoints in multichannel files?
"""
