# Python Version: 2.7
# mainly due to dependencies on inhouse packages

# Change log
# - Fixed column sorting of rawdata_mask, to avoid messing up column order when mask is applied

# ToDos
#
# Make changes such that we can make changes to latex dataframe output
# conditionally on a cell basis...
# We need to prepend '>' to the Dzaa column, if the sigma_Tzaa is larger than say 0.15
# We need to be able to add a footnote indicator to any cell (conditionally)
#
# One option is to make a custom parser to produce a dataframe of strings
# that can then be written using the to_latex method.
# Or could make our own complete dataframe to latex parser.
# (Should be not so difficult)
#


import copy
import datetime as dt
import os
import ipdb as pdb
from collections import OrderedDict

import dateutil
import matplotlib as mpl
# mpl.use('WXAgg')
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import matplotlib.dates as mpld
import numpy as np
import pandas as pd
import pathlib
import pydatastorage.h5io as h5io
#import pypf.db.boreholes as bh
import pypf.db.zoom_span as zoom_span


# requires pathlib 2
# conda install -c anaconda pathlib2


def ensure_paths_exist(paths):
    if isinstance(paths, str):
        paths = [paths]
    for path in paths:
        if not os.path.exists(path):
            os.mkdir(path)


def float_eq(alike, val, atol=1e-8, rtol=1e-5):
    """makes a float comparison allowing for round off errors to within the
    speciefied absolute and relative tolerances
    """
    return np.abs(np.array(alike) - val) <= (atol + rtol * np.abs(val))


def find(condition):
    """
    Remake of mpl.mlab.find to also work with masked arrays.
    It will not take masked elements into account.
    For normal np.ndarray the function works like mpl.mlab.find.
    """
    np.array(condition)
    res, = np.nonzero(np.ma.ravel(condition > 0))
    return res


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


def nyears2lim(end_date=None, nyears=1):
    # check date format, do necessary conversion to datetime.date object
    if end_date is None:
        return False

    if type(end_date) in [int, float, np.float32, np.float64]:
        raise NotImplementedError('Int and float type date serials are not implemented')
        # end_date = mpl.dates.num2date(end_date).date()
    elif type(end_date) == str:
        end_date = dateutil.parser.parse(end_date, yearfirst=True, dayfirst=True).date()
    elif type(end_date) in [dt.datetime]:
        end_date = end_date.date()
    elif type(end_date) != dt.date:
        raise ValueError('dates should be either ordinals, date or datetime objects')

    start = copy.copy(end_date).replace(year=end_date.year - nyears)
    start = start + dt.timedelta(days=1)

    return [start, end_date]


def fix_lim(lim):
    if lim is None:
        return

    newlist = []
    for lid, thistime in enumerate(lim):
        if type(thistime) == str:
            newlist.append(dateutil.parser.parse(thistime, yearfirst=True, dayfirst=False).date())
        elif type(thistime) in [dt.datetime, dt.date]:
            newlist.append(thistime)
        else:
            raise TypeError('Date ranges must be specified as strings or datetime or date objects.')
    return newlist


def find_zero(xi, yi, exclude_zeros=True):
    # Finds zero crossings in the xi elements (e.g. ground temperature) and interpolates
    # the depth of the zero crossing based on the distances in yi.
    #
    # The standard use of np.sign will also identify zeros as a sign change, so
    # if a node is exactly 0.000 (which could happen e.g. during freezing
    # there would be two immediately successive crossings.
    #
    # Using the flag exclude_zeros=True will exclude these double indices.
    # [1, 0, -1, 0, -1, 1]  ->   id = [0, 4]
    # [-1, 0, 1, 0, 1, -1]  ->   id = [1, 2, 3, 4]
    # ...such that interpolation can be done from nodes id to id+1
    # In context of active layers, this means that the freezing front is
    # identified at the top of a 0C interval.
    #
    # There may be still some issues with [1, 0, 1] type situations, which may have to be handled.


    if exclude_zeros:
        # This code will check for xi elements equal zero, and
        mydiff = np.diff(np.sign(xi))

        for id, d in enumerate(mydiff):
            if (xi[id] == 0) & (xi[id + 1] < 0):
                mydiff[id] = 0
            elif (xi[id] < 0) & (xi[id + 1] == 0):
                mydiff[id] = 0

        idx3 = np.where(mydiff != 0)[0]

    else:
        idx3 = np.where(np.diff(np.sign(xi)) != 0)[0] + 1
        # In this call a change from positive to zero is also counted as a sign change!
        # this version is not very useful for ALT estimation...

    ys = []
    for ix in idx3:
        y0 = (yi[ix] * (xi[ix+1] - 0) + yi[ix+1] * (0 - xi[ix])) / (xi[ix+1] - xi[ix])
        ys.append(y0)

    return ys



# selecting a certain SensorID, e.g. SensorID=0: bh.rawdata.xs(0, level='SensorID', axis=1)


class Borehole:
    def __init__(self, name=None,
                 h5path=None, h5dataset='GroundTemperature',
                 calpath=None):

        self.name = name
        self.date = None
        self.latitude = None
        self.longitude = None
        self.crs = None
        self.description = None

        self.rawdata = None
        self.rawdata_mask = None
        self.daily_ts = None
        self.caldata = None
        self.calibrated = False

        self.h5path = None
        self.h5dataset = h5dataset
        self.calpath = None

        if calpath is not None:
            self.calpath = pathlib.Path(calpath)
            self.load_calibration_data()

        if h5path is not None:
            self.h5path = pathlib.Path(h5path)
            self.load_h5_borehole()
            self.calibrate()
            self.calc_daily_average()



    def load_calibration_data(self, calpath=None):
        """Loads calibration data for each sensor"""
        if calpath is not None:
            self.caldata = pd.read_csv(calpath, skipinitialspace=True, sep=';', comment='#')
        elif self.calpath is not None:
            self.caldata = pd.read_csv(self.calpath, skipinitialspace=True, sep=';', comment='#')

    def calibrate(self):
        """Applies calibration data to borehole measurements"""
        if self.calibrated:
            return

        if self.caldata is None:
            # TODO: Implement warning system!
            # raise ValueError('Calibration data not loaded - can''t calibrate...!')
            return

        # Procedure:
        # 1. Loop over all columns in data frame
        # 2.     Find row in caldata that matches SensorID and CoordZ and possibly data type
        # TODO: When available, also check Instrument serial
        # 3.     Apply cal value to all data points
        # 4. If cal-value is not available, warn? or fail?

        for cid in range(len(self.rawdata.columns)):
            od = OrderedDict(list(zip(self.rawdata.columns.names, self.rawdata.columns[cid])))
            # calid = (self.caldata['Channel']==od['SensorID']) & \
            #         (self.caldata['Depth']==od['CoordZ'])
            calid = (self.caldata['SensorID'] == od['SensorID'])

            # TODO: Consider adding a data type to cal-data.

            if sum(calid) == 1:
                self.rawdata.iloc[:,cid]  += self.caldata[calid]['dT'].values[0]
            elif (sum(calid) == 0) and (od['CoordZ'] <= 0):
                # sensor above ground, we don't care that it is not calibrated
                pass
                # TODO: Implement calibration possibility for all sensors
            else:
                # if this is a ground channel and it has no calibration
                # or if there are more cal values matching
                # ... raise error!
                raise ValueError('Problem with matching calibration data to instrument channels!')

        self.calibrated = True
        self.calc_daily_average()


    def uncalibrate(self):
        """Removes calibration from calibrated borehole measurements"""
        if not self.calibrated:
            return

        # Procedure:
        # 1. Loop over all columns in data frame
        # 2.     Find row in caldata that matches SensorID and CoordZ and possibly data type
        # TODO: When available, also check Instrument serial
        # 3.     Remove cal value to all data points
        # 4. If cal-value is not available, warn? or fail?

        for cid in range(len(self.rawdata.columns)):
            od = OrderedDict(list(zip(self.rawdata.columns.names, self.rawdata.columns[cid])))
            #calid = (self.caldata['Channel']==od['SensorID']) & \
            #        (self.caldata['Depth']==od['CoordZ'])
            calid = (self.caldata['Channel'] == od['SensorID'])

            # TODO: Consider adding a data type to cal-data.

            if sum(calid) == 1:
                self.rawdata.iloc[:,cid]  -= self.caldata[calid]['dT'].values[0]
            elif (sum(calid) == 0) and (od['CoordZ'] <= 0):
                # sensor above ground, we don't care that it is not calibrated
                pass
            else:
                # if this is a ground channel and it has no calibration
                # or if there are more cal values matching
                # ... raise error!
                raise ValueError('Problem with matching calibration data to instrument channels!')

        self.calibrated = False
        self.calc_daily_average()

    def apply_mask(self):
        """Applies masking of the data, by adding the self.rawdata_mask dataframe to the
        self.rawdata dataframe, which will propagate the NaN of the mask to the rawdata.
        Currently this function is irreversible. To get original data back, you have to
        re-read the data from disk."""

        # Make sure all indices from self.rawdata are available in the mask
        # If not available, introduce with value 0.0, so they don't
        # affect original values when we apply the mask to the data.
        mask = self.rawdata_mask.reindex(self.rawdata.index, fill_value=0.)

        # We add zero to the mask, to convert bool values to floats...
        self.rawdata = self.rawdata+(mask+0)
        
    def load_h5_borehole(self, calc_daily_avg=True):
        # def read_borehole(datfile, calfile=None, dataset_name='GroundTemperature'):

        # open the data file
        datafile = h5io.DataStore(str(self.h5path.absolute()))

        # get the borehole name
        self.name = self.h5path.stem

        # TODO: get the borehole name from the h5 file, requires implementation in pydatastorage

        # read all stored data (use read_where to get part of the data)
        DataSet = datafile.DataSets.__dict__[self.h5dataset]
        data = DataSet.Measurement.node.read()
        sensor = DataSet.Sensor.node.read()
        datatype = datafile.DataType.node.read()
        logger_serial = 'Not defined'

        # Convert to a pandas DataFrames
        df = pd.DataFrame(data)
        dfdt = pd.DataFrame(datatype)
        dfsensor = pd.DataFrame(sensor)
        
        if any(df.duplicated(subset=['TimeStamp', 'SensorID', 'DataTypeID', 'Value'])):
            print('Data contains duplicate entries... consider cleaning up database.')
            print('Discarding duplicate values...')
            df.drop_duplicates(subset=['TimeStamp', 'SensorID', 'DataTypeID', 'Value'], inplace=True)
        
        # decode bytestrings read from hdf5
        def decode_strings(thisdf):
            str_df = thisdf.select_dtypes([np.object])
            try:
                str_df = str_df.stack().str.decode('ascii').unstack()
            except:
                # the above line may fail if there are no string columns in 
                # the dataframe. In that case fail silently.
                pass
            for col in str_df:
                thisdf[col] = str_df[col]
            return thisdf
        
        df = decode_strings(df)
        dfdt = decode_strings(dfdt)
        dfsensor = decode_strings(dfsensor)

        #pdb.set_trace()
        
        # Make the time stamp human readable
        df['TimeStamp'] = DataSet.get_time(list(df['TimeStamp']))
        #df['TimeStamp'] = df['TimeStamp'].dt.tz_localize(tz='utc').dt.tz_convert(tz=DataSet.timeepoch.tzinfo)
        df['TimeStamp'] = df['TimeStamp'].dt.tz_localize(tz='utc') # Timestamps from HDF files are always stored as UTC
        datafile.close()

        # Now reorganize data in standard table format
        #pvdf = pd.pivot_table(df, index='TimeStamp', columns=['SensorID', 'DataTypeID'], values=['Value'])
        pvdf = df.pivot_table(index='TimeStamp', columns=['SensorID', 'DataTypeID'], values=['Value'])

        # And get the masks of the data
        #pvdf_mask = pd.pivot_table(df, index='TimeStamp', columns=['SensorID', 'DataTypeID'], values=['Masked'])
        pvdf_mask = df.pivot(index='TimeStamp', columns=['SensorID', 'DataTypeID'], values=['Masked'])

        # We want the multiindex to have the following levels:
        # 0.  SensorID
        # 1.  SensorName
        # 2.  SensorType
        # 3.  DataType
        # 4.  DataUnit
        # 5.  CoordZ

        # To begin with, we have:
        # 0.  Value
        # 1.  SensorID
        # 2.  DataTypeID

        # Drop the unwanted level 0
        pvdf.columns = pvdf.columns.droplevel(0)
        pvdf_mask.columns = pvdf_mask.columns.droplevel(0)

        # Now we have:
        # 0.  SensorID
        # 1.  DataTypeID

        # Prepare for modificaion of column multiindex
        clevels = list(pvdf.columns.levels)
        clabels = list(pvdf.columns.codes)
        cnames = list(pvdf.columns.names)

        # Add SensorName
        sensor_names = [dfsensor[dfsensor['SensorID'] == x]['SensorName'].values[0] for x in clevels[0] if any(dfsensor['SensorID'] == x)]
        # final condition "if any...." added here and below on 2020-1019 because an unused
        # level value showed up in Kangerlussuaq data, and caused an indexing error
        
        sensor_name_list = list(np.unique(sensor_names))
        # Insert the SensorType information as level 2
        clevels.insert(1, sensor_name_list)
        clabels.insert(1, [sensor_name_list.index(st) for st in sensor_names])
        cnames.insert(1, 'SensorName')

        # Now we have:
        # 0.  SensorID
        # 1.  SensorName
        # 2.  DataTypeID

        # Add SensorType
        sensor_types = [dfsensor[dfsensor['SensorID'] == x]['SensorType'].values[0] for x in clevels[0] if any(dfsensor['SensorID'] == x)]
        sensor_type_list = list(np.unique(sensor_types))
        # Insert the SensorType information as level 2
        clevels.insert(2, sensor_type_list)
        clabels.insert(2, [sensor_type_list.index(st) for st in sensor_types])
        cnames.insert(2, 'SensorType')

        # Now we have:
        # 0.  SensorID
        # 1.  SensorName
        # 2.  SensorType
        # 3.  DataTypeID

        # Append DataUnit
        clevels.append([dfdt['Unit'][dfdt['DataTypeID'] == x].values[0] for x in clevels[3] if any(dfdt['DataTypeID'] == x)])
        clabels.append(clabels[3])    # We use same codes as for DataTypeID, which is now index 2, after insertion of SensorType data
        cnames.append('DataUnit')

        # Now we have:
        # 0.  SensorID
        # 1.  SensorName
        # 2.  SensorType
        # 3.  DataTypeID
        # 4.  DataUnit

        # Append Sensor depth as new level at end of multiindex
        sensor_depths = [dfsensor[dfsensor['SensorID'] == x]['CoordZ'].values[0] for x in clevels[0] if any(dfsensor['SensorID'] == x)]
        sensor_depth_list = list(np.unique(sensor_depths))
        clevels.append(sensor_depth_list)
        clabels.append([sensor_depth_list.index(sd) for sd in sensor_depths])
        cnames.append('CoordZ')

        # Now we have:
        # 0.  SensorID
        # 1.  SensorName
        # 2.  SensorType
        # 3.  DataTypeID
        # 4.  DataUnit
        # 5.  CoordZ

        # Finally insert datatype names, instead of the the DataTypeIDs
        # This means updating level 2 (of the original multiindex)
        clevels[3] = [dfdt['Name'][dfdt['DataTypeID'] == x].values[0] for x in clevels[3] if any(dfdt['DataTypeID'] == x)]
        cnames[3] = 'DataType'

        # Now we have:
        # 0.  SensorID
        # 1.  SensorName
        # 2.  SensorType
        # 3.  DataType
        # 4.  DataUnit
        # 5.  CoordZ

        # Now create the multiindex
        mi = pd.MultiIndex(levels=clevels, names=cnames, codes=clabels)

        # Assign new index to dataframe
        pvdf.columns = mi
        pvdf_mask.columns = mi

        # And sort according to depth
        pvdf.sort_index(axis=1, level='CoordZ', inplace=True, sort_remaining=False)
        pvdf_mask.sort_index(axis=1, level='CoordZ', inplace=True, sort_remaining=False)


        # Now stor index information also as separate lists
        self.sensor_types = list(pvdf.columns.get_level_values('SensorType'))
        self.sensor_data_types = list(pvdf.columns.get_level_values('DataType'))
        self.sensor_data_units = list(pvdf.columns.get_level_values('DataUnit'))
        self.sensor_id = list(pvdf.columns.get_level_values('SensorID'))
        self.sensor_depths = np.array(pvdf.columns.get_level_values('CoordZ'))

        # TODO: implement calibration in separate method
        # # Apply calibrations
        # if calfile is not None:
        #     sensorids = pvdf.columns.get_level_values('SensorID')
        #     for id in sensorids[:-2]:
        #          #pdb.set_trace()
        #          pvdf.iloc[:,id] += caldata[caldata['Channel']==id+1]['dT'].values[0]

        # Now remove sensors that we know are above ground
        # This could be also programmed based on the CoordZ value...
        # pvdf2 = pvdf.drop([0,1,2,3,18], level=1, axis=1, errors='ignore')

        self.rawdata = pvdf.astype(float)
        self.rawdata_mask = pvdf_mask.astype(float)

        # consistency check:
        temp_sensors =  np.array(self.sensor_data_types) == 'Temp'
        degC_cols = (np.array(self.sensor_data_units) == 'C') & temp_sensors
        K_cols = (np.array(self.sensor_data_units) == 'K') & temp_sensors

        self.rawdata[self.rawdata.iloc[:, degC_cols] < -273.15] = np.NaN
        self.rawdata[self.rawdata.iloc[:, K_cols] < 0] = np.NaN

        # TODO: Implement mask handling, currently mask is only read and stored

        self.calc_daily_average()

    def apply_masking_file(self, fname):
        """Reads and applies masking from a text file.
        
        The format of the file is:
        from_datetime; to_datetime; SensorID 1; SensorID 2; ... SensorID N
        
        from_datetime and to_datetime should be timezone aware - currently not tested for other timezones than UTC.
        datetimes should be followed by a semicolon separated list of SensorIDs to be masked for all 
        datapoints within the given timerange (endpoints inclusive).
        """
        if not hasattr(fname, 'open'):
            fname = pathlib.Path(fname)
            
        with fname.open(mode='r') as f:
            lines = f.readlines()

        print("\nApplying masks from file: {0}".format(fname))
            
        for line in lines:
            if line.startswith('#'):
                continue
                
            # remove any comments at end of line
            line = line.strip()
            head, sep, tail = line.partition('#')
            line = head.strip()
            
            if len(line) == 0:
                continue  # line is empty, so continue
            
            print(line)
            dat = line.strip().strip(';').split(';')
            dat = [s.strip() for s in dat]
            date1 = dat[0]
            date2 = dat[1]
            if len(dat) > 2:
                SensorIDs = [int(s) for s in dat[2:]]
            ids = (self.rawdata_mask.index >= date1) & (self.rawdata_mask.index <= date2)
            
            #bh2 = copy.deepcopy(bh)  
            self.rawdata_mask.iloc[ids, self.rawdata_mask.columns.get_level_values(0).isin(SensorIDs)] = np.nan
            self.apply_mask()
            self.calc_daily_average()

    def merge(self, other):
        """Merges two boreholes, so that measurements from the two time series are merged and appear as one.
        No checking or corrections are performed.
        Only measurements from other that occur later than last measurement of self are considered.
        There is no keeping track of calibration status.
        """
        outbh = copy.deepcopy(self)
        # add to rawdata only the rows from other that are later than any row in self
        outbh.rawdata = self.rawdata.append(other.rawdata[other.rawdata.index>self.rawdata.index[-1]])
        outbh.rawdata_mask = self.rawdata_mask.append(other.rawdata_mask[other.rawdata_mask.index>self.rawdata_mask.index[-1]])
        #outbh.rawdata_mask = self.rawdata_mask + other.rawdata_mask[other.rawdata_mask.index>self.rawdata_mask.index[-1]]
        outbh.apply_mask()
        outbh.calc_daily_average()
        return outbh

    def __add__(self, other):
        return self.merge(other)

    def sort_columns_by_depth(self):
        self.daily_ts.sort_index(1,'CoordZ', sort_remaining=False, inplace=True)
        
    def get_header_info(self):
        """Converts the column headers (MultiIndex) to an OrderedDict
        with the level names as keys, and lists of the level values as 
        values of the dict.
        Can be modified and then used to recreate a modified MultiIndex.
        
        A use case could be to modify depths of the sensors.
        """
        
        c = self.rawdata.columns
        level_names = c.names
        level_values =  [list(c.get_level_values(id)) for id in range(len(c.names))]
        od = OrderedDict(list(zip(level_names, level_values)))
        
        # Convert OrderedDict into new MultiIndex:
        # mi = pd.MultiIndex.from_frame(pd.DataFrame(od))
        
        return od
        
    def set_header_info(self, od=None, names=None, values=None):
        if od is not None:
            mi = pd.MultiIndex.from_frame(pd.DataFrame(od))
            self.rawdata.columns = mi
            self.rawdata_mask.columns = mi
        else:
            if (names is not None) and (values is not None):
                mi = pd.MultiIndex.from_arrays(values, names)
                self.rawdata_mask.columns = mi
            else:
                raise ValueError('data format not appropriate to generate MultiIndex')
                
    def export(self, fname, delim=', '):
        """
        This method should take a filename as input and export the ground temperature data
        from the borehole as a textfile.
        """

        if self.daily_ts is None:
            self.calc_daily_average()

        # TODO: Masking in exported data

        from io import StringIO

        # Writing to a buffer
        # output = StringIO()
        # self.daily_ts.to_csv(output, mode='w', line_terminator='\n', float_format="%.3f", na_rep='---', header=False)


        with open(fname, 'w') as fid:
            fid.write('{0:15s}{1}\n'.format('Borehole:', self.name))
            fid.write('{0:15s}{1}\n'.format('Latitude:', self.latitude))
            fid.write('{0:15s}{1}\n'.format('Longitude:', self.longitude))
            fid.write('{0:15s}{1}\n'.format('Description:', self.description))
            fid.write('{0:15s}{1}\n'.format('SensorType:', ', '.join(['{0}'.format(f) for f in self.sensor_types])))
            fid.write('{0:15s}{1}\n'.format('DataUnits:', ', '.join(['{0}'.format(f) for f in self.sensor_data_units])))
            fid.write('{0:15s}{1}\n'.format('Depths:', ', '.join(['{0:.2f}'.format(f) for f in self.sensor_depths])))
            fid.write('# ')
            fid.write('-'*100)
            fid.write('\n')
            #fid.write(output.getvalue())
            self.daily_ts.to_csv(fid, mode='a', line_terminator='\n', float_format="%.3f", na_rep='---',
                                 header=False)

        # output.close()


    def get_limmits(self, fullts=False):
        """Get the date limits of the time series.

        :param fullts: (boolean, default False) use rawdata full time series
        :return:
        """
        if fullts:
            try:
                return [self.rawdata.index[0].date(), self.rawdata.index[-1].date()]
            except AttributeError:
                return [self.rawdata.index[0], self.rawdata.index[-1]]
        else:
            try:
                return [self.daily_ts.index[0].date(), self.daily_ts.index[-1].date()]
            except AttributeError:
                return [self.daily_ts.index[0], self.daily_ts.index[-1]]


    def calc_daily_average(self, mindata=1, threshold=0.9):
        """Calculates daily averages of borehole data

        Input arguments:
        mindata:      The minimum number of data per day, if less, date will be masked
        threshold:    The relative number of measurements that must be available per day.
                      Default 0.9 means that minimum 9 of 10 measurements must be available,
                      and not masked. If measurement interval is every 3 h, no measurements
                      must be missing. If interval is every 1 h, 2 measurements may be missing

        tz_unaware:   Flag to allow time zone unaware averaging
                      With physical data it makes most sense to calculate the
                      daily average based on local time

        Adds a daily_ts DataFrame to the borehole instance

        The dominant data frequency is calculated for each day in original timeseries,
        any date with more than 90% (default threshold) of the data (according to the
        dominant frequency) available will have an average calculated.

        Averages may be biased on days where the measurement frequencies changed to a faster
        frequency.

        This could be avoided by implementing a masking whenever there are more measurements
        available than the maximum possible with the dominant frequency.

        """

        # TODO: Handle masking when frequency changes consistently (so not just when a data point is missing...)

        if self.rawdata is None:
            return

        # TODO: Handle masks in calculation of daily averages
        grouped = self.rawdata.groupby(self.rawdata.index.date)
        daily_ts = grouped.mean()
        daily_count = grouped.count()
        daily_frequency = daily_count.median()

        # For data sources with a data interval of more than 1 day,
        # the aggregation to daily values result in an index consisting
        # of datetime.date objects.
        # To be sure we always have a pd.DatetimeIndex, we specifically convert.
        daily_ts.index = pd.DatetimeIndex(daily_ts.index)

        daily_ts = daily_ts[(daily_count >= daily_frequency * threshold) & (daily_count >= mindata)]

        start = self.rawdata.index[0]
        try:
            start = start.date()
        except:
            pass

        end = self.rawdata.index[-1]
        try:
            end = end.date()
        except:
            pass

        # if the frequency is daily or higher, fill missing days...

        td = ((daily_ts.index[-1] - daily_ts.index[0]) / len(daily_ts))
        if td.days <= 1:
            # fill any missing dates with NaN values (which is equivalent to "don't gap-fill": method=None)
            self.daily_ts = daily_ts.reindex(pd.date_range(start, end), method=None)
        else:
            # TODO: Implement fancy fuzzy frequency checking and reindexing
            self.daily_ts = daily_ts
            

    def get_MeanGT(self, lim=None, datelist=None, fullts=False, ignore_mask=False):
        # TODO: Implement masking in get_MeanGT, get_MinGT and get_MaxGT

        lim = fix_lim(lim)

        if not fullts:
            if self.daily_ts is None:
                self.calc_daily_average()

            #     if ignore_mask:
            #         print "Warning: ignore_mask option not implemented for averaged data!"
            
            df = self.daily_ts
            
            if datelist is not None:
                df = df.loc[datelist]
            
            if lim is not None:
                meanGT = df.loc[lim[0]:lim[1]].mean()
            else:
                meanGT = df.mean()

            meanGT = meanGT[meanGT.index.get_level_values('CoordZ') >= 0]

            #     # It is a little tricky to handle the possibility of a masked
            #     # value in the value-column. We have to treat data and masks
            #     # separately.
            #     dtype = [('depth', float), ('value', float)]
            #     dtype2 = [('depth', bool), ('value', bool)]
            #
            #     meanGT_data = np.array(meanGT, dtype=dtype)
            #     meanGT_mask = np.array(np.ma.getmaskarray(meanGT), dtype=dtype2)
            #
            #     meanGT2 = np.ma.array(meanGT_data, mask=meanGT_mask, dtype=dtype)
            #
            #     # insert depths in first column
            #     meanGT2['depth'] = np.ma.atleast_2d(np.ma.array(self.depths)).transpose()
            #
            #     return meanGT2

            return meanGT
        else:
            # TODO: Implement use of the full timeseries in get_MeanGT, get_MinGT and get_MaxGT
            raise NotImplementedError('Use of full time series not implemented.')
        #     meanGT = self.loggers[0].get_mean_GT(lim, ignore_mask)
        #
        #     # If more loggers, append data from these
        #     if len(self.loggers) > 1:
        #         for l in self.loggers[1:]:
        #             meanGT = np.append(meanGT, l.get_mean_GT(lim, ignore_mask))
        #
        #     # Return sorted recarray
        #     return np.sort(meanGT, order='depth', axis=0)
        #     # raise "Full timeseries support in get_MAGT is not implemented yet!"

    def get_MaxGT(self, lim=None, datelist=None, fullts=False, ignore_mask=False):

        lim = fix_lim(lim)

        if not fullts:
            if self.daily_ts is None:
                self.calc_daily_average()

            #     if ignore_mask:
            #         print "Warning: ignore_mask option not implemented for averaged data!"

            # Get max of timeseries
            df = self.daily_ts
            
            if datelist is not None:
                df = df.loc[datelist]
            
            if lim is not None:
                maxGT = df.loc[lim[0]:lim[1]].max()
            else:
                maxGT = df.max()

            maxGT = maxGT[maxGT.index.get_level_values('CoordZ') >= 0]

            #     dtype = [('depth', float), ('value', float), ('time', type(tsmax['time']))]
            #     maxGT = np.array(tsmax, dtype=dtype)
            #
            #     # insert depths in first column
            #     maxGT['depth'] = np.array(self.depths)
            #
            #     # Return sorted recarray
            #     return np.sort(maxGT, order='depth')

            return maxGT

        else:
            raise NotImplementedError('Use of full time series not implemented.')
        #     # Get max from first logger
        #     maxGT = self.loggers[0].get_max_GT(lim, ignore_mask)
        #
        #     # If more loggers, append data from these
        #     if len(self.loggers) > 1:
        #         for l in self.loggers[1:]:
        #             maxGT = np.append(maxGT, l.get_max_GT(lim, ignore_mask))
        #     # Return sorted recarray
        #     return np.sort(maxGT, order='depth', axis=0)

    def get_MinGT(self, lim=None, datelist=None, fullts=False, ignore_mask=False):

        lim = fix_lim(lim)

        if not fullts:
            if self.daily_ts is None:
                self.calc_daily_average()

            #     if ignore_mask:
            #         print "Warning: ignore_mask option not implemented for averaged data!"

            # Get max of timeseries
            df = self.daily_ts
            
            if datelist is not None:
                df = df.loc[datelist]
            
            if lim is not None:
                minGT = df.loc[lim[0]:lim[1]].min()
            else:
                minGT = df.min()
            
            minGT = minGT[minGT.index.get_level_values('CoordZ') >= 0]

            #     dtype = [('depth', float), ('value', float), ('time', type(tsmin['time']))]
            #     minGT = np.array(tsmin, dtype=dtype)
            #
            #     # insert depths in first column
            #     minGT['depth'] = np.array(self.depths)
            #
            #     # Return sorted recarray
            #     return np.sort(minGT, order='depth')

            return minGT

        else:
            raise NotImplementedError('Use of full time series not implemented.')
        #     # Get min from first logger
        #     minGT = self.loggers[0].get_min_GT(lim, ignore_mask)
        #
        #     # If more loggers, append data from these
        #     if len(self.loggers) > 1:
        #         for l in self.loggers[1:]:
        #             minGT = np.append(minGT, l.get_min_GT(lim, ignore_mask))
        #
        #     # Return sorted recarray
        #     return np.sort(minGT, order='depth', axis=0)

    def get_ALT(self, end_date=None, nyears=1, lim=None, depths=None):
        # TODO: lim string input is not working!
        if end_date is None:
            end_date = self.rawdata.index[-1].date()

        if lim is None:
            lim = nyears2lim(end_date, 1)

        if (lim[0] > self.daily_ts.index[-1].date()) or (lim[1] < self.daily_ts.index[0].date()):
            raise IndexError('Requested date range is outside timeseries.')

        # prepare to filter on depths
        if (depths is None) or (type(depths) == str and depths.lower() == 'all'):
            depths = self.sensor_depths

        if not hasattr(depths, '__iter__'):
            depths = [depths]

        # get max values within time limits
        maxGT = self.get_MaxGT(lim=lim)
        maxGT = maxGT[maxGT.notna()]

        # convert depths to column indices
        did = get_indices(maxGT.index.get_level_values('CoordZ'), depths)
        did = [i for i in did if not np.isnan(i)]  # remove nan values
        

        maxT = maxGT[did].values
        maxT_d = maxGT[did].index.get_level_values('CoordZ').values

        # z0 = find_zero(maxT, maxT_d)[0]
        z0 = find_zero(maxT, maxT_d)
        
        if len(z0) == 0:
            z0 = np.NaN

        elif maxT[0] > 0.:
            z0 = z0[0]
        else:
            z0 = np.NaN
            #raise ValueError('Upper most temperature is negative, get_ALT cannot calculate thickness of active layer')

        # return z0, maxT, maxT_d
        return z0, maxGT

    def get_zaa_stats(self, fullts=False, lim=None, end_date=None, nyears=None):
        if not fullts:
            if self.daily_ts is None:
                self.calc_daily_average()

            if lim is None and nyears is not None:
                if end_date is None:
                    end_date = self.daily_ts.index[-1]
                lim = nyears2lim(end_date, nyears)
        else:
            raise NotImplementedError('Full timeseries handling not implemented!')
            # TODO: Implement full time series in zaa stats

        zaa_stats = self.daily_ts.loc[lim[0]:lim[1]].describe().transpose()

        zaa_stats = zaa_stats[zaa_stats.index.get_level_values('CoordZ') >= 0]
        zaa_stats = zaa_stats.sort_values('CoordZ')
        
        d_stats = {}

        if len(np.where(zaa_stats['max'] - zaa_stats['min'] <= 0.1)[0]) < 1:
            # No depths has a difference between Tmax and Tmin of less than 0.1C
            # Thus, take the deepest depth
            d_stats['Dzaa'] = np.max(self.sensor_depths)
            did = zaa_stats.index.get_level_values('CoordZ') == d_stats['Dzaa']
            d_stats['Tzaa'] = zaa_stats[did]['mean'].values[0]
            d_stats['Tstd'] = zaa_stats[did]['std'].values[0]
            #d_stats['Tzaa'] = zaa_stats[did]['mean']
            #d_stats['Tstd'] = zaa_stats[did]['std']
            d_stats['exact'] = False
        else:
            # We do have zaa in range. Use first depth with Tmax-Tmin < 0.1C
            id = np.where(zaa_stats['max'] - zaa_stats['min'] <= 0.1)[0][0]
            d_stats['Dzaa'] = zaa_stats.index.get_level_values('CoordZ')[id]
            d_stats['Tzaa'] = zaa_stats.iloc[id]['mean']
            d_stats['Tstd'] = zaa_stats.iloc[id]['std']
            d_stats['exact'] = True

        # if len(np.where(stats['std'] * 4 < 0.1)[0]) < 1:
        #     id = len(stats) - 1
        # else:
        #     id = np.where(stats['std'] * 4 < 0.1)[0][0]

        return d_stats

    def plot_timeseries(self, depths=None, plotmasked=False, fullts=False, legend=True,
             annotations=True, lim=None, show=False, figsize=(10,6), **kwargs):
        """Method to plot the daily timeseries of temperature data from the ground temperature sensors.

        :param depths:
        :param plotmasked:
        :param fullts: 
        :param legend:
        :param annotations:
        :param lim:
        :param kwargs:
        :return:
        """
         # TODO: TRANSLATE PLOT FUNCTION TO NEW MODULE - NOTHING DONE YET


        if 'axes' in kwargs:
            ax = kwargs.pop('axes')
        elif 'Axes' in kwargs:
            ax = kwargs.pop('Axes')
        elif 'ax' in kwargs:
            ax = kwargs.pop('ax')
        else:
            fh = plt.figure(figsize=figsize)
            ax = plt.axes()

        fh = ax.figure

        # Flag to toggle wether full timeseries from the loggers is used or
        # the averaged daily timeseries.
        if not fullts:
            if self.daily_ts is None:
                self.calc_daily_average()

            if lim is None:
                lim = self.get_limmits(fullts=False)

            # Ensure limit consists of two datetime objects
            lim = fix_lim(lim)

            if (depths is None) or (type(depths) == str and depths.lower() == 'all'):
                depths = self.sensor_depths

            if not hasattr(depths, '__iter__'):
                depths = [depths]

            depths = [d for d in depths if d > 0.0000001]

            # convert depths to column indices
            did = get_indices(self.daily_ts.columns.get_level_values('CoordZ'), depths)

            # get copies of the data and times
            # TODO: Reconsider way to handle getting only ground temperatures
            data = self.daily_ts[lim[0]:lim[1]].iloc[:, did].values
            ordinals = list(map(dt.datetime.toordinal, self.daily_ts[lim[0]:lim[1]].index))

            # TODO: Handle masked values
            #     if ignore_mask:
            #         mask = np.zeros(data.shape, dtype=bool)
            #         mask = np.where(data < -273.15, True, False)
            #         data.mask = mask
            #
            # Find the maximum and minimum temperatures, and round up/down
            mx = np.nanmax(np.ceil(np.nanmax(data)))
            mn = np.nanmin(np.floor(np.nanmin(data)))

            # Iterate over all ground temperature columns
            for c2id, cid in enumerate(did):

                # Option to plot also the masked data points, but keeping
                # a mask on anything that could not be a real temperature.
                # if plotmasked:
                #     mask = np.zeros(data.shape, dtype=bool)
                #     mask = np.where(data < -273.15, True, False)
                #     data.mask = mask
                #
                # if any(data.mask == False):
                lh = ax.plot_date(ordinals, data[:,c2id], '-', label="{0:.2f} m".format(depths[c2id]), picker=5, **kwargs)
                lh[0].tag = 'dataseries'

        else:
            if lim is None:
                lim = self.get_limmits(fullts=True)

            # Ensure limit consists of two datetime objects
            lim = fix_lim(lim)

            if (depths is None) or (type(depths) == str and depths.lower() == 'all'):
                depths = self.sensor_depths

            if not hasattr(depths, '__iter__'):
                depths = [depths]

            depths = [d for d in depths if d > 0.0000001]

            # convert depths to column indices
            did = get_indices(self.rawdata.columns.get_level_values('CoordZ'), depths)

            # get copies of the data and times
            # TODO: Reconsider way to handle getting only ground temperatures
            data = self.rawdata[lim[0]:lim[1]].iloc[:, did].values
            #ordinals = map(dt.datetime.toordinal, self.rawdata[lim[0]:lim[1]].index)
            ordinals = list(map(mpld.date2num, self.rawdata[lim[0]:lim[1]].index))
            

            # TODO: Handle masked values
            #     if ignore_mask:
            #         mask = np.zeros(data.shape, dtype=bool)
            #         mask = np.where(data < -273.15, True, False)
            #         data.mask = mask
            #
            # Find the maximum and minimum temperatures, and round up/down
            mx = np.nanmax(np.ceil(np.nanmax(data)))
            mn = np.nanmin(np.floor(np.nanmin(data)))

            # Iterate over all ground temperature columns
            for c2id, cid in enumerate(did):

                # Option to plot also the masked data points, but keeping
                # a mask on anything that could not be a real temperature.
                # if plotmasked:
                #     mask = np.zeros(data.shape, dtype=bool)
                #     mask = np.where(data < -273.15, True, False)
                #     data.mask = mask
                #
                # if any(data.mask == False):
                lh = ax.plot_date(ordinals, data[:,c2id], '-', label="{0:.2f} m".format(depths[c2id]), picker=5, **kwargs)
                lh[0].tag = 'dataseries'
        
            #raise NotImplementedError('Use of full timeseries data is not yet implemented.')
            # depth_arr = self.get_depths(return_indices=True)
            #
            # if (depths is None) or (type(depths) == str and depths.lower() == 'all'):
            #     depths = depth_arr['depth']
            #
            # for d, lid, cid in depth_arr:
            #     if d in depths:
            #         # following line is necessary in order for mpl not to connect
            #         # the points by a straight line.
            #         self.loggers[lid].timeseries.fill_missing_steps(tolerance=2)
            #
            #         # get copies of the data and times
            #         data = self.loggers[lid].timeseries.data[:, cid].copy()
            #         times = self.loggers[lid].timeseries.times.copy()
            #
            #         ax.hold(hstate)
            #
            #         # Option to plot also the masked data points, but keeping
            #         # a mask on anything that could not be a real temperature.
            #         if plotmasked:
            #             mask = np.zeros(data.shape, dtype=bool)
            #             mask = np.where(data < -273.15, True, False)
            #             data.mask = mask
            #         lh = ax.plot_date(times, data, '-', label="%.2f m" % d, picker=5, **kwargs)
            #         lh[0].tag = 'dataseries'
            #         hstate = True
        # end if

        if annotations:
            if ax.get_title() == '':
                ax.set_title(self.name)
            else:
                ax.set_title(ax.get_title() + ' + ' + self.name)

            ax.set_ylabel('Temperature [$^\circ$C]')
            ax.set_xlabel('Time')

        fh.axcallbacks = zoom_span.AxesCallbacks(ax)

        if legend:
            fh.legend(loc=7)
            fh.tight_layout()
            fh.subplots_adjust(right=0.85)
        else:
            fh.tight_layout()


        if show:
            plt.show(block=False)

        return plt.gca()


    def plot_date(self, date, annotations=True, fs=12, xlim=None, ylim=None, show=True, **kwargs):
        """Plot temperature depth profile on a specific date.

        :param date:
        :param annotations:
        :param fs:
        :param xlim:
        :param show:
        :param kwargs:
        :return:
        """
        if 'axes' in kwargs:
            ax = kwargs.pop('axes')
        elif 'Axes' in kwargs:
            ax = kwargs.pop('Axes')
        elif 'ax' in kwargs:
            ax = kwargs.pop('ax')
        else:
            fh = plt.figure()
            ax = plt.axes()

        gid = find(self.daily_ts.columns.get_level_values('CoordZ') >= 0)
        depths = np.take(self.daily_ts.columns.get_level_values('CoordZ'), gid).values

        self.sort_columns_by_depth()
        
        data = self.daily_ts.iloc[self.daily_ts.index.isin([date]), gid].values.flatten()
        
        if len(data) == 0:
            raise ValueError('The requested date ({0}) is not in the dataset.'.format(date))

        if 'color' not in kwargs and 'c' not in kwargs:
            kwargs['color'] = 'k'
        if 'linestyle' not in kwargs and 'ls' not in kwargs:
            kwargs['linestyle'] = '-'

        lh = ax.plot(data, depths, marker='.', markersize=7, **kwargs)

        lh[0].tag = 'gt'

        ax.axvline(x=0, linestyle=':', color='k')
        ax.axhline(y=0, linestyle='-', color='k')
        
        if xlim is not None:
            ax.set_xlim(xlim)
        
        if ylim is None:
            ylim = plt.get(ax, 'ylim')
        
        ax.set_ylim(ymax=min(ylim), ymin=max(ylim))        
        
        ax.get_xaxis().tick_top()
        ax.set_xlabel('Temperature [$^\circ$C]', fontsize=fs)
        ax.get_xaxis().set_label_position('top')
        ax.set_ylabel('Depth [m]', fontsize=fs)

        plt.xticks(fontsize=fs)
        plt.yticks(fontsize=fs)

        if annotations:
            t1h = ax.text(0.95, 0.10, self.name, horizontalalignment='right', \
                          verticalalignment='bottom', transform=ax.transAxes, fontsize=fs)
            t1h.tag = 'name'

            t2h = ax.text(0.95, 0.05, date, horizontalalignment='right', verticalalignment='bottom',
                          transform=ax.transAxes, fontsize=fs)
            t2h.tag = 'date'
            
            if len(data[~np.isnan(data)])==0:
                t3h = ax.text(0.95, 0.175, 'No data available', horizontalalignment='right', \
                              verticalalignment='bottom', transform=ax.transAxes, fontsize=fs)
                t3h.tag = 'no data'


        if show:
            plt.show(block=False)

        return ax

    def plot_trumpet(self, fullts=False, lim=None, end_date=None, args=None,
                     xlim=None, ylim=None, nyears=None, plotMeanGT=True, fs=12, ignore_mask=False,
                     depths='all', **kwargs):
        """
        Method to plot trumpet curve for specific time interval (usually 1 year)
        given by the date-range specified using the argument 'lim', or arguments 'end_date' and 'nyears'

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

        defaultargs = dict(
            plotMeanGT=plotMeanGT,
            maxGT=dict(
                linestyle='-',
                color='r',
                marker='.',
                markersize=7,
                zorder=5),
            minGT=dict(
                linestyle='-',
                color='b',
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
        
        figsize = kwargs.pop('figsize', (6.4,4.8))

        if 'axes' in kwargs:
            ax = kwargs.pop('axes')
        elif 'Axes' in kwargs:
            ax = kwargs.pop('Axes')
        elif 'ax' in kwargs:
            ax = kwargs.pop('ax')
        else:
            fh = plt.figure(figsize=figsize)
            ax = plt.axes()

        if not fullts:
            if self.daily_ts is None:
                self.calc_daily_average()

            if lim is None and nyears is not None:
                if end_date is None:
                    end_date = self.daily_ts.index[-1]
                lim = nyears2lim(end_date, nyears)
        else:
            raise NotImplementedError('Trumpet plotting based on full timeseries not implemented!')
            # TODO: Implement trumpet plotting based on full time series

        # prepare to filter on depths
        depth_arr = self.sensor_depths

        if (depths is None) or (type(depths) == str and depths.lower() == 'all'):
            depths = self.sensor_depths[self.sensor_depths >= 0]

        if not hasattr(depths, '__iter__'):
            depths = [depths]

        # get maximum values within time limits
        maxGT = self.get_MaxGT(lim=lim, fullts=fullts, ignore_mask=ignore_mask)
        # get minimum values within time limits
        minGT = self.get_MinGT(lim=lim, fullts=fullts, ignore_mask=ignore_mask)
        
        # remove any missing depths (NaN's)
        maxGT = maxGT[maxGT.notna()]
        minGT = minGT[minGT.notna()]

        # convert depths to column indices
        maxGT_depths = maxGT.index.get_level_values('CoordZ')
        did = get_indices(maxGT_depths, depths)
        did = [i for i in did if not np.isnan(i)]  # remove nan values
        ax.plot(maxGT[did], maxGT_depths[did], **args['maxGT'])

        minGT_depths = minGT.index.get_level_values('CoordZ')
        did = get_indices(minGT_depths, depths)
        did = [i for i in did if not np.isnan(i)]  # remove nan values
        ax.plot(minGT[did], minGT_depths[did], **args['minGT'])

        # ax.plot(maxGT[did], maxGT['depth'][did], **args['maxGT'])  # ,'-k',marker='.', markersize=7,**kwargs)
        # ax.plot(minGT[did], minGT['depth'][did], **args['minGT'])  # ,'-k',marker='.', markersize=7,**kwargs)

        # TODO: Implement fill between max and min GT in trumpet
        # if args.has_key('fill') and args['fill'] is not None:
        #     thisX = np.append(maxGT['value'][did], minGT['value'][did][::-1])
        #     thisY = np.append(maxGT['depth'][did], minGT['depth'][did][::-1])
        #     ax.fill(thisX, thisY, **args['fill'])  # facecolor=fclr,edgecolor='none')
        
        if 'vline' in args and args['vline'] is not None:
            ax.axvline(x=0, **args['vline'])

        if 'plotMeanGT' in args and args['plotMeanGT']:
            # get maximum values within time limits
            meanGT = self.get_MeanGT(lim=lim, fullts=fullts, ignore_mask=ignore_mask)

            # remove any missing depths (NaN's)
            maxGT = meanGT[meanGT.notna()]

            # remove values above ground (z-axis is positive downwards)
            meanGT = meanGT[meanGT.index.get_level_values('CoordZ') >= 0]

            # convert depths to column indices
            meanGT_depths = meanGT.index.get_level_values('CoordZ')
            did = get_indices(meanGT_depths, depths)
            did = [i for i in did if not np.isnan(i)]  # remove nan values
            ax.plot(meanGT[did], meanGT_depths[did], **args['MeanGT'])


        if ylim is None:
            ylim = plt.get(ax, 'ylim')
            
        ax.set_ylim(ymax=min(ylim), ymin=max(ylim))

        if xlim is not None:
            ax.set_xlim(xlim)

        if 'grid' in args and args['grid'] is not None:
            ax.grid(True, **args['grid'])

        ax.get_xaxis().tick_top()
        ax.set_xlabel('Temperature [$^\circ$C]', fontsize=fs)
        ax.get_xaxis().set_label_position('top')
        ax.set_ylabel('Depth [m]', fontsize=fs)

        plt.xticks(fontsize=fs)
        plt.yticks(fontsize=fs)

        if args['title'] is not None:
            ax.text(0.95, 0.05, args['title'], verticalalignment='bottom',
                    horizontalalignment='right', transform=ax.transAxes, fontsize=fs)

        ax.lim = lim
        return ax

    def plot_surf(self, depths=None, ignore_mask=False, fullts=False, legend=True,
                  annotations=True, lim=None, figsize=(15, 6),
                  cmap=plt.cm.bwr, show_sensors=True, cax=None,
                  cont_levels=[0], **kwargs):
        """Method to plot surface plot of daily time series data.

        :param depths:
        :param ignore_mask:
        :param fullts:
        :param legend:
        :param annotations:
        :param lim:
        :param figsize:
        :param cmap:
        :param show_sensors:
        :param cax:
        :param cont_levels:
        :param kwargs:
        :return:
        """

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
            fh = plt.figure(figsize=figsize, facecolor=figBG)
            ax = plt.axes(rect1)
            ax.set_facecolor(axesBG)

        if fh is None:
            fh = ax.get_figure()

        # Flag to toggle wether full timeseries from the loggers is used or
        # the averaged daily timeseries.
        if not fullts:
            if self.daily_ts is None:
                self.calc_daily_average()

            if lim is None:
                lim = [self.daily_ts.index[0], self.daily_ts.index[-1]]
            else:
                lim = fix_lim(lim)

            if (depths is None) or (type(depths) == str and depths.lower() == 'all'):
                depths = list(np.array(self.sensor_depths)[np.array(self.sensor_depths) >= 0])

            if not hasattr(depths, '__iter__'):
                depths = [depths]

            # convert depths to column indices
            did = get_indices(self.sensor_depths, depths)

            # following line is necessary in order for mpl not to connect
            # the points by a straight line.

            # get copies of the data and times
            # TODO: Reconsider way to handle getting only ground temperatures
            data = self.daily_ts[lim[0]:lim[1]].iloc[:, did].values
            ordinals = list(map(dt.datetime.toordinal, self.daily_ts[lim[0]:lim[1]].index))
            #depths = self.daily_ts.columns.get_level_values('CoordZ')[did]
            #depths = depths[depths >= 0]
            depths = np.array(depths)
            sensor_depths = depths

            # Handle masked (NaN) values by interpolating 1-dimensionally along depth axis
            # ... if a point exists both above and below the missing point(s)
            
            # first convert to dataframe, as there is a nice 
            # convenience function for interpolation along columns
            df = pd.DataFrame(data, columns=depths)
            df2 = pd.DataFrame(data, columns=depths)
            
            # insert additional depths, node points midway between each sensor
            new_depths = depths[0:-1] + np.diff(depths)/2
            
            # add new columns with NaN values
            for d in new_depths:
                df[d] = np.NaN
            df = df.reindex(sorted(df.columns), axis=1)  # reindex to make sure columns are in ascending depth order
            
            # Then do the interpolation inplace on the dataframe
            # and interpolate only points surrounded by valid data.
            df.interpolate(method='linear', axis=1, limit_area='inside', inplace=True)

            # Now reintroduce the original NaNs
            for c in df2.columns:
                df[c] = df2[c]
            
            # Make backup of data, and reasign the interpolated data
            data_bak = data
            depths_bak = depths
            data = df.values
            depths = np.array(df.columns)
            
            
            
            # Find the maximum and minimum temperatures, and round up/down
            mxn = np.nanmax(np.abs([np.floor(np.nanmin(data)),
                                    np.ceil(np.nanmax(data))]))
            levels = np.arange(-mxn, mxn + 1)
            
            xx, yy = np.meshgrid(ordinals, depths)

            cf = ax.contourf(xx, yy, data.T, levels, cmap=cmap)

            if cont_levels is not None:
                ct = ax.contour(xx, yy, data.T, cont_levels, colors='k')
                cl = ax.clabel(ct, cont_levels, inline=True, fmt='%1.1f $^\circ$C', fontsize=8, colors='k')

            if show_sensors:
                xlim = ax.get_xlim()
                ax.plot(np.ones_like(sensor_depths) * xlim[0] + 0.01 * np.diff(xlim), sensor_depths, 'ko', ms=3)

            ax.invert_yaxis()
            ax.xaxis_date()

            cbax = plt.colorbar(cf, orientation='horizontal', ax=cax, shrink=1.0, aspect=30, fraction=0.05)
            cbax.set_label('Temperature [$^\circ$C]')
            fh.autofmt_xdate()
        else:
            raise NotImplementedError('Full timeseries support not implemented!')
        # end if

        # if legend:
        #     ax.legend(loc='best')

        if annotations:
            if ax.get_title() == '':
                ax.set_title(self.name)
            else:
                ax.set_title(ax.get_title() + ' + ' + self.name)

            ax.set_ylabel('Depth [m]')
            # ax.set_xlabel('Time [year]')

        # plt.draw()

        return plt.gca()


def split_year_date_lists(start='1960-08-01', end=dt.date.today(), month=8, day=1):
    """Calculates lists of start and end dates of year slices, specified by 'start' and 'end' arguments,
    and splitting on the date specified by the 'month' and 'day' arguments.
    The first start date in the list may be before the specified start date, while the first end date
    will be always be after the specified start date.
    The last start date will always be before the specified end date, while


    :param start:
    :param end:
    :param month:
    :param day:
    :return:
    """

    lim = fix_lim([start, end])

    # We want have the first split year encompas the specified start date
    # Thus, if the start date is before ????-month-day, go one year back

    if lim[0] < dt.date(lim[0].year, month, day):
        lim[0] = lim[0].replace(year=lim[0].year-1)

    # We want the last split year to encompas the specified end date
    # Thus, if the end date is after ????-month-day minus 1 day, add an extra year

    if lim[1] > dt.date(lim[1].year, month, day)-dt.timedelta(days=1):
        lim[1] = lim[1].replace(year=lim[1].year + 1)

    # Create list of start dates (XXXX-08-01) from first year of measurements to the current year
    sdate = [dt.date(yr, month, day) for yr in range(lim[0].year, lim[1].year)]
    edate = [dt.date(yr, month, day)-dt.timedelta(days=1) for yr in range(lim[0].year+1, lim[1].year+1)]

    return sdate, edate



def plot_trumpet(bhole, end_date=None, nyears=1, lim=None, xlim=None, ylim=None, depths=None, annotate=True, Tzaa=None, Dzaa=None, Tstd=None, **kwargs):
    args = dict(
        maxGT=dict(
            linestyle='-',
            lw=2,
            color='r',
            zorder=5,
            label='Maximum'),
        minGT=dict(
            linestyle='-',
            lw=2,
            color='b',
            zorder=5,
            label='Minimum'),
        MeanGT=dict(
            linestyle='-',
            lw=2,
            color='g',
            zorder=5,
            label='Average'),
        fill=None,
        vline=dict(
            linestyle='--',
            lw=1.5,
            color='k',
            zorder=2),
        grid=dict(
            linestyle='-',
            color='0.5',
            zorder=0),
        title=None)


    if lim is None:
        bhole.plot_trumpet(end=end_date, nyears=1, xlim=xlim, ylim=ylim, args=args, depths=depths, **kwargs)
        lim = nyears2lim(bhole.daily_ts.index[-1], 1)
    else:
        # pdb.set_trace()
        lim = fix_lim(lim)  # ensure dates
        bhole.plot_trumpet(lim=lim, xlim=xlim, ylim=ylim, args=args, depths=depths, **kwargs)

    z0, maxGT = bhole.get_ALT(lim=lim, depths=depths)
    # z0, maxT, maxT_d = bhole.get_ALT(end_date=end_date, nyears=nyears,
    #                           lim=lim, depths=depths)

    #z0 = z0[0]  # for now use only first zero crossing

    zaa_stats = bhole.get_zaa_stats(lim=lim)
    Tzaa = zaa_stats['Tzaa']
    Tstd = zaa_stats['Tstd']
    Dzaa = zaa_stats['Dzaa']
    Dzaa_exact = zaa_stats['exact']

    p1 = mpl.patches.Rectangle((-50, z0 - 50), 100, 100, ec='none', fc='#FFE4E4', zorder=-10)
    plt.gca().add_patch(p1)
    p2 = mpl.patches.Rectangle((-50, z0), 100, 100, ec='k', fc='#99CCFF', zorder=-5)
    plt.gca().add_patch(p2)

    tit = 'Hole: {0}'.format(bhole.name)

    period = [l.year for l in lim]
    if period[0] == period[1]:
        tit = tit + '\nPeriod: {0:d}'.format(period[0])
    else:
        tit = tit + '\nPeriod: {0:d}-{1:d}'.format(*[l.year for l in lim])

    tit = tit + '\nALT: {0:.2f} m'.format(z0)

    if Dzaa_exact:
        tit = tit + '\nT$_{{zaa}}$: {0:.1f} $^{{\circ}} \mathrm{{C}}$ @ {1:.1f} m'.format(Tzaa, Dzaa)
    else:
        tit = tit + '\nT: {0:.1f} $^{{\circ}} \mathrm{{C}}$ @ {1:.1f} m'.format(Tzaa, Dzaa)

    leg = plt.legend(loc='lower left', title=tit, fontsize=12)
    leg._legend_box.align = "left"

    ax = plt.gca()
    
    if ylim is None:
        ylim = ax.get_ylim()
        ax.set_ylim([ylim[0], 0])
        
    ylim = ax.get_ylim()
    yspan = np.diff(ylim)
    trans = transforms.blended_transform_factory(ax.transAxes, ax.transData)

    if annotate:
        ax.text(0.97, z0 + 0.02 * yspan, 'Active layer', transform=trans,
                ha='right', va='bottom', fontsize=14)
        ax.text(0.97, z0 - 0.02 * yspan, 'Permafrost', transform=trans,
                ha='right', va='top', fontsize=14)

    return (plt.gcf(), maxGT)


# This code refers to the old borhole module... should not be needed now...
#def plot_surf(bhole, depths=None, ignore_mask=False, fullts=False, legend=True,
#              annotations=True, lim=None, figsize=(15, 6),
#              cmap=plt.cm.bwr, sensor_depths=True, cax=None,
#              cont_levels=[0]):
#    ax = bh.plot_surf(bhole, depths=depths, ignore_mask=ignore_mask, fullts=fullts, legend=legend,
#                      annotations=annotations, lim=lim, figsize=figsize,
#                      cmap=cmap, sensor_depths=sensor_depths, cax=cax,
#                      cont_levels=cont_levels)
#    return ax