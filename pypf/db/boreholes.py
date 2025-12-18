# Python Version: 3.9
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
from collections import OrderedDict, Counter
from rich import print

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


# Install instructions

# conda create -n boreholes --python=3.9
# conda install ipython matplotlib pandas numpy lxml openpyxl
# conda install -c conda-forge ipdb pyproj rich
# conda install -c anaconda pytables

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

    return fix_lim([start, end_date])


def fix_lim(lim, tzinfo=dt.timezone.utc):
    if lim is None:
        return

    newlist = []
    for lid, thistime in enumerate(lim):
        if type(thistime) == str:
            mytime = dateutil.parser.parse(thistime, yearfirst=True, dayfirst=False)
            newlist.append(mytime.replace(tzinfo=tzinfo))
            #newlist.append(dateutil.parser.parse(thistime, yearfirst=True, dayfirst=False).date())
        elif type(thistime) in [dt.date]:
            newlist.append(dt.datetime(*thistime.timetuple()[0:6],tzinfo=tzinfo))
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
    #
    # This function is used in simple calculation of thaw depth.

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



def calc_thaw_depth_at_node(data, depths, node):
    """
    Method to calculate thawd epth on a specific date below the specified node.
    Thus, the temperature at the specified node should be above zero, and
    the temperature at node+1 should be below zero.
    The method will throw an error, if this is not the case.

    It will calculate three different estimates of the thaw depth:
    a) Extrapolation from two points just above the frost table
    b) Extrapolation from two points below the frost table
    c) Interpolation from one point above and one below the frost table
    
    Args:
    data : array
        Array of temperature values.
    depths : array
        Array of depth values.
    node : int
        Index of the node immediately above the frost table.

    Returns:
    thawd : array
        Array of estimated thaw depths.
    rel_d : float
        Relative depth of the frost table.

    """

    #raise NotImplementedError('Not implemented!')

    def x_at_y_eq_0(x_1, x_2, y_1, y_2):
        """
        Calculates the value of x at y=0, thus the crossing point of the
        x-axis. According to the general equation:

        x_0 = x_1 - y_1 / ((y_2-y_1)/(x_2-x_1))
        """
        if y_1 == y_2:
            raise ValueError('Cannot interpoloate: Temperatures at the two points are identical')
        return x_1 - y_1 / ((y_2 - y_1) / (x_2 - x_1))

    # if fullts:
    #     raise NotImplementedError("Thaw depth calculation based on full time series is not implemented!")

    # if self.daily_ts == None:
    #     self.calc_daily_avg()

    # # Get the index of the day in question
    # did = find(self.daily_ts.times == date)

    # if not did:
    #     raise ValueError("Date does not exist in data set") 

    # # Select only ground temperatures
    # GTid = find(np.array(self.depths) >= 0.)
    # data = self.daily_ts.data[did, GTid].flatten()
    # depths = np.take(self.depths, GTid)

    # ids = np.argsort(depths)
    # depths = depths[ids]
    # data = data[ids]

    # if node < 2 or node > len(depths)-3:
    #     raise ValueError('node-argument must be larger than 2 and less than number of depths (excluding above ground depths) minus 2.')

    if data[node] < 0. or data[node+1] > 0.:
        raise ValueError('node argument should be depth index for node immediately above frost table.')

    # # Ensure that no nan values are present at node-1, node, node+1 and node+2
    # if np.any(np.isnan(data[node-1:node+2])) or np.any(np.isnan(depths[node-1:node+2])):
    #     raise ValueError("Data or depths contain NaN values.")

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

    if node >= 1:
        # Calculate based on two points above frost table
        thawd[0] = x_at_y_eq_0(
                depths[node-1],
                depths[node],
                data[node-1],
                data[node])
        thawd.mask[0] = False

    if node >= 0 and data[node] >= 0. and data[node+1] < 0.:
        # Calculate based on one point above and below frost table
        thawd[1] = x_at_y_eq_0(
                depths[node],
                depths[node+1],
                data[node],
                data[node+1])
        thawd.mask[1] = False

    if node+2 < len(data) and np.all(data[node+1:node+3] < 0.) :
        # Calculate based on two points below frost table
        thawd[2] = x_at_y_eq_0(
                depths[node+1],
                depths[node+2],
                data[node+1],
                data[node+2])
        thawd.mask[2] = False

    return thawd




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
            print('[bold yellow]Data contains duplicate entries... consider cleaning up database.[/bold yellow]')
            print('[bold yellow]Discarding duplicate values...[/bold yellow]')
            df.drop_duplicates(subset=['TimeStamp', 'SensorID', 'DataTypeID', 'Value'], inplace=True, keep='last')
        
        df_dup = df.duplicated(subset=['TimeStamp', 'SensorID', 'DataTypeID'], keep=False)
        if any(df_dup):
            date_list = DataSet.get_time(list(df[df_dup]['TimeStamp']))
            date_list = pd.to_datetime(date_list).normalize().unique().tolist()
            date_list = [d.date() for d in date_list]
            print('[bold red]>>  Data contains duplicate entries with different values...[/bold red]')
            print('[bold red]>>  TimeStamp, SensorID and DataTypeID are identical, but values are different.[/bold red]')
            print('[bold red]>>  This indicates a timezone issue when registering the data...[/bold red]')
            print('[red]>>  Please investigate dates: [/red]')
            for d in date_list:
                print('[red]>>         {0}[/red]'.format(d))
            print('[bold red]>>  For now... Discarding duplicate values (keep="last")...[/bold red]')
            df.drop_duplicates(subset=['TimeStamp', 'SensorID', 'DataTypeID'], inplace=True, keep='last')

        # decode bytestrings read from hdf5
        def decode_strings(thisdf):
            str_df = thisdf.select_dtypes([object])
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
            SensorIDs = self.rawdata_mask.columns.get_level_values(0)
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

    def merge(self, other, method='replace'):
        """Merges two boreholes, so that measurements from the two time series are merged and appear as one.
        No checking or corrections are performed.
        Only measurements from other that occur later than last measurement of self are considered.
        There is no keeping track of calibration status.

        Method indicates how duplicate timestamps should be handled.
        'keep' will keep own data
        'replace' will replace duplicate timestamps with data from other
        """
        
        if method == 'keep':
            # The merging will keep data from self, and only fill with data from other
            # outside the date interval covered by self (no gap filling)
            outbh = copy.deepcopy(self)
            idx = np.logical_or(other.rawdata.index<self.rawdata.index[0], other.rawdata.index>self.rawdata.index[-1])
            idx_mask = np.logical_or(other.rawdata_mask.index<self.rawdata_mask.index[0], other.rawdata_mask.index>self.rawdata_mask.index[-1])
            outbh.rawdata = pd.concat([self.rawdata, other.rawdata[idx]], axis='index', sort=True)
            outbh.rawdata_mask = pd.concat([self.rawdata_mask, other.rawdata_mask[idx_mask]], axis='index', sort=True)
        elif method == 'replace':
            # The merging will replace data from self with data from other, where there is
            # overlap.
            outbh = copy.deepcopy(other)
            idx = np.logical_or(self.rawdata.index<other.rawdata.index[0], self.rawdata.index>other.rawdata.index[-1])
            idx_mask = np.logical_or(self.rawdata_mask.index<other.rawdata_mask.index[0], self.rawdata_mask.index>other.rawdata_mask.index[-1])
            outbh.rawdata = pd.concat([self.rawdata[idx], other.rawdata], axis='index', sort=True)
            outbh.rawdata_mask = pd.concat([self.rawdata_mask[idx_mask], other.rawdata_mask], axis='index', sort=True)
        else:
            raise ValueError('Unknown method of handling duplicates: {0}'.format(method))
        
        outbh.sort_columns_by_depth()
        outbh.sensor_depths = np.array(outbh.rawdata.columns.get_level_values('CoordZ'))

        outbh.apply_mask()
        outbh.calc_daily_average_adaptive()
        return outbh

    def __add__(self, other):
        return self.merge(other)

    def sort_columns_by_depth(self):
        if hasattr(self, 'rawdata') and self.rawdata is not None:
            self.rawdata.sort_index(axis=1, level='CoordZ', sort_remaining=False, inplace=True)
        if hasattr(self, 'rawdata_mask') and self.rawdata_mask is not None:
            self.rawdata_mask.sort_index(axis=1, level='CoordZ', sort_remaining=False, inplace=True)
        if hasattr(self, 'daily_ts') and self.daily_ts is not None:
            self.daily_ts.sort_index(axis=1, level='CoordZ', sort_remaining=False, inplace=True)
        

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
            self.daily_ts.to_csv(fid, mode='a', lineterminator='\n', float_format="%.3f", na_rep='---',
                                 header=False)

        # output.close()


    def get_limits(self, fullts=False):
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


    def calc_daily_average(self, mindata=1, threshold=0.9, method='adaptive'):
        """Wrapper function, standard is now to use adaptive algorithm"""
        if method.lower() == 'adaptive':
            self.calc_daily_average_adaptive(mindata=1, threshold=0.9)
        elif method.lower() == 'simple':
            self.calc_daily_average_simple(mindata=1, threshold=0.9)
        else:
            raise ValueError('Averaging method not recognized ({0})'.format(method))


    def calc_daily_average_simple(self, mindata=1, threshold=0.9):
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


    def _infer_frequencies(self, plot=False):

        def f(x,a,b,c): 
            """
            a: value before step
            b: value after step
            c: the x-value at which the step occurs
            
                        |------------ b 
                        | 
            a ----------|
                        c
            """
            return np.heaviside(x-c,0)*(b-a)+a # Heaviside fitting function
    
        def obj(c, xdat, yobs, a, b): 
            """Objective function for fitting heaviside step"""
            return np.sum(np.square(yobs-f(xdat, a, b, c)))

        def calc_ssq(xdat, ydat, a, b):
            # calculate sum-of-squares for step-up function
            obj_f = lambda c: obj(c, xdat, ydat, a, b)
            ssq = list(map(obj_f, xdat))
            lsq_id = np.argmin(ssq)
            return ssq, lsq_id

        def apply_change_points(row, date=None, freq1=None, freq2=None, **kwargs):
            """Function to assign nominal time step to entries close to a step change in timestep size.
            Intended to be applied to a subset of rows from a dataframe, taken right before
            and after the step change."""
            cp_date = pd.to_datetime(date).tz_localize(row.name.tz)
            if row.name <= cp_date:
                return freq1
            else:
                return freq2

        # calculate timesteps in hours
        td = [d.total_seconds()/3600 for d in np.diff(self.rawdata.index)]
        # we consider the timestep a look-ahead value, it is the time difference
        # between the current timestamp and the one immediately after.
        
        # create dataframe with date information
        annotated_df = pd.DataFrame(td, index=self.rawdata.index[0:-1])
        annotated_df.columns = ['freq']
        annotated_df['year'] = annotated_df.index.year
        annotated_df['month'] = annotated_df.index.month
        annotated_df['day'] = annotated_df.index.day
        annotated_df['week'] = annotated_df.index.isocalendar().week
                    
        # keep only timesteps of less than/equal to 24h 
        # anything larger will be data gaps
        pruned_df = annotated_df[annotated_df['freq']<=24.]

        # find weeks where major changes occur
        freq_by_week_df = pruned_df.groupby(['year','week'])['freq'].mean().round()
        tmp = np.nonzero(freq_by_week_df.diff().values)

        detected_change_points = []
        if len(tmp[0]) > 1:
            change_ids = list(tmp[0][1:] )
            
            # Loop over identified changes                  
            for idx in change_ids:
                # actual week of change
                week_of_year_index = (pruned_df['year']==freq_by_week_df.index[idx][0]) & (pruned_df['week']==freq_by_week_df.index[idx][1])
                # first date is 3 days before
                start_date = (pruned_df.iloc[np.nonzero(week_of_year_index.values)[0][0]].name - dt.timedelta(days=3)).date()
                # last date is 3 days after
                end_date = (pruned_df.iloc[np.nonzero(week_of_year_index.values)[0][-1]].name + dt.timedelta(days=3)).date()

                # Select the appropriate dates in the range
                subset_index = (pruned_df.index > start_date.isoformat()) & (pruned_df.index <= end_date.isoformat())
                subset = pruned_df[subset_index].round()
                        
                # Count the occurences of different timesteps    
                cnt = Counter(subset.groupby(['year','month','day'])['freq'].mean().round())

                # get the two most common timestep durations
                # sorted in ascending order
                most_common = sorted([k for (k,c) in cnt.most_common(2)])

                if len(most_common) < 2:
                    # No relevant changes identified
                    continue
                
                if any(np.array([cnt.get(most_common[idc]) for idc in np.arange(2)]) <= 2):
                    # We should have at least 3 days with this timestep in the time series
                    # We don't, so we treat them as outliers
                    
                    # pdb.set_trace()
                    
                    continue
                    
                xdat = np.arange(len(subset))
                ydat = subset['freq']
                
                # calculate sum-of-squares for step-up function
                ssq_a, lsq_a_id = calc_ssq(xdat, ydat, most_common[0], most_common[1])
            
                # calculate sum-of-squares for step-down function
                ssq_b, lsq_b_id = calc_ssq(xdat, ydat, most_common[1], most_common[0])
                
                # calculate sum-of-squares for no step level a 
                ssq_a2, lsq_a2_id = calc_ssq(xdat, ydat, most_common[0], most_common[0])

                # calculate sum-of-squares for no step level b
                ssq_b2, lsq_b2_id = calc_ssq(xdat, ydat, most_common[1], most_common[1])
                
                if min([ssq_a2[lsq_a2_id],ssq_b2[lsq_b2_id]]) < min([ssq_a[lsq_a_id],ssq_b[lsq_b_id]]):
                    # straight line is better fit than step function
                    continue
                    
                # Select best fitting parameters of step-function
                tsdat = subset.index      
                if ssq_a[lsq_a_id] <= ssq_b[lsq_b_id]:
                    a = most_common[0]
                    b = most_common[1]
                    lsq_id = lsq_a_id
                else:
                    a = most_common[1]
                    b = most_common[0]
                    lsq_id = lsq_b_id
                
                # Calculate the inferred nominal timestep intervals
                ydat = f(xdat, a, b, xdat[lsq_id])
                
                # Select best fitting parameters of straight line
                if ssq_a2[lsq_a2_id] <= ssq_b2[lsq_b2_id]:
                    af = most_common[0]
                    bf = most_common[0]
                    ydat_flat = f(xdat, af, bf, xdat[lsq_a2_id])
                else:
                    a = most_common[1]
                    b = most_common[1]
                    ydat_flat = f(xdat, af, bf, xdat[lsq_b_id])        
                        
                # Store result of fitting
                change_point = {}
                change_point['freq1'] = a
                change_point['freq2'] = b
                change_point['date'] = (subset.iloc[lsq_id].name.date() + dt.timedelta(days=1)).isoformat()
                change_point['year'] = (subset.iloc[lsq_id].name.date() + dt.timedelta(days=1)).year
                change_point['month'] = (subset.iloc[lsq_id].name.date() + dt.timedelta(days=1)).month
                change_point['week'] = (subset.iloc[lsq_id].name.date() + dt.timedelta(days=1)).isocalendar().week
                
                detected_change_points.append(change_point)
                
                if plot:
                    # plot resulting fit
                    plt.figure(figsize=(12,2))
                    plt.plot(subset.index, subset['freq'], '.r')
                    plt.plot(subset.index, ydat, '-b')
                    plt.plot(subset.index, ydat_flat, ':b')
                    plt.show(block=False)
        
        freq_by_week_df = freq_by_week_df.reset_index()
        joined_df = pd.merge(annotated_df, freq_by_week_df, on=['year', 'week'], how='left')
        annotated_df['nominal_freq'] = joined_df['freq_y'].values

        # Ensure week 52 extending into new year has correct value in the new year
        years = annotated_df[(annotated_df['week'] == 52) & (annotated_df['month'] == 1)]['year'].unique()
        for yr in years:
            idx = (annotated_df['year'] == yr) & (annotated_df['week'] == 52) & (annotated_df['month'] == 1)
            annotated_df[idx] = freq_by_week_df[(freq_by_week_df['year'] == yr-1) & (freq_by_week_df['week'] == 52)]['freq'].values[0]

        if len(detected_change_points) > 0:
            # If change points were detected
            # adjust frequencies in vicinitiy of change points
            dcp_df = pd.DataFrame(detected_change_points)
            for cpid, cp in dcp_df.iterrows():
                # find week of year where change is occuring
                idx = (annotated_df['year'] == cp['year']) & (annotated_df['week'] == cp['week'])
                # apply freq1 before change point, and freq2 after
                annotated_df.loc[idx,'nominal_freq'] = annotated_df.loc[idx].apply(apply_change_points, axis=1, args=(cp['date'], cp['freq1'], cp['freq2']))
                
                # handle week 52 spilling over into new year
                if (cp['week'] == 52) & (cp['month'] == 12):
                    # ... if change point is in old year
                    idx = (annotated_df['year'] == cp['year']+1) & (annotated_df['week'] == 52) & (annotated_df['month'] == 1)
                elif (cp['week'] == 52) & (cp['month'] == 1):
                    # ... if change point is in new year
                    idx = (annotated_df['year'] == cp['year']-1) & (annotated_df['week'] == 52) & (annotated_df['month'] == 12)
                    
                if any(idx):
                    annotated_df.loc[idx,'nominal_freq'] = annotated_df.loc[idx].apply(apply_change_points, axis=1, args=(cp['date'], cp['freq1'], cp['freq2']))
        
        if plot:
            # plot resulting fit
            plt.figure(figsize=(12,2))
            plt.plot(annotated_df.index, annotated_df['nominal_freq'], '-b')
            plt.plot(annotated_df.index, annotated_df['freq'], ':r')
            plt.gca().set_ylim([0,10])
            plt.show(block=False)
        
        return detected_change_points, annotated_df


    def calc_daily_average_adaptive(self, mindata=1, threshold=0.9):
        """Calculates daily averages of borehole data using an adaptive algorithm
        to estimate the expected daily frequency of measurements.

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

        The expected data frequency is calculated with an adaptive algorithm, allowing step 
        changes in data frequency to occur in the time series.
        Any date with more than 90% (default threshold) of the expected data (according to the
        adaptive frequency calculation) available will have an average calculated.

        The adaptive frequency determination assigns the faster measurement frequency to days
        where the frequency changes, which will typically result in that day being masked due to 
        missing data. This is the preferred behaviour, to avoid biased averages.
        
        Consider implementing also a masking algorithm for whenever there are more measurements
        available than the maximum possible with the estimated frequency.
        """

        # TODO: Handle masking when frequency changes consistently (so not just when a data point is missing...)

        if self.rawdata is None:
            return

        # TODO: Handle masks in calculation of daily averages
        grouped = self.rawdata.groupby(self.rawdata.index.date)
        daily_ts = grouped.mean()
        daily_count = grouped.count()

        dcp, nominal_freq_df = self._infer_frequencies()
        #ndf_df = nominal_freq_df.groupby(nominal_freq_df.index).agg({'nominal_freq': 'min'})
        ndf_df = nominal_freq_df.groupby(nominal_freq_df.index)['nominal_freq'].min()

        #nominal_count = 24/ndf_df.groupby(ndf_df.index.date).agg({'nominal_freq': 'min'})
        nominal_count = 24/ndf_df.groupby(ndf_df.index.date).min()

        # For data sources with a data interval of more than 1 day,
        # the aggregation to daily values result in an index consisting
        # of datetime.date objects.
        # To be sure we always have a pd.DatetimeIndex, we specifically convert.
        daily_ts.index = pd.DatetimeIndex(daily_ts.index)

        # Old code, no longer valid:
        # daily_ts = daily_ts[(daily_count >= nominal_count.values * threshold) & (daily_count >= mindata)]

        mask = ((daily_count >= nominal_count.values[:, None] * threshold) & (daily_count >= mindata)).any(axis=1)
        daily_ts = daily_ts[mask]

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


    def get_date(self, date, fullts=False):
        """
        Retrieve ground temperature data for a specific date.

        Parameters
        ----------
        date : datetime.date, datetime.datetime, or str
            The date for which to extract the temperature profile. Must match an index in self.daily_ts.
        fullts : bool, optional (default: False)
            If True, attempts to extract from the full time series (not implemented).
            If False, extracts from the daily averaged time series.

        Returns
        -------
        pandas.DataFrame
            DataFrame with sensor metadata as columns and a 'Value' column containing the temperature values for the specified date.
            The DataFrame is indexed by the MultiIndex columns of self.daily_ts (e.g., SensorID, CoordZ, etc.).

        Raises
        ------
        ValueError
            If the requested date is not present in the dataset.
        NotImplementedError
            If fullts=True (not implemented).
        """
        if not fullts:
            gid = find(self.daily_ts.columns.get_level_values('CoordZ') >= 0)
            #depths = np.take(self.daily_ts.columns.get_level_values('CoordZ'), gid).values

            self.sort_columns_by_depth()
            
            #data = self.daily_ts.iloc[self.daily_ts.index.isin([date]), gid].values.flatten()
            data = self.daily_ts.iloc[self.daily_ts.index.isin([date]), gid]
            
            if len(data) == 0:
                raise ValueError('The requested date ({0}) is not in the dataset.'.format(date))
            
            data = data.T.reset_index()
            data = data.rename(columns={data.columns[-1]: 'Value'})
            
            return data
        else:
            raise NotImplementedError('Full timeseries date extraction not yet implemented')


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
                meanGT = df.loc[lim[0].date():lim[1].date()].mean()
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
                maxGT = df.loc[lim[0].date():lim[1].date()].max()
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
                minGT = df.loc[lim[0].date():lim[1].date()].min()
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
        
        lim = fix_lim(lim)

        if (lim[0].date() > self.daily_ts.index[-1].date()) or (lim[1].date() < self.daily_ts.index[0].date()):
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
        

        maxT = maxGT.iloc[did].values
        maxT_d = maxGT.iloc[did].index.get_level_values('CoordZ').values

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

        zaa_stats = self.daily_ts.loc[lim[0].date():lim[1].date()].describe().transpose()

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
                lim = self.get_limits(fullts=False)

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
            start = lim[0].date() if hasattr(lim[0], "date") else lim[0]
            end = lim[1].date() if hasattr(lim[1], "date") else lim[1]
            data = self.daily_ts[start:end].iloc[:, did].values
            dates = list(self.daily_ts[start:end].index)
            ordinals = list(map(dt.datetime.toordinal, self.daily_ts[start:end].index))

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
                #pdb.set_trace()
                lh = ax.plot_date(dates, data[:,c2id], '-', label="{0:.2f} m".format(depths[c2id]), picker=5, **kwargs)
                #lh = ax.plot_date(ordinals, data[:,c2id], '-', label="{0:.2f} m".format(depths[c2id]), picker=5, **kwargs)
                lh[0].tag = 'dataseries'

        else:
            if lim is None:
                lim = self.get_limits(fullts=True)

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

        # gid = find(self.daily_ts.columns.get_level_values('CoordZ') >= 0)
        # depths = np.take(self.daily_ts.columns.get_level_values('CoordZ'), gid).values
        # 
        # self.sort_columns_by_depth()
        # 
        # data = self.daily_ts.iloc[self.daily_ts.index.isin([date]), gid].values.flatten()
        # 
        # if len(data) == 0:
        #     raise ValueError('The requested date ({0}) is not in the dataset.'.format(date))

        data = self.get_date(date)
        depths = data['CoordZ'].values
        values = data['Value'].values
        
        if 'color' not in kwargs and 'c' not in kwargs:
            kwargs['color'] = 'k'
        if 'linestyle' not in kwargs and 'ls' not in kwargs:
            kwargs['linestyle'] = '-'

        lh = ax.plot(values, depths, marker='.', markersize=7, **kwargs)

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
            
            if len(data[~np.isnan(data['Value'])])==0:
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
        ax.plot(maxGT.iloc[did], maxGT_depths[did], **args['maxGT'])

        minGT_depths = minGT.index.get_level_values('CoordZ')
        did = get_indices(minGT_depths, depths)
        did = [i for i in did if not np.isnan(i)]  # remove nan values
        ax.plot(minGT.iloc[did], minGT_depths[did], **args['minGT'])

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
            ax.plot(meanGT.iloc[did], meanGT_depths[did], **args['MeanGT'])


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
            data = self.daily_ts[lim[0].date():lim[1].date()].iloc[:, did].values
            ordinals = list(map(dt.datetime.toordinal, self.daily_ts[lim[0].date():lim[1].date()].index))
            
            times = self.daily_ts[lim[0].date():lim[1].date()].index
            
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
            
            #xx, yy = np.meshgrid(ordinals, depths)
            xx, yy = np.meshgrid(times, depths)

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


    # def calc_ALT(self, lim=None, fullts=False, silent=False):
    #     """
    #     Under CONSTRUCTION....

    #     [Yr maxd maxddayID] = SnuwALT(dateinput,data,depths)

    #     dateinput:   serial number date as produced by datenum
    #     depths:      depths included in data (length(depths) = size(data,2))
    #     data:        matrix with depths as columns, days as rows
    #     maxddayID:   Index into dateinput that gives maximum thawdepth
    #     """
    #     pass

    def calc_ALT_estimates(self, end_date=None, nyears=1, lim=None, depths=None, fullts=False, silent=False):
        """
        Calculate Active Layer Thickness (ALT) estimates based on three different temperature profiles:
        m: Temperature envelope method (based on maximum ground temperatures at every depth)
        p: Profile on the day of maximum temperature at the node above the frost table
        n: Profile on the day of maximum temperature at the node below the frost table

        For each temperature profile, thaw depth is estimated in three different ways:
        ++: Linear extrapolation from the two nodes above the frost table
        +-: Linear interpolation between the two nodes above and below the frost table
        --: Linear extrapolation from the two nodes below the frost table

        Args:
        end_date (dt.date, optional): End date for the calculation period (typically YYYY-07-31).
        nyears (int, optional): Number of years to include in the calculation (default 1).
        lim (tuple, optional): Custom time limits for the calculation.
        depths (list, optional): Depths to include in the calculation (default all).
        fullts (bool, optional): Whether to use full timeseries data (not implemented).
        silent (bool, optional): Whether to suppress output messages.

        Returns:
        dict: A dictionary containing the estimated thaw depths for each profile and method.
        """

        if fullts:
            raise NotImplementedError("Full timeseries support in calc_ALT is not implemented yet!")

        if end_date is None:
            end_date = self.rawdata.index[-1].date()

        if lim is None:
            lim = nyears2lim(end_date, 1)
        
        lim = fix_lim(lim)

        if (lim[0].date() > self.daily_ts.index[-1].date()) or (lim[1].date() < self.daily_ts.index[0].date()):
            raise IndexError('Requested date range is outside timeseries.')

        # prepare to filter on depths
        self.sort_columns_by_depth()

        if self.daily_ts is not None:
            self.calc_daily_average_adaptive()

        # Apply time limits and depth limits (ground temperatures only)
        ts = self.daily_ts.loc[lim[0].date():lim[1].date(), self.daily_ts.columns.get_level_values('CoordZ') >= 0]

        if (depths is None) or (type(depths) == str and depths.lower() == 'all'):
            depths = ts.columns.get_level_values('CoordZ')
            if hasattr(depths, 'values'):
                depths = depths.values

        if not hasattr(depths, '__iter__'):
            depths = [depths]  

        max_depth_id = None   # index of deepest thawed node for each year
        maxday_id_p = 0       # index to the date with max temperature at the node above ALT for each year
        maxday_id_n = -1      # index to the date with max temperature at the node below ALT for each year

        thaw_depth_p = np.ma.zeros((1, 3))
        thaw_depth_n = np.ma.zeros((1, 3))
        thaw_depth_m = np.ma.zeros((1, 3))
        ALT_date = np.ma.ones(2, dtype=dt.date)

        # Loop over depth levels, starting from top
        # did will be depth index
        # data will hold temperature timeseries for that depth 
        for did, data in enumerate(ts.T.values):
            # if all data are NaN, continue
            if np.all(np.isnan(data)):
                # If all data are masked, continue to next iteration.
                continue

            # Find days with positive thaw at present depth (in this specific year)
            tid = find(data >= 0)      # array of indices to positive temperatures
            if len(tid) != 0:
                # If exist, set max thaw depth to this depth level
                max_depth_id = did
                # set maxdday_id_p to the index into data (for a certain year) for
                # which the temperature is the highest, assuming this is the day
                # with deepest active layer. If more days with same temp, take
                # last day.
                mid = find(data == np.nanmax(data))
                if len(mid) != 0:
                    maxday_id_p = mid.max()
            elif False:
                # Here we should handle the situation when data is missing
                # in the thaw period from one depth.
                # Presently that will be considered as no thaw.
                # Should allow to look at deeper depth.
                pass
            else:
                # Otherwise we have passed 0C isotherm, so break loop

                # set maxday_id_n to the index into data (for a certain year) for
                # which the temperature is the highest, and the node above is at
                # positive temperatures, assuming this is the day with deepest active
                # layer. If more days with same temp, take last day.

                if not did == 0:
                    # If this is not the upper most node, find all days
                    # with positive temperatures in depth above the current.
                    data_above = ts.T.iloc[did-1]
                    idxp = find(data_above >= 0)  # index into data_above for positive temperatures

                    # Now check these days for the highest temperature at the
                    # present node-depth.
                    data_filtered = data[idxp]
                    idx = find(data_filtered == np.nanmax(data_filtered))
                    # idx is index into days with positive temperature at depth above...
                    # make idx index into days of the present year.
                    idx = idxp[idx]
                    maxday_id_n = idx.max()
                break

        # Still only looking at one year, if we have thaw, do interpolation
        if max_depth_id != 0:
            # interpolate to find 0 degr.

            if max_depth_id == len(depths) - 1:
                print("Max depth beyond grid")
                max_depth_id = depths[-1]
            elif max_depth_id >= len(find(depths >= 0)):
                print("Max depth above second grid point")
                max_depth_id = -9999.
            else:
                # Get the temperature profile on maxday_id_p and maxday_id_n
                thaw_depth_n = calc_thaw_depth_at_node(ts.iloc[maxday_id_n].values, depths, max_depth_id)
                thaw_depth_p = calc_thaw_depth_at_node(ts.iloc[maxday_id_p].values, depths, max_depth_id)

                ALT_date[0] = ts.index[maxday_id_p].date()
                ALT_date[1] = ts.index[maxday_id_n].date()

        # get max values within time limits
        maxGT = self.get_MaxGT(lim=lim)
        maxGT = maxGT[maxGT.notna()]

        # convert depths to column indices
        did = get_indices(maxGT.index.get_level_values('CoordZ'), depths)
        did = [i for i in did if not np.isnan(i)]  # remove nan values
        
        maxT = maxGT.iloc[did].values
        maxT_d = maxGT.iloc[did].index.get_level_values('CoordZ').values

        # find the deepest node (from the top) that has positive temperatures
        for node in range(len(maxT)):
            if maxT[node] >= 0:
                max_depth_id = node
            else:
                break

        thaw_depth_m = calc_thaw_depth_at_node(maxT, maxT_d, max_depth_id)


        # maxd will hold the largest estimate of the three sets of estimates
        # (thaw_depth_m, thaw_depth_n, thaw_depth_p)
        maxdepth = np.max([thaw_depth_m.max(), thaw_depth_n.max(), thaw_depth_p.max()])

        if not silent:
            print(" max     m++     m+-     m--     p++     p+-     p--    date(p)       n++     n+-     n--    date(n)")
            print("%5.2f   %5.2f   %5.2f   %5.2f   %5.2f   %5.2f   %5.2f   %10s   %5.2f   %5.2f   %5.2f   %10s" % (
                maxdepth,
                thaw_depth_m[0], thaw_depth_m[1], thaw_depth_m[2],
                thaw_depth_p[0], thaw_depth_p[1], thaw_depth_p[2], ALT_date[0],
                thaw_depth_n[0], thaw_depth_n[1], thaw_depth_n[2], ALT_date[1]))

        return {'max_depth': maxdepth, 
                'm++': thaw_depth_m[0],
                'm+-': thaw_depth_m[1],
                'm--': thaw_depth_m[2],
                'p++': thaw_depth_p[0],
                'p+-': thaw_depth_p[1],
                'p--': thaw_depth_p[2],
                'date(p)': ALT_date[0],
                'n++': thaw_depth_n[0],
                'n+-': thaw_depth_n[1],
                'n--': thaw_depth_n[2],
                'date(n)': ALT_date[1]}
















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

    if lim[0].date() < dt.date(lim[0].year, month, day):
        lim[0] = lim[0].replace(year=lim[0].year-1)

    # We want the last split year to encompas the specified end date
    # Thus, if the end date is after ????-month-day minus 1 day, add an extra year

    if lim[1].date() > dt.date(lim[1].year, month, day)-dt.timedelta(days=1):
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


#def plot_date(self, date, annotations=True, fs=12, xlim=None, ylim=None, show=True, **kwargs):
def plot_date(bhole, date, xlim=None, ylim=None, **kwargs):
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



    bhole.plot_date(date, xlim=xlim, ylim=ylim, **kwargs)

    tmp = bhole.daily_ts.loc[date].values
    z = bhole.daily_ts.loc[date].index.get_level_values('CoordZ').values

    tmp = tmp[z>0]
    z = z[z>0]

    # see this stackoverflow for inspiration how to handle signchanges
    # https://stackoverflow.com/questions/2652368/how-to-detect-a-sign-change-for-elements-in-a-numpy-array
    # especially this answer:
    # https://stackoverflow.com/a/67968974

    # problem is how to handle where temperature is exactly 0
    # It should count as frozen if ground is frozen on either side, but not (?) if thawed on both sides...
    # complication: multiple depths may have 0 temperature...

    # The following works when there are not multiple 0-elements in succession
    # s = np.sign(tmp)
    # s = np.where(np.logical_and(s==0., np.logical_or(np.roll(s,1)==-1, np.roll(s,-1)==-1)), -1., s)

    z_zero = find_zero(tmp, z)

    top_sign = np.sign(tmp[0])
    top_z = 0

    if len(z_zero) > 0:
        for zid, z in enumerate(z_zero):
            this_sign = top_sign*(-1)**zid
            
            if this_sign == -1:
                p1 = mpl.patches.Rectangle((-50, top_z), 100, z-top_z, ec='none', fc='#99CCFF', zorder=-10)    # frozen
            else:
                p1 = mpl.patches.Rectangle((-50, top_z), 100, z-top_z, ec='k', fc='#FFE4E4', zorder=-5)   # thawed
            plt.gca().add_patch(p1)
            top_z = z
    else:
        zid=-1

    this_sign = top_sign*(-1)**(zid+1)
    z = 1000

    if this_sign == -1:
        p1 = mpl.patches.Rectangle((-50, top_z), 100, z-top_z, ec='none', fc='#99CCFF', zorder=-10)    # frozen
    else:
        p1 = mpl.patches.Rectangle((-50, top_z), 100, z-top_z, ec='k', fc='#FFE4E4', zorder=-5)   # thawed
    plt.gca().add_patch(p1)


    ax = plt.gca()
    
    if ylim is None:
        ylim = ax.get_ylim()
        ax.set_ylim([ylim[0], 0])
        
    ylim = ax.get_ylim()
    yspan = np.diff(ylim)
    trans = transforms.blended_transform_factory(ax.transAxes, ax.transData)

    #if annotate:
    #    ax.text(0.97, z0 + 0.02 * yspan, 'Active layer', transform=trans,
    #            ha='right', va='bottom', fontsize=14)
    #    ax.text(0.97, z0 - 0.02 * yspan, 'Permafrost', transform=trans,
    #            ha='right', va='top', fontsize=14)

    return (plt.gcf())




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



def calc_all_ALT_estimates(bhole, end_date=None, nyears=1, lim=None):
    # Calculate all active layer thickness (ALT) estimates for a borehole
    alt_estimates = []
    bh_lim = bhole.get_limits()
    y1 = bh_lim[0].year
    y2 = bh_lim[1].year

    if lim is not None:
        lim = fix_lim(lim)

    for yr in range(y1, y2 + 1):
        # replace the year of lim[0] and lim[1]
        this_lim = lim
        if lim is not None:
            this_lim = [lim[0].replace(year=yr), 
                        lim[1].replace(year=yr+lim[1].year-lim[0].year)]
        elif end_date is None:
            raise ValueError("Either 'lim' or 'end_date' must be provided.")

        try:
            result = bhole.calc_ALT_estimates(end_date=end_date,
                                              nyears=nyears,
                                              lim=this_lim,
                                              fullts=False,
                                              silent=True)
        except Exception as e:
            print(f"Error calculating ALT for year {yr}: {e}")
            result = {'max_depth': np.nan, 
                      'm++': np.nan,
                      'm+-': np.nan,
                      'm--': np.nan,
                      'p++': np.nan,
                      'p+-': np.nan,
                      'p--': np.nan,
                      'date(p)': pd.NaT,
                      'n++': np.nan,
                      'n+-': np.nan,
                      'n--': np.nan,
                      'date(n)': pd.NaT}

        alt_estimates.append(result)
    
    return alt_estimates


# calc_ALT_estimates(end_date=None, nyears=1, lim=None, depths=None, fullts=False, silent=False):