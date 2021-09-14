import pydap
from .ClimateData import *



class DMI_gridpoint(climate_data):
    def __init__(self, gridpoint=[1, 1], start_date=None, end_date=None,
                 description=None, fname=None):

        self.gridpoint = gridpoint
        self.start_date = start_date
        self.end_date = end_date
        self.description = description
        if fname == None:
            self.fname = None
            self.path = None
        else:
            self.path, self.fname = os.path.split(fname)
        self.raw = dict(AT=None, P=None, SWE=None, SD=None)
        self.latitude = None
        self.longitude = None
        self.lat_bounds = None
        self.lon_bounds = None
        self.get_data()
        self.fitlim = None

    def __getstate__(self):
        """
        Creates a dictionary of climate_station variables. Data timeseries
        objects are converted to dictionaries available for pickling.
        """
        odict = self.__dict__.copy()  # copy the dict since we change it
        # Get timeseries instances as dictionaries

        odict['raw'] = []
        if 'raw' in self.__dict__:
            for key, ts in list(self.raw.items()):
                if ts != None:
                    odict['raw'].append((key, ts.__getstate__()))
                else:
                    odict['raw'].append((key, None))

        odict['daily'] = []
        if 'daily' in self.__dict__:
            for key, ts in list(self.daily.items()):
                if ts != None:
                    odict['daily'].append((key, ts.__getstate__()))
                else:
                    odict['daily'].append((key, None))

        odict['patched'] = []
        if 'patched' in self.__dict__:
            for key, ts in list(self.patched.items()):
                if ts != None:
                    odict['patched'].append((key, ts.__getstate__()))
                else:
                    odict['patched'].append((key, None))

        odict['scaled'] = []
        if 'scaled' in self.__dict__:
            for key, ts in list(self.scaled.items()):
                if ts != None:
                    odict['scaled'].append((key, ts.__getstate__()))
                else:
                    odict['scaled'].append((key, None))

        return odict

    def __setstate__(self, sdict):
        """
        Updates instance variables from dictionary.
        """
        # pdb.set_trace()
        raw = sdict.pop('raw', None)
        daily = sdict.pop('daily', None)
        patched = sdict.pop('patched', None)
        scaled = sdict.pop('scaled', None)
        self.__dict__.update(sdict)  # update attributes

        if 'raw' not in self.__dict__:
            self.raw = {}
        if 'daily' not in self.__dict__:
            self.daily = {}
        if 'patched' not in self.__dict__:
            self.patched = {}
        if 'scaled' not in self.__dict__:
            self.scaled = {}

        if raw != None and type(raw) == list:
            for key, ts in raw:
                if ts != None:
                    self.raw[key] = myts(None, None)
                    self.raw[key].__setstate__(ts)

        if daily != None and type(daily) == list:
            for key, ts in daily:
                if ts != None:
                    self.daily[key] = myts(None, None)
                    self.daily[key].__setstate__(ts)

        if patched != None and type(patched) == list:
            for key, ts in patched:
                if ts != None:
                    self.patched[key] = myts(None, None)
                    self.patched[key].__setstate__(ts)

        if scaled != None and type(scaled) == list:
            for key, ts in scaled:
                if ts != None:
                    self.scaled[key] = myts(None, None)
                    self.scaled[key].__setstate__(ts)

    def pickle(self, fullfile_noext=None, addname=''):
        """
        Pickle DMI_gridpoint instance to the specified file.
        filename will have the extension .pkl appended.
        """
        if fullfile_noext == None:
            if self.path == None:
                path = basepath
            else:
                path = self.path
            if self.fname == None:
                self.fname = os.path.join(basepath, 'HIRHAM', '25km', \
                                          "GP" + "[" + str(self.gridpoint[0]) + "][" + str(self.gridpoint[1]))

                fname = 'GP' + self.gridpoint.__str__() + addname

                self.fname = os.path.join(path, fname + '.pkl')
            else:
                ext, fname = self.fname[::-1].split('.')
                fname = fname[::-1] + addname
            fullfile_noext = os.path.join(path, fname)

        if type(fullfile_noext) in [list, tuple]:
            # Concatenate path elements
            fullfile_noext = os.path.join(*fullfile_noext)

        with open(fullfile_noext + '.pkl', 'wb') as pklfh:
            # Pickle dictionary using highest protocol.
            pickle.dump(self, pklfh, -1)

    def getDAPData(self, coordsID, climSet, climKey=None, dataKey=None,
                   fname=None, force_reload=False):
        """
        Get DMI Climate model data from DAP internet source using the pydap protocol.

        function will load data from file if local cache exists!

        Input arguments:
        coordsID:    list of length=2 with the indices of the grid point to retrieve
                       Use %run NCtxtread.py to plot map with grid points
        dataSet:     Name of the data set to retrieve 'T2m','SN','RR'
        dataKey:     key (into DAP source) of the data set to retrieve 'tas','snw','pr'

        Creates an entry in self.data dictionary of type myts.myts
        """

        # Key: dataSet,  value: [dataKey, key name to store in data dict, multiplier, to be added]
        namedict = {'T2m': ['tas', 'AT', 1., -273.15],
                    'SN': ['snw', 'SWE', 1000., 0.],
                    'RR': ['pr', 'P', 1., 0.]
                    }

        if climKey == None:
            climKey = namedict[climSet][0]
        if dataKey == None:
            dataKey = namedict[climSet][1]

        data = {}
        dataChanged = False

        if fname == None:
            fname = os.path.join(basepath, 'HIRHAM', '25km', \
                                 climSet + "[" + str(coordsID[0]) + "][" + str(coordsID[1]) + "].pkl")
            if not os.path.isdir(os.path.dirname(fname)):
                os.makedirs(os.path.dirname(fname))

        if not force_reload and os.path.isfile(fname):
            print('Loading from file...', fname)
            with open(fname, "rb") as fh:
                data = pickle.load(fh)

        periods = ['1950-1999', '2000-2049', '2050-2080']

        tmp = []

        pdb.set_trace()

        if 'time' not in data:
            data['epoch'] = np.array([])
            data['rawtime'] = np.array([])
            data['time'] = np.array([])

            print('acquiring data from DAP server...')

            for id, p in enumerate(periods):
                print("Opening DAP source for reading ...")
                tmp.append(pydap.client.open('http://ensemblesrt3.dmi.dk/cgi-bin/' +
                                             'nph-dods/data/klimagroenland/daily/SURF/' +
                                             climSet + '_' + p + '.nc.gz'))

                print('Getting times for the period {0}...'.format(p))
                data['epoch'] = np.append(data['epoch'],
                                          dateutil.parser.parse(tmp[id]['time'].units[12:]))
                rawtime = tmp[id]['time'][:]

                print('Getting {0} time steps'.format(p))
                tid = 0
                for t in rawtime:
                    data['time'] = np.append(data['time'], data['epoch'][-1] +
                                             dt.timedelta(hours=t))
                    if tid == 100:
                        tid = 0
                        print("rawtime: {0}".format(t))
                    else:
                        tid += 1

                data['rawtime'] = np.append(data['rawtime'], rawtime)

                dataChanged = True

        if dataChanged:
            print('Saving data...', fname)
            fh = open(fname, 'wb')
            pickle.dump(data, fh, pickle.HIGHEST_PROTOCOL)
            fh.close()
            dataChanged = False

        if climSet not in data:
            data[climSet] = np.array([])
            print('Getting ' + climSet + '...')

            for id, p in enumerate(periods):
                if not len(tmp) >= id + 1:
                    print("Opening DAP source for reading ...")
                    tmp.append(pydap.client.open('http://ensemblesrt3.dmi.dk/cgi-bin/' +
                                                 'nph-dods/data/klimagroenland/daily/SURF/' +
                                                 climSet + '_' + p + '.nc.gz'))

                print('Getting {0} for the period {1}...'.format(climSet, p))
                data[climSet] = np.append(data[climSet],
                                          tmp[id][climKey][:, 0,
                                          coordsID[0], coordsID[1]])

                if 'lon' not in data:
                    data['lon'] = tmp[id]['lon'][coordsID[0], coordsID[1]]
                if 'lat' not in data:
                    data['lat'] = tmp[id]['lat'][coordsID[0], coordsID[1]]
                if 'lon_bounds' not in data:
                    data['lon_bounds'] = tmp[id]['lon_bounds'][coordsID[0], coordsID[1]]
                if 'lat_bounds' not in data:
                    data['lat_bounds'] = tmp[id]['lat_bounds'][coordsID[0], coordsID[1]]

            dataChanged = True

        if dataChanged:
            print('Saving data...', fname)
            fh = open(fname, 'wb')
            pickle.dump(data, fh, pickle.HIGHEST_PROTOCOL)
            fh.close()

        self.raw[dataKey] = myts(data[climSet] *
                                 namedict[climSet][2] + namedict[climSet][3], data['time'])
        self.longitude = data['lon']
        self.latitude = data['lat']
        self.lon_bounds = data['lon_bounds']
        self.lat_bounds = data['lat_bounds']

        self.fname = os.path.join(basepath, 'HIRHAM', '25km', \
                                  "GP" + "[" + str(coordsID[0]) + "][" + str(coordsID[1]) + "].pkl")
        self.pickle()

    def get_data(self):
        """
        Retrieve all climate model data and process.

        calls getDAPData to do the actual data retrieval.
        """

        # -------------------------------------------------------------------------
        # Get and plt.plot climate model data


        self.getDAPData(self.gridpoint, 'T2m')
        self.getDAPData(self.gridpoint, 'SN')
        self.getDAPData(self.gridpoint, 'RR')

    def fit(self, other, parameter, lim=None, col=0):
        from scipy.optimize import fmin

        def my_fit(params, ts1, ts2, parameter):
            """
            Objective function for minimization/optimization of scaling of one
            timeseries to another.

            the objective function is:   sqrt(mean**2 + std**2)
            """

            if parameter == 'AT':
                ts1.scaleAT(params[0], params[1], recalc=False, recalc_SWE=False)
            elif parameter == 'P':
                ts1.scaleP(params[0], params[1], recalc=False)
            else:
                raise ValueError('Unknown parameter passed to DMI_gridpoint.fit! Parameter = {0}'.format(parameter))

            std_diff = ts1.std['value'] - ts2.std['value']
            mean_diff = ts1.mean['value'] - ts2.mean['value']
            return np.sqrt(mean_diff ** 2 + std_diff ** 2)

        # Limit the timeseries to specified interval, and dates/times in both series.
        sts = self.scaled[parameter].limit(lim=lim)
        ots = other.limit(lim=lim)

        ids, ido = sts.intersect(ots)
        sts = sts[ids]
        ots = ots[ido]

        # Produce initial guess, x0
        x0 = [sts.std['value'] / ots.std['value'], sts.mean['value'] - ots.mean['value']]

        # Do minimization
        xopt = fmin(my_fit, x0, args=(sts, ots, parameter), xtol=0.00001, ftol=0.00001, maxiter=None, maxfun=None,
                    full_output=0, disp=1, retall=0, callback=None)

        # Print results
        print("Optimized parameters: a={0}, b={1}".format(xopt[0], xopt[1]))
        print(self.gridpoint)
        self.print_stats()

    def permute_timeseries(self, arr, dataSeries='raw', dataKey='P', month=8, day=1):
        """
        Permute time series instance on a yearly basis according to arr, which
        should hold all years from the time series in the permuted order.

        Input arguments:
        arr:      list of years in permuted order. The method will only permute
                    years contained in arr, any other years in the data time-
                    series will remain unchanged in the result.
        dataKey:  Key into data dictionary of the dataset to permute. default='P'
        month:    Month to use for cut point, default = 8 (August)
        day:      Day to use for cut point, default = 1   (1st)

        month and day signify the first day in the permutation year. The
        method is intended for permuting precipitation to be used for calculation
        of Snow Water Equivalent. Thus we want to permute years with a cut-date
        in the summer time, where there is no snow to avoid discontinuities.

        Thus for the default settings, the permuted years will run from
        (YYYY)-08-01 to (YYYY+1)-07-31
        """

        years_in_dataset = self.__dict__[dataSeries][dataKey].get_year_list()
        min_year = np.min(years_in_dataset)
        max_year = np.max(years_in_dataset)

        min_permuted_year = np.min(arr)
        max_permuted_year = np.max(arr)

        permutation_start_date = dt.date(min_permuted_year, month, day)

        # To begin with, ts holds all data from unpermuted series which
        # fall before the lowest year in the permuted series.
        ts = self.__dict__[dataSeries][dataKey].limit([dt.date(min_year, 1, 1),
                                                       permutation_start_date -
                                                       dt.timedelta(days=1)])

        ts_list = []
        for id, year in enumerate(arr):
            # Get data from the specified year
            lim = [dt.date(year, month, day),
                   dt.date(year + 1, month, day) - dt.timedelta(days=1)]
            ts_list.append(self.__dict__[dataSeries][dataKey].limit(lim))

            # Change timeseries to reflect the new position in time

            # Let us say original timeseries contains 1950-01-01 to 2080-12-31
            # The permuted sequence contains the years 1950 to 2079.
            # We are currently referencing as the first entry in the permuted
            # series the year 1975. Thus we are extracting 1975-08-01 to
            # 1976-07-31 and this should become 1950-08-01 to 1951-07-31.
            # Thus from each year we should subtract (year - min_permuted_year)
            # which equals 25 years.
            # BUT: We have to keep track of number of days in the years we are
            # affecting... How do we do that?


            ndays_year = np.diff(lim)
            lim_permuted = [dt.date(min_permuted_year + id, month, day),
                            dt.date(min_permuted_year + id + 1, month, day) - \
                            dt.timedelta(days=1)]

            ndays_permuted_year = np.diff(lim_permuted)

            ts_list[id].times = ts_list[id].times - \
                                (lim[0] - permutation_start_date) + \
                                (lim_permuted[0] - permutation_start_date)

            while ts_list[id].times[-1].month < month:
                ts_list[id].data = np.append(ts_list[id].data,
                                             np.atleast_2d(ts_list[id].data[-1, :]),
                                             axis=0)
                ts_list[id].times = np.append(ts_list[id].times,
                                              ts_list[id].times[-1] + \
                                              dt.timedelta(days=1))

            while ts_list[id].times[-1].month >= month and \
                            ts_list[id].times[-1].day >= day:
                ts_list[id].data = ts_list[id].data[0:-1, :]
                ts_list[id].times = ts_list[id].times[0:-1]

            if ndays_year == ndays_permuted_year:
                # Same number of days in the two years... do nothing
                pass
            elif ndays_year > ndays_permuted_year:
                # More days at the original location than at the permuted
                # location... remove last day of series.
                pass
            elif ndays_year < ndays_permuted_year:
                # Less days at the original location than at the permuted
                # location... insert one day at end of series.
                pass

            ts_list[id].update_ordinals()

        # At the end, we should handle all data from original time series that
        # fall after the end of the permuted series:

        lim = [dt.date(max_permuted_year + 1, month, day), dt.date(max_year, 12, 31)]
        ts_list.append(self.__dict__[dataSeries][dataKey].limit(lim))

        ts.merge(ts_list)

        return ts

    def calc_AT_parameters(self, dataSeries='scaled', dataKey='AT', plot=False, fontsize=12, figsize=(15, 12),
                           style='-k', lim=None, ax1=None, ax2=None):
        """
        Returns a record array with the fields:

        Year:          The year for which the parameters are calculated
        MAAT:          Mean Annual Air Temperature
        FDD:           Freezing Degree Days
        TDD:           Thawing Degree Days
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
                raise KeyError('{0} dataseries not found'.format(dataSeries))

        if not dt_type in [dt.datetime, dt.date]:
            try:
                time = mpl.dates.num2date(time, tz=self.tzinfo)
            except:
                raise "Bad input to calcMAAT"

        ## Get data from specified time series
        # time = self.__dict__[dataSeries]['AT'].limit(lim).times
        # data = self.__dict__[dataSeries]['AT'].limit(lim).data
        #
        # if not self.__dict__[dataSeries]['AT'].dt_type in [dt.datetime, dt.date]:
        #    try:
        #        time = mpl.dates.num2date(time, tz=self.tzinfo)
        #    except:
        #        raise "Bad input to calcMAAT"


        # Make dictionary with years as keys and an array of the daily
        # AT data for that year as value
        years = {}
        for id, t in enumerate(time):
            if t.year in years:
                years[t.year] = np.append(years[t.year], data[id])
            else:
                years[t.year] = np.array(data[id])

        # Construct a rec-array with  Year, MAAT, FDD, TDD, ndays, days_in_year

        dtype = [('Year', int), ('MAAT', float),
                 ('FDD', float), ('TDD', float),
                 ('ndays', int), ('days_in_year', int)]
        params = []
        for id, entry in enumerate(sorted(years.items())):
            params.append(( \
                int(entry[0]), \
                entry[1].mean(), \
                entry[1][entry[1] < 0.].sum(), \
                entry[1][entry[1] > 0.].sum(), \
                entry[1].size,
                np.diff(mpl.dates.date2num([dt.date(entry[0], 1, 1), dt.date(entry[0] + 1, 1, 1)]))))

        statistics = np.array(params, dtype=dtype)

        if is_ts == False:
            self.__dict__[dataSeries]['ATstats'] = statistics

        # Plot if specified
        if plot:
            fig = self.plot_AT_parameters(
                dataSeries=dataSeries,
                ATstats=statistics,
                fontsize=fontsize,
                figsize=figsize,
                style=style,
                lim=lim,
                ax1=ax1,
                ax2=ax1)
            return statistics, fig
        else:
            return statistics

    def plot_AT_parameters(self, dataSeries='scaled',
                           dataKey='AT',
                           ATstats=None,
                           fontsize=12,
                           figsize=(15, 12),
                           style='-k',
                           lim=None,
                           ax1=None,
                           ax2=None,
                           recalc=False):

        if ATstats != None:
            stats2 = copy.deepcopy(ATstats)
        else:
            if recalc:
                stats2 = self.calc_AT_parameters(dataSeries, dataKey,
                                                 lim=lim,
                                                 plot=False)
            else:
                try:
                    stats2 = copy.deepcopy(self.__dict__[dataSeries]['ATstats'])
                except:
                    stats2 = self.calc_AT_parameters(dataSeries, dataKey,
                                                     lim=lim,
                                                     plot=False)

        figBG = 'w'  # the plt.figure background color
        axesBG = '#f6f6f6'  # the axies background color
        textsize = 14  # size for axes text

        if ax1 == None and ax2 == None:
            fig = plt.figure(figsize=figsize, facecolor=figBG)
        else:
            try:
                fig = ax1.figure
            except:
                fig = ax2.figure

        if ax1 == None or ax2 == None:
            fig.clf()
            left, width = 0.1, 0.8
            rect1 = [left, 0.55, width, 0.35]  # left, bottom, width, height
            rect2 = [left, 0.1, width, 0.35]  # left, bottom, width, height
            ax1 = plt.axes(rect1, axisbg=axesBG)
            ax2 = plt.axes(rect2, axisbg=axesBG, sharex=ax1)

        stats2 = np.append(stats2, stats2[-1])
        plotdates = np.array([dt.date(s, 1, 1) for s in stats2['Year']])
        plotdates[-1] = plotdates[-1].replace(month=12, day=31)

        # plot MAAT
        leg = True

        ax1.plot_date(plotdates, stats2['MAAT'], style, drawstyle='steps-post', label='MAAT')
        ax2.plot_date(plotdates, stats2['FDD'] / 365, style, drawstyle='steps-post', label='FDD')
        ax2.plot_date(plotdates, stats2['TDD'] / 365, style, drawstyle='steps-post', label='TDD')

        if leg:
            l = plt.gca().legend(loc='best')
            for t in l.get_texts():
                t.set_fontsize(textsize - 2)

        ax1.set_ylabel('MAAT ($^\circ$C)', size=textsize)
        ax2.set_ylabel('FDD or TDD  [DD / 365 days]', size=textsize)
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

    def regress_MAAT(self, rtype='linear', plot=True,
                     lab1='Data', lab2='Time',
                     tit='Linear regression',
                     ax=None,
                     annotation=['eq', 'R', 'N'],
                     fontsize=12,
                     figsize=(15, 6),
                     style='-k',
                     label='',
                     dataSeries='scaled',
                     dataKey='AT',
                     lim=['2009-01-01', '2080-12-31']):

        from scipy import stats as scp_stats

        try:
            # Treat as MyTimeseries
            MAAT = dataSeries.limit(lim=lim).calc_annual_avg()
            offset = dataSeries.limit(lim=lim).mean()
        except:
            # If it fails, it should be an entry into self
            if dataSeries in self.__dict__:
                MAAT = self.__dict__[dataSeries][dataKey].limit(lim=lim).calc_annual_avg()
                offset = self.__dict__[dataSeries][dataKey].limit(lim=lim).mean()
            else:
                raise KeyError('{0} dataseries not found'.format(dataSeries))

        MAAT = MAAT[mpl.mlab.find(MAAT.data[:, 1] > 350)]
        deltat = np.array(
            [np.float((t.replace(month=12, day=31) - t.replace(month=1, day=1)).days + 1) for t in MAAT.times]) / 2

        time = MAAT.ordinals

        data = MAAT.data[:, 0]

        if rtype.lower() == 'linear':
            (a_s, b_s, r, tt, stderr) = scp_stats.linregress(time + deltat, data)
            print(('regression: a=%.8f b=%.8f, std error= %.3f' % (a_s, b_s, stderr)))
            print('data = a * ordinal + b')
            print('Data point associated with mid-point of year')
        else:
            raise ValueError('Unknown regression type specified')

        MAATtrend = dict(list(zip(['a_s', 'b_s', 'r', 'tt', 'stderr', 'offset'],
                             (a_s, b_s, r, tt, stderr, offset))))

        if plot:
            figBG = 'w'  # the plt.figure background color
            axesBG = '#f6f6f6'  # the axies background color
            textsize = 14  # size for axes text

            figILU = plt.figure(figsize=figsize, facecolor=figBG)

            left, width = 0.1, 0.8
            rect1 = [left, 0.2, width, 0.6]  # left, bottom, width, height

            ax = plt.axes(rect1, axisbg=axesBG)

            myd = np.vstack((time, data)).T
            myd = np.vstack((myd, myd[-1, :]))
            myd[-1, 0] = pylab.date2num(pylab.num2date(myd[-1, 0]).replace(month=12, day=31))

            leg = False

            if style != None:
                ax.plot(myd[:, 0], myd[:, 1], style, drawstyle='steps-post', label=label)
            else:
                ax.plot(myd[:, 0], myd[:, 1], drawstyle='steps-post', label=label)
            if label != '':
                leg = True

            mindate = np.min(myd[:, 0])
            maxdate = np.max(myd[:, 0])
            daterange = np.array([mindate, maxdate])

            if ax == None:
                fig = plt.figure()
                ax = plt.axes()

            if rtype.lower() == 'linear':
                ax.plot(daterange, (daterange) * a_s + b_s, '-r', lw=1, label="linear regression")

            if leg:
                l = plt.gca().legend(loc='best')
                for t in l.get_texts():
                    t.set_fontsize(textsize - 2)

            ax.set_ylabel('MAAT ($^\circ$C)', size=textsize)
            ax.set_xlabel('Time', size=textsize)

            pylab.xticks(fontsize=fontsize)
            pylab.yticks(fontsize=fontsize)

            xycoords = 'axes fraction'
            astr = ''
            if 'eq' in annotation:
                astr = 'y = {0:.4f}*x + {1:.4f}'.format(a_s, b_s)
            if 'R' in annotation:
                if astr != '': astr = astr + '\n'
                astr = astr + 'R$^2$ = {0:.4f}'.format(r)
            if 'N' in annotation:
                if astr != '': astr = astr + '\n'
                astr = astr + 'N = {0:.0f}'.format(len(data))

            if astr != '':
                ax.annotate(astr, xy=(0.05, 0.95), xycoords=xycoords,
                            verticalalignment='top',
                            fontsize=fontsize)

            plt.show()

        return MAATtrend

    def regress_DD(self, rtype='linear',
                   dataSeries='scaled',
                   lim=['2009-01-01', '2080-12-31']):

        from scipy import stats as scp_stats

        # Calculate MAAT, FDD and TDD from time series
        try:
            # Treat dataSeries as MyTimeseries
            data = self.calc_AT_parameters(dataSeries=dataSeries,
                                           plot=False, lim=lim)
        except:
            # If it fails, it should be an entry into self
            if dataSeries in self.__dict__:
                data = self.calc_AT_parameters(dataSeries=dataSeries,
                                               plot=False, lim=lim)
            else:
                raise KeyError('{0} dataseries not found'.format(dataSeries))

        # Use only years with more than 350 days available
        data = data[mpl.mlab.find(data['ndays'] > 350)]

        # Prepare time variable, specifying january 1 of each year
        time = np.array([dt.date(t, 1, 1) for t in data['Year']])

        # deltat holds number of days to middle of year
        deltat = np.array([np.float((t.replace(month=12, day=31) -
                                     t.replace(month=1, day=1)).days + 1)
                           for t in time]) / 2

        # convert datetimes to ordinals
        time = mpl.dates.date2num(time)

        # Now perform regressions
        DDtrends = dict()

        if rtype.lower() == 'linear':
            print('')

            # FDD trend
            (a_s, b_s, r, tt, stderr) = scp_stats.linregress(time + deltat,
                                                             data['FDD'] / data['days_in_year'])
            print('Freezing degree days trend:')
            # print('regression: a=%.8f b=%.8f, std error= %.3f' % (a_s,b_s,stderr))
            print(('regression: a=%.8f b=%.8f, r^2= %.3f' % (a_s, b_s, r)))
            print('data = a * ordinal + b')
            print('Data point associated with mid-point of year')
            print('')
            offset = data['FDD'].mean() / data['ndays'].mean()
            DDtrends['FDD'] = dict(list(zip(['a_s', 'b_s', 'r', 'tt', 'stderr', 'offset'],
                                       (a_s, b_s, r, tt, stderr, offset))))

            # TDD trend
            (a_s, b_s, r, tt, stderr) = scp_stats.linregress(time + deltat,
                                                             data['TDD'] / data['days_in_year'])
            print('Thawing degree days trend:')
            # print('regression: a=%.8f b=%.8f, std error= %.3f' % (a_s,b_s,stderr))
            print(('regression: a=%.8f b=%.8f, r^2= %.3f' % (a_s, b_s, r)))
            print('data = a * ordinal + b')
            print('Data point associated with mid-point of year')
            print('')
            offset = data['TDD'].mean() / data['ndays'].mean()
            DDtrends['TDD'] = dict(list(zip(['a_s', 'b_s', 'r', 'tt', 'stderr', 'offset'],
                                       (a_s, b_s, r, tt, stderr, offset))))

            # MAAT trend
            (a_s, b_s, r, tt, stderr) = scp_stats.linregress(time + deltat, data['MAAT'])
            print('MAAT trend:')
            # print('regression: a=%.8f b=%.8f, std error= %.3f' % (a_s,b_s,stderr))
            print(('regression: a=%.8f b=%.8f, r^2= %.3f' % (a_s, b_s, r)))
            print('data = a * ordinal + b')
            print('Data point associated with mid-point of year')
            print('')
            offset = data['MAAT'].mean() / data['ndays'].mean()
            DDtrends['MAAT'] = dict(list(zip(['a_s', 'b_s', 'r', 'tt', 'stderr', 'offset'],
                                        (a_s, b_s, r, tt, stderr, offset))))

        return DDtrends

    def detrend_DD(self, dataSeries='scaled', lim=['2009-01-01', '2080-12-31'], plot=True):
        """
        Detrends the specified dataSeries (must be a key into self.__dict__)
        and stores the result in self._detrend

        The method calculates any necessary parameters not already specified
        in the class variables.

        Makes a plot of the resulting MAAT time series, if arg plot=True
        """

        if 'ATstats' not in self.__dict__[dataSeries]:
            self.calc_AT_parameters(dataSeries=dataSeries, lim=lim, plot=False)

        # Calculate trend information for MAAT, FDD and TDD
        if '_detrend' not in self.__dict__:
            self._detrend = dict()
        if 'regr' not in self._detrend:
            self._detrend['DDtrends'] = \
                self.regress_DD(rtype='linear',
                                dataSeries=dataSeries,
                                lim=lim)

        ATstats = self.__dict__[dataSeries]['ATstats']
        DDtrends = self._detrend['DDtrends']

        # Construct timeseries of the FDD/TDD trends

        print("calculate trends and detrend")
        # ydict = self._detrend['AT'].split_years()
        ydict = self.__dict__[dataSeries]['AT'].limit(lim=lim).split_years()
        for id, y in enumerate(sorted(ydict.keys())):
            time = mpl.dates.date2num(dt.date(y, 1, 1)) + ATstats['days_in_year'][id] / 2

            TDD_trend = time * DDtrends['TDD']['a_s'] + DDtrends['TDD']['b_s']
            FDD_trend = time * DDtrends['FDD']['a_s'] + DDtrends['FDD']['b_s']

            # TDD_trend = ydict[y].ordinals.reshape(-1,1)* \
            #                DDtrends['TDD']['a_s'] + DDtrends['TDD']['b_s']
            # FDD_trend = ydict[y].ordinals.reshape(-1,1)* \
            #                DDtrends['FDD']['a_s'] + DDtrends['FDD']['b_s']

            ydict[y] = ydict[y] * \
                       np.where(ydict[y].data >= 0,
                                (DDtrends['TDD']['offset'] - TDD_trend) / \
                                (ATstats['TDD'][id] / ATstats['ndays'][id]) + 1,
                                (DDtrends['FDD']['offset'] - FDD_trend) / \
                                (ATstats['FDD'][id] / ATstats['ndays'][id]) + 1)

        tslist = list(ydict.values())
        self._detrend['AT'] = tslist[0]
        self._detrend['AT'].merge(tslist[1:], mtype='forced')
        self._detrend['AT'].sort_chronologically()

        if plot == True:
            print("perform regression and plot")
            MAATtrend = self.regress_MAAT(plot=True,
                                          lab1='Data', lab2='Time',
                                          tit='After detrending',
                                          ax=None,
                                          annotation=['eq', 'R', 'N'],
                                          fontsize=12,
                                          figsize=(15, 6),
                                          style='-k',
                                          label='',
                                          dataSeries='_detrend',
                                          lim=lim)
        return

    def retrend_DD(self, dataSeries='_detrend', DDtrends=None, ATstats=None,
                   lim=None, plot=True):
        """
        Reapplies trend to timeseries.

        Input is:
        dataSeries:  Time series of Air Temperatures to apply trend to
                       May be a MyTimeseries or a key into self.__dict__
        rparams:     Regression parameters as reported by self.regress_DD

        if ts is not specified, it is assumed to be the detrended timeseries
          from self. An Error will be thrown if this timeseries does not exist
        if rparams is not specified, it will be assumed to be the regression
          parameters of the original grid point time series. An Error will be
          thrown if self._detrend[rparams] does not exist.
        """

        ## Determine correct timeseries to use
        # if dataSeries == None:
        #    # If none, default to self._detrend
        #    if self.__dict__.has_key('_detrend') and \
        #        self._detrend.has_key['AT']:
        #        ts = self._detrend['AT'].limit(lim)
        #    else:
        #        KeyError('Detrended timeseries has not been calculated' + \
        #                 '   Aborting...')
        #    if DDtrends == None:
        #        if self.__dict__.has_key('_detrend') and \
        #           self._detrend.has_key['DDtrends']:
        #            rparams = self._detrend['DDtrends']
        #        else:
        #            KeyError('Regression parameters have not been calculated' + \
        #                     '   Aborting...')
        #
        try:
            # Treat as MyTimeseries
            ts = dataSeries.limit(lim)
            if DDtrends == None:
                ValueError('You must specify the DDtrends to apply!' + \
                           '   Aborting...')
            if ATstats == None:
                print("Calculating AT statistics")
                ATstats = self.calc_AT_parameters(dataSeries, plot=False,
                                                  lim=lim)
        except:
            if dataSeries in self.__dict__ and \
                    'AT' in self.__dict__[dataSeries]:
                ts = self.__dict__[dataSeries]['AT']
            else:
                KeyError('Specified timeseries not found' + \
                         '   Aborting...')

            # Check trend information
            if DDtrends == None:
                if dataSeries in self.__dict__ and \
                        'DDtrends' in self.__dict__[dataSeries]:
                    DDtrends = self._detrend['DDtrends']
                else:
                    KeyError('Regression parameters have not been calculated' + \
                             '   Aborting...')

                    # Check timeseries statistics
            # if ATstats == None and \
            #    not self.__dict__[dataSeries].has_key('ATstats'):
            #    print "Calculating AT statistics"
            #    ATstats = self.calc_AT_parameters(dataSeries=dataSeries,
            #                                      lim=lim, plot=False)
            # else:
            #    ATstats = self.__dict__[dataSeries]['ATstats']
            if ATstats == None:
                print("Calculating AT statistics")
                ATstats = self.calc_AT_parameters(dataSeries=dataSeries,
                                                  lim=lim, plot=False)
        # Construct timeseries of the FDD/TDD trends

        print("calculate trends and retrend")
        # ydict = self._detrend['AT'].split_years()
        ydict = ts.limit(lim=lim).split_years()
        for id, y in enumerate(sorted(ydict.keys())):
            time = mpl.dates.date2num(dt.date(y, 1, 1)) + ATstats['days_in_year'][id] / 2

            TDD_trend = time * DDtrends['TDD']['a_s'] + DDtrends['TDD']['b_s']
            FDD_trend = time * DDtrends['FDD']['a_s'] + DDtrends['FDD']['b_s']

            # TDD_trend = ydict[y].ordinals.reshape(-1,1)* \
            #                DDtrends['TDD']['a_s'] + DDtrends['TDD']['b_s']
            # FDD_trend = ydict[y].ordinals.reshape(-1,1)* \
            #                DDtrends['FDD']['a_s'] + DDtrends['FDD']['b_s']

            ydict[y] = ydict[y] * \
                       np.where(ydict[y].data >= 0,
                                -(DDtrends['TDD']['offset'] - TDD_trend) / \
                                (ATstats['TDD'][id] / ATstats['ndays'][id]) + 1,
                                -(DDtrends['FDD']['offset'] - FDD_trend) / \
                                (ATstats['FDD'][id] / ATstats['ndays'][id]) + 1)

        tslist = list(ydict.values())
        ts2 = tslist[0]
        ts2.merge(tslist[1:], mtype='forced')
        ts2.sort_chronologically()

        if plot:
            print("perform regression and plot")
            MAATtrend = self.regress_MAAT(plot=True,
                                          lab1='Data', lab2='Time',
                                          tit='After reaplying trend',
                                          ax=None,
                                          annotation=['eq', 'R', 'N'],
                                          fontsize=12,
                                          figsize=(15, 6),
                                          style='-k',
                                          label='',
                                          dataSeries=ts2,
                                          lim=lim)
        return ts2

    def detrend_MAAT(self, lim=['2009-01-01', '2080-12-31'], plot=False):
        """Plots mean annual air temperatures and fit a straight line
        through them...
        """

        MAATtrend = self.regress_MAAT(dataSeries='raw', lim=lim)

        if '_detrendMAAT' in self.__dict__:
            self._detrendMAAT['AT'] = self.__dict__['raw']['AT'].limit(lim=lim)
        else:
            self._detrendMAAT = dict(AT=self.__dict__['raw']['AT'].limit(lim=lim))

        self._detrendMAAT['AT'].data = self._detrendMAAT['AT'].data - \
                                       (MAATtrend['a_s'] * \
                                        self._detrendMAAT['AT'].ordinals.reshape(-1, 1) + \
                                        MAATtrend['b_s'])

        self._detrendMAAT['MAATtrend'] = MAATtrend

        if plot == True:
            MAATtrend2 = self.regress_MAAT(plot=True,
                                           lab1='Data', lab2='Time',
                                           tit='After detrending',
                                           ax=None,
                                           annotation=['eq', 'R', 'N'],
                                           fontsize=12,
                                           figsize=(15, 6),
                                           style='-k',
                                           label='',
                                           dataSeries='_detrendMAAT',
                                           lim=lim)
        return

    def calc_P_parameters(self, dataSeries='scaled', dataKey='P',
                          plot=False,
                          fontsize=12,
                          figsize=(15, 12),
                          style='-k',
                          lim=None,
                          ax1=None,
                          ax2=None):

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
                raise KeyError('{0} dataseries not found'.format(dataSeries))

        if not dt_type in [dt.datetime, dt.date]:
            try:
                time = mpl.dates.num2date(time, tz=self.tzinfo)
            except:
                raise "Bad input to calc_P_parameters"

        # Make dictionary with years as keys and an array of the daily
        # AT data for that year as value
        years = {}
        for id, t in enumerate(time):
            if t.year in years:
                years[t.year] = np.append(years[t.year], data[id])
            else:
                years[t.year] = np.array(data[id])

        # Construct a rec-array with  Year, MAAT, FDD, TDD, ndays, days_in_year

        dtype = [('Year', int), ('SAP', float),
                 ('MAP', float), ('npdays', float),
                 ('ndays', int), ('days_in_year', int)]
        params = []
        for id, entry in enumerate(sorted(years.items())):
            params.append(( \
                int(entry[0]), \
                entry[1][entry[1] > 0.].sum(), \
                entry[1][entry[1] > 0.].sum() / entry[1].size, \
                np.where(entry[1] > 0., np.ones_like(entry[1]),
                         np.zeros_like(entry[1])).sum(),
                entry[1].size,
                np.diff(mpl.dates.date2num([dt.date(entry[0], 1, 1), dt.date(entry[0] + 1, 1, 1)]))))

        statistics = np.array(params, dtype=dtype)

        # pdb.set_trace()


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
            return statistics, fig
        else:
            return statistics

    def plot_P_parameters(self, dataSeries='scaled',
                          dataKey='P',
                          Pstats=None,
                          fontsize=12,
                          figsize=(15, 12),
                          style='-k',
                          lim=None,
                          ax1=None,
                          ax2=None,
                          recalc=False):

        if Pstats != None:
            stats2 = copy.deepcopy(Pstats)
        else:
            if recalc:
                stats2 = self.calc_P_parameters(dataSeries, dataKey,
                                                lim=lim,
                                                plot=False)
            else:
                try:
                    stats2 = copy.deepcopy(self.__dict__[dataSeries]['Pstats'])
                except:
                    stats2 = self.calc_P_parameters(dataSeries, dataKey,
                                                    lim=lim,
                                                    plot=False)

        figBG = 'w'  # the plt.figure background color
        axesBG = '#f6f6f6'  # the axies background color
        textsize = 14  # size for axes text

        if ax1 == None and ax2 == None:
            fig = plt.figure(figsize=figsize, facecolor=figBG)
        else:
            try:
                fig = ax1.figure
            except:
                fig = ax2.figure

        if ax1 == None or ax2 == None:
            fig.clf()
            left, width = 0.1, 0.8
            rect1 = [left, 0.55, width, 0.35]  # left, bottom, width, height
            rect2 = [left, 0.1, width, 0.35]  # left, bottom, width, height
            ax1 = plt.axes(rect1, axisbg=axesBG)
            ax2 = plt.axes(rect2, axisbg=axesBG, sharex=ax1)

        stats2 = np.append(stats2, stats2[-1])
        plotdates = np.array([dt.date(s, 1, 1) for s in stats2['Year']])
        plotdates[-1] = plotdates[-1].replace(month=12, day=31)

        # plot MAAT
        leg = True

        ax1.plot_date(plotdates, stats2['SAP'], style, drawstyle='steps-post', label='SAP')
        ax2.plot_date(plotdates, stats2['MAP'], style, drawstyle='steps-post', label='MAP')

        if leg:
            l = plt.gca().legend(loc='best')
            for t in l.get_texts():
                t.set_fontsize(textsize - 2)

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

    def regress_MAP(self, rtype='linear', plot=True,
                    lab1='Data', lab2='Time',
                    tit='Linear regression',
                    ax=None,
                    annotation=['eq', 'R', 'N'],
                    fontsize=12,
                    figsize=(15, 6),
                    style='-k',
                    label='',
                    dataSeries='scaled',
                    dataKey='P',
                    lim=['2009-01-01', '2080-12-31']):

        from scipy import stats as scp_stats

        # pdb.set_trace()

        try:
            # treat dataSeries as timeseries
            tmpdat = dataSeries.limit(lim=lim).data
            offset = tmpdat[tmpdat > 0.].sum() / len(tmpdat)
            statistics = self.calc_P_parameters(dataSeries, None, plot=False, lim=lim)
        except:
            # If it fails, it should be an entry into self
            if dataSeries in self.__dict__:
                statistics = self.calc_P_parameters(dataSeries, dataKey,
                                                    plot=False, lim=lim)

                tmpdat = self.__dict__[dataSeries][dataKey].limit(lim=lim).data
                offset = tmpdat[tmpdat > 0.].sum() / len(tmpdat)
            else:
                raise KeyError('{0} dataseries not found'.format(dataSeries))

        id = statistics['ndays'] > 350
        data = statistics['MAP'][id]
        deltat = statistics['days_in_year'] / 2

        time = mpl.dates.date2num([dt.datetime(y, 1, 1) for y in statistics['Year']])

        # pdb.set_trace()

        if rtype.lower() == 'linear':
            (a_s, b_s, r, tt, stderr) = scp_stats.linregress(time + deltat, data)
            print(('regression: a=%.8f b=%.8f, std error= %.3f' % (a_s, b_s, stderr)))
            print('data = a * ordinal + b')
            print('Data point associated with mid-point of year')
        else:
            raise ValueError('Unknown regression type specified')

        MAPtrend = dict(list(zip(['a_s', 'b_s', 'r', 'tt', 'stderr', 'offset'],
                            (a_s, b_s, r, tt, stderr, offset))))

        if plot:
            figBG = 'w'  # the plt.figure background color
            axesBG = '#f6f6f6'  # the axies background color
            textsize = 14  # size for axes text

            figILU = plt.figure(figsize=figsize, facecolor=figBG)

            left, width = 0.1, 0.8
            rect1 = [left, 0.2, width, 0.6]  # left, bottom, width, height

            ax = plt.axes(rect1, axisbg=axesBG)

            myd = np.vstack((time, data)).T
            myd = np.vstack((myd, myd[-1, :]))
            myd[-1, 0] = pylab.date2num(pylab.num2date(myd[-1, 0]).replace(month=12, day=31))

            leg = False

            if style != None:
                ax.plot(myd[:, 0], myd[:, 1], style, drawstyle='steps-post', label=label)
            else:
                ax.plot(myd[:, 0], myd[:, 1], drawstyle='steps-post', label=label)
            if label != '':
                leg = True

            mindate = np.min(myd[:, 0])
            maxdate = np.max(myd[:, 0])
            daterange = np.array([mindate, maxdate])

            if ax == None:
                fig = plt.figure()
                ax = plt.axes()

            if rtype.lower() == 'linear':
                ax.plot(daterange, (daterange) * a_s + b_s, '-r', lw=1, label="linear regression")

            if leg:
                l = plt.gca().legend(loc='best')
                for t in l.get_texts():
                    t.set_fontsize(textsize - 2)

            ax.set_ylabel('MAP (mm)', size=textsize)
            ax.set_xlabel('Time', size=textsize)

            pylab.xticks(fontsize=fontsize)
            pylab.yticks(fontsize=fontsize)

            xycoords = 'axes fraction'
            astr = ''
            if 'eq' in annotation:
                astr = 'y = {0:.4f}*x + {1:.4f}'.format(a_s, b_s)
            if 'R' in annotation:
                if astr != '': astr = astr + '\n'
                astr = astr + 'R$^2$ = {0:.4f}'.format(r)
            if 'N' in annotation:
                if astr != '': astr = astr + '\n'
                astr = astr + 'N = {0:.0f}'.format(len(data))

            if astr != '':
                ax.annotate(astr, xy=(0.05, 0.95), xycoords=xycoords,
                            verticalalignment='top',
                            fontsize=fontsize)

            plt.show()

        return MAPtrend

    def detrend_MAP(self, dataSeries='scaled',
                    lim=['2009-01-01', '2080-12-31'],
                    plot=True):
        """
        Detrends the specified dataSeries (must be a key into self.__dict__)
        and stores the result in self._detrend

        The method calculates any necessary parameters not already specified
        in the class variables.

        Makes a plot of the resulting MAP time series, if arg plot=True
        """

        if 'Pstats' not in self.__dict__[dataSeries]:
            self.calc_P_parameters(dataSeries=dataSeries, lim=lim, plot=False)

        # Calculate trend information for MAAT, FDD and TDD
        if '_detrend' not in self.__dict__:
            self._detrend = dict()
        if 'regr' not in self._detrend:
            self._detrend['MAPtrend'] = \
                self.regress_MAP(rtype='linear',
                                 dataSeries=dataSeries,
                                 lim=lim)

        Pstats = self.__dict__[dataSeries]['Pstats']
        MAPtrend = self._detrend['MAPtrend']

        # pdb.set_trace()

        print("calculate trends and detrend")

        ydict = self.__dict__[dataSeries]['P'].limit(lim=lim).split_years()
        for id, y in enumerate(sorted(ydict.keys())):
            # trend = ydict[y].ordinals.reshape(-1,1)* \
            #                MAPtrend['a_s'] + MAPtrend['b_s']

            time = mpl.dates.date2num(dt.date(y, 1, 1)) + Pstats['days_in_year'][id] / 2
            trend = time * MAPtrend['a_s'] + MAPtrend['b_s']

            # ydict[y] = ydict[y] * \
            #    np.where(ydict[y].data>=0,
            #    (MAPtrend['offset']-trend)/  \
            #    (Pstats['MAP'][id]/Pstats['ndays'][id]) + 1, 0)
            ydict[y] = ydict[y] * \
                       np.where(ydict[y].data >= 0,
                                (MAPtrend['offset'] - trend) / \
                                (Pstats['MAP'][id]) + 1, 0)

        tslist = list(ydict.values())
        self._detrend['P'] = tslist[0]
        self._detrend['P'].merge(tslist[1:], mtype='forced')
        self._detrend['P'].sort_chronologically()

        # ts = self.__dict__[dataSeries]['P'].limit(lim=lim)
        #
        # trendline = ts.ordinals.reshape(-1,1)* \
        #                MAPtrend['a_s'] + MAPtrend['b_s']
        #
        # ts.data = np.where(ts.data>=0, ts.data-trendline+MAPtrend['offset'], 0)
        #
        # self._detrend['P'] = ts

        if plot == True:
            print("perform regression and plot")
            MAPtrend = self.regress_MAP(plot=True,
                                        lab1='Data', lab2='Time',
                                        tit='After detrending',
                                        ax=None,
                                        annotation=['eq', 'R', 'N'],
                                        fontsize=12,
                                        figsize=(15, 6),
                                        style='-k',
                                        label='',
                                        dataSeries='_detrend',
                                        lim=lim)
        return

    def retrend_MAP(self, dataSeries=None, MAPtrend=None, Pstats=None,
                    lim=None, plot=True):
        """
        Reapplies trend to timeseries.

        Input is:
        dataSeries:  Time series of Precipitation to apply trend to
                       May be a MyTimeseries or a key into self.__dict__
        rparams:     Regression parameters as reported by self.regress_MAP

        if ts is not specified, it is assumed to be the detrended timeseries
          from self. An Error will be thrown if this timeseries does not exist
        if rparams is not specified, it will be assumed to be the regression
          parameters of the original grid point time series. An Error will be
          thrown if self._detrend[rparams] does not exist.
        """

        try:
            # Treat as MyTimeseries
            ts = dataSeries.limit(lim)
            if MAPtrend == None:
                ValueError('You must specify the MAPtrend to apply!' + \
                           '   Aborting...')
            if Pstats == None:
                print("Calculating P statistics")
                Pstats = self.calc_P_parameters(dataSeries, plot=False,
                                                lim=lim)
        except:
            if dataSeries in self.__dict__ and \
                    'P' in self.__dict__[dataSeries]:
                ts = self.__dict__[dataSeries]['P'].limit(lim)
            else:
                KeyError('Specified timeseries not found' + \
                         '   Aborting...')

            # Check trend information
            if MAPtrend == None:
                if dataSeries in self.__dict__ and \
                        'MAPtrend' in self.__dict__[dataSeries]:
                    MAPtrend = self._detrend['MAPtrend']
                else:
                    KeyError('Regression parameters have not been calculated' + \
                             '   Aborting...')

                    # Check timeseries statistics
            if Pstats == None:
                print("Calculating P statistics")
                Pstats = self.calc_P_parameters(dataSeries=dataSeries,
                                                lim=lim, plot=False)

        # Construct timeseries of the FDD/TDD trends

        print("calculate trends and retrend")

        ydict = ts.split_years()
        for id, y in enumerate(sorted(ydict.keys())):
            # trend = ydict[y].ordinals.reshape(-1,1)* \
            #                MAPtrend['a_s'] + MAPtrend['b_s']

            time = mpl.dates.date2num(dt.date(y, 1, 1)) + Pstats['days_in_year'][id] / 2
            trend = time * MAPtrend['a_s'] + MAPtrend['b_s']

            # ydict[y] = ydict[y] * \
            #    np.where(ydict[y].data>=0,
            #    -(MAPtrend['offset']-trend)/  \
            #    (Pstats['MAP'][id]/Pstats['ndays'][id]) + 1, 0)
            ydict[y] = ydict[y] * \
                       np.where(ydict[y].data >= 0,
                                -(MAPtrend['offset'] - trend) / \
                                (Pstats['MAP'][id]) + 1, 0)

        tslist = list(ydict.values())
        ts2 = tslist[0]
        ts2.merge(tslist[1:], mtype='forced')
        ts2.sort_chronologically()

        # trendline = ts.ordinals.reshape(-1,1)* \
        #                MAPtrend['a_s'] + MAPtrend['b_s']
        #
        # ts.data = np.where(ts.data>=0, ts.data+trendline-MAPtrend['offset'], 0)

        if plot:
            print("perform regression and plot")
            MAPtrend = self.regress_MAP(plot=True,
                                        lab1='Data', lab2='Time',
                                        tit='After reaplying trend',
                                        ax=None,
                                        annotation=['eq', 'R', 'N'],
                                        fontsize=12,
                                        figsize=(15, 6),
                                        style='-k',
                                        label='',
                                        dataSeries=ts2,
                                        lim=lim)
        return ts2


class test_DMI_gridpoint(DMI_gridpoint):
    def __init__(self):
        DMI_gridpoint.__init__(self, gridpoint=[78, 50])

    def test_permute_timeseries(self):
        start_year = 1950
        end_year = 1959
        month, day = 8, 1
        times = mpl.dates.num2date(mpl.dates.drange(dt.date(start_year, 1, 1),
                                                    dt.date(end_year, 12, 31), dt.timedelta(days=1)))

        data = []
        flag = False
        counter = 0
        for t in times:
            if t.month == 1 and t.day == 1:
                # When passing into new calendar year, clear flag:
                flag = False

            if t.month >= month and t.day >= day and not flag:
                # If we pass cut off month, add to counter and reset flag
                counter += 1
                flag = True

            data.append(counter)

        data = np.array(data).reshape([-1, 1])

        self.data['P'] = myts(data, times)
        print("plotting original series")
        plt.plot_date(times, data, '-g')
        permutations = n_permutations(self.data['P'].get_year_list()[0:-1], 5)
        self.permutations = permutations
        for p in permutations:
            ts = self.permute_timeseries(p)
            print("plotting permuted series")
            plt.plot_date(ts.times, ts.data, '-r')
