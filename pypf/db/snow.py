"""
Module containing all functions relating to the handling of snow data.

"""

version = 1.1

import datetime as dt
import pylab
import numpy as np
import pdb

         
def calcSWE(Tts,Pts):
    """
    Calculate Snow Water Equivalent by accumulating all precip when temperature
    is below freezing.
    
    When temperature rises above 0 degC all snow disappears immediately on the
    first day of melting.
    
    Exception:
    If the period of melting is 10 days or less, snow pack 
    is reduced by half.
    
    Input arguments:
    Tts:     MyTimeseries timeseries of temperature data
    Pts:     MyTimeseries timeseries of precipitation data
    
    RETURNS: timeseries of snow water equivalents
    """
        
    meltperiod = 10

    if Tts.has_missing_steps(freq=1):
        print "There are missing days in temperature data."
        print "Handling of these data have not been implemented in calcSWE."
        print "You could interpolate..."
        raise
    
    
    freezingdays = Tts.ordinals[Tts.data[:,0]<0]
    warmID = pylab.find(Tts.data[:,0]>=0)
    meltingdays = Tts.ordinals[warmID] 
    
    
    
    # -------------------------------------------------------------------------
    # define winter as days with temperature below zero
    # allow for up to 5 days of consecutive thawing in winter without removing
    # snowcover    
    
    keepID = np.zeros(len(meltingdays), dtype=bool)
    
    lastMeltDays = np.append(pylab.find(np.diff(meltingdays)>1),meltingdays.size-1)
    meltLengths = np.append(lastMeltDays[0]+1, np.diff(lastMeltDays))
    firstMeltDays = lastMeltDays-meltLengths+1
    # lastMeltDays holds index into meltingdays for the last day in each melting period
    # firstMeltDays holds index into meltingdays for the first day in each melting period
    # meltLengths holds the length of the melting periods

    # if timeseries begins with melting days, they should be treated as such
    # no matter the length of the sequence
    if Tts.ordinals[0] == meltingdays[0]:
        meltLengths[0] = meltperiod+1 # trick to cheat the for loop

    # Find and remove short melting periods
    for k in np.arange(0,firstMeltDays.size-1):
        if meltLengths[k] > meltperiod:
            keepID[firstMeltDays[k]:lastMeltDays[k]+1] = True

    meltingdays = meltingdays[keepID]
    
    # Pick only precip points that have corresponding temperatures
    (days, id1, id2) = list_intersection(Tts.ordinals,Pts.ordinals)
    SWEacc = np.cumsum(Pts.data[:,0])[id2]
    
    if len(id2) == len(Pts.ordinals):
        print "All precip datapoints have corresponding temperatures"
    else:
        print "Some precip points do not have corresponding temperatures"
    
    # -------------------------------------------------------------------------
    # Select precipitation data:
    # If we are in a melting period, SWEacc is set to 0
    # this means that as soon as a melting period starts (day 1 of more than 5
    # consecutive days with temp above 0) all snow disappears
    # If we are in a freezing period (winterday), the precipitation is
    # accumulated.
    # If we are in a non-freezing period with less than 5 consequtive days,
    # there are no changes to SWEacc    

    SWEcorrection = 0
    
    for k,day in enumerate(days):
        if day in meltingdays:
            # We are in a long melting period

            # set correction to the cumulative precip of the present day
            # in the end SWEcorrection will be the accumulated precip on the
            # last melting day
            SWEcorrection = SWEacc[k]
            # make sure no precipitation is accumulated (remove all snow)
            SWEacc[k] = 0
        elif day in freezingdays:
            # we are in freezing period, subtract cumulative precipitation 
            # of the last melting day, to obtain cumulative precipitation during
            # freezing period so far
            SWEacc[k] = SWEacc[k]-SWEcorrection
        else:
            # We are in short melting period
            # This is never true for the first iteration, as we are looping
            
            # Reset correction and SWEacc to reflect a constant snow level
            SWEcorrection = SWEacc[k]-SWEacc[k-1]
            SWEacc[k] = SWEacc[k-1]

    
    SWEts = Pts.empty_copy()
    SWEts.data = SWEacc.reshape(-1,1)
    SWEts.ordinals = days.reshape(-1)
    SWEts.update_times()
    return SWEts 
    
# This is the original code, commented out on June 7 2010
#def calcSWE(Tts,Pts):
#    """
#    Calculate Snow Water Equivalent by accumulating all precip when temperature
#    is below freezing.
#    
#    When temperature rises above 0 degC all snow disappears immediately on the
#    first day of melting.
#    
#    Exception:
#    If the period of melting is 5 days or less, there is no change in snow pack.
#    
#    Input arguments:
#    Tts:     MyTimeseries timeseries of temperature data
#    Pts:     MyTimeseries timeseries of precipitation data
#    
#    RETURNS: timeseries of snow water equivalents
#    """
#        
#    meltperiod = 5
#
#    if Tts.has_missing_steps(freq=1):
#        print "There are missing days in temperature data."
#        print "Handling of these data have not been implemented in calcSWE."
#        print "You could interpolate..."
#        raise
#    
#    
#    freezingdays = Tts.ordinals[Tts.data[:,0]<0]
#    warmID = pylab.find(Tts.data[:,0]>=0)
#    meltingdays = Tts.ordinals[warmID] 
#    
#    
#    
#    # -------------------------------------------------------------------------
#    # define winter as days with temperature below zero
#    # allow for up to 5 days of consecutive thawing in winter without removing
#    # snowcover    
#    
#    keepID = np.zeros(len(meltingdays), dtype=bool)
#    
#    lastMeltDays = np.append(pylab.find(np.diff(meltingdays)>1),meltingdays.size-1)
#    meltLengths = np.append(lastMeltDays[0]+1, np.diff(lastMeltDays))
#    firstMeltDays = lastMeltDays-meltLengths+1
#    # lastMeltDays holds index into meltingdays for the last day in each melting period
#    # firstMeltDays holds index into meltingdays for the first day in each melting period
#    # meltLengths holds the length of the melting periods
#
#    # if timeseries begins with melting days, they should be treated as such
#    # no matter the length of the sequence
#    if Tts.ordinals[0] == meltingdays[0]:
#        meltLengths[0] = meltperiod+1 # trick to cheat the for loop
#
#    # Find and remove short melting periods
#    for k in np.arange(0,firstMeltDays.size-1):
#        if meltLengths[k] > meltperiod:
#            keepID[firstMeltDays[k]:lastMeltDays[k]+1] = True
#
#    meltingdays = meltingdays[keepID]
#    
#    # Pick only precip points that have corresponding temperatures
#    (days, id1, id2) = list_intersection(Tts.ordinals,Pts.ordinals)
#    SWEacc = np.cumsum(Pts.data[:,0])[id2]
#    
#    if len(id2) == len(Pts.ordinals):
#        print "All precip datapoints have corresponding temperatures"
#    else:
#        print "Some precip points do not have corresponding temperatures"
#    
#    # -------------------------------------------------------------------------
#    # Select precipitation data:
#    # If we are in a melting period, SWEacc is set to 0
#    # this means that as soon as a melting period starts (day 1 of more than 5
#    # consecutive days with temp above 0) all snow disappears
#    # If we are in a freezing period (winterday), the precipitation is
#    # accumulated.
#    # If we are in a non-freezing period with less than 5 consequtive days,
#    # there are no changes to SWEacc    
#
#    SWEcorrection = 0
#    
#    for k,day in enumerate(days):
#        if day in meltingdays:
#            # We are in a long melting period
#
#            # set correction to the cumulative precip of the present day
#            # in the end SWEcorrection will be the accumulated precip on the
#            # last melting day
#            SWEcorrection = SWEacc[k]
#            # make sure no precipitation is accumulated (remove all snow)
#            SWEacc[k] = 0
#        elif day in freezingdays:
#            # we are in freezing period, subtract cumulative precipitation 
#            # of the last melting day, to obtain cumulative precipitation during
#            # freezing period so far
#            SWEacc[k] = SWEacc[k]-SWEcorrection
#        else:
#            # We are in short melting period
#            # This is never true for the first iteration, as we are looping
#            
#            # Reset correction and SWEacc to reflect a constant snow level
#            SWEcorrection = SWEacc[k]-SWEacc[k-1]
#            SWEacc[k] = SWEacc[k-1]
#
#    
#    SWEts = Pts.empty_copy()
#    SWEts.data = SWEacc.reshape(-1,1)
#    SWEts.ordinals = days.reshape(-1)
#    SWEts.update_times()
#    return SWEts 

    
    
def calcSnowDepth(SWE,rosnow):
    """ 
    Calculate snow depth.
    
    Input arguments:
    SWE:     Snow Water Equivalent in mm/m^2, 
    rosnow:  Snow density in kg/m^3
    
    We want to multiply SWE by rowater (1000 kg/m^3) 
    and divide by rosnow (kg/m^3) to get snow depth
    However, first SWE is divided by 1000 to get (m/m^2)
    
    RETURNS (SWE/1000)*rowater/rosnow    unit is m
    """
    rowater = 1000. # kg/m^3
    
    return (SWE/1000)*rowater/rosnow
   

def getWinterdaysID(alldays,winterstartmonth=10):
    """
    Calculate day number relative to October 1 for each date in alldays array.
    
    If time series starts before first winter day start calculations one year 
    previously (a day in march 2008 should have a winterDayID relative to 
    2007-10-01)
    
    winterdayID thus contains number of days since October 1
    for days between August1 and October 1, the daynumber is set to 1
    
    Input arguments:
    alldays:             array of datetime.datetime/date objects
    winterstartmonth:    month to start winter calculation from (default=10)
    
    Returns:
    winterdayID:         array of day numbers relative to October 1
    wIDcompr:            contains compressed winter day id array. Since
                           SNUW does linear interpolation on missing dates,
                           we can significantly reduce array length by only
                           including points around discontinuities (e.g. 
                           around YYYY-08-01 and YYYY-10-01)
    
    The function is used as basis for calculating increases in snow density.
    """
    if not type(alldays[0]) in [dt.datetime, dt.date]:
        try:
            alldays = np.array(pylab.num2date(alldays, tz=None))
            #enddate = num2date(alldays[-1], tz=utc)         
        except:
            raise TypeError("Bad input to rosnow_lin")
        datestart = alldays[0]
    else:
        datestart = alldays[0]
        #enddate = alldays[-1]      

    winterdayID = np.ones(alldays.shape)
    wIDcompr = []
    winterf1 = False    # Set to True when we pass into new year
    winterf2 = False    # Set to True after processing first entry in the new year
    
    #count winterdays from october 1
    if datestart.month < winterstartmonth:
        # if time series starts before first winter day
        # start calculations one year previously (a day in march 2008 should 
        # have a winterDayID relative to 2007-10-01)
        firstday = datestart.replace(year=datestart.year-1,month=winterstartmonth,day=1)
    else:
        # Otherwise just use this years first winter day as reference
        firstday = datestart.replace(month=winterstartmonth,day=1)

    for id, day in enumerate(alldays):
        # if we pass october 1 change the winter start day by one year
        if np.logical_and(day.year>firstday.year,
                     day.month >= winterstartmonth):
            # If we have just passed into a new winter cycle (passed october 1)      
            firstday = day.replace(month=winterstartmonth,day=1)
            winterf1 = True
            
        # Get number of days since beginning of winter (oct 1)
        if np.logical_or(day.month < 8,
                     day.month >= winterstartmonth):
            # If we are in winter period, add day number to winterdayID array
            winterdayID[id] = (day-firstday).days +1
            if winterf1 and not winterf2:
                # If this is first entry in the new year add values to 
                # compressed array and set winterf2
                if id >= 1:
                    wIDcompr.append([alldays[id-1],winterdayID[id-1]])                
                wIDcompr.append([day,winterdayID[id]])
                winterf2 = True                
        else:
            # We are now in the period between YYYY-08-01 and YYYY-10-01
            # winterdayID is not altered, which means it will contain ones.
            if winterf1 and winterf2:
                # If this is the first day in the summer period
                # add information to compressed array and reset flags
                if id>=1:
                    wIDcompr.append([alldays[id-1],winterdayID[id-1]])
                wIDcompr.append([day,winterdayID[id]])
                winterf1 = False
                winterf2 = False

        # winterdayID now contains number of days since october 1
        # for days between august1 and october 1, the daynumber is 1
    
    if not wIDcompr[-1][-1] == alldays[-1]:
        wIDcompr.append([alldays[-1],winterdayID[-1]])
    if not wIDcompr[-1][0] == alldays[0]:
        wIDcompr.insert(0,[alldays[0],winterdayID[0]])
    
    return winterdayID, np.array(wIDcompr)

def rosnow_lin(alldays,winterstartmonth=10,rsnw_a=1,rsnw_b=0.7,compress = False):
    """
    Calculate snow densities in kg/m^3 according to a linear equation:
    
    rosnow = rsnw_a * winterdayID + rsnw_b
    
    Input arguments:
    alldays:              Array of datetime/date objects
    winterstartmonth:     kwarg, month to start winter calculation (default=10)
    rsnw_a:               kwarg, slope of linear equation (default=1)
    rsnw_b:               kwarg, 0-intercept of linear equation (default=0.7)
    compress:             kwarg, flag to signal the use of the compressed
                            array of rosnow (default=False)
    
    Returns:
    rosnow:     Array of snow densities for each entry in alldays
    
    If compress is True the function will also return:
    rsn_compr:  Array of datetime/date objects in first column and snow densities
                  in second column
    """
    winterdayID,wIDcompr = getWinterdaysID(alldays,winterstartmonth=10)
    
    # ##########################################################################      
    # snow density calculation
    # rosnow is density of snow in kg/m^3

    rosnow = winterdayID*rsnw_a + rsnw_b
    
    if compress:
        rsn_compr = wIDcompr.copy()
        rsn_compr[:,1] = wIDcompr[:,1]*rsnw_a + rsnw_b
        return rosnow,rsn_compr
    else:
        return rosnow


def list_intersection(list1, list2):
    """Returns the intersection of list1 and list2 as well as two arrays
    containing indices into list1 and list2 for the elements in the intersection.
    
    Example:

    list1 = [0,1,2,3]
    list2 = [2,3,4,5]
    (intersect, id1, id2) = list_intersection(list1,list2)
    
    Result:
    intersect = [2,3]
    id1 = [2,3]
    id1 = [0,1]
    
    take(list1,id1) = intersect
    take(list2,id2) = intersect
    
    """

    int_dict = {}
    list2_dict = {}
    
    # make dictionaty from list1, with elemetns as keys
    #list1_dict = dict(zip(list1,id1))
    for id,e in enumerate(list2): 
        if not list2_dict.has_key(e):
            list2_dict[e] = id        
    
    # now perform intersection with list2
    for id,e in enumerate(list1):
        # test if element is in list1, and that it has not already been added to 
        # the intersection dictionary
        if np.logical_and(list2_dict.has_key(e), not int_dict.has_key(e)):
            # True, add to intersection dictionary...
            int_dict[e] = [id, list2_dict[e]]
            # ...and set index int list2 to True
    
    # ids = array(int_dict.values())
        
    mylist = list()
    id1 = list()
    id2 = list()
    
    for entry in sorted(int_dict.items()):
        x,y = entry
        mylist.append(x)
        id1.append(y[0])
        id2.append(y[1])
    
    try:
        if type(list1) == np.ndarray:
            return (np.array(mylist), id1, id2)
        else:
            return (mylist, id1, id2)            
        
    except:
        return ([], [], [])