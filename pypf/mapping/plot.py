# -*- coding:utf-8 -*-
"""
Created on 25/02/2010

@author: tin
"""

#try:
#    import ipdb as pdb
#except:
import pdb
import copy

#import sys
#import os
import os.path
#import numpy as np

from pkg_resources import resource_filename

import matplotlib.pyplot as plt
import matplotlib.path as mpath 
import matplotlib.patches as mpatches
import pylab

#import osgeo.ogr as ogr
#import osgeo.osr as osr
import gdal

from pypf.mapping.maps import EDCbasemap
from pypf.mapping.geomutils import * 
    
#stop_on_FID1 = None
#stop_on_FID2 = None


pkg_name = 'pypf'
brown1997_relative_path = os.path.join('resources','brown1997')

def PlotIcecap(basemap=None, ax=None, corners=None, plotspecs=None):
    """
    Call signature:
    PlotIcecap(basemap=None, ax=None, corners=None, plotspecs=None)
    
    Method to plot osgeo.ogr geometries.
    
    basemap:    An mpl_toolkits.basemap.basemap object which provides coordinate
                   transformation for the specified axes
    ax:         The axes to plot in
    corners:    Corners of a rectangle specifying the area of the icecap to plot
                    [This option is not presently functional!]
    """    
    ## example: [[-55.0,63.0],[-46.0, 71.0]]

    if not plotspecs:
        plotspecs = [('NUM_CODE', 'integer', 
                            {21:  dict(edgecolor='none',           # Glaciers
                                          facecolor='w',
                                          linewidth=0.5,
                                          zorder=21),
                            }),]
    
    if ax == None:
        pylab.figure()
        ax = pylab.axes(axisbg='0.75')    
    
    kwargs = dict(edgecolor='none', facecolor='w') # None for default and 'none' for no color

    # This part is not functioning! Thus commented out!
    G2 = None
    if basemap != None:
        x = [basemap.xmin,basemap.xmin,basemap.xmax,basemap.xmax]
        y = [basemap.ymin,basemap.ymax,basemap.ymax,basemap.ymin]
        lons, lats = basemap(x, y, inverse=True)
        G2 = ogr.Geometry(ogr.wkbLinearRing)
        G2.AddPoint(lons[0],lats[0])
        G2.AddPoint(lons[1],lats[1])
        G2.AddPoint(lons[2],lats[2])
        G2.AddPoint(lons[3],lats[3])
        #plotGeometry(G2)
    
    # Open geometry file, Mapinfo format
    #g = ogr.Open ('c:\thin\Artek\Data\GroundTemp\Is_vest_reduced_lat-lon.tab')
    #g = ogr.Open ('c:\thin\Artek\Data\GroundTemp\IS_Maniitsoq.tab')
    shpFN = resource_filename(pkg_name, os.path.join(brown1997_relative_path, 'glaciers.shp'))
        
    myEDC = plotShpFile(shpFN, map=basemap, plotspecs=plotspecs)


#def PlotIcecap(basemap=None, ax=None, corners=None):
#    """
#    Call signature:
#    PlotIcecap(basemap = None, ax = None, corners = None)
#    
#    Method to plot osgeo.ogr geometries.
#    
#    basemap:    An mpl_toolkits.basemap.basemap object which provides coordinate
#                   transformation for the specified axes
#    ax:         The axes to plot in
#    corners:    Corners of a rectangle specifying the area of the icecap to plot
#                    [This option is not presently functional!]
#    """    
#    ## example: [[-55.0,63.0],[-46.0, 71.0]]
#
#    if ax == None:
#        pylab.figure()
#        ax = pylab.axes(axisbg='0.75')    
#    
#    kwargs = dict(edgecolor='none', facecolor='w') # None for default and 'none' for no color
#
#    # This part is not functioning! Thus commented out!
#    G2 = None
#    if basemap != None:
#        x = [basemap.xmin,basemap.xmin,basemap.xmax,basemap.xmax]
#        y = [basemap.ymin,basemap.ymax,basemap.ymax,basemap.ymin]
#        lons, lats = basemap(x, y, inverse=True)
#        G2 = ogr.Geometry(ogr.wkbLinearRing)
#        G2.AddPoint(lons[0],lats[0])
#        G2.AddPoint(lons[1],lats[1])
#        G2.AddPoint(lons[2],lats[2])
#        G2.AddPoint(lons[3],lats[3])
#        #plotGeometry(G2)
#    
#    # Open geometry file, Mapinfo format
#    #g = ogr.Open ('c:\thin\Artek\Data\GroundTemp\Is_vest_reduced_lat-lon.tab')
#    #g = ogr.Open ('c:\thin\Artek\Data\GroundTemp\IS_Maniitsoq.tab')
#    #shpFN = resource_filename(pkg_name, os.path.join(brown1997_relative_path, 'glaciers.shp'))
#    
#    g = ogr.Open ('c:\thin\Maps_and_Orthos\Greenland\Inland Ice DCW\glnobox.gmt')
#    
#    # This is a workaround... the ogr routine reports the Geometries as lines
#    # even though they are polygons...
#    # by supplying the argument as_patch, we force the plotting routine to treat
#    # them as polygons.
#    kwargs['as_patch'] = True
#    
#    # Get the number of layers in the current file
#    #L_count = g.GetLayerCount()
#
#    # Get the first layer
#    L = g.GetLayer(0)
#
#    if G2 != None:
#        L.SetSpatialFilter(G2)
#    else:
#        L.SetSpatialFilter(None)
#
#    # Make sure we read from the beginning
#    L.ResetReading()
#
#    # Get the first feature
#    F = L.GetNextFeature()    
#    
#    xl = ax.get_xlim()
#    yl = ax.get_ylim()
#
#    #pdb.set_trace()    
#    
#    # Loop over all features in the present layer
#    while F is not None:
#        # Get number of descriptive fields in feature
#        #field_count = F.GetFieldCount()
#        
#        # Get Geometry connected to the feature
#        G = F.GetGeometryRef()
#        
#        plotGeometry(G, basemap=basemap, ax=ax, **kwargs)
#        
#        # Remove feature from memory
#        F.Destroy()
#        
#        # Get the next feature in the file
#        F = L.GetNextFeature()
#
#    # Remove datasource from memory
#    g.Destroy()
#    
#    ax.set_xlim(*xl)
#    ax.set_ylim(*yl)
###    pylab.xlabel("Longitude")
###    pylab.ylabel("Latitude")
###    pylab.grid(True)
###    pylab.title("Test")
###    pylab.show()




def plot_Greenland(ax=None, corners=None, resolution='l',cities='all', \
                   plot_icecap=False, colors=None, coastlw=1.0, \
                   gdist=10.):
    """
    Call signature:
    plot_Greenland(ax=None, corners=None, resoluiton='l',cities='all')
    
    Method to plot a map of Greenland.
    
    ax:         The axes to plot in
    corners:    Corners of a rectangle specifying the area of the icecap to plot
    """    
    ## example: [[-55.0,63.0],[-46.0, 71.0]]

    if ax == None:
        pylab.figure()
        ax = pylab.axes(axisbg='0.75')    
    
    if colors == None:
        colors = dict(
                coastlines = '0.3', 
                oceans = '1',
                continents = '0.4',
                lakes = '0.9',
                )
        
    if type(corners) == str and corners.lower() == "westcoast":
        corners = [[-55.0, 63.0], [-46.0, 71.0]]
    elif corners == None:
        corners = [[-65.0, 58.0], [20.0, 81.0]]
        
    myEDC = EDCbasemap(resolution=resolution, corners=corners, gdist=gdist, ax=ax,
                       colors=colors, coastlw=coastlw)
    
    if plot_icecap:
        PlotIcecap(basemap=myEDC.map, ax=myEDC.ax, corners=corners)
    
    print('Drawing cities...')
    
    myEDC.GreenlandCities = {
         'Ilulissat':     np.array((-51.100000, 69.216667)),
         'Sisimiut':      np.array((-53.669284, 66.933628)),
         'Kangerlussuaq': np.array((-50.709167, 67.010556)),
         'Nuuk':          np.array((-51.742632, 64.170284))}

    if cities == 'all':
        myEDC.plotPoints(np.array(list(myEDC.GreenlandCities.values()))[:, 0],
            np.array(list(myEDC.GreenlandCities.values()))[:, 1], 
                     'or', ms=8, picker=8, 
                     names=list(myEDC.GreenlandCities.keys()), 
                     tag='cities')
        
        for city in list(myEDC.GreenlandCities.keys()):
            x, y = myEDC.map(myEDC.GreenlandCities[city][0], 
                             myEDC.GreenlandCities[city][1])
            ax.text(x, y, city+'$\quad $', 
                        fontsize=12, fontweight='bold',
                        horizontalalignment='right', 
                        verticalalignment='top',
                        bbox=dict(facecolor='red'))
            
    elif type(cities) in [tuple, list]:
        for city in cities:
            myEDC.plotPoints(np.array(myEDC.GreenlandCities[city])[0],
                             np.array(myEDC.GreenlandCities[city])[1], 
                             'or', ms=8, picker=8, 
                             names=city, tag='cities')
            
            x, y = myEDC.map(myEDC.GreenlandCities[city][0], 
                             myEDC.GreenlandCities[city][1])
            ax.text(x, y, city+'$\quad $', 
                        fontsize=12, fontweight='bold',
                        horizontalalignment='right', 
                        verticalalignment='top',
                        bbox=dict(facecolor='red'))    
    
    ax.figure.myEDC = myEDC
    
    return myEDC


def plotGrlCities(cities='all', names=True, basemap=None, corners='Greenland', resolution='l', \
                  colors=None, coastlw=1.0):
    """
    Function to plot Greenlandic cities on existing or new map.
    
    cities:     List of city names to plot or 'all' for all defined cities
    ax:         The axes to plot in
    corners:    Corners of a rectangle specifying the area of the map to plot
    """    
    ## example: [[-55.0,63.0],[-46.0, 71.0]]

    if colors == None:
        colors = dict(
                coastlines = '0.3', 
                oceans = '1',
                continents = '0.4',
                lakes = '0.9',
                )

    if basemap == None:
        pylab.figure()
        ax = pylab.axes(axisbg='0.75')    
            
        if type(corners) == str and corners.lower() == "westcoast":
            corners = [[-55.0, 63.0], [-46.0, 71.0]]
        elif type(corners) == str and corners.lower() == "greenland":
            corners = [[-65.0, 58.0], [20.0, 81.0]]
        elif corners == None:
            corners = [[-65.0, 58.0], [20.0, 81.0]]
            
        basemap = EDCbasemap(resolution=resolution, corners=corners, gdist=10., 
                           ax=ax, colors=colors, coastlw=coastlw)
    else:
        ax = basemap.ax

    
    print('Drawing cities...')
    
    GreenlandCities = {
         'Ilulissat':     dict(coords=np.array((-51.100000, 69.216667)),
                               dx=0, dy=0,
                               fontsize=12, fontweight='bold',
                               horizontalalignment='right', 
                               verticalalignment='top', zorder=50),
         'Sisimiut':      dict(coords=np.array((-53.669284, 66.933628)),
                               dx=0, dy=0,
                               fontsize=12, fontweight='bold',
                               horizontalalignment='right', 
                               verticalalignment='top', zorder=50),
         'Kangerlussuaq': dict(coords=np.array((-50.709167, 67.010556)),
                               dx=0, dy=0,
                               fontsize=12, fontweight='bold',
                               horizontalalignment='right', 
                               verticalalignment='top', zorder=50),         
         'Nuuk':          dict(coords=np.array((-51.742632, 64.170284)),
                               dx=0, dy=0,
                               fontsize=12, fontweight='bold',
                               horizontalalignment='right', 
                               verticalalignment='top', zorder=50),
         'Qeqertarsuaq':  dict(coords=np.array((-53.535519, 69.246191)),
                               dx=0, dy=0,
                               fontsize=12, fontweight='bold',
                               horizontalalignment='right', 
                               verticalalignment='top', zorder=50),
         'Uumannaq':  dict(coords=np.array((-52.132444, 70.675043)),
                               dx=0, dy=0,
                               fontsize=12, fontweight='bold',
                               horizontalalignment='right', 
                               verticalalignment='top', zorder=50),
         'Upernavik':  dict(coords=np.array((-56.150866, 72.785034)),
                               dx=0, dy=0,
                               fontsize=12, fontweight='bold',
                               horizontalalignment='right', 
                               verticalalignment='top', zorder=50),                       
         'Qaqortoq':  dict(coords=np.array((-46.036663, 60.718234)),
                               dx=0, dy=0,
                               fontsize=12, fontweight='bold',
                               horizontalalignment='right', 
                               verticalalignment='top', zorder=50)                       
                       }
    

    if type(cities) == str and cities == 'all':
        plotCities = list(GreenlandCities.keys())
    elif type(cities) == dict:
        GreenlandCities = copy.deepcopy(cities)
        plotCities = list(GreenlandCities.keys())
        
#        myEDC.plotPoints(np.array(myEDC.GreenlandCities.values())[:, 0],
#            np.array(myEDC.GreenlandCities.values())[:, 1], 
#                     'or', ms=8, picker=8, 
#                     names=myEDC.GreenlandCities.keys(), 
#                     tag='cities', **kwargs)
#        
#        for city in myEDC.GreenlandCities.keys():
#            x, y = myEDC.map(myEDC.GreenlandCities[city][0], 
#                             myEDC.GreenlandCities[city][1])
#            ax.text(x, y, city+'$\quad $', 
#                        fontsize=12, fontweight='bold',
#                        horizontalalignment='right', 
#                        verticalalignment='top', **kwargs)
#            
#    elif type(cities) in [tuple, list]:
    
    for city in plotCities:
        zorder = GreenlandCities[city].pop('zorder', 50)
        dx = GreenlandCities[city].pop('dx', 0)
        dy = GreenlandCities[city].pop('dy', 0)
        coords = GreenlandCities[city].pop('coords')
        marker = GreenlandCities[city].pop('marker', 'o')
        mc = GreenlandCities[city].pop('color', 'r')
        ms = GreenlandCities[city].pop('size', 8)
        
        basemap.plotPoints(coords[0],
                         coords[1], 
                         marker=marker, color=mc, ms=ms,
                         picker=8, 
                         names=city, tag='cities',zorder=zorder)
        
        x, y = basemap.map(coords[0]+dx, 
                         coords[1]+dy)
        if names:
            ax.text(x, y, city,zorder=zorder,
                    **GreenlandCities[city])    
    
    return ax


def print_file_info(fname):
    # script to count features

    # get the driver
    driver = ogr.GetDriverByName('GMT')
    # open the data source
    datasource = driver.Open(fname, 0)
    if datasource is None:
        print('Could not open file')
        sys.exit(1)

    # get the data layer
    layer = datasource.GetLayer()
    # loop through the features and count them
    cnt = 0
    feature = layer.GetNextFeature()
    while feature:
        cnt = cnt + 1
        feature.Destroy()
        feature = layer.GetNextFeature()
    print('There are ' + str(cnt) + ' features')
    # close the data source
    datasource.Destroy()


def combine_PF_zones(g=None):
    """
    Method to combine PF zones from Brown 1998 map
    """    
    
    inFN = os.path.join('C:\\','thin','Maps_and_Orthos',
                        'Circum-Arctic Map of Permafrost','permaice.shp')

    outFN = r'c:\thin\Maps_and_Orthos\Circum-Arctic Map of Permafrost\C_perma_new2.shp'
        
    combineNeighbourFeatures(inFN, outFN, filterStr="EXTENT = 'C'", 
                             driverName=None,
                             clearAttributes=None)
    
    outFN = r'c:\thin\Maps_and_Orthos\Circum-Arctic Map of Permafrost\D_perma.shp'
    
    combineNeighbourFeatures(inFN, outFN, filterStr="EXTENT = 'D'", 
                             driverName=None,
                             clearAttributes=None)
    
    outFN = r'c:\thin\Maps_and_Orthos\Circum-Arctic Map of Permafrost\S_perma.shp'
    
    combineNeighbourFeatures(inFN, outFN, filterStr="EXTENT = 'S'", 
                             driverName=None,
                             clearAttributes=None)
    
    outFN = r'c:\thin\Maps_and_Orthos\Circum-Arctic Map of Permafrost\I_perma.shp'
    
    combineNeighbourFeatures(inFN, outFN, filterStr="EXTENT = 'I'", 
                             driverName=None,
                             clearAttributes=None)    

    outFN = r'c:\thin\Maps_and_Orthos\Circum-Arctic Map of Permafrost\SI_perma.shp'
    
    combineNeighbourFeatures(inFN, outFN, filterStr="(EXTENT = 'S') OR (EXTENT = 'I')", 
                             driverName=None,
                             clearAttributes=None)  






def plotBrownPF(ax=None, corners=None):
    """
    Method to combine PF zones from Brown 1998 map
    """    
    
    outFN = r'c:\thin\Maps_and_Orthos\Circum-Arctic Map of Permafrost\C_perma.shp'
    myEDC = plotShpFile(outFN)
    plt.draw()
        
    outFN = r'c:\thin\Maps_and_Orthos\Circum-Arctic Map of Permafrost\D_perma.shp'
    plotShpFile(outFN, myEDC.ax)
    plt.draw()
    
    outFN = r'c:\thin\Maps_and_Orthos\Circum-Arctic Map of Permafrost\S_perma.shp'
    plotShpFile(outFN, myEDC.ax)
    plt.draw()
    
    outFN = r'c:\thin\Maps_and_Orthos\Circum-Arctic Map of Permafrost\I_perma.shp'
    plotShpFile(outFN, myEDC.ax)    
    plt.draw()

#    outFN = r'C:\THIN\Maps_and_Orthos\Circum-Arctic Map of Permafrost\C_perma.shp'
#    myEDC = plotShpFile(outFN)
#    plt.draw()
#    
#    outFN = r'C:\THIN\Maps_and_Orthos\Circum-Arctic Map of Permafrost\D_perma.shp'
#    plotShpFile(outFN, myEDC.ax)
#    plt.draw()
#    
#    outFN = r'C:\THIN\Maps_and_Orthos\Circum-Arctic Map of Permafrost\SI_perma.shp'
#    plotShpFile(outFN, myEDC.ax)  
#    plt.draw()




def Plot_PF_Greenland(ax=None, corners=None, colors=None, resolution='l',
                      cities='all', coastlw=1.0):
    """
    Call signature:
    PlotIcecap(basemap = None, ax = None, corners = None)
    
    Method to plot osgeo.ogr geometries.
    
    basemap:    An mpl_toolkits.basemap.basemap object which provides coordinate
                   transformation for the specified axes
    ax:         The axes to plot in
    corners:    Corners of a rectangle specifying the area of the icecap to plot
                    [This option is not presently functional!]
    """    
    # example: [[-55.0,63.0],[-46.0, 71.0]]


    if colors == None:
        colors = dict(EXTENT = dict(\
                           C = dict(edgecolor='none',
                                    facecolor='b',
                                    linewidth=0.5,
                                    zorder=20),
                           D = dict(edgecolor='none',
                                    facecolor="#FFAA00",
                                    linewidth=0.5,
                                    zorder=20),
                           S = dict(edgecolor='none',
                                    facecolor="#00E600",
                                    linewidth=0.5,
                                    zorder=20),
                           I = dict(edgecolor='none',
                                    facecolor="#00E600",
                                    linewidth=0.5,
                                    zorder=20)
                           ),
                      continents =  dict(edgecolor='k',
                                    facecolor='0.4',
                                    linewidth=0.5,
                                    zorder=16),
                      oceans =  dict(edgecolor='k',
                                    facecolor="#DCE6FF",
                                    linewidth=0.5,
                                    zorder=15),
                      lakes =  dict(edgecolor='k',
                                    facecolor="#DCE6FF",
                                    linewidth=0.5,
                                    zorder=18),
                      glaciers =  dict(edgecolor='none',
                                    facecolor='w',
                                    linewidth=0.5,
                                    zorder=21),
                      other =   dict(edgecolor='y',
                                    facecolor='y',
                                    linewidth=1,
                                    zorder=30)             
                            )

        
    if ax == None:
        pylab.figure()
        ax = pylab.axes(axisbg='0.75')      
        
    if type(corners) == str and corners.lower() == "westcoast":
        corners = [[-55.0, 63.0], [-46.0, 71.0]]
    elif corners == None:
        corners = [[-65.0, 58.0], [20.0, 81.0]]
        
    myEDC = EDCbasemap(resolution=resolution, corners=corners, gdist=10., ax=ax,
                       colors=None, coastlw=coastlw, plot_continents=False)
    
    
    # Open geometry file, Mapinfo format
    fname = os.path.join('C:\\','THIN','Maps_and_Orthos','Circum-Arctic Map of Permafrost','permaice.shp')
    g = ogr.Open (fname)
    
    # Get the number of layers in the current file
    L_count = g.GetLayerCount()

    # Get the first layer
    L = g.GetLayer(0)

    # Get spatial reference of map
    mapSR = osr.SpatialReference()
    mapSR.ImportFromProj4(myEDC.map.proj4string)
    
    # Get spatial reference of layer
    layerSR = L.GetSpatialRef()
    
    # Create transformations to and from map coordinates:
    coordTrans2map = osr.CoordinateTransformation(layerSR, mapSR)
    coordTrans2file = osr.CoordinateTransformation(mapSR, layerSR)
    

    # This part is not functioning! Thus commented out!
    G2 = None
#    if myEDC.map != None:
#        x = [myEDC.map.xmin,myEDC.map.xmin,myEDC.map.xmax,myEDC.map.xmax]
#        y = [myEDC.map.ymin,myEDC.map.ymax,myEDC.map.ymax,myEDC.map.ymin]
#        #lons, lats = myEDC.map(x, y, inverse=True)
#        rect = ogr.Geometry(ogr.wkbLinearRing)
#        rect.AddPoint(x[0],y[0])
#        rect.AddPoint(x[1],y[1])
#        rect.AddPoint(x[2],y[2])
#        rect.AddPoint(x[3],y[3])
#        rect.AddPoint(x[0],y[0])
#        #G2.AddPoint(lons[0],lats[0])
#        #G2.AddPoint(lons[1],lats[1])
#        #G2.AddPoint(lons[2],lats[2])
#        #G2.AddPoint(lons[3],lats[3])
#        #G2.AddPoint(lons[0],lats[0])
#        rect.AssignSpatialReference(mapSR)
#        rect.Transform(coordTrans2file)
#        G2 = rect
#        #G2 = ogr.Geometry(ogr.wkbPolygon)
#        #G2.AddGeometry(outring)
#        #G2.AddGeometry(inring)
#        #plotGeometry(G2)

    if G2 != None:
        L.SetSpatialFilter(G2)
    else:
        L.SetSpatialFilter(None)

    # Make sure we read from the beginning
    L.ResetReading()

    # Get the first feature
    F = L.GetNextFeature()    
    
    xl = ax.get_xlim()
    yl = ax.get_ylim()
    
   
    i = 0
    # Loop over all features in the present layer
    while F is not None:
        # Get number of descriptive fields in feature
        field_count = F.GetFieldCount()
        
        
        num_code = F.GetFieldAsInteger('NUM_CODE')
        extent = F.GetFieldAsString('EXTENT')
        
        #print "{0:05d}    num: {1}    extent: {2}".format(i, num_code, extent)
        
        if num_code == 21:
            # Glaciers
            kwargs = colors['glaciers']
        elif num_code == 22:
            # Relict permafrost
            kwargs = colors['other']
        elif num_code == 23:
            # Inland lakes
            kwargs = colors['lakes']
        elif num_code == 24:
            # Oceans / inland seas
            kwargs = colors['oceans']        
        elif num_code == 25:    
            # Land
            kwargs = colors['continents']
        elif num_code in np.arange(1,21):
            kwargs = colors['EXTENT'][extent]
        else:
            kwargs = colors['other']
        
        # Get Geometry connected to the feature
        G = F.GetGeometryRef()
        
        G.TransformTo(mapSR)
        
        plotGeometry(G, basemap=myEDC.map, ax=ax, noconv=True, as_patch=True,**kwargs)
        
        # Remove feature from memory
        F.Destroy()
        
        # Get the next feature in the file
        F = L.GetNextFeature()
        
        i+=1

    # Remove datasource from memory
    g.Destroy()
    
    ax.set_xlim(*xl)
    ax.set_ylim(*yl)
##    pylab.xlabel("Longitude")
##    pylab.ylabel("Latitude")
##    pylab.grid(True)
##    pylab.title("Test")
##    pylab.show()

    return myEDC



#def combine_PF_zones(g=None):
#    """
#    Method to combine PF zones from Brown 1998 map
#    """    
#        
#    if g == None:
#        # Open geometry file, Mapinfo format
#        fname = os.path.join('D:\\','Thomas','Maps_and_Orthos','Circum-Arctic Map of Permafrost','permaice.shp')
#        g = ogr.Open (fname)
#        
#    # Get the number of layers in the current file
#    L_count = g.GetLayerCount()
#
#    # Get the first layer
#    L = g.GetLayer(0)
#
#    # Make sure we read from the beginning
#    L.ResetReading()
#
#
#    L.layer.SetAttributeFilter("EXTENT = 'C'")
#    F_Count = L.GetFeatureCount()
#    L.ResetReading()
#    
#    
###    FIDs = [L.GetNextFeature().GetFID() for k in np.arange(F_count)]
###    L.ResetReading()
## 
###    while len(FIDs) > 0:
###        F1 = L.GetFeature(FIDs.pop(0))
###        FIDs2 = []
###        for fid in FIDs:
###            F2 = L.GetFeature(fid)
##        
##    F1 = L.GetNextFeature()
##    G1 = F1.GetGeometryRef()
##    F2 = L.GetNextFeature()
##    while F1 != None and F2 != None:        
##        if F1.Disjoint(F2)
##        
##        
##        for id in np.arange(F_Count):
##        
##        
##    
##    while len(FIDs) > 0:
##        F1 = L.GetFeature(FIDs.pop(0))
##        FIDs2 = []
##        for fid in FIDs:
##            F2 = L.GetFeature(fid)
##         
#        
#    
#    # Get the first feature
#    F1 = L.GetNextFeature()    
#    
#    # Loop over all features in the present layer
#    while F is not None:
#        # Get number of descriptive fields in feature
#        field_count = F.GetFieldCount()
#        
#        num_code = F.GetFieldAsInteger('NUM_CODE')
#        extent = F.GetFieldAsString('EXTENT')
#        
#        #print "{0:05d}    num: {1}    extent: {2}".format(i, num_code, extent)
#        
#        if num_code == 21:
#            # Glaciers
#            kwargs = colors['glaciers']
#        elif num_code == 22:
#            # Relict permafrost
#            kwargs = colors['other']
#        elif num_code == 23:
#            # Inland lakes
#            kwargs = colors['lakes']
#        elif num_code == 24:
#            # Oceans / inland seas
#            kwargs = colors['oceans']        
#        elif num_code == 25:    
#            # Land
#            kwargs = colors['continents']
#        elif num_code in np.arange(1,21):
#            kwargs = colors['EXTENT'][extent]
#        else:
#            kwargs = colors['other']
#        
#        # Get Geometry connected to the feature
#        G = F.GetGeometryRef()
#        
#        G.TransformTo(mapSR)
#        
#        plotGeometry(G, basemap=myEDC.map, ax=ax, noconv=True, as_patch=True,**kwargs)
#        
#        # Remove feature from memory
#        F.Destroy()
#        
#        # Get the next feature in the file
#        F = L.GetNextFeature()
#        
#        i+=1
#
#    # Remove datasource from memory
#    g.Destroy()
#    
#    ax.set_xlim(*xl)
#    ax.set_ylim(*yl)
###    pylab.xlabel("Longitude")
###    pylab.ylabel("Latitude")
###    pylab.grid(True)
###    pylab.title("Test")
###    pylab.show()
#
#    return myEDC


def plotShpFile(shpFN, basemap=None, map=None, mapSR=None, plotspecs=None, ax=None, as_patch=True, crop_geom=None, **kwargs):
    """
    crop_geom:  An osgeo.ogr geometry. The geometries to plot will be intersected
                with crop_geom to plot only parts of the geometries that 
                fall within crop_geom.
                Example:
                crop_geom=osgeo.ogr.Geometry(wkt='POLYGON ((491700 7421200, 491700 7424200, 503000 7424200, 503000 7421200, 491700 7421200))')

                
    plotspecs: plot spec mapping which may depend on feature attribute values.
                If plotspecs is a dictionary, it will be considered key value pairs
                for plot attributes and passed to plotting directly.
                If plotspecs is a list, it must consist of tuples of feature
                attribute names followed by a field type (string, integer, float) and 
                dictionary of plot specifications (e.g. edgecolor etc).
                If match is not found in first dict (based on first feature attribute value)
                the second feature attribute will be checked... etc.
                (below, a match will first be attempted for the NUM_CODE attribute and if
                no match is found, the EXTENT attribute will be tested).  
                
                Example:
                    plotspecs = [('NUM_CODE', 'integer', dict(
                                            25 =  dict(edgecolor='k',              # Continents
                                                          facecolor='0.4',
                                                          linewidth=0.5,
                                                          zorder=16),
                                            24 =  dict(edgecolor='k',              # Oceans
                                                          facecolor="#DCE6FF",
                                                          linewidth=0.5,
                                                          zorder=15),
                                            23 =  dict(edgecolor='k',              # Lakes
                                                          facecolor="#DCE6FF",
                                                          linewidth=0.5,
                                                          zorder=18),
                                            21 =  dict(edgecolor='none',           # Glaciers
                                                          facecolor='w',
                                                          linewidth=0.5,
                                                          zorder=21)
                                            22 =  dict(edgecolor='k',              # other
                                                          facecolor='none',
                                                          linewidth=0.5,
                                                          zorder=25)                                                          
                                            )),
                                  ('EXTENT', 'string', dict(                                                
                                            C = dict(edgecolor='k',
                                                    facecolor='b',
                                                    linewidth=0.5,
                                                    zorder=20)
                                            ))]
    
    """    
    
    if not kwargs:
        kwargs = dict()
         
    # Open geometry file, Mapinfo format
    g = ogr.Open (shpFN)
    
    # Get the number of layers in the current file
    L_count = g.GetLayerCount()

    # Get the first layer
    L = g.GetLayer(0)

    # Get spatial reference of map
    if basemap:
        map = basemap.map
        
    if map:
        if mapSR is None:
            mapSR = osr.SpatialReference()
            mapSR.ImportFromProj4(map.proj4string)
        noconv = False
        ax = map.ax
    else:
        if not mapSR:
            raise ValueError('Either basemap or mapSR must be passed')
        noconv = True
    
    # Get spatial reference of layer
    layerSR = L.GetSpatialRef()
    
    # Create transformations to and from map coordinates:
    coordTrans2map = osr.CoordinateTransformation(layerSR, mapSR)
    coordTrans2file = osr.CoordinateTransformation(mapSR, layerSR)
    
    # Make sure we read from the beginning
    L.ResetReading()

    # Get the first feature
    F = L.GetNextFeature()    
    
    xl = ax.get_xlim()
    yl = ax.get_ylim()
   
    i = 0
    
    # Loop over all features in the present layer
    while F is not None:
        # Get number of descriptive fields in feature
        field_count = F.GetFieldCount()
        
        do_plot = False
        
        if plotspecs is not None:
            if isinstance(plotspecs, dict):
                kwargs.update(plotspecs)
                do_plot = True
            else:
                for field, ftype, specs in plotspecs:
                    if ftype.lower() == 'double':
                        fieldval = F.GetFieldAsDouble(field)
                    elif ftype.lower() == 'integer':
                        fieldval = F.GetFieldAsInteger(field)
                    else:
                        fieldval = F.GetFieldAsString(field)
                    
                    if fieldval in list(specs.keys()):
                        #pdb.set_trace()
                        kwargs.update(specs[fieldval])
                        #print fieldval
                        #if fieldval == "LDHIGH":
                        #   pdb.set_trace()
                        do_plot = True
                        break
        else:
            # No plotspecs given, plot with standard settings
            do_plot = True
        
        if do_plot:    
            plotFeature(F, mapSRS=mapSR, basemap=map, ax=ax, noconv=noconv, as_patch=as_patch, **kwargs)
        
        # Remove feature from memory
        F.Destroy()
        
        # Get the next feature in the file
        F = L.GetNextFeature()
        
        i+=1

    # Remove datasource from memory
    g.Destroy()
    
    ax.set_xlim(*xl)
    ax.set_ylim(*yl)

    return basemap


def plotFeature(F, mapSRS=None, SRStransform=None, basemap=None, ax=None, 
                G2=None, noconv=False, as_patch=False, **kwargs):
    """
    Method to plot osgeo.ogr features.
    
    F:            The feature to plot
    mapSRS:       Spatial Reference System of the output map, feature geometries
                     will be transformed to this system.
    SRStransform: Transform to apply to the feature geometries. 
    basemap:      An mpl_toolkits.basemap.basemap object which provides coordinate
                     transformation for the specified axes
    ax:           The axes to plot in
    G2:           An osgeo.ogr geometry. The geometry to plot will be intersected
                     with G2 to plot only parts of G that fall within G2
    edgecolor:    Geometry edge color
    facecolor:    Geometry face color
    """
    
    if ax == None:
        fh = pylab.figure()
        ax = pylab.axes()
        
    if kwargs == {}:
        if as_patch:
            kwargs = dict(edgecolor='k', facecolor='y')

    # Get the associated geometry
    G = F.GetGeometryRef()
    
    # Transform the geometry
    if SRStransform != None:
        G.transform(SRStransform)
    elif mapSRS != None:    
        G.TransformTo(mapSRS)
    
    if G2 != None:
        G = G.intersection(G2)
    
    attributes = list(F.items())
    attributes['FID'] = F.GetFID()
            
    plotGeometry(G, basemap=basemap, ax=ax, G2=G2, noconv=noconv, 
                 as_patch=as_patch, attributes=attributes, **kwargs)
            
    return ax



def onGeometryPick(event):
    thisGeometry = event.artist
    print('pypf.mapping.plot:onGeometryPick:')
    print('  Geometry: {0}'.format(thisGeometry))
    try:
        print('  Attributes: {0}'.format(thisGeometry.attributes))
    except:
        pass
    
    return False
    

def plotGeometry(G, basemap=None, ax=None, G2=None, noconv=False, 
                 as_patch=False, attributes=None, **kwargs):
    """
    Call signature:
    PlotGeo(G, basemap = None, ax = None, G2 = None, **kwargs)
    
    Method to plot osgeo.ogr geometries.
    
    G:          The geometry to plot
    basemap:    An mpl_toolkits.basemap.basemap object which provides coordinate
                   transformation for the specified axes
    ax:         The axes to plot in
    G2:         An osgeo.ogr geometry. The geometry to plot will be intersected
                   with G2 to plot only parts of G that fall within G2
    edgecolor:  Geometry edge color
    facecolor:  Geometry face color
    """
        
    if ax == None:
        fh = pylab.figure()
        ax = pylab.axes()
        
    if kwargs == {}:
        if as_patch:
            kwargs = dict(edgecolor='k', facecolor='y')

    if G2 != None:
        G = G.intersection(G2)
    
#    if attributes['GM_LABEL'] == "LDHIGH":
#        pdb.set_trace()
    
    # Get vertices and pen movement codes    
    vertices, codes = get_vertices(G)
    
    if as_patch:
        # Create the Path object 
        path = mpath.Path(vertices, codes) 
        # Add plot it 
        patch = mpatches.PathPatch(path, picker=True, **kwargs) 
    
    #    if attributes['GM_LABEL'] == "LDHIGH":        
            # Add on pick event and attributes
        h = ax.add_patch(patch)
        h.attributes = attributes
    else:
        hg = ax.plot(vertices[:,0], vertices[:,1], picker=True, **kwargs)
        for h in hg:
            h.attributes = attributes

    ax.figure.canvas.mpl_connect('pick_event', onGeometryPick)    
#        print "plotted...{0}".format(attributes['GM_LABEL'])
#        print "Area...{0}".format(attributes['AREA'])
                
    return ax    
    
    
def get_vertices(G):    
    """Traverse a (possibly nested) geometry, and retrieve the
    vertice coordinates, and the pen movement codes.
    """
    #print "New iteration"    
    # Get the number of geometries in combined geometry
    G_count = G.GetGeometryCount()
    #print G_count
    #print "Number of geometries: {0}".format(G_count)
    # Loop over all geometries in combined geometry and recursively plot
    
    # Get vertices of the geometry
    vertices = getGeometryCoords(G)
#    if len(vertices):
#        print "no Vertices found"
    
    # Produce list of codes for path generation
    codes = np.ones(len(vertices), dtype=mpath.Path.code_type) * mpath.Path.LINETO
    if len(codes):
        # If the geometry has any vertices, move to the first vertice
        codes[0] = mpath.Path.MOVETO
    
    # Now loop over inside geometries 
    #G_count = G_count
    for gid in range(0,G_count):
#        print "using subgeometry {0}".format(gid)
        # return subgeometry
        subG = G.GetGeometryRef(gid)
        
        in_verts, in_codes = get_vertices(subG)
        
#        if len(in_verts):
#            print "no Vertices found (2)"
        
        # add this to the list of vertices etc.
        
        vertices = np.concatenate((vertices, 
                                   in_verts)) 
                
        codes = np.concatenate((codes, in_codes)) 

#    print "Returning from iteration..."
    return vertices, codes



# Commented out as backup when implementing the new 
# recursive get_vertices functions.
#
#
#def plotGeometry(G, basemap=None, ax=None, G2=None, noconv=False, 
#                 as_patch=False, attributes=None, **kwargs):
#    """
#    Call signature:
#    PlotGeo(G, basemap = None, ax = None, G2 = None, **kwargs)
#    
#    Method to plot osgeo.ogr geometries.
#    
#    G:          The geometry to plot
#    basemap:    An mpl_toolkits.basemap.basemap object which provides coordinate
#                   transformation for the specified axes
#    ax:         The axes to plot in
#    G2:         An osgeo.ogr geometry. The geometry to plot will be intersected
#                   with G2 to plot only parts of G that fall within G2
#    edgecolor:  Geometry edge color
#    facecolor:  Geometry face color
#    """
#        
#    if ax == None:
#        fh = pylab.figure()
#        ax = pylab.axes()
#        
#    if kwargs == {}:
#        if as_patch:
#            kwargs = dict(edgecolor='k', facecolor='y')
#
#    if G2 != None:
#        G = G.intersection(G2)
#    
#    if attributes['GM_LABEL'] == "LDHIGH":
#        pdb.set_trace()
#        
#    # Get the number of geometries in combined geometry
#    G_count = G.GetGeometryCount()
#    #print G_count
#    
#    if G_count == 0:
#        # This is a single geometry
#        # Get all the x and y coordinates
#        x = [G.GetX(i) for i in range(G.GetPointCount())]
#        y = [G.GetY(i) for i in range(G.GetPointCount())]
#
#        # plot a filled polygon
#        if basemap != None and noconv != True:
#            x, y = basemap(x, y)
#        
#        if as_patch:
#            hg = ax.fill(x, y, picker=True, **kwargs)
#        else:
#            hg = ax.plot(x, y, picker=True, **kwargs)
#        
#        for h in hg:
#            h.attributes = attributes
#            
#        ax.figure.canvas.mpl_connect('pick_event', onGeometryPick)
#        
#    else:
#        # This is a combined geometry
#        # Loop over all geometries in combined geometry and recursively plot
#        
#        # Get reference to outside geometry
#        outG = G.GetGeometryRef(0)
#        
#        # Get vertices of the geometry
#        vertices = getGeometryCoords(outG)
#        
#        # Produce list of codes for path generation
#        codes = np.ones(len(vertices), dtype=mpath.Path.code_type) * mpath.Path.LINETO
#        if len(codes):
#            # If the geometry has any vertices, move to the first vertice
#            codes[0] = mpath.Path.MOVETO
#        
#        # Now loop over inside geometries 
#        
#        for gid in range(G_count-1):
#            # return subgeometry
#            subG = G.GetGeometryRef(gid+1)
#            
#            # Get vertices of the geometry
#            in_verts = getGeometryCoords(subG)
#        
#            # Produce list of codes for path generation
#            in_codes = np.ones(len(in_verts), dtype=mpath.Path.code_type) * mpath.Path.LINETO
#            if len(codes):
#                # If the geometry has any vertices, move to the first vertice
#                in_codes[0] = mpath.Path.MOVETO
#            
#            # add this to the list of vertices etc.
#            
#            vertices = np.concatenate((vertices, 
#                                       in_verts)) 
#                    
#            codes = np.concatenate((codes, in_codes)) 
#        
#        # Create the Path object 
#        path = mpath.Path(vertices, codes) 
#        # Add plot it 
#        patch = mpatches.PathPatch(path, picker=True, **kwargs) 
#
#        if attributes['GM_LABEL'] == "LDHIGH":        
#            # Add on pick event and attributes
#            h = ax.add_patch(patch)
#            h.attributes = attributes
#        
#            ax.figure.canvas.mpl_connect('pick_event', onGeometryPick)    
##            # recursively iterate over subgeometries
##            plotGeometry(subG, basemap=basemap, ax=ax, G2=G2, noconv=noconv, 
##                         as_patch=as_patch, attributes=attributes, **kwargs)        
#            print "plotted...{0}".format(attributes['GM_LABEL'])
#            print "Area...{0}".format(attributes['AREA'])
#                
#    return ax








        

#-------------------------------------------------------------------------------
#---NOT USED
#-------------------------------------------------------------------------------


# THIS FUNCTION MAY NOT BE USED!! CHECK IT OUT.
def plotOgrLayer(L, colors=None, ax=None, corners=None, resolution='l',
                 coastlw=0.5, plot_continents=False, **plotArgs):
    """
    ax:         The axes to plot in
    corners:    Corners of a rectangle specifying the area of the icecap to plot
                    [This option is not presently functional!]
    """    
    
    if plotArgs == {}:
        plotArgs = dict(facecolor = 'none', edgecolor='k')
        
    if ax == None:
        pylab.figure()
        ax = pylab.axes(axisbg='0.75')      
        
    if type(corners) == str and corners.lower() == "westcoast":
        corners = [[-55.0, 63.0], [-46.0, 71.0]]
    elif type(corners) == str and corners.lower() == "greenland":    
        corners = [[-65.0, 58.0], [20.0, 81.0]]
        
    myEDC = EDCbasemap(resolution=resolution, corners=corners, gdist=10., ax=ax,
                       colors=None, coastlw=coastlw, 
                       plot_continents=plot_continents)
    
    # Get spatial reference of map
    mapSR = osr.SpatialReference()
    mapSR.ImportFromProj4(myEDC.map.proj4string)
    
    # Get spatial reference of layer
    layerSR = L.GetSpatialRef()
    
    # Create transformations to and from map coordinates:
    coordTrans2map = osr.CoordinateTransformation(layerSR, mapSR)
    coordTrans2file = osr.CoordinateTransformation(mapSR, layerSR)
    
    # Make sure we read from the beginning
    L.ResetReading()

    # Get the first feature
    F = L.GetNextFeature()    
    
    xl = ax.get_xlim()
    yl = ax.get_ylim()
    
    i = 0
    # Loop over all features in the present layer
    while F is not None:
        # Get number of descriptive fields in feature
        field_count = F.GetFieldCount()
        
        # Get Geometry connected to the feature
        G = F.GetGeometryRef()
        
        G.TransformTo(mapSR)
        
        plotGeometry(G, basemap=myEDC.map, ax=ax, noconv=True, as_patch=True,
                     **plotArgs)
        
        # Remove feature from memory
        F.Destroy()
        
        # Get the next feature in the file
        F = L.GetNextFeature()
        i+=1
    
    ax.set_xlim(*xl)
    ax.set_ylim(*yl)

    return myEDC


def plot_file(fname, basemap=None, ax=None, corners=None):
    """
    Call signature:
    PlotIcecap(basemap = None, ax = None, corners = None)
    
    Method to plot osgeo.ogr geometries.
    
    basemap:    An mpl_toolkits.basemap.basemap object which provides coordinate
                   transformation for the specified axes
    ax:         The axes to plot in
    corners:    Corners of a rectangle specifying the area of the icecap to plot
                    [This option is not presently functional!]
    """    
    ## example: [[-55.0,63.0],[-46.0, 71.0]]

    if ax == None:
        pylab.figure()
        ax = pylab.axes(axisbg='0.75')    
    
    kwargs = dict(edgecolor='k', facecolor='w')
    
    # Open geometry file, Mapinfo format
    g = ogr.Open (fname)
        
    # Get the number of layers in the current file
    L_count = g.GetLayerCount()

    # Get the first layer
    L = g.GetLayer(0)

    # Make sure we read from the beginning
    L.ResetReading()

    # Get the first feature
    F = L.GetNextFeature()    

    # This part is not functioning! Thus commented out!
##    G2 = None
##    if corners != None:
##        G2 = F.GetGeometryRef().Clone()
##        G2.Empty()
##        G2.AddPoint(corners[0][0],corners[0][1])
##        G2.AddPoint(corners[0][0],corners[1][1])
##        G2.AddPoint(corners[1][0],corners[1][1])
##        G2.AddPoint(corners[1][0],corners[1][0])
##        plotGeometry(G2)
    
#    xl = ax.get_xlim()
#    yl = ax.get_ylim()
    # Loop over all features in the present layer
    while F is not None:
        # Get number of descriptive fields in feature
        field_count = F.GetFieldCount()
        
        # Get Geometry connected to the feature
        G = F.GetGeometryRef()
        
        plotGeometry(G, basemap=basemap, ax=ax, **kwargs)
        
        # Get the next feature in the file
        F = L.GetNextFeature()

#    ax.set_xlim(*xl)
#    ax.set_ylim(*yl)
##    pylab.xlabel("Longitude")
##    pylab.ylabel("Latitude")
##    pylab.grid(True)
##    pylab.title("Test")
##    pylab.show()



def plotShpFile_legacy(shpFN, ax=None, corners=None, colors=None, resolution='l',
                      cities='all', coastlw=1.0, gdist=10):
    """
    Function to plot shape file and chose a colorset according to feature attribute  
    "NUM_CODE". It is inteded for plotting Brown et al. (1997) permafrost map.
            
    Call signature:
    plotShpFile(basemap = None, ax = None, corners = None)
    
    Method to plot osgeo.ogr geometries.
    
    basemap:    An mpl_toolkits.basemap.basemap object which provides coordinate
                   transformation for the specified axes
    ax:         The axes to plot in
    corners:    Corners of a rectangle specifying the area of the icecap to plot
                    [This option is not presently functional!]
    """    


    if colors == None:
        colors = dict(EXTENT = dict(\
                           C = dict(edgecolor='k',
                                    facecolor='b',
                                    linewidth=0.5,
                                    zorder=20),
                           D = dict(edgecolor='k',
                                    facecolor="#FFAA00",
                                    linewidth=0.5,
                                    zorder=20),
                           S = dict(edgecolor='k',
                                    facecolor="#00E600",
                                    linewidth=0.5,
                                    zorder=20),
                           I = dict(edgecolor='k',
                                    facecolor="#000010",
                                    linewidth=0.5,
                                    zorder=20)
                           ),
                      continents =  dict(edgecolor='k',
                                    facecolor='0.4',
                                    linewidth=0.5,
                                    zorder=16),
                      oceans =  dict(edgecolor='k',
                                    facecolor="#DCE6FF",
                                    linewidth=0.5,
                                    zorder=15),
                      lakes =  dict(edgecolor='k',
                                    facecolor="#DCE6FF",
                                    linewidth=0.5,
                                    zorder=18),
                      glaciers =  dict(edgecolor='none',
                                    facecolor='w',
                                    linewidth=0.5,
                                    zorder=21),
                      other =   dict(edgecolor='y',
                                    facecolor='y',
                                    linewidth=1,
                                    zorder=30)             
                            )

        
    if ax == None:
        pylab.figure()
        ax = pylab.axes(axisbg='0.75')      

    if type(corners) == str and corners.lower() == "westcoast":
        corners = [[-55.0, 63.0], [-46.0, 71.0]]
    if type(corners) == str and corners.lower() == "greenland":
        corners = [[-65.0, 58.0], [20.0, 81.0]]
    elif corners == None:
        corners = [[-65.0, 58.0], [20.0, 81.0]]
        
    myEDC = EDCbasemap(resolution=resolution, corners=corners, gdist=gdist, ax=ax,
                       colors=None, coastlw=coastlw, plot_continents=False)
    
    
    # Open geometry file, Mapinfo format
    g = ogr.Open (shpFN)
    
    # Get the number of layers in the current file
    L_count = g.GetLayerCount()

    # Get the first layer
    L = g.GetLayer(0)

    # Get spatial reference of map
    mapSR = osr.SpatialReference()
    mapSR.ImportFromProj4(myEDC.map.proj4string)
    
    # Get spatial reference of layer
    layerSR = L.GetSpatialRef()
    
    # Create transformations to and from map coordinates:
    #coordTrans2map = osr.CoordinateTransformation(layerSR, mapSR)
    #coordTrans2file = osr.CoordinateTransformation(mapSR, layerSR)
    
    # Make sure we read from the beginning
    L.ResetReading()

    # Get the first feature
    F = L.GetNextFeature()    
    
    xl = ax.get_xlim()
    yl = ax.get_ylim()
   
    i = 0
    # Loop over all features in the present layer
    while F is not None:
        # Get number of descriptive fields in feature
        field_count = F.GetFieldCount()
        
        
        num_code = F.GetFieldAsInteger('NUM_CODE')
        extent = F.GetFieldAsString('EXTENT')
        
        #print "{0:05d}    num: {1}    extent: {2}".format(i, num_code, extent)
        
        if num_code == 21:
            # Glaciers
            kwargs = colors['glaciers']
        elif num_code == 22:
            # Relict permafrost
            kwargs = colors['other']
        elif num_code == 23:
            # Inland lakes
            kwargs = colors['lakes']
        elif num_code == 24:
            # Oceans / inland seas
            kwargs = colors['oceans']        
        elif num_code == 25:    
            # Land
            kwargs = colors['continents']
        elif num_code in np.arange(1,21):
            kwargs = colors['EXTENT'][extent]
        else:
            kwargs = colors['other']
        
#        # Get Geometry connected to the feature
#        G = F.GetGeometryRef()
#        
#        G.TransformTo(mapSR)
#        
#        plotGeometry(G, basemap=myEDC.map, ax=ax, noconv=True, as_patch=True,**kwargs)
#        
        plotFeature(F, mapSRS=mapSR, basemap=myEDC.map, ax=ax, noconv=True, 
            as_patch=True, **kwargs)
        # Remove feature from memory
        F.Destroy()
        
        # Get the next feature in the file
        F = L.GetNextFeature()
        
        i+=1

    # Remove datasource from memory
    g.Destroy()
    
    ax.set_xlim(*xl)
    ax.set_ylim(*yl)
##    pylab.xlabel("Longitude")
##    pylab.ylabel("Latitude")
##    pylab.grid(True)
##    pylab.title("Test")
##    pylab.show()

    return myEDC



#def relimPatchPlot(ax):
#    ph = ax.findobj(match=mpl.patches.PathPatch)
#    
#    vert = plt.get(ph[0],'path').vertices
#    
#    if len(ph) > 1: 
#        for h in ph[1:]:
#            vert = np.vstack((vert,plt.get(h,'path').vertices))
#    
#    xl = [np.min(vert[:,0]), np.max(vert[:,0])]
#    yl = [np.min(vert[:,1]), np.max(vert[:,1])]
#    
#    ax.set_xlim(*xl)
#    ax.set_ylim(*yl)


#def plotPatchOutlines(ax):
#    ph = ax.findobj(match=mpl.patches.PathPatch)
#    
#    print "{0} pathpatch objects found...".format(len(ph))
#    fh = plt.figure()
#    ax = plt.axes()
#       
#    for h in ph:
#        vert = plt.get(h,'verts')
#        ax.plot(vert[:,0],vert[:,1], '-k')
    
#    relimPatchPlot(ax)


#def plotWkt(wkt):
#    fh = plt.figure()
#    ax = plt.axes()
#    objs = wkt.split('),(')
#    objs[0] = objs[0].split('(')[-1]
#    objs[-1] = objs[-1].split(')')[0]
#    
#    for o in objs:
#        coords = np.array([])
#        verts = o.split(',')
#        for v in verts:
#            coords = np.concatenate((coords, np.fromstring(v, sep=' ', count=2)))
#        coords = np.reshape(coords,(-1,2))
#        ax.plot(coords[:,0],coords[:,1],'-k')


##def combineNeighbourFeatures(inFN, outFN, filterAttributes=None, 
##                             filterValues=None, driverName=None,
##                             clearAttributes=None):
#def combineNeighbourFeatures_old(inFN, outFN, filterStr=None, 
#                             driverName=None, clearAttributes=None):
#    
#    global stop_on_FID1
#    global stop_on_FID2
#    
#    stop_on_FID1 = None
#    stop_on_FID2 = [4528]
#    
#    if driverName == None:
#        inDS = ogr.Open (inFN)
#        if inDS is None:
#            print 'Could not open ' + inFN
#            sys.exit(1)
#        inLayer = inDS.GetLayer()
#        driver = inDS.GetDriver()
#    else:   
#        # get the shapefile driver
#        driver = ogr.GetDriverByName(driverName)
#        # open the input data source and get the layer
#        inDS = driver.Open(inFN, 0)
#        if inDS is None:
#            print 'Could not open ' + inFN
#            sys.exit(1)
#        inLayer = inDS.GetLayer()
#
#    # get input SpatialReference
#    inSpatialRef = inLayer.GetSpatialRef()
#
#    # create a new data source and layer
#    if os.path.exists(outFN):
#        driver.DeleteDataSource(outFN)
#    outDS = driver.CreateDataSource(outFN)
#    if outDS is None:
#        print 'Could not create ' + outFN
#        sys.exit(1)
#    outLayer = outDS.CreateLayer(os.path.basename(outFN)[:-4],
#                                 geom_type=inLayer.GetLayerDefn().GetGeomType())
#
#    # copy the fields from the input layer to the output layer
#    copyFields(inLayer, outLayer)
#
#    # get the FeatureDefn for the output shapefile
#    featureDefn = outLayer.GetLayerDefn()
#
#    # Filter out only Features with specified Attribute value
##    if filterAttribute != None:
##        inLayer.SetAttributeFilter("{0} = '{1}'".format(filterAttribute,
##                                                        filterValue))
#
#    if filterStr != None:
#        inLayer.SetAttributeFilter(filterStr)
#
#    featCount = inLayer.GetFeatureCount()
#
#    # get IDs of all features in (filtered) layer
#    inFIDs = [inLayer.GetNextFeature().GetFID() for k in np.arange(featCount)]
#    
#    while len(inFIDs) >= 1:
#        print "Processing Feature {0}".format(inFIDs[0])
#        
#        # get first geometry
#        inFeature = inLayer.GetFeature(inFIDs.pop(0))
#    
#        # get the input geometry
#        joinGeom = inFeature.GetGeometryRef()
#        
#        gCount = joinGeom.GetGeometryCount()
##        if gCount > 1:
##            print "multi geometry: {0} elements".format(gCount)
##            pdb.set_trace()
#        
#        # create a new feature
#        outFeature = ogr.Feature(featureDefn)
#        
#        # copy the attributes
#        copyAttributes(inFeature, outFeature)
#        
#        # loop through the input features        
#        outFIDs = []
#        while len(inFIDs)>=1:
#            #  get the next input feature
#            thisFID = inFIDs.pop(0)
#            addFeature = inLayer.GetFeature(thisFID)
#            
#            # Get geometry of the present feature
#            geom = addFeature.GetGeometryRef()
#            
#            if stop_on_FID1 != None and thisFID in stop_on_FID1:
#                print "Geometry {0} is to be tested...!".format(thisFID)
#                pdb.set_trace()
#        
#            if outLayer.GetFeatureCount == 806 and thisFID == 4528:
#                print "We're going to join 3492 and 4528..."
#                pdb.set_trace()
#            
##            gCount = geom.GetGeometryCount()
##            if gCount > 1:
##                print "multi geometry: {0} elements".format(gCount)
##                ax = plotGeometry(joinGeom, as_patch=True, edgecolor='g', facecolor='None')
##                ax = plotGeometry(geom, ax=ax, as_patch=True, edgecolor='b', facecolor='None')
##                pylab.show()
##                pdb.set_trace()
#            
##            if joinGeom.Disjoint(geom):
##                # if disjoint, store the FID and continue to next feature    
##                outFIDs.append(thisFID)
##            else:
#            if joinGeom.Overlaps(geom):
##                 or joinGeom.Touches(geom) or \
##                joinGeom.Contains(geom) or joinGeom.Equal(geom) or \
##                joinGeom.Within(geom):
#                    
#                if stop_on_FID2 != None and thisFID in stop_on_FID2:
#                    print "Geometry {0} is to be joined...!".format(thisFID)
#                    pdb.set_trace()
#                    
#                # else update joinGeom to include the union of the features
#                print 'Geometry joined with FID {0}'.format(thisFID)
#                joinGeom.Union(geom)
#                if not joinGeom.Intersect(geom):
#                    print "Problem with union!!!"
#                    pdb.set_trace()
#                    outFIDs.append(thisFID)
#            else:
#                # if geometries are far appart, append second FID to new list    
#                outFIDs.append(thisFID)
#            
#            # destroy the feature and get the next input feature
#            addFeature.Destroy()
#        
#        print 'Adding joined Feature to output layer'
#        
#        # set the geometry and attribute
#        outFeature.SetGeometry(joinGeom)
#    
#        # add the feature to the shapefile
#        outLayer.CreateFeature(outFeature)
#        
#        print '{0} Features in output layer'.format(outLayer.GetFeatureCount())
#        
#        # destroy the output feature
#        outFeature.Destroy()
#        inFeature.Destroy() 
#        
#        # prepare for next iteration    
#        inFIDs = outFIDs
#
#    pdb.set_trace()
#
#    # close the shapefiles
#    inDS.Destroy()
#    outDS.Destroy()
#
#    # create a *.prj file
#    inSpatialRef.MorphToESRI()
#    # compose prj filename
#    prjFN = outFN[::-1].partition('.')[2][::-1]+'.prj'
#    file = open(prjFN,'w')
#    file.write(inSpatialRef.ExportToWkt())
#    file.close()