# -*- coding:utf-8 -*-
"""
Created on 25/02/2010

@author: tin
"""

import pdb

import numpy as np

from mpl_toolkits.basemap import Basemap
from matplotlib import transforms as mtransforms
from matplotlib.ticker import FuncFormatter, MaxNLocator, AutoLocator
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import pyproj
  
stop_on_FID1 = None
stop_on_FID2 = None

class MyMap:
    def __init__(self, ax=None):
        if ax == None:
            self.fh = plt.figure()
            plt.clf
            self.ax = self.fh.gca()
        else:
            self.fh = ax.figure
            self.ax = ax
        self.fh.canvas.mpl_connect('pick_event', self.onpick)
        self.fh.canvas.mpl_connect('key_press_event', self.on_key_press)
        return

    def onpick(self, event):
        print('pypf.mapping.maps:onGeometryPick:')
        if isinstance(event.artist, plt.Line2D):
            thisline = event.artist
            
            xdata = thisline.get_xdata()
            #ydata = thisline.get_ydata()
            ind = event.ind
            
            try:
                self.lastClick = thisline.coordsID[ind].flatten()
            except AttributeError:
                print('  Line has no coordsID attribute...')
            else:
                print('  onpick1 line: ', self.lastClick)
            
            try:
                thisline.names
            except AttributeError:
                print('  Line has no names attribute...')
            else:
                if len(thisline.names) == 1:
                    print('  Name: ', thisline.names)
                elif len(thisline.names) == len(xdata):
                    print('  Name: ', np.take(thisline.names, ind))
            
            #try:
            #    thisline.tag
            #    
            #except AttributeError:
            #    print 'Line has no tag attribute...'
            #else:
            #    # if len(self.lastClick)
            #    if thisline.tag == "CLIMgrid":
            #        #pd.plotData(coordsID = self.lastClick)
            #        pass
                
        elif isinstance(event.artist, plt.Rectangle):
            patch = event.artist
            vertsxy = patch.get_verts()
            lons, lats = self.map(vertsxy[0], vertsxy[1], inverse=True)
            print('  onpick1 patch :', list(zip(lons, lats)))
        elif isinstance(event.artist, plt.Text):
            text = event.artist
            print('  onpick1 text: ', text.get_text())
        
        return False
    
    def on_key_press(self, event):
        " Allow for rotation etc of text elements"
        print('pypf:mapping:maps:on_key_press:')
        
        if event.key == 't' and len(self.ax.texts) > 0:
            tstr = ['{']
            for t in self.ax.texts:
                txt = t.get_text()
                rot = t.get_rotation()
                pos = t.get_position()
                tstr.append("    '{0}': {{'rotation':{1:3.0f}, 'x':{2:.2f}, 'y':{3:.2f}}},".format(txt, rot, pos[0], pos[1]))
            tstr.append('}')
            
            print('\n'.join(tstr))

        return False
   
    def plotPoints(self, lons, lats, style='bo', ms=2, picker=True, names=[], tag=[], **kwargs):
        """
        Function to convert lat and lon to map x and y and plot them.
        """
        
        x, y = self.map(lons, lats)
        # Consider adding optional labels to points!S
        
        # compute the native map projection coordinates for cities.
        xl = self.ax.set_xlim()
        yl = self.ax.set_ylim()
        
        pid = np.logical_and(
           np.logical_and(x >= xl[0], x <= xl[1]),
           np.logical_and(y >= yl[0], y <= yl[1]))
        try:
            y[0]   # Test to see if this is an array!
            y = y.compress(pid.flat)
            x = x.compress(pid.flat)
        except:
            pass
        
        #lons = lons.compress(id.flat)
        #lats = lats.compress(id.flat)      
        
        # Trick to get array of indices!
        coordsID = np.argwhere(pid)
        
        #pdb.set_trace()
        
        if len(names) == len(coordsID):
            #for nid, obj in enumerate(names):
            for nid in np.arange(len(names)):
                if pid[nid] == False:
                    del names[nid]
        elif names != []:
            names = names[0]
            
        # plot filled circles at the locations of the cities.
        lh = self.map.plot(x, y, style, ms=ms, ax=self.ax, picker=True,**kwargs)
        lh[0].coordsID = coordsID
        try:
            lh[0].names = names
        except:
            lh[0].names = []
        
        try:
            lh[0].tag = tag
        except:
            lh[0].tag = []            
    
        plt.show(block=False)
    
    def xlim(self, xl):
        try:
            self.ax.set_xlim(xl)
        except:
            return False
        return True
    
    def ylim(self, yl):
        try:
            self.ax.set_ylim(yl)
        except:
            return False
        return True
    
    def set_limmits(self, xl, yl):
        try:
            self.ax.set_xlim(xl)            
            self.ax.set_ylim(yl)
        except:
            return False
        return True
      

class EDCbasemap(MyMap):
    def __init__(self, resolution='l', corners=None, gdist=10., ax=None, 
                 coastlw=1.0, colors=None, plot_continents=True):
        """Initialization function, which creates the map figure
        """
        MyMap.__init__(self, ax) #@IndentOk
        # setup equidistant conic basemap.
        # lat_1 is first standard parallel.
        # lat_2 is second standard parallel.
        # lon_0,lat_0 is central point.
        # resolution = 'l' for low-resolution coastlines.
        
        if colors == None:
            colors = dict( 
                coastlines = '0.3',
                oceans = 'w',
                continents = 'w',
                lakes = 'w',
                )
        
        if colors == 'grey':
            colors = dict(
                coastlines = '0.3', 
                oceans = '0.9',
                continents = '0.4',
                lakes = '0.9',
                )
        
        if colors == 'nice':        
            colors = dict(
                  continents = '0.4',
                  oceans = "#DCE6FF",
                  lakes = "#DCE6FF",
                  coastlines = "k")
            
        if corners != None:
            llcrnrlon = corners[0][0]
            llcrnrlat = corners[0][1]
            urcrnrlon = corners[1][0]
            urcrnrlat = corners[1][1]
            self.map = Basemap(\
                  llcrnrlon=llcrnrlon, llcrnrlat=llcrnrlat, \
                  urcrnrlon=urcrnrlon, urcrnrlat=urcrnrlat, \
                  resolution=resolution, projection='eqdc', \
                  lat_1=15., lat_2=75, lat_0=72, lon_0= -43., ax=self.ax)
        else:      
            self.map = Basemap(width=2500000, height=3000000,
                  resolution=resolution, projection='eqdc', \
                  lat_1=15., lat_2=75, lat_0=72, lon_0= -43., ax=self.ax)
        if plot_continents:
            self.map.drawcoastlines(color=colors['coastlines'],linewidth=coastlw)
            self.map.fillcontinents(color=colors['continents'], lake_color=colors['lakes'])   #coral, aqua
            
        # draw parallels and meridians.
        if gdist != None:
            self.map.drawparallels(np.arange(0., 81., gdist), labels=[1, 1, 0, 0], zorder=50)
            self.map.drawmeridians(np.arange(-180., 181., gdist), labels=[0, 0, 0, 1], zorder=50)
        self.map.drawmapboundary(fill_color=colors['oceans'], zorder=50)   #aqua 
        #self.ax = self.map.ax
        self.ax.set_title("Equidistant Conic Projection")
        plt.show(block=False)



class LAEAbasemap(MyMap):
    def __init__(self, ax=None):
        """
        Initialization function, which creates the map figure
        """
        MyMap.__init__(self, ax)
        self.fh = plt.figure()
        plt.clf
        # setup lambert azimuthal equal area basemap.
        # lat_ts is latitude of true scale.
        # lon_0,lat_0 is central point.
        self.map = Basemap(width=2500000, height=3000000,
                    resolution='h', projection='laea', \
                    lat_ts=72, lat_0=72, lon_0= -43.)
        self.map.drawcoastlines()
        self.map.fillcontinents(color='coral', lake_color='aqua')
        # draw parallels and meridians.
        self.map.drawparallels(np.arange(-80., 81., 10.), labels=[1, 1, 0, 1])
        self.map.drawmeridians(np.arange(-180., 181., 10.), labels=[1, 1, 0, 1])
        self.map.drawmapboundary(fill_color='aqua') 
        # draw tissot's indicatrix to show distortion.
        #self.ax = plt.gca()
        plt.title("Lambert Azimuthal Equal Area Projection")
        plt.show(block=False)
      

     
class UTMBasemap(Basemap):
    #def __call__(self, *args, **kwargs):
    #    x,y = Basemap.__call__(self, *args, **kwargs)
    #    # add coordinate offsets (Basemap coordinate system has x=0,y=0 at lower
    #    # left corner of domain)
    #    x = x-self.projparams['x_0']+5.e5
    #    y = y-self.projparams['y_0']
    #    return x,y
    
    def convert_UTM_to_axes(self,x,y):
        x = x+self.projparams['x_0']-5.e5
        y = y+self.projparams['y_0']
        return x,y
        

     
class UTMWGS84basemap(MyMap):
    lon_0 = None
    lat_0 = None

    def __init__(self, resolution='i', utmcorners=None, corners=None, gdist=10., ax=None, 
                 coastlw=1.0, colors=None, plot_continents=True, UTMGridZone=22):
        """Initialization function, which creates the map figure
        """
        MyMap.__init__(self, ax) #@IndentOk
        
        if colors == None:
            colors = dict( 
                coastlines = '0.3',
                oceans = 'w',
                continents = 'w',
                lakes = 'w',
                )
        
        if colors == 'grey':
            colors = dict(
                coastlines = '0.3', 
                oceans = '0.9',
                continents = '0.4',
                lakes = '0.9',
                )
        
        if colors == 'nice':        
            colors = dict(
                  continents = '0.4',
                  oceans = "#DCE6FF",
                  lakes = "#DCE6FF",
                  coastlines = "k")
        
        kwargs = self.define_basemap_params(UTMGridZone)

        if corners != None:
            llcrnrlon = corners[0][0]
            llcrnrlat = corners[0][1]
            urcrnrlon = corners[1][0]
            urcrnrlat = corners[1][1]
        else:
            llcrnrlon = self.get_zcm(UTMGridZone)-20
            llcrnrlat = 64.
            urcrnrlon = self.get_zcm(UTMGridZone)+20
            urcrnrlat = 72
        
        if utmcorners != None:
            # proj4 definition of UTM zone 17.
            # coordinates agree to the 3rd decimal place (less than 1mm)
            # when compared to result from
            # http://home.hiwaay.net/~taylorc/toolbox/geography/geoutm.html
            p1 = pyproj.Proj(proj='utm',zone=UTMGridZone)
            p2 = pyproj.Proj(proj='latlong',datum='WGS84')
            lon1, lat1 = pyproj.transform(p1, p2, utmcorners[0][0], utmcorners[0][1])
            lon2, lat2 = pyproj.transform(p1, p2, utmcorners[1][0], utmcorners[1][1])
            llcrnrlon = lon1
            llcrnrlat = lat1
            urcrnrlon = lon2
            urcrnrlat = lat2
            
        self.map = UTMBasemap(\
              llcrnrlon=llcrnrlon, llcrnrlat=llcrnrlat, \
              urcrnrlon=urcrnrlon, urcrnrlat=urcrnrlat, \
              resolution=resolution, ax=self.ax, suppress_ticks=False, **kwargs)
              
        if plot_continents:
            self.map.drawcoastlines(color=colors['coastlines'],linewidth=coastlw)
            self.map.fillcontinents(color=colors['continents'], lake_color=colors['lakes'])   #coral, aqua
            
        self.ax.set_title("UTM Projection")
        #self.ax.yaxis.set_major_locator(UTMYAutoLocator(self.map))
        #self.ax.xaxis.set_major_locator(UTMXAutoLocator(self.map))
        
        self.ax.yaxis.set_major_formatter(FuncFormatter(self.UTMyformatter))
        self.ax.xaxis.set_major_formatter(FuncFormatter(self.UTMxformatter))
        
        self.map.set_axes_limits(ax=self.ax)
        
        plt.show(block=False)
      
    def get_zcm(self, UTMGridZone):
        """Get the central meridian of the UTM Grid Zone"""
        return 3 + 6 * (UTMGridZone - 1) - 180
        
    def get_utmzone(self, zcm):
        """Get the UTM Grid Zone from the central meridian"""
        return 1 + int((zcm + 180)/6)
        
    def define_basemap_params(self, UTMGridZone):
        """return a dictionary of projection parameters that
        can be passed directly to Basemap.
        """
        lon_0 = self.get_zcm(UTMGridZone)
        lat_0 = 0.
        
        # use WGS84 ellipsoid, with scaling factor of 0.9996 at
        # central meridian).  
        return dict(projection='tmerc',lon_0=lon_0,lat_0=lat_0,
            k_0=0.9996,rsphere=(6378137.00,6356752.314245179))
    
    def UTMyformatter(self, y, pos):
        'make UTM ticks'
        return '{0:.0f}'.format(y-self.map.projparams['y_0'])
        #return '{0}'.format(y)
        
    def UTMxformatter(self, x, pos):
        'make UTM ticks'
        #print x, pos
        return '{0:.0f}'.format(x-self.map.projparams['x_0']+5.e5)
        #return '{0}'.format(x)

class UTMAutoLocator(MaxNLocator):
    def __init__(self,map):
        MaxNLocator.__init__(self, nbins=9, steps=[1, 2, 5, 10], integer=True)
        self.x_0 = map.projparams['x_0']
        self.y_0 = map.projparams['y_0']
    
    def __call__(self):
        vmin, vmax = self.axis.get_view_interval()
        vmin = self.trans(vmin)
        vmax = self.trans(vmax)
        vmin, vmax = mtransforms.nonsingular(vmin, vmax, expander = 0.05)
        locs = self.bin_boundaries(vmin, vmax)
        #print 'locs=', locs
        prune = self._prune
        if prune=='lower':
            locs = locs[1:]
        elif prune=='upper':
            locs = locs[:-1]
        elif prune=='both':
            locs = locs[1:-1]
        locs = self.itrans(locs)
        #print 'locs=', locs
        return self.raise_if_exceeds(locs)        
    
    def trans(self):
        NotImplementedError('Derived must override')


class UTMXAutoLocator(UTMAutoLocator):
    def trans(self, val):
        #print val-self.x_0+5.e5
        return val-self.x_0+5.e5
    def itrans(self, val):
        return val+self.x_0-5.e5
        
class UTMYAutoLocator(UTMAutoLocator):
    def trans(self, val):
        #print val-self.y_0
        return val-self.y_0        
    def itrans(self, val):
        return val+self.y_0