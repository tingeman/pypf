'''
Created on 11/02/2013

@author: thin
'''

from pkg_resources import resource_filename

import numpy as np
import os.path
import copy
import pdb

import matplotlib.pyplot as plt
import pylab

from pypf.mapping.maps import EDCbasemap
from pypf.mapping.plot import plotShpFile, plotGrlCities


pkg_name = 'pypf'
brown1997_relative_path = os.path.join('resources','brown1997')

basemap_colors = dict(
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


plotspecs = [('NUM_CODE', 'integer', 
                        {25: dict(edgecolor='k',              # Continents
                                      facecolor='0.4',
                                      linewidth=0.5,
                                      zorder=16),
                        24:  dict(edgecolor='k',              # Oceans
                                      facecolor="#DCE6FF",
                                      linewidth=0.5,
                                      zorder=15),
                        23:  dict(edgecolor='k',              # Lakes
                                      facecolor="#DCE6FF",
                                      linewidth=0.5,
                                      zorder=18),
                        21:  dict(edgecolor='none',           # Glaciers
                                      facecolor='w',
                                      linewidth=0.5,
                                      zorder=21),
                        22:  dict(edgecolor='k',              # other
                                      facecolor='none',
                                      linewidth=0.5,
                                      zorder=25)                                                          
                        }),
              ('EXTENT', 'string', {                                                
                       'C': dict(edgecolor='k',
                                facecolor='b',
                                linewidth=0.5,
                                zorder=20),
                       'D': dict(edgecolor='k',
                                facecolor="#FFAA00",
                                #facecolor="#557FFF",
                                #facecolor="#B3CFFB",
                                linewidth=0.5,
                                zorder=20),
                       'S': dict(edgecolor='k',
                                #facecolor="#FFAA00",
                                facecolor="#00E600",
                                #facecolor="#D4AAFF",
                                #facecolor="#B3CFFB",
                                linewidth=0.5,
                                zorder=20),
                       'I': dict(edgecolor='k',
                                facecolor="#00E600",
                                #facecolor="#FFD4FF",
                                linewidth=0.5,
                                zorder=20)
                        })]


grl_cities = {
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
                               verticalalignment='top', zorder=50),
         'Maniitsoq':  dict(coords=np.array((-52.898140, 65.414892)),
                               dx=0, dy=0,
                               fontsize=12, fontweight='bold',
                               horizontalalignment='right', 
                               verticalalignment='top', zorder=50)                      
                       }

                       
def plot_PF_zones(basemap, plotspecs):
    shpFN = resource_filename(pkg_name, os.path.join(brown1997_relative_path, 'permaice.shp')) 
    basemap = plotShpFile(shpFN, basemap=basemap, plotspecs=plotspecs)   
    
    return basemap

                       
def map_of_greenland(save=False, resolution='l', gdist=10, cities=False, colors='grey'):
    """ Plots map of greenland. 
        
    """
    
    if not colors:
        colors = dict(continents = '0.4',
                  oceans = "#DCE6FF",
                  lakes = "#DCE6FF",
                  coastlines = "k")
                  
    plotspecs = [('NUM_CODE', 'integer', 
                        {21:  dict(edgecolor='none',           # Glaciers
                                      facecolor='w',
                                      linewidth=0.5,
                                      zorder=21)                                                          
                        }),]
    
    
    shpFN = resource_filename(pkg_name, os.path.join(brown1997_relative_path, 'permaice.shp')) 
    
    corners = [[-65.0, 58.0], [20.0, 81.0]]
        
    myEDC = EDCbasemap(resolution=resolution, corners=corners, gdist=gdist,
                       plot_continents=True, colors=colors)
    
    pdb.set_trace()
    myEDC = plotShpFile(shpFN, basemap=myEDC, plotspecs=plotspecs)   
    
    if cities:
        plotGrlCities(cities=grl_cities, basemap=myEDC)
    
    myEDC.ax.set_title('')
    
    if save == True:
        pylab.savefig('map_of_greenland.png',dpi=300,format='png',
                      bbox_inches='tight',pad_inches=0.5)
        pylab.savefig('map_of_greenland.svg',dpi=300,format='svg')
    
    return myEDC
    

def PF_map_of_greenland(save=False, resolution='l', gdist=10, cities=True):
    """ Plots map of greenland with permafrost zones indicated. 
    The plot has major cities indicated as well. 
    
    """
    
    shpFN = resource_filename(pkg_name, os.path.join(brown1997_relative_path, 'permaice.shp')) 
    
    corners = [[-65.0, 58.0], [20.0, 81.0]]
        
    myEDC = EDCbasemap(resolution=resolution, corners=corners, gdist=gdist,
                       plot_continents=False, colors=None)
    
    myEDC = plotShpFile(shpFN, basemap=myEDC, plotspecs=plotspecs)   
    
    if cities:
        plotGrlCities(cities=grl_cities, basemap=myEDC)
    
    myEDC.ax.set_title('')
    
    if save == True:
        pylab.savefig('map_of_greenland.png',dpi=300,format='png',
                      bbox_inches='tight',pad_inches=0.5)
        pylab.savefig('map_of_greenland.svg',dpi=300,format='svg')
    
    return myEDC


def PF_map_of_greenland_plain(save=False, resolution='l', gdist=None):
    """ Plots map of greenland with permafrost zones indicated. 
    The plot does not have cities indicated, and the background is white. 
    
    """
    
    shpFN = resource_filename(pkg_name, os.path.join(brown1997_relative_path, 'permaice.shp')) 
    
    corners = [[-65.0, 58.0], [20.0, 81.0]]
        
    myEDC = EDCbasemap(resolution=resolution, corners=corners, gdist=gdist,
                       plot_continents=False, colors=None)
    
    myplotspecs = copy.deepcopy(plotspecs)
    myplotspecs[0][2][24]['facecolor'] = 'w'
    
    myEDC = plotShpFile(shpFN, basemap=myEDC, plotspecs=myplotspecs)   

    #plotGrlCities(cities=grl_cities, basemap=myEDC)
    
    myEDC.ax.set_title('')
    
    if save == True:
        pylab.savefig('map_of_greenland.png',dpi=300,format='png',
                      bbox_inches='tight',pad_inches=0.5)
        pylab.savefig('map_of_greenland.svg',dpi=300,format='svg')

    return myEDC


def PF_map_of_Westcoast(save=False, resolution='l', gdist=5, names=False):

    shpFN = resource_filename(pkg_name, os.path.join(brown1997_relative_path, 'permaice.shp'))
    
    corners = [[-55.0, 63.0], [-46.0, 71.0]]
    
    myEDC = EDCbasemap(resolution=resolution, corners=corners, gdist=gdist,
                       plot_continents=False, colors=None)
        
    myEDC = plotShpFile(shpFN, basemap=myEDC, plotspecs=plotspecs)  

    plotGrlCities(cities=grl_cities, basemap=myEDC, names=names)
    
    myEDC.ax.set_title('')
    
    if save == True:
        pylab.savefig('map_of_westcoast.png',dpi=300,format='png',
                      bbox_inches='tight',pad_inches=0.5)
        pylab.savefig('map_of_westcoast.svg',dpi=300,format='svg')

    return myEDC


#def PF_map_of_greenland_no_cities(save=False):
#    
#    colors = dict(EXTENT = dict(\
#                       C = dict(edgecolor='k',
#                                facecolor='b',
#                                linewidth=0.5,
#                                zorder=20),
#                       D = dict(edgecolor='k',
#                                facecolor="#FFAA00",
#                                linewidth=0.5,
#                                zorder=20),
#                       S = dict(edgecolor='k',
#                                facecolor="#00E600",
#                                linewidth=0.5,
#                                zorder=20),
#                       I = dict(edgecolor='k',
#                                facecolor="#00E600",
#                                linewidth=0.5,
#                                zorder=20)
#                       ),
#                  continents =  dict(edgecolor='k',
#                                facecolor='0.4',
#                                linewidth=0.5,
#                                zorder=16),
#                  oceans =  dict(edgecolor='k',
#                                facecolor="#DCE6FF",
#                                linewidth=0.5,
#                                zorder=15),
#                  lakes =  dict(edgecolor='k',
#                                facecolor="#DCE6FF",
#                                linewidth=0.5,
#                                zorder=18),
#                  glaciers =  dict(edgecolor='none',
#                                facecolor='w',
#                                linewidth=0.5,
#                                zorder=21),
#                  other =   dict(edgecolor='y',
#                                facecolor='y',
#                                linewidth=1,
#                                zorder=30)             
#                        )
#
#    cities = {}
#    
#    shpFN = os.path.join('D:\\','Thomas','Maps_and_Orthos',
#                    'Circum-Arctic Map of Permafrost','permaice.shp')
#    
#    fh = plt.figure(figsize=(11,8))
#    ax = fh.add_subplot(111)
#    
#    myEDC = mapping.plotShpFile(shpFN, ax=ax, corners='Greenland', 
#                                colors=colors, resolution='l')   
#
#    mapping.plotGrlCities(cities=cities, myEDC=myEDC)
#    
#    myEDC.ax.set_title('')
#    
#    if save == True:
#        pylab.savefig('map_of_greenland_no_cities.png',dpi=300,format='png',
#                      bbox_inches='tight',pad_inches=0.5)
#        pylab.savefig('map_of_greenland_no_cities.svg',dpi=300,format='svg')
#
#def PF_map_of_greenland_plain(save=False):
#    
#    colors = dict(EXTENT = dict(\
#                       C = dict(edgecolor='k',
#                                facecolor='b',
#                                linewidth=0.5,
#                                zorder=20),
#                       D = dict(edgecolor='k',
#                                facecolor="#FFAA00",
#                                linewidth=0.5,
#                                zorder=20),
#                       S = dict(edgecolor='k',
#                                facecolor="#00E600",
#                                linewidth=0.5,
#                                zorder=20),
#                       I = dict(edgecolor='k',
#                                facecolor="#00E600",
#                                linewidth=0.5,
#                                zorder=20)
#                       ),
#                  continents =  dict(edgecolor='k',
#                                facecolor='0.4',
#                                linewidth=0.5,
#                                zorder=16),
#                  oceans =  dict(edgecolor='k',
#                                facecolor="#FFFFFF",      #"#DCE6FF",
#                                linewidth=0.5,
#                                zorder=15),
#                  lakes =  dict(edgecolor='k',
#                                facecolor="#DCE6FF",
#                                linewidth=0.5,
#                                zorder=18),
#                  glaciers =  dict(edgecolor='none',
#                                facecolor='w',
#                                linewidth=0.5,
#                                zorder=21),
#                  other =   dict(edgecolor='y',
#                                facecolor='y',
#                                linewidth=1,
#                                zorder=30)             
#                        )
#
#    
#    shpFN = os.path.join('D:\\','Thomas','Maps_and_Orthos',
#                    'Circum-Arctic Map of Permafrost','permaice.shp')
#    
#    fh = plt.figure(figsize=(11,8))
#    ax = fh.add_subplot(111)
#    
#    myEDC = mapping.plotShpFile(shpFN, ax=ax, corners='Greenland', 
#                                colors=colors, resolution='l', gdist=None)   
#
#    myEDC.ax.set_title('')
#    
#    if save == True:
#        pylab.savefig('map_of_greenland_plain.png',dpi=300,format='png',
#                      bbox_inches='tight',pad_inches=0.5)
#        #pylab.savefig('map_of_greenland_plain.svg',dpi=300,format='svg')
#        
#    
#def PF_map_of_Westcoast(save=False):
#    
#    colors = dict(EXTENT = dict(\
#                       C = dict(edgecolor='k',
#                                facecolor='b',
#                                linewidth=0.5,
#                                zorder=20),
#                       D = dict(edgecolor='k',
#                                facecolor="#FFAA00",
#                                linewidth=0.5,
#                                zorder=20),
#                       S = dict(edgecolor='k',
#                                facecolor="#00E600",
#                                linewidth=0.5,
#                                zorder=20),
#                       I = dict(edgecolor='k',
#                                facecolor="#00E600",
#                                linewidth=0.5,
#                                zorder=20)
#                       ),
#                  continents =  dict(edgecolor='k',
#                                facecolor='0.4',
#                                linewidth=0.5,
#                                zorder=16),
#                  oceans =  dict(edgecolor='k',
#                                facecolor="#DCE6FF",
#                                linewidth=0.5,
#                                zorder=15),
#                  lakes =  dict(edgecolor='k',
#                                facecolor="#DCE6FF",
#                                linewidth=0.5,
#                                zorder=18),
#                  glaciers =  dict(edgecolor='none',
#                                facecolor='w',
#                                linewidth=0.5,
#                                zorder=21),
#                  other =   dict(edgecolor='y',
#                                facecolor='y',
#                                linewidth=1,
#                                zorder=30)             
#                        )
#
#    cities = {
#         'Ilulissat':     dict(coords=np.array((-51.100000, 69.216667)),
#                               dx=-0.2, dy=-0.1,
#                               fontsize=10, fontweight='bold',
#                               horizontalalignment='right', 
#                               verticalalignment='top', zorder=50),
#         'Sisimiut':      dict(coords=np.array((-53.669284, 66.933628)),
#                               dx=-0.2, dy=0,
#                               fontsize=10, fontweight='bold',
#                               horizontalalignment='right', 
#                               verticalalignment='top', zorder=50),
#         'Kangerlussuaq': dict(coords=np.array((-50.709167, 67.010556)),
#                               dx=1.0, dy=0,
#                               fontsize=10, fontweight='bold',
#                               horizontalalignment='left', 
#                               verticalalignment='bottom', zorder=50),         
#         'Nuuk':          dict(coords=np.array((-51.742632, 64.170284)),
#                               dx=-0.3, dy=0,
#                               fontsize=10, fontweight='bold',
#                               horizontalalignment='right', 
#                               verticalalignment='top', zorder=50)}
#
#    fh = plt.figure(figsize=(11,8))
#    ax = fh.add_subplot(111)
#
#    shpFN = os.path.join('D:\\','Thomas','Maps_and_Orthos',
#                    'Circum-Arctic Map of Permafrost','permaice.shp')
#    
#    myEDC = mapping.plotShpFile(shpFN, ax=ax, corners='westcoast', 
#                                colors=colors, resolution='l', gdist=5)   
#
#    mapping.plotGrlCities(cities=cities, myEDC=myEDC)
#    
#    myEDC.ax.set_title('')
#    
#    if save == True:
#        pylab.savefig('map_of_westcoast.png',dpi=300,format='png',
#                      bbox_inches='tight',pad_inches=0.5)
#        pylab.savefig('map_of_westcoast.svg',dpi=300,format='svg')
