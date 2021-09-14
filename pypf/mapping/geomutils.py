'''
Created on 11/02/2013

@author: thin
'''

import pdb


import sys
import os

import numpy as np

import osgeo.ogr as ogr
import osgeo.osr as osr



# function to copy fields (not the data) from one layer to another
# parameters:
#   fromLayer: layer object that contains the fields to copy
#   toLayer: layer object to copy the fields into
def copyFields(fromLayer, toLayer):
    featureDefn = fromLayer.GetLayerDefn() 
    for i in range(featureDefn.GetFieldCount()):
        toLayer.CreateField(featureDefn.GetFieldDefn(i))


# function to copy attributes from one feature to another
# (this assumes the features have the same attribute fields!)
# parameters:
#   fromFeature: feature object that contains the data to copy
#   toFeature: feature object that the data is to be copied into
def copyAttributes(fromFeature, toFeature):
    for i in range(fromFeature.GetFieldCount()):
        fieldName = fromFeature.GetFieldDefnRef(i).GetName()
        toFeature.SetField(fieldName, fromFeature.GetField(fieldName))



# define the function
def reproject(inFN, inEPSG, outFN, outEPSG):

    # get the shapefile driver
    driver = ogr.GetDriverByName('ESRI Shapefile')

    # create the input SpatialReference
    inSpatialRef = osr.SpatialReference()
    inSpatialRef.ImportFromEPSG(inEPSG)

    # create the output SpatialReference
    outSpatialRef = osr.SpatialReference()
    outSpatialRef.ImportFromEPSG(outEPSG)

    # create the CoordinateTransformation
    coordTrans = osr.CoordinateTransformation(inSpatialRef, outSpatialRef)

    # open the input data source and get the layer
    inDS = driver.Open(inFN, 0)
    if inDS is None:
        print('Could not open ' + inFN)
        sys.exit(1)
    inLayer = inDS.GetLayer()

    # create a new data source and layer
    if os.path.exists(outFN):
        driver.DeleteDataSource(outFN)
    outDS = driver.CreateDataSource(outFN)
    if outDS is None:
        print('Could not create ' + outFN)
        sys.exit(1)
    outLayer = outDS.CreateLayer(os.path.basename(outFN)[:-4],
                               geom_type=inLayer.GetLayerDefn().GetGeomType())

    # copy the fields from the input layer to the output layer
    copyFields(inLayer, outLayer)

    # get the FeatureDefn for the output shapefile
    featureDefn = outLayer.GetLayerDefn()

    # loop through the input features
    inFeature = inLayer.GetNextFeature()
    while inFeature:

        # get the input geometry
        geom = inFeature.GetGeometryRef()

        # reproject the geometry
        geom.Transform(coordTrans)

        # create a new feature
        outFeature = ogr.Feature(featureDefn)

        # set the geometry and attribute
        outFeature.SetGeometry(geom)

        # copy the attributes
        copyAttributes(inFeature, outFeature)

        # add the feature to the shapefile
        outLayer.CreateFeature(outFeature)

        # destroy the features and get the next input feature
        outFeature.Destroy
        inFeature.Destroy
        inFeature = inLayer.GetNextFeature()

    # close the shapefiles
    inDS.Destroy()
    outDS.Destroy()

    # create the *.prj file
    outSpatialRef.MorphToESRI()
    prjfile = open(outFN.replace('.shp', '.prj'), 'w')
    prjfile.write(outSpatialRef.ExportToWkt())
    prjfile.close()
    

def combineNeighbourFeatures(inFN, outFN, filterStr=None, 
                             driverName=None, clearAttributes=None,
                             FIDlim=None):
    
    if driverName == None:
        inDS = ogr.Open (inFN)
        if inDS is None:
            print('Could not open ' + inFN)
            sys.exit(1)
        inLayer = inDS.GetLayer()
        driver = inDS.GetDriver()
    else:   
        # get the shapefile driver
        driver = ogr.GetDriverByName(driverName)
        # open the input data source and get the layer
        inDS = driver.Open(inFN, 0)
        if inDS is None:
            print('Could not open ' + inFN)
            sys.exit(1)
        inLayer = inDS.GetLayer()

    # get input SpatialReference
    inSpatialRef = inLayer.GetSpatialRef()

    # create a new data source and layer
    if os.path.exists(outFN):
        driver.DeleteDataSource(outFN)
    outDS = driver.CreateDataSource(outFN)
    if outDS is None:
        print('Could not create ' + outFN)
        sys.exit(1)

    if filterStr != None:
        inLayer.SetAttributeFilter(filterStr)
           
    outLayer = outDS.CreateLayer("filterStr".format(iter),
                             geom_type=inLayer.GetLayerDefn().GetGeomType())
    
    # copy the fields from the input layer to the output layer
    copyFields(inLayer, outLayer)
    
    joinLayerFeatures(inLayer, outLayer, FIDlim=FIDlim)
    
    # close the shapefiles
    inDS.Destroy()
    outDS.Destroy()

    # create a *.prj file
    inSpatialRef.MorphToESRI()
    # compose prj filename
    prjFN = outFN[::-1].partition('.')[2][::-1]+'.prj'
    prjfile = open(prjFN,'w')
    prjfile.write(inSpatialRef.ExportToWkt())
    prjfile.close()



def joinFeatureToLayer(inFeature, inLayer, clearAttributes=None):

    featCount = inLayer.GetFeatureCount()

    # get IDs of all features in layer
    layerFIDs = getLayerFIDs(inLayer)   #[inLayer.GetNextFeature().GetFID() for k in np.arange(featCount)]
    
    geom = inFeature.GetGeometryRef()
    inFID = inFeature.GetFID()
    
    stop = False
    while not stop and len(layerFIDs) >= 1:
        #print "testing FID {0} with FID {1}".format(inFID,layerFIDs[0])
        
        # get first geometry
        layerFeature = inLayer.GetFeature(layerFIDs.pop(0))
        
        thisFID = layerFeature.GetFID()
    
        # get the input geometry
        joinGeom = layerFeature.GetGeometryRef()
        
#        gCount = joinGeom.GetGeometryCount()

        if joinGeom.Overlaps(geom):
            stop = True
            #print "Feature FID {0} joined with layer FID {1}".format(inFID,layerFIDs[0])
            joinGeom = joinGeom.Union(geom)
            print("Feature FID {0} joined with layer FID {1}...".format(inFID,thisFID))
#            if not joinGeom.Intersect(geom):
#                print "Problem with union!!!"
#                pdb.set_trace()
            #pdb.set_trace()
            # set the geometry
            
            # create a new feature
            newFeature = ogr.Feature(inLayer.GetLayerDefn())
                        
            # copy the attributes
            copyAttributes(layerFeature, newFeature)
            
            # set the geometry and attribute
            newFeature.SetGeometry(joinGeom)
            
            newFeature.SetFID = thisFID
            
            print("New feature created")
            
            # add the feature to the shapefile
            inLayer.SetFeature(newFeature)

#            layerFeature.SetGeometry(joinGeom)
#            print "Geometry stored to feature"
#            # update the feature to the layer
#            inLayer.SetFeature(layerFeature)
            print("Feature stored to Layer")
            
        # destroy the output feature
        layerFeature.Destroy()
             
    if not stop:
        print("Adding Feature FID {0} to layer as feature # {0}".format(featCount+1))
        # add feature as new feature to layer
         
        inFeature.SetFID(-1)       
#        # create a new feature
#        outFeature = ogr.Feature(featureDefn)
#        
#        # copy the attributes
#        copyAttributes(inFeature, outFeature)
#        
#        # set the geometry and attribute
#        outFeature.SetGeometry(joinGeom)
#    
#        # add the feature to the shapefile
#        inLayer.CreateFeature(outFeature)
#        
#        
        # update the feature to the layer
        inLayer.CreateFeature(inFeature)
    
        inFeature.Destroy()
        
        # print "{0} features in inLayer".format(inLayer.GetFeatureCount())        
        return False
    else:
        return True


def joinLayerFeatures(inLayer, outLayer, FIDlim=None, clearAttributes=None,
                      minArea=100):
    
    fieldDefn = ogr.FieldDefn('origFIDs', ogr.OFTString)
    fieldDefn.SetWidth(1000)
    outLayer.CreateField(fieldDefn)
    
    
    #featCount = inLayer.GetFeatureCount()

    # get IDs of all features in layer
    inFIDs = getLayerFIDs(inLayer)   #[inLayer.GetNextFeature().GetFID() for k in np.arange(featCount)]
    
    if FIDlim != None:
        def f(x): return x>=FIDlim[0] and x<=FIDlim[1] 
        inFIDs = list(filter(f, inFIDs))    
        
    
#    geom = inFeature.GetGeometryRef()
#    inFID = inFeature.GetFID()
    
    stop = False
    while not stop and len(inFIDs) >= 1:
        # get first geometry
        Feature = inLayer.GetFeature(inFIDs.pop(0))
        
        thisFID = Feature.GetFID()

        print("Processing FID {0}".format(thisFID))
        FIDattrib = "{0}".format(thisFID)  
        # get the input geometry
        joinGeom = Feature.GetGeometryRef()
        
#        gCount = joinGeom.GetGeometryCount()
        join_flag = True
        #any_joins = False
        it = 0
        while join_flag == True:
            
            print("Iteration {0}".format(it))
            join_flag = False
            
            for fid in inFIDs[:]:        # [:] makes sure we iterate over a copy!
                
                loopFeature = inLayer.GetFeature(fid)
                geom = loopFeature.GetGeometryRef()

                
                if joinGeom.Overlaps(geom):
                    print("joining features")
                    # Remove the geometry fid from the list of FIDs
                    inFIDs.remove(fid)
                    
                    # Perform the join
                    joinGeom = joinGeom.Union(geom)
                                      
                    join_flag = True
                    #any_joins = True
                    
                    FIDattrib = FIDattrib+",{0}".format(fid)
                    
                    print("Geometry of FID {0} joined with geometry of FID {1}...".format(thisFID,fid))

                                    
                # Clean up
                loopFeature.Destroy()
            
            it += 1
        
        # create a new feature
        newFeature = ogr.Feature(outLayer.GetLayerDefn())
                    
        # copy the attributes
        copyAttributes(Feature, newFeature)
        newFeature.SetField('origFIDs',FIDattrib)
        
        joinGeom = filterAreas(joinGeom, minArea)
        
        # set the geometry and attribute
        newFeature.SetGeometry(joinGeom)
        
        newFeature.SetFID = -1
        
        #print "New feature created"
        
        # add the feature to the shapefile
        outLayer.CreateFeature(newFeature)

        #print "Feature stored to Layer"
        
        # Clean up
        #joinGeom.Destroy()
        Feature.Destroy()
        newFeature.Destroy()
        
        
def getLayerFIDs(L):
    L.ResetReading()
    FIDs = [L.GetNextFeature().GetFID() for k in np.arange(L.GetFeatureCount())]
    L.ResetReading()
    return FIDs


def getGeometryCoords(G):
    x = [G.GetX(i) for i in range(G.GetPointCount())]
    y = [G.GetY(i) for i in range(G.GetPointCount())]
    return np.vstack((x, y)).T 


def getAreas(geom):
    
    areas = []
    
    for gid in range(geom.GetGeometryCount()):
        areas.append(geom.GetGeometryRef(int(gid)).GetArea())
    
    return areas


def filterAreas(geom, minArea):
    
    areas = getAreas(geom)
    
    ids = np.where(np.array(areas)>=minArea)[0]
    
    newGeom = ogr.Geometry(geom.GetGeometryType())
    
    for gid in ids:
        newGeom.AddGeometry(geom.GetGeometryRef(int(gid)))
    
    ndel = len(areas)-newGeom.GetGeometryCount()
    if ndel != 0:
        print("{0} geometries removed".format(ndel))
    
    return newGeom    
