"""
IMPORT PACKAGES
"""
import os
import sys
# Add matlab dependencies required for I2MC algorithim to the system path
sys.path.insert(1, "MatlabDependencies")
# Import Pupil Labs functions for data reading (writing functions not actually used currently and could be removed)
from PupilDependencies.file_methods import Serialized_Dict, load_pldata_file, PLData_Writer, load_object, save_object
import matlab.engine
# logging package from initial approach to development of pupil plugin, i believe - could be removed?
import logging
# msgpack is required for some of the Pupil Labs functions
import msgpack
import shutil
import numpy as np
import copy
from datetime import datetime
import scipy.io as sio
import cv2
import glob
from PIL import Image, ImageDraw
import math


"""
DEFINE FUNCTIONS
"""
# Function to extract normalised gaze points from Pupil files.
def extractPupilNormPos(file):
    # load gaze data, data timestamps, and topic from the pldata file using the Pupil function
    data, data_ts, topics = load_pldata_file(file, 'gaze')
    #create empty list for normalized gaze points
    #First set of lists used to form data structure with information on the source of the gaze point
    Xnorm0_list = []
    Ynorm0_list = []
    Xnorm1_list = []
    Ynorm1_list = []
    Xaver_list = []
    Yaver_list = []
    # second set of lists are used to feed gaze points to the I2MC algorithim in matlab
    I2MC_xlist = []
    I2MC_ylist = []


    # for each line of gaze data ...
    for line in data:
        # ... extract normal positions from the data structure
        # save normalised gaze points to a list
        AveGaze_norms = list(line['norm_pos'])
        # read confidence value for data point (currently unused, though a filter is built in below).
        conf = line['confidence']
        # read the base data - this includes information on the source of the gaze data point (left, right, average)
        base = line['base_data']
        # for the I2MC algoritihim we want independent lists of x and y co-ordinates
        # all gaze points are treated as averaged for the I2MC algorithim. Because where confidence is low Pupil outputs
        # data as individual gaze points (left and right eyes seperately) this creates a data structure that consists
        # largely of 'blank'/'NaN' values
        I2MC_xlist.append(AveGaze_norms[0])
        I2MC_ylist.append(AveGaze_norms[1])
        # confidence filter - not applied here as the I2MC algorithim is being deployed to address the noise in the data
        if float(conf) >= 0.0:
            # if the length of the base variable is 1, the data comes from 1 eye, so we need to check which eye
            if len(base) == 1:
                # read the base data from the base object
                base_data = base[0]
                # read the eye value data from the base_data object - this specifies the eye which the data came from.
                eyeVal = base_data['topic']
                # if the eye values is pupil.0 save the co-ordinate values to the '0' lists (others are blank)
                if eyeVal == 'pupil.0':
                    Xnorm0_list.append(AveGaze_norms[0])
                    Ynorm0_list.append(AveGaze_norms[1])
                    Xnorm1_list.append('')
                    Ynorm1_list.append('')
                    Xaver_list.append('')
                    Yaver_list.append('')
                # if the eye value is pupil.1, save the co-ordinates to the '1' lists (others are blank)
                elif eyeVal == 'pupil.1':
                    Xnorm0_list.append('')
                    Ynorm0_list.append('')
                    Xnorm1_list.append(AveGaze_norms[0])
                    Ynorm1_list.append(AveGaze_norms[1])
                    Xaver_list.append('')
                    Yaver_list.append('')
                # This is for a later version of the Pupil software than this project was carried out with
                # the pupil.0.3d data includes averaged gaze points.
                elif eyeVal == 'pupil.0.3d':
                    Xnorm0_list.append('')
                    Ynorm0_list.append('')
                    Xnorm1_list.append('')
                    Ynorm1_list.append('')
                    Xaver_list.append(AveGaze_norms[0])
                    Yaver_list.append(AveGaze_norms[1])
            # if the length of the base data is 2, this means the gaze data is the product of averaged left/right eyes
            elif len(base) == 2:
                Xnorm0_list.append('')
                Ynorm0_list.append('')
                Xnorm1_list.append('')
                Ynorm1_list.append('')
                Xaver_list.append(AveGaze_norms[0])
                Yaver_list.append(AveGaze_norms[1])
        # If the confidence filter is applied and not met, add blank values to the co-ordinate lists.
        else:
            Xnorm0_list.append('')
            Ynorm0_list.append('')
            Xnorm1_list.append('')
            Ynorm1_list.append('')
            Xaver_list.append('')
            Yaver_list.append('')
    #return normalised gaze points and list of data timestamps
    return data_ts, Xnorm0_list, Ynorm0_list, Xnorm1_list, Ynorm1_list, Xaver_list, Yaver_list, I2MC_xlist, I2MC_ylist

#This function converts the timestamps from the Pupil data file into milliseconds, for use in the I2MC algorithim
def formatTimestamps(data_ts):
    # Found some data files included 0 timestamps at points in the middle of the data collection - timestamps for these
    # values are interpolated.
    for i, v in enumerate(data_ts):
        if v == 0 and i != 0:
            less = data_ts[i-1]
            more = data_ts[i+1]
            difference = more - less
            data_ts[i] = less + (0.5 * difference)
    # find the minimum value in the data timestamps (start point)
    start = float(min(data_ts))
    # find the maximum value in the data timestamps (end point)
    end = float(max(data_ts))
    # data collection time equals the end point minus the start point
    length_time = float(end - start)
    # calculate the number of data points
    datapoints = float(len(data_ts))
    # calculate the data frequency as the number of data points over the length of the recording - round to nearest 10
    freq = round(int(datapoints/length_time), -1)
    # create an empty list to hold the converted timestamps
    ts_list = []
    # for each timestamp in the original list ...
    for value in data_ts:
        # ... the new timestamp (relative ms) is the difference of the value from the start time in seconds,
        # multiplied by 1000, and rounded to an integer.
        newTS = float(round((value - start) * 1000, 0))
        # append the new timestamp to the list of new timestamps
        ts_list.append([newTS])
    # return the calculated frequency from the data file, as well as a list of timestamps in relative milliseconds.
    return ts_list, freq

# This function reads the resolution of the world recording from the world.intrinsics file returned by Pupil.
def loadResolution(file):
    intrinsics_dictionary = load_object(file + "\world.intrinsics")
    for i, pair in enumerate(intrinsics_dictionary):
        if i == 1:
            value = str(pair)
    intrinsics = intrinsics_dictionary[value]
    res = intrinsics['resolution']
    return res

# This function takes the normalised gaze points extracted from Pupil data files and converts them into pixel values
# for both I2MC processing and visualisation.
def convert_norm2pix(Xnorm0_list, Ynorm0_list, Xnorm1_list, Ynorm1_list, Xaver_list, Yaver_list, I2MC_xlist, I2MC_ylist, res):
    # retrieve resolution for x axis
    x_max = res[0]
    # retrieve resolution for y axis
    y_max = res[1]
    # create empty lists to hold outputs
    Xpix0_list = []
    Ypix0_list = []
    Xpix1_list = []
    Ypix1_list = []
    XaverPix_list = []
    YaverPix_list = []
    I2MC_xPixlist = []
    I2MC_yPixlist = []
    # for each value in the list of normalised X position co-ordinates. (eye1)
    for value in Xnorm0_list:
        # if the value is blank, return a NaN
        if value == '':
            Xpix = float("NaN")
        else:
            #if the value is not blank, take the normalised value and multiply by the resolution to receive the pixel
            # position
            Xpix = float(value * x_max)
        # append the pixel based X coordinate to the output list.
        Xpix0_list.append([Xpix])
    # do the same for the normalised y coordniates (eye1)
    for value in Ynorm0_list:
        if value == '':
            Ypix = float("NaN")
        else:
            Ypix = float(value * y_max)
        Ypix0_list.append([Ypix])
    # do the same for the normalised x coordinates (eye2)
    for value in Xnorm1_list:
        if value == '':
            Xpix = float("NaN")
        else:
            Xpix = float(value * x_max)
        Xpix1_list.append([Xpix])
    # do the same for the normalised y coordinates (eye2)
    for value in Ynorm1_list:
        if value == '':
            Ypix = float("NaN")
        else:
            Ypix = float(value * y_max)
        Ypix1_list.append([Ypix])
    # do the same for the normalised x coordinates (averaged)
    for value in Xaver_list:
        if value == '':
            Xpix = float("NaN")
        else:
            Xpix = float(value * x_max)
        XaverPix_list.append([Xpix])
    # do the same for the normalised y coordinates (averaged)
    for value in Yaver_list:
        if value == '':
            Ypix = float("NaN")
        else:
            Ypix = float(value * y_max)
        YaverPix_list.append([Ypix])
    # for the normalised values in the I2MC specific lists - convert to pixels.
    for value in I2MC_xlist:
        Xpix = float(value * x_max)
        I2MC_xPixlist.append([Xpix])
    for value in I2MC_ylist:
        Ypix = float(value * y_max)
        I2MC_yPixlist.append([Ypix])
    #return lists of coordniates based on pixels rather than normalised positions.
    # I2MC lists for processing, other lists for visualisation.
    return Xpix0_list, Ypix0_list, Xpix1_list, Ypix1_list, XaverPix_list, YaverPix_list, I2MC_xPixlist, I2MC_yPixlist

# This function takes options for the I2MC processing and saves it into a matlab file (.mat) for loading within the engine
# later
def createOptionStruct(file, res, freq):
    options = {
        'opt': {
            'xres': float(res[0]),
            'yres': float(res[1]),
            'missingx': float("NaN"),
            'missingy': float("NaN"),
            'freq': float(freq),
            'scrSz': [float(0), float(0)],
            'disttoscreen': float(0),
            'downsampFilter': int(0),
            'downsamples': [float(2)]
        }
    }
    dirName = 'I2MC_Inputs/' + file
    try:
        os.mkdir(dirName)
    except:
        pass
    # save options in to mat file
    optPath = dirName + '\opt.mat'
    sio.savemat(optPath, options)
    return optPath

# This function takes the data extracted from the pupil files and saves them into a .mat file for processing through
# matlab.
def createDataStruct(ts_list, I2MC_xPixlist, I2MC_yPixlist, file):
    data = {
        'data': {
            'time': ts_list,
            'average': {
                'X': I2MC_xPixlist,
                'Y': I2MC_yPixlist
            }
        }
    }
    dirName = 'I2MC_Inputs/' + file
    try:
        os.mkdir(dirName)
    except:
        pass
    # save data in to mat file
    dataPath = dirName + '\data.mat'
    sio.savemat(dataPath, data)
    return dataPath

# this function essentially just collates the above functions within a parent function.
# extracts data from pupil file, saves mat files, and returns paths for the mat files as well as the raw data for
# storing
def dataExtraction(file):
    data_ts, Xnorm0_list, Ynorm0_list, Xnorm1_list, Ynorm1_list, Xaver_list, Yaver_list, I2MC_xlist, I2MC_ylist = extractPupilNormPos(file)
    ts_list, freq = formatTimestamps(data_ts)
    res = loadResolution(file)
    Xpix0_list, Ypix0_list, Xpix1_list, Ypix1_list, XaverPix_list, YaverPix_list, I2MC_xPixlist, I2MC_yPixlist = convert_norm2pix(Xnorm0_list, Ynorm0_list, Xnorm1_list,
                                                                      Ynorm1_list, Xaver_list, Yaver_list, I2MC_xlist, I2MC_ylist, res)
    optPath = createOptionStruct(file, res, freq)
    dataPath = createDataStruct(ts_list, I2MC_xPixlist, I2MC_yPixlist, file)
    return optPath, dataPath, Xpix0_list, Ypix0_list, Xpix1_list, Ypix1_list, XaverPix_list, YaverPix_list, res

#Function to launch the matlab engine and prepare it for the I2MC processing
def I2MCEngine():
    # Launch matlab engine
    eng = matlab.engine.start_matlab()
    try:
        #try to add the I2MC functions to the matlab path
        eng.addpath('I2MC-1.1/functions/I2MC')
        eng.addpath('I2MC-1.1/functions/helpers')
        eng.addpath('I2MC-1.1/functions/interpolation')
    except:
        #Notify users if I2MC functions not found/added to matlab path (should probably add an escape as this would
        # cause a critical error)
        print('Error: I2MC files missing')
    #returns variable represnting the matlab engine
    return eng


def runI2MC(optPath, dataPath):
    # open the engine
    eng.desktop(nargout=0)
    # load .mat file options
    eng.load(optPath, nargout=0)
    # load .mat data
    eng.load(dataPath, nargout=0)
    # try to run the I2MC algorithim on data - if failed, return message.
    try:
        [fix, ndata, par] = eng.I2MCfunc(eng.workspace['data'],eng.workspace['opt'], nargout=3)
    except:
        print('I2MC function failed to execute')
        exit()
    # return outputs from I2MC
    return fix, ndata, par

# This function takes the outputs from the I2MC algoritihim (i.e. the matlab outputs) and extracts information from
# the data structures. Gaze data, fixation data and I2MC parameters are saved to text files, and copies of the gaze and
# fixation data are returned for visualisation.
def formatOutputs(fix, ndata, par, file, Xpix0_list, Ypix0_list, Xpix1_list, Ypix1_list):
    print('formatting outputs')
    # reading outputs from I2MC data structures - see https://github.com/royhessels/I2MC for info.
    cutoff = fix['cutoff']
    startList = fix['start']
    start = startList[0]
    endList = fix['end']
    end = endList[0]
    startT = fix['startT']
    endT = fix['endT']
    dur = fix['dur']
    xposList = fix['xpos']
    xpos = xposList[0]
    yposList = fix['ypos']
    ypos = yposList[0]
    flankdatalossList = fix['flankdataloss']
    flankdataloss = flankdatalossList[0]
    fracinterpedList = fix['fracinterped']
    fracinterped = fracinterpedList[0]
    RMSxyList = fix['RMSxy']
    RMSxy = RMSxyList[0]
    BCEAList = fix['BCEA']
    BCEA = BCEAList[0]
    fixRangeXList = fix['fixRangeX']
    fixRangeX = fixRangeXList[0]
    fixRangeYList = fix['fixRangeY']
    fixRangeY = fixRangeYList[0]
    time = ndata['time']
    # from lists of co-ordinates for individual eyes created earlier, check if left or right eye data is missing.
    leftX = Xpix0_list
    leftY = Ypix0_list
    leftMissing = []
    for value in Xpix0_list:
        if math.isnan(value[0]) == True:
            state = 1
        else:
            state = 0
        leftMissing.append([state])
    # edits
    rightX = Xpix1_list
    rightY = Ypix1_list
    rightMissing = []
    for value in Xpix1_list:
        if math.isnan(value[0]) == True:
            state = '1'
        else:
            state = '0'
        rightMissing.append([state])
    # more data extraction from I2MC data structures
    average = ndata['average']
    averageX = average['X']
    averageY = average['Y']
    averageMissing = average['missing']
    finalweights = ndata['finalweights']
    xres = par['xres']
    yres = par['yres']
    freq = par['freq']
    missingx = par['missingx']
    missingy = par['missingy']
    scrSz = par['scrSz']
    disttoscreen = par['disttoscreen']
    windowtimeInterp = par['windowtimeInterp']
    edgeSampInterp = par['edgeSampInterp']
    maxdisp = par['maxdisp']
    windowtime = par['windowtime']
    steptime = par['steptime']
    downsamples = par['downsamples']
    downsampFilter = par['downsampFilter']
    chebyOrder = par['chebyOrder']
    maxerrors = par['maxerrors']
    cutoffstd = par['cutoffstd']
    maxMergeDist = par['maxMergeDist']
    maxMergeTime = par['maxMergeTime']
    minFixDur = par['minFixDur']
    # compile fixation data
    OutputFixationData = []
    for i in range(len(start)):
        fixData = [start[i], end[i], startT[i][0], endT[i][0], dur[i][0], xpos[i], ypos[i], flankdataloss[i], fracinterped[i], RMSxy[i], BCEA[i], fixRangeX[i], fixRangeY[i]]
        OutputFixationData.append(fixData)
    # compile gaze data
    OutputGazeData = []
    for i in range(len(time)):
        gazeData = [time[i][0], leftX[i][0], leftY[i][0], rightX[i][0], rightY[i][0], averageX[i][0], averageY[i][0], leftMissing[i][0], rightMissing[i][0], averageMissing[i][0], finalweights[i][0]]
        OutputGazeData.append(gazeData)
    # save output parameters to a list
    OutputParameterData = [
        ['Cutoff', cutoff],
        ['xres', xres],
        ['yres', yres],
        ['freq', freq],
        ['missingx', missingx],
        ['missingy', missingy],
        ['scrSz', scrSz],
        ['disttoscreen', disttoscreen],
        ['windowtimeInterp', windowtimeInterp],
        ['edgeSampInterp', edgeSampInterp],
        ['maxdisp', maxdisp],
        ['windowtime', windowtime],
        ['steptime', steptime],
        ['downsamples', downsamples],
        ['downsampFilter', downsampFilter],
        ['chebyOrder', chebyOrder],
        ['maxerrors', maxerrors],
        ['cutoffstd', cutoffstd],
        ['maxMergeDist', maxMergeDist],
        ['maxMergeTime', maxMergeTime],
        ['minFixDur', minFixDur]
    ]
    # check for directory and save outputs
    dirName = 'I2MC_Outputs/' + file + '/'
    try:
        os.mkdir('I2MC_Outputs/' + file)
    except:
        pass
    np.savetxt(dirName + 'ParameterInfo.txt', OutputParameterData, fmt='%s', delimiter=",")
    np.savetxt(dirName + 'GazeData.txt', OutputGazeData, fmt='%s', delimiter=",")
    np.savetxt(dirName + 'FixationData.txt', OutputFixationData, fmt='%s', delimiter=",")
    return OutputFixationData, OutputGazeData

# this function extracts the world video timestamps from the world_timestamps.npy file, converts them to relative ms,
# and returns a list of new timestamps.
def extractWorldTS(file):
    print('Extracting timestamps from world video')
    TSPath = file + '/world_timestamps.npy'
    VidTSArray = np.load(TSPath)
    VidTSList = np.ndarray.tolist(VidTSArray)
    for index, value in enumerate(VidTSList):
        if value == 0 and index != 0:
            more = VidTSList[index + 1]
            less = VidTSList[index - 1]
            difference = more - less
            VidTSList[index] = less + (0.5 * difference)
    if VidTSList[0] == 0:
        start = VidTSList[1]
    else:
        start = min(VidTSList)
    TSs = []
    for index, value in enumerate(VidTSList):
        newTS = (value - start) * 1000
        TSs.append(newTS)
    return TSs

# this function takes the mp4 output from the pupil files and stores images of each frame, as well as constructing a
# list of all the frames
def extractVidFrames(file, TSs):
    print('Extracting world video frames')
    VidPath = file + '/world.mp4'
    OutputDir = 'I2MC_Outputs/' + file + '/Frames'
    FrameList_raw = []
    try:
        os.mkdir(OutputDir)
    except:
        pass
    capture = cv2.VideoCapture(VidPath)
    i = 0
    while (capture.isOpened()):
        ret, frame = capture.read()
        if ret == False:
            break
        else:
            cv2.imwrite(OutputDir + '/Vid_Frame' + str(i) + '.jpg', frame)
            FrameList_raw.append(OutputDir + '/Vid_Frame' + str(i) + '.jpg')
        i = i + 1
    capture.release()
    cv2.destroyAllWindows()
    FrameList = []
    for index, Frame in enumerate(FrameList_raw):
        FrameShort = Frame.replace(OutputDir + '/', '')
        FrameShort_noExtension = FrameShort.replace('.jpg', '')
        ind = FrameShort_noExtension.replace('\Vid_Frame', '')
        FrameList.append([ind, TSs[index], FrameShort, Frame])
    return FrameList

# This function compiles all data to the gaze point level and saves a text file with this data
def compile2gaze(OutputGazeData, OutputFixationData, FrameList, file):
    print('Compiling to gaze data file')
    #Map fixation values on to the gaze data
    fixMapList = [float("NaN")] * len(OutputGazeData)
    for gazeInd, gazeLine in enumerate(OutputGazeData):
        for fixInd, fixLine in enumerate(OutputFixationData):
            fixStart = fixLine[2]
            fixEnd = fixLine[3]
            if fixStart <= gazeLine[0] and fixEnd >= gazeLine[0]:
                fixMapList[gazeInd] = fixInd
            elif gazeLine[0] > fixStart and gazeLine[0] > fixEnd:
                pass
    for gazeInd, gazeLine in enumerate(OutputGazeData):
        gazeLine.append(fixMapList[gazeInd])
    #Map world video frame values on to the gaze data.
    FrameMapList = [float("NaN")] * len(OutputGazeData)
    for gazeInd, gazeLine in enumerate(OutputGazeData):
        for frameInd, frameLine in enumerate(FrameList):
            frameStart = frameLine[1]
            try:
                frameEndLine = FrameList[frameInd + 1]
                frameEnd = frameEndLine[1]
            except:
                frameEnd = 'end'
            if frameEnd == 'end':
                if gazeLine[0] >= frameStart:
                    FrameMapList[gazeInd] = frameInd
            elif frameStart <= gazeLine[0] and frameEnd >= gazeLine[0]:
                FrameMapList[gazeInd] = frameInd
            #elif gazeLine[0] > frameStart and gazeLine[0] > frameEnd and frameEnd != float("NaN"):
                #pass
    #MAY WANT TO INCORPORATE X/Y CORDS IN TO THIS DATA - DO IN SECTION ABOVE/BELOW
    for gazeInd, gazeLine in enumerate(OutputGazeData):
        gazeLine.append(FrameMapList[gazeInd])
    np.savetxt('I2MC_Outputs/' + file + '/CompiledData.txt', OutputGazeData, fmt='%s', delimiter=",")
    return OutputGazeData

# this function compiles all data to the level of video frames, and saves a text file containing the data
def compile2Frame(CompiledData, FrameList, OutputFixationData):
    print('Compiling to frame data file')
    for FrameInd, FrameLine in enumerate(FrameList):
        frameStart = FrameLine[1]
        try:
            frameEndLine = FrameList[FrameInd + 1]
            frameEnd = frameEndLine[1]
        except:
            frameEnd = 'end'
        gazePointList = []
        fixList = []
        frameHasFixation = False
        for dataInd, dataLine in enumerate(CompiledData):
            if dataLine[12] == FrameInd:
                gazePoint = []
                """NEEDS CHANGING HERE OR REBUILDING DATA STRUCTURES"""
                if int(dataLine[7]) == 1 and int(dataLine[8]) == 1:
                    gazePoint = [dataLine[5], dataLine[6], 'ave']
                elif int(dataLine[7]) == 0:
                    gazePoint = [dataLine[1], dataLine[2], 'left']
                elif int(dataLine[8]) == 0:
                    gazePoint = [dataLine[3], dataLine[4], 'right']
                if len(gazePoint) > 0:
                    gazePointList.append(gazePoint)
        FrameLine.append(gazePointList)
        for fixInd, fixLine in enumerate(OutputFixationData):
            fixStart = fixLine[2]
            fixEnd = fixLine[3]
            if frameEnd == 'end':
                if fixStart <= frameStart and fixEnd > frameStart:
                    fixList.append([fixLine[5], fixLine[6], fixInd])
                    frameHasFixation = True
            elif fixStart <= frameStart and fixEnd >= frameEnd:
                fixList.append([fixLine[5], fixLine[6], fixInd])
                frameHasFixation = True
        if frameHasFixation == False:
            fixList.append('none')
        FrameLine.append(fixList)
    np.savetxt('I2MC_Outputs/' + file + '/FrameData.txt', FrameList, fmt='%s', delimiter=",")
    return FrameList

# This function takes the frame level data and creates an overlay for each frame that visualises gaze and fixation
# co-ordinates. gaze points are plotted in different colours depending on which eyes the gaze point was derived from;
# green - left eye. blue - right eye. purple - both eyes. Fixations plotted with red circles.
# the function returns a list of the overlay file names
def createOverlays(FrameList, res):
    print('Creating overlays for visualisation.')
    OverlayList = []
    overlayDirectory = 'I2MC_Outputs/' + file + '/Overlays'
    try:
        os.mkdir(overlayDirectory)
    except:
        pass
    frameTotal = len(FrameList)
    # set font colour - NOT WORKING!
    fontcolour = (0, 0, 0, 255)
    for frameInd, frameLine in enumerate(FrameList):
        print('Creating overlay for frame ' + str(frameInd) + ' / ' + str(frameTotal - 1))
        gazepoints = frameLine[4]
        fixations = frameLine[5]
        overlay_Image = Image.new('RGBA', (res[0], res[1]), (0, 0, 0, 0))
        draw_overlay = ImageDraw.Draw(overlay_Image)
        for point in gazepoints:
            gazeX = int(point[0])
            gazeY = int(point[1])
            eye = point[2]
            if eye == 'ave':
                colour = (128, 0, 128)
                outline = (64, 0, 64)
            if eye == 'left':
                colour = (0, 255, 0)
                outline = (0, 128, 0)
            elif eye == 'right':
                colour = (0, 0, 255)
                outline = (0, 0, 128)
            bounding_gaze = [
                (
                    (gazeX - 10),
                    ((res[1] - gazeY) - 10)
                ),
                (
                    (gazeX + 10),
                    ((res[1] - gazeY) + 10)
                )
            ]
            draw_overlay.rectangle(bounding_gaze, fill=colour, outline=outline, width=2)
        for fix in fixations:
            if fix == 'none':
                FixImgText = 'Fixation: --'
            elif len(fix) == 3:
                xpos = int(fix[0])
                ypos = int(fix[1])
                FixationIndex = fix[2]
                # Create bounding boxes for visualisation based on x and y pos.
                bounding = [
                    (
                        (xpos - 10),
                        ((res[1] - ypos) - 10)
                    ),
                    (
                        (xpos + 10),
                        ((res[1] - ypos) + 10)
                    )
                ]
                outerbounding = [
                    (
                        (xpos - 30),
                        ((res[1] - ypos) - 30)
                    ),
                    (
                        (xpos + 30),
                        ((res[1] - ypos) + 30)
                    )
                ]
                draw_overlay.ellipse(bounding, fill=(255, 0, 0), outline=(128, 0, 0), width=2)
                draw_overlay.ellipse(outerbounding, outline=(128, 0, 0), width=10)
                FixImgText = 'Fixation: ' + str(FixationIndex) + ', X: ' + str(xpos) + ', Y:' + str(ypos)
            FixImgTextLong = 'Fixation: 00000000, X: 0000000000, Y: 0000000000'
            text_size = draw_overlay.textsize(FixImgTextLong)
            infobounding = [
                (
                    (0),
                    (res[1])
                ),
                (
                    (text_size[0] + 40),
                    (res[1] - (text_size[1] + 40))
                )
            ]
            textLoc = (20, res[1] - (20 + text_size[1]))
        draw_overlay.rectangle(infobounding, fill=(255, 255, 255, 255), outline=(0, 0, 0, 255), width=5)
        draw_overlay.text(textLoc, FixImgText, fontcolour)
        overlay_name = overlayDirectory + '/Overlay_Frame' + str(frameInd) + '.png'
        OverlayList.append(overlay_name)
        overlay_Image.save(overlay_name, 'PNG')
    return OverlayList

# this function takes the list of frame and overlay file names and combines each frame with its overlay.
def mergeOverlays2Frames(FrameList, OverlayList):
    for frameInd, frame in enumerate(FrameList):
        print('Merging overlay to frame ' + str(frameInd) + ' / ' + str(len(FrameList) - 1))
        frame_name = frame[3]
        overlay_name = OverlayList[frameInd]
        frameImg = cv2.imread(frame_name)
        overlayImg = cv2.imread(overlay_name)
        OutImgMat = cv2.addWeighted(frameImg, 1, overlayImg, 255, 0)
        cv2.imwrite(frame_name, OutImgMat)

# This function takes the frames that have been combined with the overlays and codes a .avi file.
def compile2Movie(file):
    print('Preparing to render video')
    ImgDir = 'I2MC_Outputs/' + file + '/Frames'
    VidName = file + '_Visualisation.avi'
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    images = [img for img in os.listdir(ImgDir) if img.endswith('.jpg')]
    frame = cv2.imread(os.path.join(ImgDir, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(VidName, fourcc, 30, (width, height))

    for index, image in enumerate(images):
        print('rendering frame ' + str(index))
        video.write(cv2.imread(ImgDir + '/Vid_Frame' + str(index) + '.jpg'))

    cv2.destroyAllWindows()
    video.release()
    shutil.copyfile(VidName, 'I2MC_Outputs/' + file + '/' + VidName)
    os.remove(VidName)



"""
INPUTS
"""

# Set this to the name of the pupil directory you wish to process. The directory should be located in the root.
file = 'Sf031-ET'

"""
MAIN LOOP
"""

#extract data
optPath, dataPath, Xpix0_list, Ypix0_list, Xpix1_list, Ypix1_list, XaverPix_list, YaverPix_list, res = dataExtraction(file)

#start matlab engine
eng = I2MCEngine()
# run I2MC
fix, ndata, par = runI2MC(optPath, dataPath)
# format data outputs
OutputFixationData, OutputGazeData = formatOutputs(fix, ndata, par, file, Xpix0_list, Ypix0_list, Xpix1_list, Ypix1_list)

# extract world video timestamps.
TSs = extractWorldTS(file)
# extract world video frames
FrameList = extractVidFrames(file, TSs)
# compile data to gaze point level
compiledData = compile2gaze(OutputGazeData, OutputFixationData, FrameList, file)
# compile data to frame level
frameData = compile2Frame(compiledData, FrameList, OutputFixationData)

# Visualise
# create overlays
OverlayList = createOverlays(FrameList, res)
# merge overlays to frames
mergeOverlays2Frames(FrameList, OverlayList)
# compile merged frame/overlays into video file.
compile2Movie(file)


