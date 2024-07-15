# set parameters for analysis
exec(open("erpr_settings.py").read())

# import packages
exec(open("erpr_import_packages.py").read())

# eventMsgs = ['START_TRIAL']
# eventMsgs = ['START_TRIAL','END_TRIAL']
def erpr(
    eyelinkData,
    selectEye,
    eventMsgs=None,
    eventDict=None,
    eventWindowSize=3,
    nBins=10,
    removeBlinks=True,
    showTraces=True,
):
    """
    eyelinkData: string indicating the full path of the eyelink data file

    eventMsgs: list with strings used to look up timestamps of event messages in the edf file.

    eventDict: contains a dictionary with event names in eventDict['MSG'],
    and corresponding start timestamps in eventDict['TIMESTAMPS']

    eventWindowSize: float indicating period/duration [in seconds] for each pupil response to an event

    nBins: integer indicating the number of bins (categories/conditions) that each pupil response may
    be assigned to depending on the score in eventDict. eventDict must be provided
    and eventDict['MSG] should contain a numpy array of floats (e.g., reflecting visual change magnitudes)



    IMPORTANT: Three possible combinations between eventMsgs and eventDict:

    1) If eventMsgs = None, and eventDict = None,  then all events in eyelinkData are put in eventDict.
    Configure it this way if event names are unknown.

    2) If eventMsgs is a list but eventDict = None,
    then the strings in eventMsgs are used to select events in eyelinkData and to create an eventDict.
    Configure it this way if event names in edf file are known.
    A separate ERPR is then created per eventMsg

    3) If eventMsgs is a list and eventDict is a dict,
    then the timestamp of the first eventMsg in eyelinkData will serve as timepoint 0
    to match the timestamps between the eyelinkData and eventDict.
    Configure it this way if the eventDict contains timestamps in a different format (e.g. seconds instead of milliseconds)
    than eyelinkData (e.g., a video starts at timestamp 0s but the first eyelinkData event may start at timestamp 412490ms).

    output

    erprDict:
    Multiple and separate ERPRs are created per eventMsg if eventDict['MSG'] contains a list of multiple unique strings

    One pooled ERPR is created if eventDict['MSG'] contains the same strings

    Multiple and separate ERPR are created if eventDict['MSG'] contains a list of floats.
    The floats are then binned in nBins (see below; default=10) with equal number of events per bin.

    """

    # selectEye = 'left'
    # eventMsgs=["start_" + onsetEvent]
    # eventDict=None
    # eventWindowSize=2
    # nBins=10
    # removeBlinks=True
    # showTraces=True

    # pathFolders, __, __, base_filename, __ = splitPathFilenameComponents(edfFilename)

    # ascFile = pathFolders + "\\" + base_filename + ".asc"
    # if os.path.isfile(ascFile):
    #     logger.log("Ascii file already exists. Skipping conversion of:")
    #     logger.log(ascFile)
    # else:
    #     # convert eyelink data file to ascii file
    #     tempEdfFilename = base_filename + ".edf"
    #     tempAscFilename = base_filename + ".asc"
    #     command = ["edf2asc", tempEdfFilename, tempAscFilename]
    #     process = subprocess.run(
    #         command, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, cwd=pathFolders
    #     )
    #     logger.log(process.stdout.decode("UTF-8"))

    # read ascii file
    # from classes.eyetracker_parsing import eyelink_parser

    # parseObj = eyelink_parser(ascFile)
    # parseObj.lookForMsg
    # eyelinkData = parseObj.parseAscFile()

    eyelinkDataTemp = eyelinkData.copy()

    sampleDict = eyelinkDataTemp["SAMPLES"].copy()

    # remove blinks
    tdata = sampleDict["TIMESTAMPS"].copy()
    if selectEye == "left":
        pupName = "PUPIL-SIZE-LEFT"
        gazeXName = "X-GAZE-LEFT"
        gazeYName = "Y-GAZE-LEFT"
    elif selectEye == "right":
        pupName = "PUPIL-SIZE-RIGHT"
        gazeXName = "X-GAZE-RIGHT"
        gazeYName = "Y-GAZE-RIGHT"
    elif selectEye == "both":
        pupName = "PUPIL-SIZE-BOTH"
        gazeXName = "X-GAZE-BOTH"
        gazeYName = "Y-GAZE-BOTH"
        sampleDict["PUPIL-SIZE-BOTH"] = np.mean(
            np.vstack(
                (
                    sampleDict["PUPIL-SIZE-LEFT"],
                    sampleDict["PUPIL-SIZE-RIGHT"],
                )
            ),
            axis=0,
        )
        sampleDict["X-GAZE-BOTH"] = np.mean(
            np.vstack(
                (
                    sampleDict["X-GAZE-LEFT"],
                    sampleDict["X-GAZE-RIGHT"],
                )
            ),
            axis=0,
        )
        sampleDict["Y-GAZE-BOTH"] = np.mean(
            np.vstack(
                (
                    sampleDict["Y-GAZE-LEFT"],
                    sampleDict["Y-GAZE-RIGHT"],
                )
            ),
            axis=0,
        )

    pdata = sampleDict[pupName].copy()
    xdata = sampleDict[gazeXName].copy()
    ydata = sampleDict[gazeYName].copy()

    if np.shape(pdata)[0] == np.shape(tdata)[0]:
        pass
    else:
        logger.log("Eyelink timestamp array has different size as pupil array", "error")
        print("size pdata:")
        print(np.size(pdata))
        print("size tdata:")
        print(np.size(tdata))

    if showTraces:
        plt.figure(figsize=(30, 15))
        plt.subplot(4, 1, 1)
        plt.plot(tdata, pdata)
        # plt.plot(tdata[10000:40000], pdata[10000:40000])
        plt.xlabel("Timestamps [ms]")
        plt.ylabel("Pupil size [au]")
        plt.subplot(4, 1, 2)
        plt.plot(tdata, xdata)
        plt.plot(tdata, ydata)
        # plt.plot(tdata[10000:40000], pdata[10000:40000])
        plt.xlabel("Timestamps [ms]")
        plt.ylabel("Pupil size [au]")
        plt.legend(['x','y'])

    # removeEyelinkBlinks function CAUSES REMOVAL OF FIRST SAMPLES IN EYETRACKING DATA ... and some weird memory error causes to update eyelinkDataTemp ... despite using .copy()
    if removeBlinks:
        
        newtdata, newpdata, newxdata, newydata = removeEyelinkBlinks(
            tdata, pdata, xdata, ydata, dilutionSize=[50, 80], showTimeTraces=False
        )
        
        finPupilSamples = np.array(np.isfinite(newpdata))


        newtdata, newpdata, newxdata, newydata = interpolateBlinks(newtdata, newpdata, newxdata, newydata)

        sampleDict["TIMESTAMPS"] = newtdata
        sampleDict[pupName] = newpdata
        sampleDict[gazeXName] = newxdata
        sampleDict[gazeYName] = newydata

        if showTraces:
            plt.title("Before blink removal")
            plt.subplot(4, 1, 3)
            plt.plot(newtdata, newpdata)
            # plt.plot(newtdata[10000:40000], newpdata[10000:40000])
            plt.xlabel("Timestamps [ms]")
            plt.ylabel("Pupil size [au]")

            plt.subplot(4, 1, 4)
            plt.plot(newtdata, newxdata)
            plt.plot(newtdata, newydata)
            # plt.plot(newtdata[10000:40000], newpdata[10000:40000])
            plt.xlabel("Timestamps [ms]")
            plt.ylabel("Pupil size [au]")

    npTimeStamps = np.array(sampleDict["TIMESTAMPS"].copy())
    npPupilData = np.array(sampleDict[pupName].copy())
    npXgazeData = np.array(sampleDict[gazeXName].copy())
    npYgazeData = np.array(sampleDict[gazeYName].copy())
    nFinSamplesPerTrial = []

    binAnalysis = False
    if eventDict == None:  # fill eventDict using MSG dict in eyelinkDataTemp

        logger.log("eventDict is empty. ")
        if eventMsgs == None:  # use all events detected in eyelinkDataTemp
            tempEventDict = eyelinkDataTemp["EVENTS"].copy()
            eventDict = {}
            eventDict["TIMESTAMPS"] = []
            eventDict["MSG"] = []
            eventDict["UNIQUE_MSG"] = []
            eventDict["UNIQUE_MSG_IND"] = []
            eventDict["N_UNIQUE_MSG"] = []

            logger.log(
                "eventMsg is also empty. eventDict is automatically created with messages from edf file"
            )
            logger.log("Detected events: " + ", ".join(eventDict["MSG"]))
            countUniqueMsgINDs = 0
            for i in range(len(tempEventDict["MSG"])):
                if tempEventDict["MSG"][i] in eventDict["MSG"]:  # previously seen MSG
                    uniqueMsgIND = eventDict["MSG"].index(tempEventDict["MSG"][i])
                    eventDict["UNIQUE_MSG_IND"].append(uniqueMsgIND)
                    eventDict["N_UNIQUE_MSG"][uniqueMsgIND] = (
                        eventDict["N_UNIQUE_MSG"][uniqueMsgIND] + 1
                    )
                else:  # new MSG
                    eventDict["UNIQUE_MSG"].append(tempEventDict["MSG"][i])
                    eventDict["UNIQUE_MSG_IND"].append(countUniqueMsgINDs)
                    eventDict["N_UNIQUE_MSG"].append(1)
                    countUniqueMsgINDs += 1
                eventDict["TIMESTAMPS"].append(tempEventDict["TIMESTAMPS"][i])
                eventDict["MSG"].append(tempEventDict["MSG"][i])

        else:  # eventMsg provided
            tempEventDict = eyelinkDataTemp["EVENTS"].copy()
            eventDict = {}
            eventDict["TIMESTAMPS"] = []
            eventDict["MSG"] = []
            eventDict["UNIQUE_MSG"] = []
            eventDict["UNIQUE_MSG_IND"] = []
            eventDict["N_UNIQUE_MSG"] = []

            countUniqueMsgINDs = 0
            logger.log(
                "eventDict is automatically created with all the messages from edf file as events: "
                + ", ".join(eventMsgs)
            )

            for i in range(len(tempEventDict["MSG"])):
                for j in range(len(eventMsgs)):
                    if tempEventDict["MSG"][i].startswith(eventMsgs[j]):
                        if tempEventDict["MSG"][i] in eventDict["MSG"]:  # previously seen MSG
                            uniqueMsgIND = eventDict["UNIQUE_MSG"].index(tempEventDict["MSG"][i])
                            # eventDict["UNIQUE_MSG_IND"].append(uniqueMsgIND)
                            eventDict["N_UNIQUE_MSG"][uniqueMsgIND] = (
                                eventDict["N_UNIQUE_MSG"][uniqueMsgIND] + 1
                            )
                        else:  # new MSG
                            eventDict["UNIQUE_MSG"].append(tempEventDict["MSG"][i])
                            eventDict["UNIQUE_MSG_IND"].append(countUniqueMsgINDs)
                            eventDict["N_UNIQUE_MSG"].append(1)
                            countUniqueMsgINDs += 1
                        eventDict["TIMESTAMPS"].append(tempEventDict["TIMESTAMPS"][i])
                        eventDict["MSG"].append(tempEventDict["MSG"][i])

                        break

        if len(eventDict["MSG"]) > 10:
            logger.log(
                "Many detected events: " + eventDict["MSG"][0] + " - " + eventDict["MSG"][-1]
            )
        else:
            logger.log("Detected events: " + ", ".join(eventDict["MSG"]))

    else:  # use user-provided eventDict
        logger.log(
            "eventDict is provided. Using predefined events in eventDict rather than those based on messages in eyelink data file."
        )
        if eventMsgs == None:
            logger.log(
                "eventMsgs is empty. Unable to match timestamps in eventDict with timestamps from edf file",
                "error",
            )
        else:  # eventMsgs consists of list of strings with MSGs, each belonging to a different category/condition
            logger.log(
                "Taking first string in list eventMsgs as reference for first timepoint of eventDict."
            )
            # logger.log("Event messages: " + ", ".join(eventDict["MSG"]))

            if type(eventDict["MSG"]).__name__ == "ndarray":
                binAnalysis = True
                logger.log(
                    "Event messages consist of floats in numpy array. Creating "
                    + str(nBins)
                    + " bins for assigning pupil responses to conditions depending on percentiles float numbers"
                )

                percentiles = np.linspace(0, 100, nBins + 1)
                binEdges = stats.scoreatpercentile(eventDict["MSG"], percentiles)
                # binEdges = np.linspace(
                # np.min(eventDict["MSG"]), np.max(eventDict["MSG"]), nBins + 1
                # )
                binCenters = np.diff(binEdges) / 2 + binEdges[:-1]
                eventDict["UNIQUE_MSG"] = list(binCenters)
                eventDict["UNIQUE_MSG_IND"] = list(np.arange(len(binCenters)))
                eventDict["N_UNIQUE_MSG"] = []

                binEdgesTemp = binEdges
                binEdgesTemp[-1] = binEdgesTemp[-1] * 1.1
                for i in range(len(binCenters)):
                    nUniqueMsg = np.sum(
                        (eventDict["MSG"] >= binEdgesTemp[i])
                        & (eventDict["MSG"] < binEdgesTemp[i + 1])
                    )
                    eventDict["N_UNIQUE_MSG"].append(nUniqueMsg)
            else:
                logger.log("Event messages in provided eventDict: " + ", ".join(eventDict["MSG"]))
                logger.log("Categorizing ERPRs per MSG.")
                logger.log("THIS STILL NEEDS TO BE PROGRAMMED!!! TO DO")

        # match timestamps in eventDict to timestamps of pupil traces (eye-tracker)
        npTimeStamps = (npTimeStamps - npTimeStamps[0]) / 1000  # to seconds

    # structure pupil responses in matrix (rows = events, columns = time)
    nTrials = len(eventDict["TIMESTAMPS"])
    nSamplesPerTrial = eventWindowSize * eyelinkDataTemp["SETTINGS"]["SAMPLING_RATE"]

    eventDict["COUNT_UNIQUE_MSG"] = []
    for temp in eventDict["UNIQUE_MSG"]:
        eventDict["COUNT_UNIQUE_MSG"].append(0)

    countUniqueMsg = 0
    erprDict = {}
    erprMatrix = np.zeros((nTrials, nSamplesPerTrial))
    gazeMatrix = np.zeros((nTrials, nSamplesPerTrial,2))
    erprMatrix[:] = np.nan
    gazeMatrix[:] = np.nan

    for uniqueMsg in eventDict["UNIQUE_MSG"]:
        erprDict[uniqueMsg] = np.zeros(
            (eventDict["N_UNIQUE_MSG"][countUniqueMsg], nSamplesPerTrial)
        )
        erprDict[uniqueMsg][:] = np.nan
        countUniqueMsg += 1

    for i in range(nTrials):
        curTimeStamp = eventDict["TIMESTAMPS"][i]

        findTimeStampInds = np.where(npTimeStamps >= curTimeStamp)[0]
        if len(findTimeStampInds) == 0:  # timestamp not found
            # logger.log(
            #     "No sample and timestamp found for MSG: "
            #     + eventDict["MSG"][i]
            #     + " with timestamp: "
            #     + str(curTimeStamp)
            # )
            logger.log("Min timestamp for samples: " + str(np.min(npTimeStamps)))
            logger.log("Max timestamp for samples: " + str(np.max(npTimeStamps)))
        else:
            startInd = findTimeStampInds[0]  # take first
            endInd = startInd + nSamplesPerTrial
            if endInd > len(npPupilData):
                tempPupilData = npPupilData[startInd:]
                tempGazeXData = npXgazeData[startInd:]
                tempGazeYData = npYgazeData[startInd:]
                tempFinData   = finPupilSamples[startInd:]
            else:
                tempPupilData = npPupilData[startInd:endInd]
                tempGazeXData = npXgazeData[startInd:endInd]
                tempGazeYData = npYgazeData[startInd:endInd]
                tempFinData   = finPupilSamples[startInd:endInd]

            if binAnalysis:
                uniqueMsgIND = np.where(eventDict["MSG"][i] >= binEdges)[0][-1]
            else:
                uniqueMsgIND = eventDict["UNIQUE_MSG"].index(eventDict["MSG"][i])

            # print(len(tempPupilData))
            erprDict[eventDict["UNIQUE_MSG"][uniqueMsgIND]][
                eventDict["COUNT_UNIQUE_MSG"][uniqueMsgIND], 0 : len(tempPupilData)
            ] = tempPupilData

            erprMatrix[i, 0 : len(tempPupilData)] = tempPupilData
            gazeMatrix[i, 0 : len(tempPupilData),0] = tempGazeXData
            gazeMatrix[i, 0 : len(tempPupilData),1] = tempGazeYData
            nFinSamplesPerTrial.append(np.sum(tempFinData))

            eventDict["COUNT_UNIQUE_MSG"][uniqueMsgIND] = (
                eventDict["COUNT_UNIQUE_MSG"][uniqueMsgIND] + 1
            )

    # return variables
    nFinSamplesPerTrial = np.array(nFinSamplesPerTrial)
    
    return eyelinkDataTemp["SETTINGS"], sampleDict, eventDict, erprMatrix, erprDict, gazeMatrix, nFinSamplesPerTrial


def removeEyelinkBlinks(
    timeStamps,
    pData,
    xData,
    yData,
    blinkDetectionThreshold=[4, 2],
    dilutionSize=[20, 40],
    consecutiveRemovalPeriod=20,
    showTimeTraces=False,
):
    """


    dilutionSize is list with two integers indicating number of samples that need to be
    removed [before, after] a blink period.

    # TO DO: convert to ms instead of # samples
    """
    # pData = eyelinkDataTemp["SAMPLES"]["PUPIL-SIZE-LEFT"]
    # timeStamps = np.array(timeStamps, dtype="float32")
    # pData = np.array(pData, dtype="float32")
    timeStamps = np.array(timeStamps, dtype="float64")
    pData = np.array(pData, dtype="float64")
    xData = np.array(xData, dtype="float64")
    yData = np.array(yData, dtype="float64")

    # TO DO: also interpolate X and Y gaze periods

    if showTimeTraces:

        from matplotlib import pyplot as plt

        plt.figure(figsize=(20, 5), dpi=200)
        plt.subplot(4, 1, 1)
        plt.plot(pData)

    # filter out blink periods (blink periods contain value 0; set at 2 just in case)
    selVect = pData < 2
    pData[selVect] = np.nan
    xData[selVect] = np.nan
    yData[selVect] = np.nan

    # filter out blinks based on speed changes
    # std [below above] mean
    pdata = pData.copy()
    xdata = xData.copy()
    ydata = yData.copy()
    pdiff = np.diff(pdata)  # difference between consecutive pupil sizes

    # remove blink periods
    # for selectIdx in range(len(pData)):
    # create blinkspeed threshold 4SD below the mean
    blinkSpeedThreshold = np.nanmean(pdiff) - (blinkDetectionThreshold[0] * np.nanstd(pdiff))

    # create blinkspeed threshold 2SD above the mean
    blinkSpeedThreshold2 = np.nanmean(pdiff) + (blinkDetectionThreshold[1] * np.nanstd(pdiff))

    # blink window containing minimum and maximum value
    blinkWindow = [-dilutionSize[0], dilutionSize[1]]
    blinkWindow2 = [-dilutionSize[1], dilutionSize[0]]

    blinkIdx = np.where(
        pdiff < blinkSpeedThreshold
    )  # find where the pdiff is smaller than the lower blinkspeed threshold
    blinkIdx = blinkIdx[0]
    blinkIdx = blinkIdx[np.where(np.diff(blinkIdx) > consecutiveRemovalPeriod)[0]]
    blinkIdx = np.insert(blinkIdx, 0, 0, axis=0)

    blinkIdx2 = np.where(
        pdiff > blinkSpeedThreshold2
    )  # find where the pdiff is larger than the upper blinkspeed threshold
    blinkIdx2 = blinkIdx2[0]
    blinkIdx2 = blinkIdx2[np.where(np.diff(blinkIdx2) > consecutiveRemovalPeriod)[0]]
    blinkIdx2 = np.insert(blinkIdx2, 0, 0, axis=0)

    # remove blinks that fall outside the blinkwindow (beginning and end of pupil traces)
    # blinkIdx(blinkIdx + blinkWindow[1] > len(pdata)-1) = np.nan

    # remove blink segments
    for bl in blinkIdx:
        # print(bl)
        # if bl + blinkWindow[0] < 0:
        #     from matplotlib import pyplot as plt

        #     plt.plot(pdiff[:1000])

        pdata[np.arange(bl + blinkWindow[0], bl + blinkWindow[1])] = np.nan
        xdata[np.arange(bl + blinkWindow[0], bl + blinkWindow[1])] = np.nan
        ydata[np.arange(bl + blinkWindow[0], bl + blinkWindow[1])] = np.nan
        pdiff[np.arange(bl + blinkWindow[0], bl + blinkWindow[1])] = np.nan

    for bl in blinkIdx2:
        # print(bl)
        # if bl + blinkWindow2[0] < 0:
        #     from matplotlib import pyplot as plt

        #     plt.plot(pdiff[:1000])

        pdata[np.arange(bl + blinkWindow2[0], bl + blinkWindow2[1])] = np.nan
        xdata[np.arange(bl + blinkWindow2[0], bl + blinkWindow2[1])] = np.nan
        ydata[np.arange(bl + blinkWindow2[0], bl + blinkWindow2[1])] = np.nan
        pdiff[np.arange(bl + blinkWindow2[0], bl + blinkWindow2[1])] = np.nan



    return timeStamps, pdata,xdata,ydata


def interpolateBlinks(tdata, pdata, xdata, ydata):
    

    # interpolate blink periods
    # find the first element in the array that isn't NaN
    pdataFilt = pdata.copy()
    pdataFilt = np.where(np.isfinite(pdataFilt))[0][0]

    # find the last element in the array that isn't NaN
    pdataFilt2 = pdata.copy()
    pdataFilt2 = np.where(np.isfinite(pdataFilt2))[0][-1]
    pdata = pdata[pdataFilt:pdataFilt2]
    xdata = xdata[pdataFilt:pdataFilt2]
    ydata = ydata[pdataFilt:pdataFilt2]
    tdata = tdata[pdataFilt:pdataFilt2]

    missDataIdx = np.where(~np.isfinite(pdata))[0]
    corrDataIdx = np.where(np.isfinite(pdata))[0]

    # ## PCHIP interpolation
    # from matplotlib import pyplot as plt

    # plt.figure()
    # plt.plot(timeStampsFilt[corrDataIdx])
    # print(timeStampsFilt[corrDataIdx])
    # print(np.diff(timeStampsFilt[corrDataIdx]))
    # print(np.sum(np.diff(timeStampsFilt[corrDataIdx]) < 0))

    # notIncrIdx = np.where(np.diff(timeStampsFilt[corrDataIdx]) == 0)
    # print(notIncrIdx)
    # notIncrIdx = notIncrIdx

    # print(np.shape(notIncrIdx))

    # corrDataIdx = corrDataIdx[notIncrIdx == False]
    # timeStampsFilt[corrDataIdx] notIncrIdx

    # print(timeStampsFilt)

    pdata[missDataIdx] = PchipInterpolator(tdata[corrDataIdx], pdata[corrDataIdx])(
        tdata[missDataIdx]
    )
    xdata[missDataIdx] = PchipInterpolator(tdata[corrDataIdx], xdata[corrDataIdx])(
        tdata[missDataIdx]
    )
    ydata[missDataIdx] = PchipInterpolator(tdata[corrDataIdx], ydata[corrDataIdx])(
        tdata[missDataIdx]
    )
    # linear interpolation
    # pdata[missDataIdx] = interp1d(tdataFilt[corrDataIdx], pdata[corrDataIdx], kind='linear')(timeStampsFilt[missDataIdx])

    # if showFilteredTraces == 1:

    # if showTimeTraces:
    #     plt.subplot(4, 1, 4)
    #     plt.plot(tdataFilt, pdata)

    
    return tdata,pdata,xdata,ydata