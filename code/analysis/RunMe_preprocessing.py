# TO DO: remove effect of baseline pupil size on pupil response amplitude
# TO DO: Save plots per subject
# TO DO: improve blink removal script. Few blinks present in data after 1500ms after cue onset

exec(open("import_packages.py").read())
exec(open("import_functions.py").read())

psychPyFilesDir = "..//Results//PsychopyFiles"  # relative to eyelink files

inclOnlyCorrectTrials       = True
removeOutliersOverall       = True  # check all data for outliers
removeBrokenFixationTrials  = True  # remove trials in which observers moved away from fixation
removeFewSampleTrials       = True # remove trials with only few samples (a lot of nans due to blinking)
outlierThres                = 4  # z value threshold to remove outliers

removeEccentricityEffect    = True # remove effect that pupil amplitudes decrease as a function of probe eccentricity

varOfInterest               = "ERPR Amplitude"  # 'ERPR Area' or 'ERPR Amplitude' or "Post ERPR Amplitude"
# varOfInterest = "ERPR Area"  # 'ERPR Area' or 'ERPR Amplitude' or "Post ERPR Amplitude"
onsetEvent                  = "cue"  # 'probe' or 'cue'
normType                    = "z"  # None, 'z'

showPlotsPerObs             = False  # show plots per individual
eyelinkFilesDir             = "C:/Users/marin/desktop/new1/Results/EyelinkFiles"
overwriteERPR               = False # recalculate erpr data
suppressFileLoadMsg         = True # do not show prints of "loading file "

#%%  make logger available in all classes and instances
logger = logger()
setattr(__builtins__, "logger", logger)  # to set logger in class/functions

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# convert eyelink data file to ascii files and parse data to pickle data files

initialDir = eyelinkFilesDir
exec(open("parseEyelinkDataFiles.py").read())


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# loop through each edf file and analyze pupil responses per presentation

allPPName = []
allSessNum = []
allNOutliers = []
allPercOutliers = []
allAccuracy = []
allPercBadFixTrials = []
allPercFewSamplesTrials = []

countPP = 0
for edfFileNameWithPath in edfFileNamesWithPath:
    countPP = countPP + 1
    pathFolders, __, __, base_filename, __ = splitPathFilenameComponents(edfFileNameWithPath)

    if base_filename[0] == "t":  # test participant
        partType = 0
    elif base_filename[0] == "s":  # real participant
        partType = 1
    else:
        print("WARNING unrecognized participant type: " + base_filename)

    partNum = int(base_filename.split("_")[0][1:])
    ppName = base_filename[0] + str(partNum)
    sessNum = int(base_filename.split("_")[2])
    pickleFileName = pathFolders + os.sep + base_filename + ".pickle"

    with open(pickleFileName, "rb") as handle:
        eyelinkData = pickle.load(handle)
    handle.close()

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # ERPR calc
    exec(open("calcERPR.py").read())

    # subtract baseline at start of each pupil/gaze trace
    exec(open("baseLineSubtr.py").read())
    exec(open("inspectPupilTraces.py").read())

    # calculate pupil properties (e.g., amplitude)
    exec(open("calcPupRespProps.py").read())

    # load psychopy csv file with info about trial conditions
    exec(open("loadTrialConditions.py").read()) 
    exec(open("calcTargetLocation.py").read()) # creates variable targetLocation
    
    # remove bad trials 
    exec(open("removeIncorrectTrials.py").read()) # calculates accuracy and removes incorrect trials
    exec(open("removeFewSampleTrials.py").read()) # see nFinSamplesPerTrial
    exec(open("removeBrokenFixations.py").read()) # see gazeMatrix

    # select main dependent variable of interest; produces mainVarPupilData --> needs to be done before removeOutliers!
    exec(open("selPupilDepVar.py").read()) 

    # removes outliers in pupil responses (main variable)
    exec(open("removeOutliers.py").read()) 

    # ROTATE SPACE OF probe location depending on direction of cue
    exec(open("probeLocationRot.py").read())

    # remove confounding effect of eccentricity on pupil response amplitudes
    exec(open("removeEccentricityEffect.py").read())

    # remove confounding effect of top-down anisotropy on pupil response amplitudes
    exec(open("removeHoriVeriEffect.py").read())

    # normalize pupil responses (e.g., to z-values) by setting variable normType
    exec(open("normPupResp.py").read())

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    # PLOT Pupil response per binned SOA between cue and probe
    nTimeBinsArr = [2, 3, 4, 5]
    exec(open("ERPR_perTimeBin.py").read())

    # Plot ERPR per binned location (colormap) in horizontal space between cue and target
    nXBinsArr = [2, 4, 6, 8, 10, 12, 14, 16]
    # nXBinsArr = [14, 16]
    exec(open("ERPR_perXBin.py").read())
    exec(open("ERPR_perXTimeBin.py").read())

    # Plot ERPR per binned location (colormap) in vertical space
    nYBinsArr = [1, 2, 3, 4, 5, 6, 7, 8]  # must be same length as nXBinsArr
    # nYBinsArr = [7,8]  # must be same length as nXBinsArr
    exec(open("ERPR_perYBin.py").read())
    exec(open("ERPR_perYTimeBin.py").read())

    # Plot ERPR amplitude (colormap) per binned 2D location (horizontal and vertical space)
    # 1200 trials, 10x5 resolution = 24 trials per 2D location
    exec(open("ERPR_perXYBin.py").read())

    # Plot ERPR amplitude (colormap) per 2D location and per binned cue-probe SOA (subplots)
    exec(open("ERPR_perXYTimeBin.py").read())

    # Plot ERPR amplitude (colormap) per 2D location and per binned cue-probe SOA (subplots)
    # AND PER TARGET LOCATION
    tarLocNames = ['UP','RIGHT','DOWN','LEFT']
    nTarLocs = len(tarLocNames)
    exec(open("ERPR_perXYTimeBin_perTarLoc.py").read())

    # rerun with more time bins
    nTimeBinsArr = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    exec(open("ERPR_perXYTimeBin_highRes.py").read())

    # Plot ERPR per binned location in horizontal space across time
    # BASED ON 2D XY MAPS
    # exec(open("ERPR_PerXBin_timeLapse.py").read())

    # Time lapse video
    # THIS FOLLOWING CODE IS NOT (YET) AS GOOD TIME INTERPOLATED AS in RunMe_statistics
    # exec(open("ERPR_timeLapse.py").read())

    allPPName.append(ppName)
    allSessNum.append(sessNum)


print('Accuracy fraction - M=' + str(np.mean(allAccuracy)) + ', SD = ' + str(np.std(allAccuracy)))
allAccuracy = np.array(allAccuracy)
print('Accuracy fraction >.75 - M=' + str(np.mean(allAccuracy[allAccuracy>.75])) + ', SD = ' + str(np.std(allAccuracy[allAccuracy>.75])))

plt.figure()
plt.subplot(2,2,1)
plt.violinplot(allAccuracy,showmeans=True)
plt.plot(np.zeros(len(allAccuracy))+np.random.uniform(0.95,1.05,len(allAccuracy)),allAccuracy,'.',alpha=0.5,markersize=20)
plt.ylabel('Accuracy')

plt.subplot(2,2,2)
plt.violinplot(allAccuracy[allAccuracy>.75],showmeans=True)
plt.plot(np.zeros(len(allAccuracy[allAccuracy>.75]))+np.random.uniform(0.95,1.05,len(allAccuracy[allAccuracy>.75])),allAccuracy[allAccuracy>.75],'.',alpha=0.5,markersize=20)
plt.ylabel('Accuracy')
plt.title('Only acc > .75')
plt.tight_layout()

print('Removed percentage trials with few samples - M=' + str(np.mean(allPercFewSamplesTrials)) + ', SD = ' + str(np.std(allPercFewSamplesTrials)))
print('Removed percentage trials with bad fixations - M=' + str(np.mean(allPercBadFixTrials)) + ', SD = ' + str(np.std(allPercBadFixTrials)))
# print('Removed # outliers - M=' + str(np.mean(allNOutliers)) + ', SD = ' + str(np.std(allNOutliers)))
print('Removed percentage trials with outliers - M=' + str(np.mean(allPercOutliers)) + ', SD = ' + str(np.std(allPercOutliers)))



pickleFileName = "data/allVars.pickle"
with open(pickleFileName, "wb") as handle:
    pickle.dump(
        [allPPName, allSessNum, allNOutliers, allAccuracy, allPercFewSamplesTrials, allPercBadFixTrials],
        handle,
        protocol=pickle.HIGHEST_PROTOCOL,
    )
    handle.close()

print("done")

logger.closeLog()
