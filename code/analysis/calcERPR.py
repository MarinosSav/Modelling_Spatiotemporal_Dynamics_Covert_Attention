pickleFileName = (
    "data/"
    + ppName
    + "_erpr_sess-"
    + str(sessNum)
    + "_onset-"
    + onsetEvent
    + ".pickle"
)
if (os.path.isfile(pickleFileName)) & (overwriteERPR==False):
    if suppressFileLoadMsg==False:
        logger.log(
            "Pickle file with ERPR data already exists. Opening data in existing pickle file:"
        )
        logger.log(pickleFileName)
    with open(pickleFileName, "rb") as handle:
        [eyelinkSettings, sampleDict, eventDict, erprMatrix, gazeMatrix, nFinSamplesPerTrial] = pickle.load(handle)
        handle.close()
else:
    logger.log(
        "Calculating ERPR and gaze data"
    )

    eyelinkSettings, sampleDict, eventDict, erprMatrix, erprDict, gazeMatrix, nFinSamplesPerTrial = erpr(
        eyelinkData,
        "left",
        eventMsgs=["start_" + onsetEvent],
        eventDict=None,
        eventWindowSize=2,
        removeBlinks=interpBlinks,
        showTraces=showPupilTraces,
    )
    with open(pickleFileName, "wb") as handle:
        pickle.dump(
            [eyelinkSettings, sampleDict, eventDict, erprMatrix, gazeMatrix, nFinSamplesPerTrial],
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
        handle.close()
