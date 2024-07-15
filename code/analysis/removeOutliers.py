if removeOutliersOverall:

    noOutlierVect = (
        mainVarPupilData
        < (np.nanmedian(mainVarPupilData) + outlierThres * np.nanstd(mainVarPupilData))
    ) & (
        mainVarPupilData
        > (np.nanmedian(mainVarPupilData) - outlierThres * np.nanstd(mainVarPupilData))
    )

    nOutliers = np.sum(noOutlierVect == False)

    allNOutliers.append(nOutliers)
    allPercOutliers.append(100*nOutliers/len(noOutlierVect))
    
    print(
        "Removing "
        + str(nOutliers)
        + "//"
        + str(len(noOutlierVect))
        + " outliers - "
        + str(np.round(100 * nOutliers / len(noOutlierVect), 2))
        + "%"
    )

    cueProbeSOA = cueProbeSOA[noOutlierVect]
    cueTargetSOA = cueTargetSOA[noOutlierVect]
    probeLocation = probeLocation[noOutlierVect]
    cueLocation = cueLocation[noOutlierVect]
    targetDirection = targetDirection[noOutlierVect]
    targetLocation = targetLocation[noOutlierVect]
    nFinSamplesPerTrial = nFinSamplesPerTrial[noOutlierVect]

    erprMatrix = erprMatrix[noOutlierVect,:]
    erprMatrixBaseSubt = erprMatrixBaseSubt[noOutlierVect, :]
    respBase = respBase[noOutlierVect]
    pupilERPRAmp = pupilERPRAmp[noOutlierVect]
    pupilPostERPRMean = pupilPostERPRMean[noOutlierVect]
    pupilERPRArea = pupilERPRArea[noOutlierVect]
    gazeMatrix = gazeMatrix[noOutlierVect,:,:]
    gazeMatrixBaseSubt = gazeMatrixBaseSubt[noOutlierVect,:,:]
    gazeMatrixPolar = gazeMatrixPolar[noOutlierVect,:,:]

    mainVarPupilData = mainVarPupilData[noOutlierVect]