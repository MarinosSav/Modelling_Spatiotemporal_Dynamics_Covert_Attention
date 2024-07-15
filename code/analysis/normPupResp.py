if normType == None:
    pupilUnit = "[au]"
    normTypeName = "None"
elif normType == "z":
    erprMatrixBaseSubt = erprMatrixBaseSubt / np.nanstd(erprMatrixBaseSubt[:])
    respBase = (respBase - np.nanmean(respBase)) / np.nanstd(respBase)
    pupilERPRAmp = (pupilERPRAmp - np.nanmean(pupilERPRAmp)) / np.nanstd(pupilERPRAmp)
    pupilPostERPRMean = (pupilPostERPRMean - np.nanmean(pupilPostERPRMean)) / np.nanstd(
        pupilPostERPRMean
    )
    pupilERPRArea = (pupilERPRArea - np.nanmean(pupilERPRArea)) / np.nanstd(pupilERPRArea)
    mainVarPupilData = (mainVarPupilData - np.nanmean(mainVarPupilData)) / np.nanstd(
        mainVarPupilData
    )
    pupilUnit = "[z]"
    normTypeName = normType
else:
    print("normType not recognized")
    pupilUnit = "[?]"
    normTypeName = normType