corrTrialsVect = targetID == responseID
accuracy = np.sum(targetID == responseID) / len(responseID)
allAccuracy.append(accuracy)

print("Accuracy: " + str(accuracy))

if inclOnlyCorrectTrials:
    cueProbeSOA = cueProbeSOA[corrTrialsVect]
    cueTargetSOA = cueTargetSOA[corrTrialsVect]
    probeLocation = probeLocation[corrTrialsVect]
    cueLocation = cueLocation[corrTrialsVect]
    targetDirection = targetDirection[corrTrialsVect]
    targetLocation = targetLocation[corrTrialsVect]
    nFinSamplesPerTrial = nFinSamplesPerTrial[corrTrialsVect]

    erprMatrix = erprMatrix[corrTrialsVect,:]
    erprMatrixBaseSubt = erprMatrixBaseSubt[corrTrialsVect, :]
    pupilERPRArea = pupilERPRArea[corrTrialsVect]
    respBase = respBase[corrTrialsVect]
    pupilERPRAmp = pupilERPRAmp[corrTrialsVect]
    pupilPostERPRMean = pupilPostERPRMean[corrTrialsVect]
    gazeMatrix = gazeMatrix[corrTrialsVect,:,:]
    gazeMatrixBaseSubt = gazeMatrixBaseSubt[corrTrialsVect,:,:]
    gazeMatrixPolar = gazeMatrixPolar[corrTrialsVect,:,:]

