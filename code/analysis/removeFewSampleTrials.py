if removeFewSampleTrials:

    nFinSamplesPerTrialTemp = nFinSamplesPerTrial.copy()
    sampThres = np.median(nFinSamplesPerTrialTemp)-outlierThres*np.std(nFinSamplesPerTrialTemp)
    selEnoughSamplesTrials = nFinSamplesPerTrialTemp>sampThres

    allPercFewSamplesTrials.append(100*np.sum(~selEnoughSamplesTrials)/len(selEnoughSamplesTrials))
    
    cueProbeSOA = cueProbeSOA[selEnoughSamplesTrials]
    cueTargetSOA = cueTargetSOA[selEnoughSamplesTrials]
    probeLocation = probeLocation[selEnoughSamplesTrials]
    cueLocation = cueLocation[selEnoughSamplesTrials]
    targetDirection = targetDirection[selEnoughSamplesTrials]
    targetLocation = targetLocation[selEnoughSamplesTrials]
    nFinSamplesPerTrial = nFinSamplesPerTrial[selEnoughSamplesTrials]

    erprMatrix = erprMatrix[selEnoughSamplesTrials,:]
    erprMatrixBaseSubt = erprMatrixBaseSubt[selEnoughSamplesTrials, :]
    respBase = respBase[selEnoughSamplesTrials]
    pupilERPRAmp = pupilERPRAmp[selEnoughSamplesTrials]
    pupilPostERPRMean = pupilPostERPRMean[selEnoughSamplesTrials]
    pupilERPRArea = pupilERPRArea[selEnoughSamplesTrials]
    gazeMatrix = gazeMatrix[selEnoughSamplesTrials,:,:]
    gazeMatrixBaseSubt = gazeMatrixBaseSubt[selEnoughSamplesTrials,:,:] # not yet made
    gazeMatrixPolar = gazeMatrixPolar[selEnoughSamplesTrials,:,:]


    print('Removed ' + str(np.sum(~selEnoughSamplesTrials)) + '/' + str(len(selEnoughSamplesTrials)) + ' trials with not enough samples (<' + str(sampThres) + ')')
    print('Removed ' + str(100*np.sum(~selEnoughSamplesTrials)/len(selEnoughSamplesTrials)) + '% trials with not enough samples (<' + str(sampThres) + ')')

    if showPlotsPerObs:
        plt.figure()
        plt.subplot(2,2,1)
        plt.plot(nFinSamplesPerTrialTemp)
        plt.plot([0,len(nFinSamplesPerTrial)],[sampThres,sampThres])