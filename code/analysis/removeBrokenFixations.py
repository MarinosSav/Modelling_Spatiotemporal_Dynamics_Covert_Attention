if removeBrokenFixationTrials:

    medgazeRad = np.nanmedian(gazeMatrixPolar[:,:,0],axis=1)
    # stdgazeRad = np.nanstd(gazeMatrixPolar[:,:,0],axis=1)
    
    gazeRadThres = np.nanmedian(medgazeRad)+outlierThres*np.nanstd(medgazeRad)
    selGoodFixTrials = medgazeRad<gazeRadThres

    allPercBadFixTrials.append(100*np.sum(~selGoodFixTrials)/len(selGoodFixTrials))
    
    cueProbeSOA = cueProbeSOA[selGoodFixTrials]
    cueTargetSOA = cueTargetSOA[selGoodFixTrials]
    probeLocation = probeLocation[selGoodFixTrials]
    cueLocation = cueLocation[selGoodFixTrials]
    targetDirection = targetDirection[selGoodFixTrials]
    targetLocation = targetLocation[selGoodFixTrials]
    nFinSamplesPerTrial = nFinSamplesPerTrial[selGoodFixTrials]

    erprMatrix = erprMatrix[selGoodFixTrials,:]
    erprMatrixBaseSubt = erprMatrixBaseSubt[selGoodFixTrials, :]
    respBase = respBase[selGoodFixTrials]
    pupilERPRAmp = pupilERPRAmp[selGoodFixTrials]
    pupilPostERPRMean = pupilPostERPRMean[selGoodFixTrials]
    pupilERPRArea = pupilERPRArea[selGoodFixTrials]
    gazeMatrix = gazeMatrix[selGoodFixTrials,:,:]
    gazeMatrixBaseSubt = gazeMatrixBaseSubt[selGoodFixTrials,:,:]
    gazeMatrixPolar = gazeMatrixPolar[selGoodFixTrials,:,:]


    print('Removed ' + str(np.sum(~selGoodFixTrials)) + '/' + str(len(selGoodFixTrials)) + ' trials with deviation fixation radius (>' + str(gazeRadThres) + 'pix)' )
    print('Removed ' + str(100*np.sum(~selGoodFixTrials)/len(selGoodFixTrials)) + '% trials with deviation fixation radius (>' + str(gazeRadThres) + 'pix)')

    if showPlotsPerObs:
        plt.figure()
        plt.subplot(2,2,1)
        plt.plot(medgazeRad)
        plt.plot([0,len(medgazeRad)],[gazeRadThres, gazeRadThres])


