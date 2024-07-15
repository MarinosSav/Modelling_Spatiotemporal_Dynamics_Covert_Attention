if len(nYBinsArr) == len(nXBinsArr):
    pass
else:
    print("ERROR: nYBinsArr has not the same length as nXBinsArr")

erprAmp2DMap = np.zeros((np.max(nYBinsArr), np.max(nXBinsArr), len(nYBinsArr)))
erprAmp2DMap[:] = np.nan


countBinArray = -1
for nYBins in nYBinsArr:
    countBinArray = countBinArray + 1
    nXBins = nXBinsArr[countBinArray]

    countBEdgesX = np.linspace(0, np.max(nXBinsArr), nXBins + 1).astype(int)
    countBEdgesY = np.linspace(0, np.max(nYBinsArr), nYBins + 1).astype(int)

    binEdgesX = np.linspace(
        np.min(probeLocationRot[:, 0]), np.max(probeLocationRot[:, 0]), nXBins + 1
    )
    binCentersX = binEdgesX[0:-1] + np.diff(binEdgesX) / 2
    binEdgesX[-1] = binEdgesX[-1] + 1  # to compensate last bin edge <=

    binEdgesY = np.linspace(
        np.min(probeLocationRot[:, 1]), np.max(probeLocationRot[:, 1]), nYBins + 1
    )
    binCentersY = binEdgesY[0:-1] + np.diff(binEdgesY) / 2
    binEdgesY[-1] = binEdgesY[-1] + 1  # to compensate last bin edge <=

    countBX = -1
    for bX in range(len(binCentersX)):
        countBY = -1
        countBX = countBX + 1
        countBRangeX = [countBEdgesX[countBX], countBEdgesX[countBX + 1]]
        for bY in range(len(binCentersY)):
            countBY = countBY + 1
            countBRangeY = [countBEdgesY[countBY], countBEdgesY[countBY + 1]]

            selXVect = (probeLocationRot[:, 0] >= binEdgesX[bX]) & (
                probeLocationRot[:, 0] < binEdgesX[bX + 1]
            )
            selYVect = (probeLocationRot[:, 1] >= binEdgesY[bY]) & (
                probeLocationRot[:, 1] < binEdgesY[bY + 1]
            )

            tempErprAmp = mainVarPupilData[(selXVect) & (selYVect)]
            if np.sum((selXVect) & (selYVect)) == 0:  # weird but no data in this bin
                pass
            elif np.sum((selXVect) & (selYVect)) == 1:  # weird but only one trial in this bin
                erprAmp2DMap[
                    countBRangeY[0] : countBRangeY[1],
                    countBRangeX[0] : countBRangeX[1],
                    countBinArray,
                ] = tempErprAmp
            else:
                erprAmp2DMap[
                    countBRangeY[0] : countBRangeY[1],
                    countBRangeX[0] : countBRangeX[1],
                    countBinArray,
                ] = np.median(tempErprAmp)

erprAmp2DMap_ave = np.nanmedian(erprAmp2DMap, axis=2)

if showPlotsPerObs:

    plt.figure(figsize=(10, 5))
    plt.imshow(erprAmp2DMap_ave, cmap=colormapper)
    plt.xlabel("Horizontal position")
    plt.ylabel("Vertical position")
    plt.xticks(range(0, nXBins), labels=binCentersX[0:nXBins].astype(int))
    plt.yticks(range(0, nYBins), labels=binCentersY[0:nYBins].astype(int))
    plt.xlim([-1, nXBins])
    plt.ylim([-1, nYBins])
    plt.colorbar(label="AUC Pupil Response")
    plt.title("Pooled across time")

    # plt.xticks(range(0, 10, 2), labels=binCenters[0:10:2].astype(int))
    # plt.ylabel("Pupil response amplitude [au]")
    # plt.tight_layout()
    plt.show()
