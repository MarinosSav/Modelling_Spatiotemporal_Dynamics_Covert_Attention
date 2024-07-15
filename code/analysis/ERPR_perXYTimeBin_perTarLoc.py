erprAmp2DMapPerTimeBinPerTarLoc = np.zeros(
    (
        np.max(nYBinsArr),
        np.max(nXBinsArr),
        np.max(nTimeBinsArr),
        nTarLocs,
        len(nYBinsArr) * len(nTimeBinsArr),
    )
)
erprAmp2DMapPerTimeBinPerTarLoc[:] = np.nan

CountBinArrayXYT = -1
countBinArrayTime = -1
for nTimeBins in nTimeBinsArr:
    countBinArrayTime = countBinArrayTime + 1
    countBEdgesT = np.linspace(0, np.max(nTimeBinsArr), nTimeBins + 1).astype(int)

    binEdgesT = np.linspace(0, 400, nTimeBins + 1)
    binCentersT = binEdgesT[0:-1] + np.diff(binEdgesT) / 2
    binEdgesT[-1] = binEdgesT[-1] + 1  # to compensate last bin edge <=

    CountBinArrayXY = -1
    for nYBins in nYBinsArr:
        CountBinArrayXY = CountBinArrayXY + 1
        CountBinArrayXYT = CountBinArrayXYT + 1

        nXBins = nXBinsArr[CountBinArrayXY]

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
                countBT = -1
                countBY = countBY + 1
                countBRangeY = [countBEdgesY[countBY], countBEdgesY[countBY + 1]]
                for bT in range(len(binCentersT)):
                    countBT = countBT + 1

                    countBRangeT = [countBEdgesT[countBT], countBEdgesT[countBT + 1]]

                    selXVect = (probeLocationRot[:, 0] >= binEdgesX[bX]) & (
                        probeLocationRot[:, 0] < binEdgesX[bX + 1]
                    )
                    selYVect = (probeLocationRot[:, 1] >= binEdgesY[bY]) & (
                        probeLocationRot[:, 1] < binEdgesY[bY + 1]
                    )
                    selCueProbeSOAVect = (cueProbeSOA >= binEdgesT[bT]) & (
                        cueProbeSOA < binEdgesT[bT + 1]
                    )

                    for TL in range(nTarLocs):
                        selTLVect = targetLocation == tarLocNames[TL]
                                    
                        tempErprAmp = mainVarPupilData[(selXVect) & (selYVect) & (selCueProbeSOAVect) & (selTLVect)]
                        if (
                            np.sum((selXVect) & (selYVect) & (selCueProbeSOAVect) & (selTLVect)) == 0
                        ):  # weird but no data in this bin
                            pass
                        elif (
                            np.sum((selXVect) & (selYVect) & (selCueProbeSOAVect) & (selTLVect)) == 1
                        ):  # only one trial in this binned position and time point and target location
                            erprAmp2DMapPerTimeBinPerTarLoc[
                                countBRangeY[0] : countBRangeY[1],
                                countBRangeX[0] : countBRangeX[1],
                                countBRangeT[0] : countBRangeT[1],
                                TL,
                                CountBinArrayXYT,
                            ] = tempErprAmp
                        else:
                            erprAmp2DMapPerTimeBinPerTarLoc[
                                countBRangeY[0] : countBRangeY[1],
                                countBRangeX[0] : countBRangeX[1],
                                countBRangeT[0] : countBRangeT[1],
                                TL,
                                CountBinArrayXYT,
                            ] = np.median(tempErprAmp)

erprAmp2DMapPerTimeBinPerTarLoc_ave = np.nanmedian(erprAmp2DMapPerTimeBinPerTarLoc, axis=4)

pickleFileName = (
    "data/"
    + ppName
    + "_perXYTimeBinPerTarLoc_sess-"
    + str(sessNum)
    + "_norm-"
    + normTypeName
    + "_mes-"
    + varOfInterestAbr
    + "_onset-"
    + onsetEvent
    + ".pickle"
)
with open(pickleFileName, "wb") as handle:
    pickle.dump(
        [erprAmp2DMapPerTimeBinPerTarLoc_ave, binCentersX, binCentersY, binEdgesT, tarLocNames],
        handle,
        protocol=pickle.HIGHEST_PROTOCOL,
    )
    handle.close()

if showPlotsPerObs:

    varCBarRange = [
        np.percentile(erprAmp2DMapPerTimeBinPerTarLoc_ave[:], 1),
        np.percentile(erprAmp2DMapPerTimeBinPerTarLoc_ave[:], 99),
    ]

    plt.figure(figsize=(5* nTimeBins, 5* nTarLocs))
    countBT = -1
    for bT in range(np.max(nTimeBinsArr)):
        countBT = countBT + 1

        for TL in range(nTarLocs):
            plt.subplot(np.max(nTimeBinsArr), nTarLocs, (countBT*nTarLocs) + 1 + TL)
            plt.imshow(
                erprAmp2DMapPerTimeBinPerTarLoc_ave[:, :, countBT,TL],
                cmap=colormapper,
                vmin=varCBarRange[0],
                vmax=varCBarRange[1],
            )
            plt.xticks(range(0, nXBins), labels=binCentersX[0:nXBins].astype(int))
            plt.yticks(range(0, nYBins), labels=binCentersY[0:nYBins].astype(int))
            plt.xlim([-1, nXBins])
            plt.ylim([-1, nYBins])

            plt.xlabel("Horizontal position")
            plt.ylabel("Vertical position")
            plt.title("SOA: " + str(int(binEdgesT[bT])) + "-" + str(int(binEdgesT[bT + 1])) + "-" + tarLocNames[TL])

            # plt.colorbar(label="AUC Pupil Response")

    # plt.xticks(range(0, 10, 2), labels=binCenters[0:10:2].astype(int))
    # plt.ylabel("Pupil response amplitude [au]")

    plt.tight_layout()
    plt.show()
