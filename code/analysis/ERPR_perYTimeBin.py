# TO DO: flip figure upside down? because no bottom is near fovea and top near periphery

erprAmpPerYTimeBin = np.zeros(
    (
        np.max(nYBinsArr),
        np.max(nTimeBinsArr),
        len(nYBinsArr) * len(nTimeBinsArr),
    )
)
erprAmpPerYTimeBin[:] = np.nan

CountBinArrayYT = -1
countBinArrayTime = -1
for nTimeBins in nTimeBinsArr:
    countBinArrayTime = countBinArrayTime + 1
    countBEdgesT = np.linspace(0, np.max(nTimeBinsArr), nTimeBins + 1).astype(int)

    binEdgesT = np.linspace(0, 400, nTimeBins + 1)
    binCentersT = binEdgesT[0:-1] + np.diff(binEdgesT) / 2
    binEdgesT[-1] = binEdgesT[-1] + 1  # to compensate last bin edge <=

    CountBinArrayY = -1
    for nYBins in nYBinsArr:
        CountBinArrayY = CountBinArrayY + 1
        CountBinArrayYT = CountBinArrayYT + 1

        nYBins = nYBinsArr[CountBinArrayY]

        countBEdgesY = np.linspace(0, np.max(nYBinsArr), nYBins + 1).astype(int)

        binEdgesY = np.linspace(
            np.min(probeLocationRot[:, 0]), np.max(probeLocationRot[:, 0]), nYBins + 1
        )
        binCentersY = binEdgesY[0:-1] + np.diff(binEdgesY) / 2
        binEdgesY[-1] = binEdgesY[-1] + 1  # to compensate last bin edge <=

        countbY = -1
        for bY in range(len(binCentersY)):
            countbY = countbY + 1
            countBRangeY = [countBEdgesY[countbY], countBEdgesY[countbY + 1]]

            countBT = -1

            for bT in range(len(binCentersT)):
                countBT = countBT + 1

                countBRangeT = [countBEdgesT[countBT], countBEdgesT[countBT + 1]]

                selYVect = (probeLocationRot[:, 1] >= binEdgesY[bY]) & (
                    probeLocationRot[:, 1] < binEdgesY[bY + 1]
                )
                selCueProbeSOAVect = (cueProbeSOA >= binEdgesT[bT]) & (
                    cueProbeSOA < binEdgesT[bT + 1]
                )

                tempErprAmp = mainVarPupilData[(selYVect) & (selCueProbeSOAVect)]
                if np.sum((selYVect) & (selCueProbeSOAVect)) == 0:  # weird but no data in this bin
                    pass
                elif (
                    np.sum((selYVect) & (selCueProbeSOAVect)) == 1
                ):  # weird but only one trial in this bin
                    erprAmpPerYTimeBin[
                        countBRangeY[0] : countBRangeY[1],
                        countBRangeT[0] : countBRangeT[1],
                        CountBinArrayYT,
                    ] = tempErprAmp
                else:
                    erprAmpPerYTimeBin[
                        countBRangeY[0] : countBRangeY[1],
                        countBRangeT[0] : countBRangeT[1],
                        CountBinArrayYT,
                    ] = np.median(tempErprAmp)

erprAmpPerYTimeBin_ave = np.nanmedian(erprAmpPerYTimeBin, axis=2)

pickleFileName = (
    "data/"
    + ppName
    + "_perYTimeBin_sess-"
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
        [erprAmpPerYTimeBin_ave, binCentersY, binEdgesT],
        handle,
        protocol=pickle.HIGHEST_PROTOCOL,
    )
    handle.close()

if showPlotsPerObs:
    colors = plt.cm.coolwarm(np.linspace(0, 1, nTimeBins))

    plt.figure(figsize=(12, 6))
    ax = plt.subplot(1, 2, 1)
    ax.set_prop_cycle("color", list(colors))

    ax.plot(erprAmpPerYTimeBin_ave)
    plt.xlabel("Vertical position (deg)")
    plt.ylabel("Pupil size  " + pupilUnit)
    plt.title("Per binned vertical position")
    cbar = plt.colorbar(
        plt.cm.ScalarMappable(cmap="coolwarm"),
        ax=ax,
        ticks=[0, 0.5, 1],
        label="SOA time [ms]",
    )
    cbar.ax.set_yticklabels(
        [
            np.round(binCentersT[0], 0),
            np.round(binCentersT[int(len(binCentersT) / 2)], 0),
            np.round(binCentersT[-1], 0),
        ]
    )
