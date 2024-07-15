# TO DO: flip figure upside down? because no bottom is near fovea and top near periphery

erprAmpPerXTimeBin = np.zeros(
    (
        np.max(nXBinsArr),
        np.max(nTimeBinsArr),
        len(nXBinsArr) * len(nTimeBinsArr),
    )
)
erprAmpPerXTimeBin[:] = np.nan

CountBinArrayXT = -1
countBinArrayTime = -1
for nTimeBins in nTimeBinsArr:
    countBinArrayTime = countBinArrayTime + 1
    countBEdgesT = np.linspace(0, np.max(nTimeBinsArr), nTimeBins + 1).astype(int)

    binEdgesT = np.linspace(0, 400, nTimeBins + 1)
    binCentersT = binEdgesT[0:-1] + np.diff(binEdgesT) / 2
    binEdgesT[-1] = binEdgesT[-1] + 1  # to compensate last bin edge <=

    CountBinArrayX = -1
    for nXBins in nXBinsArr:
        CountBinArrayX = CountBinArrayX + 1
        CountBinArrayXT = CountBinArrayXT + 1

        nXBins = nXBinsArr[CountBinArrayX]

        countBEdgesX = np.linspace(0, np.max(nXBinsArr), nXBins + 1).astype(int)

        binEdgesX = np.linspace(
            np.min(probeLocationRot[:, 0]), np.max(probeLocationRot[:, 0]), nXBins + 1
        )
        binCentersX = binEdgesX[0:-1] + np.diff(binEdgesX) / 2
        binEdgesX[-1] = binEdgesX[-1] + 1  # to compensate last bin edge <=

        countBX = -1
        for bX in range(len(binCentersX)):
            countBX = countBX + 1
            countBRangeX = [countBEdgesX[countBX], countBEdgesX[countBX + 1]]

            countBT = -1

            for bT in range(len(binCentersT)):
                countBT = countBT + 1

                countBRangeT = [countBEdgesT[countBT], countBEdgesT[countBT + 1]]

                selXVect = (probeLocationRot[:, 0] >= binEdgesX[bX]) & (
                    probeLocationRot[:, 0] < binEdgesX[bX + 1]
                )
                selCueProbeSOAVect = (cueProbeSOA >= binEdgesT[bT]) & (
                    cueProbeSOA < binEdgesT[bT + 1]
                )

                tempErprAmp = mainVarPupilData[(selXVect) & (selCueProbeSOAVect)]
                if np.sum((selXVect) & (selCueProbeSOAVect)) == 0:  # weird but no data in this bin
                    pass
                elif (
                    np.sum((selXVect) & (selCueProbeSOAVect)) == 1
                ):  # weird but only one trial in this bin
                    erprAmpPerXTimeBin[
                        countBRangeX[0] : countBRangeX[1],
                        countBRangeT[0] : countBRangeT[1],
                        CountBinArrayXT,
                    ] = tempErprAmp
                else:
                    erprAmpPerXTimeBin[
                        countBRangeX[0] : countBRangeX[1],
                        countBRangeT[0] : countBRangeT[1],
                        CountBinArrayXT,
                    ] = np.median(tempErprAmp)

erprAmpPerXTimeBin_ave = np.nanmedian(erprAmpPerXTimeBin, axis=2)

pickleFileName = (
    "data/"
    + ppName
    + "_perXTimeBin_sess-"
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
        [erprAmpPerXTimeBin_ave, binCentersX, binEdgesT],
        handle,
        protocol=pickle.HIGHEST_PROTOCOL,
    )
    handle.close()

if showPlotsPerObs:
    colors = plt.cm.coolwarm(np.linspace(0, 1, nTimeBins))

    plt.figure(figsize=(12, 6))
    ax = plt.subplot(1, 2, 1)
    ax.set_prop_cycle("color", list(colors))

    ax.plot(erprAmpPerXTimeBin_ave)
    plt.xlabel("Horizontal position (deg)")
    plt.ylabel("Pupil size  " + pupilUnit)
    plt.title("Per binned horizontal position")
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
