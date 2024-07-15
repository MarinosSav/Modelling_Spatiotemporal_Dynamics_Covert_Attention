erprMatPerCueProbeSOAbin = np.zeros(
    (np.max(nTimeBinsArr), np.shape(erprMatrixBaseSubt)[1], len(nTimeBinsArr))
)
erprMatPerCueProbeSOAbin[:] = np.nan

erprAmpPerCueProbeSOAbin = np.zeros((np.max(nTimeBinsArr), len(nTimeBinsArr)))
erprAmpPerCueProbeSOAbin[:] = np.nan

countBinArray = -1
for nTimeBins in nTimeBinsArr:
    countBinArray = countBinArray + 1
    countBEdges = np.linspace(0, np.max(nTimeBinsArr), nTimeBins + 1).astype(int)

    binEdges = np.linspace(0, 400, nTimeBins + 1)
    binCenters = binEdges[0:-1] + np.diff(binEdges) / 2
    binEdges[-1] = binEdges[-1] + 1  # to compensate last bin edge <=
    countB = -1
    for b in range(nTimeBins):
        countB = countB + 1

        countBRange = [countBEdges[countB], countBEdges[countB + 1]]

        selCueProbeSOAVect = (cueProbeSOA >= binEdges[b]) & (cueProbeSOA < binEdges[b + 1])
        # selCueProbeSOAVect = (cueProbeSOA >= binEdges[countBRange[0]]) & (
        # cueProbeSOA < binEdges[countBRange[1]]
        # )

        tempErprMat = erprMatrixBaseSubt[selCueProbeSOAVect, :]

        tempErprAmp = mainVarPupilData[selCueProbeSOAVect]
        if np.sum(selCueProbeSOAVect) == 0:  # weird but no data in this bin
            pass
        elif np.sum(selCueProbeSOAVect) == 1:  # weird but only one trial in this bin
            erprMatPerCueProbeSOAbin[
                countBRange[0] : countBRange[1], :, countBinArray
            ] = tempErprMat
            erprAmpPerCueProbeSOAbin[countBRange[0] : countBRange[1], countBinArray] = tempErprAmp
        else:
            erprMatPerCueProbeSOAbin[countBRange[0] : countBRange[1], :, countBinArray] = np.mean(
                tempErprMat, 0
            )
            # erprAmpPerCueProbeSOAbin[countBRange[0]:countBRange[1],countBinArray] = np.median(tempErprAmp)
            erprAmpPerCueProbeSOAbin[countBRange[0] : countBRange[1], countBinArray] = np.mean(
                tempErprAmp
            )

colors = plt.cm.coolwarm(np.linspace(0, 1, nTimeBins))

# print(erprAmpPerCueProbeSOAbin)
erprMatPerCueProbeSOAbin_ave = np.nanmedian(erprMatPerCueProbeSOAbin, axis=2)
erprAmpPerCueProbeSOAbin_ave = np.nanmedian(erprAmpPerCueProbeSOAbin, axis=1)


pickleFileName = (
    "data/"
    + ppName
    + "_perTimeBin_sess-"
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
        [erprMatPerCueProbeSOAbin_ave, erprAmpPerCueProbeSOAbin_ave, binCenters],
        handle,
        protocol=pickle.HIGHEST_PROTOCOL,
    )
    handle.close()

if showPlotsPerObs:
    plt.figure(figsize=(12, 6))
    ax = plt.subplot(1, 2, 1)
    ax.set_prop_cycle("color", list(colors))

    ax.plot(erprMatPerCueProbeSOAbin_ave.T)
    plt.xlabel("Time from " + onsetEvent + " onset [ms]")
    plt.ylabel("Pupil size  " + pupilUnit)
    plt.title("Per binned cue-probe SOA")
    cbar = plt.colorbar(
        plt.cm.ScalarMappable(cmap="coolwarm"), ax=ax, ticks=[0, 0.5, 1], label="SOA [ms]"
    )
    cbar.ax.set_yticklabels(["0", "200", "400"])

    plt.subplot(1, 2, 2)
    plt.plot(erprAmpPerCueProbeSOAbin_ave)
    plt.xlabel("SOA [ms]")
    plt.xticks(range(0, len(binCenters)), labels=binCenters.astype(int))
    plt.ylabel("Pupil response amplitude  " + pupilUnit)

    plt.tight_layout()
    plt.show()
