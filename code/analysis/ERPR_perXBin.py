erprMatPerHoribin = np.zeros((np.max(nXBinsArr), np.shape(erprMatrixBaseSubt)[1], len(nXBinsArr)))
erprMatPerHoribin[:] = np.nan

erprAmpPerHoribin = np.zeros((np.max(nXBinsArr), len(nXBinsArr)))
erprAmpPerHoribin[:] = np.nan


countBinArray = -1
for nXBins in nXBinsArr:
    countBinArray = countBinArray + 1
    countBEdges = np.linspace(0, np.max(nXBinsArr), nXBins + 1).astype(int)

    binEdges = np.linspace(
        np.min(probeLocationRot[:, 0]), np.max(probeLocationRot[:, 0]), nXBins + 1
    )
    binCenters = binEdges[0:-1] + np.diff(binEdges) / 2
    binEdges[-1] = binEdges[-1] + 1  # to compensate last bin edge <=
    countB = -1
    for b in range(len(binCenters)):
        countB = countB + 1
        countBRange = [countBEdges[countB], countBEdges[countB + 1]]

        selXVect = (probeLocationRot[:, 0] >= binEdges[b]) & (
            probeLocationRot[:, 0] < binEdges[b + 1]
        )

        tempErprMat = erprMatrixBaseSubt[selXVect, :]
        # tempErprAmp = pupilOriAmp[selCueProbeSOAVect]
        tempErprAmp = mainVarPupilData[selXVect]
        if np.sum(selXVect) == 0:  # weird but no data in this bin
            pass
        elif np.sum(selXVect) == 1:  # weird but only one trial in this bin
            erprMatPerHoribin[countBRange[0] : countBRange[1], :, countBinArray] = tempErprMat
            erprAmpPerHoribin[countBRange[0] : countBRange[1], countBinArray] = tempErprAmp
        else:
            erprMatPerHoribin[countBRange[0] : countBRange[1], :, countBinArray] = np.mean(
                tempErprMat, 0
            )
            erprAmpPerHoribin[countBRange[0] : countBRange[1], countBinArray] = np.mean(
                tempErprAmp
            )

colors = plt.cm.coolwarm(np.linspace(0, 1, nXBins))

erprMatPerHoribin_ave = np.nanmedian(erprMatPerHoribin, axis=2)
erprAmpPerHoribin_ave = np.nanmedian(erprAmpPerHoribin, axis=1)


pickleFileName = (
    "data/"
    + ppName
    + "_perXBin_sess-"
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
        [erprMatPerHoribin_ave, erprAmpPerHoribin_ave, binCenters],
        handle,
        protocol=pickle.HIGHEST_PROTOCOL,
    )
    handle.close()

if showPlotsPerObs:

    plt.figure(figsize=(12, 6))
    ax = plt.subplot(1, 2, 1)
    ax.set_prop_cycle("color", list(colors))

    ax.plot(erprMatPerHoribin_ave.T)
    plt.xlabel("Time from " + onsetEvent + " onset [ms]")
    plt.ylabel("Pupil size  " + pupilUnit)
    plt.title("Per binned horizontal position")
    cbar = plt.colorbar(
        plt.cm.ScalarMappable(cmap="coolwarm"),
        ax=ax,
        ticks=[0, 0.5, 1],
        label="Horizontal position",
    )
    cbar.ax.set_yticklabels(
        [
            np.round(binCenters[0], 0),
            np.round(binCenters[int(len(binCenters) / 2)], 0),
            np.round(binCenters[-1], 0),
        ]
    )

    plt.subplot(1, 2, 2)
    plt.plot(erprAmpPerHoribin_ave)
    plt.xlabel("Horizontal position")
    plt.xticks(range(0, len(binCenters)), labels=binCenters.astype(int))
    plt.ylabel("Pupil response amplitude  " + pupilUnit)

    plt.tight_layout()
    plt.show()
