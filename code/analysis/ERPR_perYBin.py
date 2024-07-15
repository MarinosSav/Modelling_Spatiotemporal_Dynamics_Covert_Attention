erprMatPerVeribin = np.zeros((np.max(nYBinsArr), np.shape(erprMatrixBaseSubt)[1], len(nYBinsArr)))
erprMatPerVeribin[:] = np.nan

erprAmpPerVeribin = np.zeros((np.max(nYBinsArr), len(nYBinsArr)))
erprAmpPerVeribin[:] = np.nan

countBinArray = -1
for nYBins in nYBinsArr:
    countBinArray = countBinArray + 1
    countBEdges = np.linspace(0, np.max(nYBinsArr), nYBins + 1).astype(int)

    binEdges = np.linspace(
        np.min(probeLocationRot[:, 1]), np.max(probeLocationRot[:, 1]), nYBins + 1
    )
    binCenters = binEdges[0:-1] + np.diff(binEdges) / 2
    binEdges[-1] = binEdges[-1] + 1  # to compensate last bin edge <=
    countB = -1
    for b in range(len(binCenters)):
        countB = countB + 1
        countBRange = [countBEdges[countB], countBEdges[countB + 1]]
        selYVect = (probeLocationRot[:, 1] >= binEdges[b]) & (
            probeLocationRot[:, 1] < binEdges[b + 1]
        )

        tempErprMat = erprMatrixBaseSubt[selYVect, :]
        # tempErprAmp = pupilOriAmp[selCueProbeSOAVect]
        tempErprAmp = mainVarPupilData[selYVect]
        if np.sum(selYVect) == 0:  # weird but no data in this bin
            pass
        elif np.sum(selYVect) == 1:  # weird but only one trial in this bin
            erprMatPerVeribin[countBRange[0] : countBRange[1], :, countBinArray] = tempErprMat
            erprAmpPerVeribin[countBRange[0] : countBRange[1], countBinArray] = tempErprAmp
        else:
            erprMatPerVeribin[countBRange[0] : countBRange[1], :, countBinArray] = np.mean(
                tempErprMat, 0
            )
            erprAmpPerVeribin[countBRange[0] : countBRange[1], countBinArray] = np.mean(
                tempErprAmp
            )

colors = plt.cm.coolwarm(np.linspace(0, 1, nYBins))

erprMatPerVeribin_ave = np.nanmedian(erprMatPerVeribin, axis=2)
erprAmpPerVeribin_ave = np.nanmedian(erprAmpPerVeribin, axis=1)


pickleFileName = (
    "data/"
    + ppName
    + "_perYBin_sess-"
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
        [erprMatPerVeribin_ave, erprAmpPerVeribin_ave, binCenters],
        handle,
        protocol=pickle.HIGHEST_PROTOCOL,
    )
    handle.close()

if showPlotsPerObs:

    plt.figure(figsize=(12, 6))
    ax = plt.subplot(1, 2, 1)
    ax.set_prop_cycle("color", list(colors))

    ax.plot(erprMatPerVeribin_ave.T)
    plt.xlabel("Time from " + onsetEvent + " onset [ms]")
    plt.ylabel("Pupil size " + pupilUnit)
    plt.title("Per binned vertical position")
    cbar = plt.colorbar(
        plt.cm.ScalarMappable(cmap="coolwarm"), ax=ax, ticks=[0, 0.5, 1], label="Vertical position"
    )
    cbar.ax.set_yticklabels(
        [
            np.round(binCenters[0], 0),
            np.round(binCenters[int(len(binCenters) / 2)], 0),
            np.round(binCenters[-1], 0),
        ]
    )

    plt.subplot(1, 2, 2)
    plt.plot(erprAmpPerVeribin_ave)
    plt.xlabel("Vertical position")
    plt.xticks(range(0, len(binCenters)), labels=binCenters.astype(int))
    plt.ylabel("Pupil response amplitude [au]")

    plt.tight_layout()
    plt.show()
