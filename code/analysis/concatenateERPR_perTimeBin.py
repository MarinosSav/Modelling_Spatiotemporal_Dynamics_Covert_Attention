if "allErprAmpPerTimeBin" in locals():
    del allErprAmpPerTimeBin

if "allErprMatPerTimeBin" in locals():
    del allErprMatPerTimeBin

countP = 0
for ppName in allPPName:
    sessNum = allSessNum[countP]
    countP += 1

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

    with open(pickleFileName, "rb") as handle:
        erprMatPerCueProbeSOAbin_ave, erprAmpPerCueProbeSOAbin_ave, binCentersT = pickle.load(
            handle
        )
        handle.close()

    if "allErprAmpPerTimeBin" in locals():
        if countP == 2:
            allErprAmpPerTimeBin = np.vstack([allErprAmpPerTimeBin, erprAmpPerCueProbeSOAbin_ave])
        else:
            allErprAmpPerTimeBin = np.vstack([allErprAmpPerTimeBin, erprAmpPerCueProbeSOAbin_ave])

    else:
        allErprAmpPerTimeBin = erprAmpPerCueProbeSOAbin_ave

    if "allErprMatPerTimeBin" in locals():
        if countP == 2:
            allErprMatPerTimeBin = np.concatenate(
                [allErprMatPerTimeBin[..., None], erprMatPerCueProbeSOAbin_ave[..., None]],
                axis=2,
            )
        else:
            allErprMatPerTimeBin = np.concatenate(
                [allErprMatPerTimeBin, erprMatPerCueProbeSOAbin_ave[..., None]], axis=2
            )

    else:
        allErprMatPerTimeBin = erprMatPerCueProbeSOAbin_ave

allErprAmpPerTimeBin = allErprAmpPerTimeBin.T
