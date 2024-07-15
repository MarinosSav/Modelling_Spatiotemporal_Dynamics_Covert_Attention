if "allErprAmpPerXPerTimeBin" in locals():
    del allErprAmpPerXPerTimeBin

countP = 0
for ppName in allPPName:
    sessNum = allSessNum[countP]
    countP += 1

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
    with open(pickleFileName, "rb") as handle:
        erprAmpPerXTimeBin_ave, binCentersX, binEdgesT = pickle.load(handle)
        handle.close()

    # if "allErprAmpPerXPerTimeBin" in locals():
    #     allErprAmpPerXPerTimeBin = np.vstack((allErprAmpPerXPerTimeBin, erprAmpPerHoribin_ave))
    # else:
    #     allErprAmpPerXPerTimeBin = erprAmpPerHoribin_ave

    if "allErprAmpPerXPerTimeBin" in locals():
        if countP == 2:
            allErprAmpPerXPerTimeBin = np.concatenate(
                [allErprAmpPerXPerTimeBin[..., None], erprAmpPerXTimeBin_ave[..., None]],
                axis=2,
            )
        else:
            allErprAmpPerXPerTimeBin = np.concatenate(
                [allErprAmpPerXPerTimeBin, erprAmpPerXTimeBin_ave[..., None]], axis=2
            )

    else:
        allErprAmpPerXPerTimeBin = erprAmpPerXTimeBin_ave
