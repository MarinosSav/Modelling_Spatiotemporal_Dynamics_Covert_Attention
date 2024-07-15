if "allErprAmpPerYPerTimeBin" in locals():
    del allErprAmpPerYPerTimeBin

countP = 0
for ppName in allPPName:
    sessNum = allSessNum[countP]
    countP += 1

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
    with open(pickleFileName, "rb") as handle:
        erprAmpPerYTimeBin_ave, binCentersY, binEdgesT = pickle.load(handle)
        handle.close()

    if "allErprAmpPerYPerTimeBin" in locals():
        if countP == 2:
            allErprAmpPerYPerTimeBin = np.concatenate(
                [allErprAmpPerYPerTimeBin[..., None], erprAmpPerYTimeBin_ave[..., None]],
                axis=2,
            )
        else:
            allErprAmpPerYPerTimeBin = np.concatenate(
                [allErprAmpPerYPerTimeBin, erprAmpPerYTimeBin_ave[..., None]], axis=2
            )

    else:
        allErprAmpPerYPerTimeBin = erprAmpPerYTimeBin_ave
