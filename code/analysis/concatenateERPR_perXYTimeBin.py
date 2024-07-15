if "allErprAmp2DMapPerTimeBin" in locals():
    del allErprAmp2DMapPerTimeBin

countP = 0
for ppName in allPPName:
    sessNum = allSessNum[countP]
    countP += 1

    pickleFileName = (
        "data/"
        + ppName
        + "_perXYTimeBin_sess-"
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
        erprAmp2DMapPerTimeBin_ave, binCentersX, binCentersY, binEdgesT = pickle.load(handle)
        handle.close()

    if "allErprAmp2DMapPerTimeBin" in locals():
        if countP == 2:
            allErprAmp2DMapPerTimeBin = np.concatenate(
                [allErprAmp2DMapPerTimeBin[..., None], erprAmp2DMapPerTimeBin_ave[..., None]],
                axis=3,
            )
        else:
            allErprAmp2DMapPerTimeBin = np.concatenate(
                [allErprAmp2DMapPerTimeBin, erprAmp2DMapPerTimeBin_ave[..., None]], axis=3
            )

    else:
        allErprAmp2DMapPerTimeBin = erprAmp2DMapPerTimeBin_ave


newPickleFileName = (
    "data/ave"
    + "_perXYTimeBin"
    + "_norm-"
    + normTypeName
    + "_mes-"
    + varOfInterestAbr
    + "_onset-"
    + onsetEvent
    + ".pickle"
)

with open(newPickleFileName, "wb") as handle:
    pickle.dump(
        [allErprAmp2DMapPerTimeBin, binCentersX, binCentersY, binEdgesT],
        handle,
        protocol=pickle.HIGHEST_PROTOCOL,
    )
    handle.close()


if "allErprAmp2DMapPerTimeBin_highRes" in locals():
    del allErprAmp2DMapPerTimeBin_highRes

countP = 0
for ppName in allPPName:
    sessNum = allSessNum[countP]
    countP += 1

    pickleFileName = (
        "data/"
        + ppName
        + "_perXYTimeBin_highRes_sess-"
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
        (
            erprAmp2DMapPerTimeBin_highRes_ave,
            binCentersX,
            binCentersY,
            binEdgesT_highRes,
        ) = pickle.load(handle)
        handle.close()

    if "allErprAmp2DMapPerTimeBin_highRes" in locals():
        if countP == 2:
            allErprAmp2DMapPerTimeBin_highRes = np.concatenate(
                [
                    allErprAmp2DMapPerTimeBin_highRes[..., None],
                    erprAmp2DMapPerTimeBin_highRes_ave[..., None],
                ],
                axis=3,
            )
        else:
            allErprAmp2DMapPerTimeBin_highRes = np.concatenate(
                [allErprAmp2DMapPerTimeBin_highRes, erprAmp2DMapPerTimeBin_highRes_ave[..., None]],
                axis=3,
            )

    else:
        allErprAmp2DMapPerTimeBin_highRes = erprAmp2DMapPerTimeBin_highRes_ave
