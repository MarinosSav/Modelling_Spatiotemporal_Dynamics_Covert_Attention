if "allErprAmp2DMapPerTimeBinPerTarLoc" in locals():
    del allErprAmp2DMapPerTimeBinPerTarLoc

countP = 0
for ppName in allPPName:
    sessNum = allSessNum[countP]
    countP += 1

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
    with open(pickleFileName, "rb") as handle:
        erprAmp2DMapPerTimeBinPerTarLoc_ave, binCentersX, binCentersY, binEdgesT, tarLocNames = pickle.load(handle)
        handle.close()

    if "allErprAmp2DMapPerTimeBinPerTarLoc" in locals():
        if countP == 2:
            allErprAmp2DMapPerTimeBinPerTarLoc = np.concatenate(
                [allErprAmp2DMapPerTimeBinPerTarLoc[..., None], erprAmp2DMapPerTimeBinPerTarLoc_ave[..., None]],
                axis=4,
            )
        else:
            allErprAmp2DMapPerTimeBinPerTarLoc = np.concatenate(
                [allErprAmp2DMapPerTimeBinPerTarLoc, erprAmp2DMapPerTimeBinPerTarLoc_ave[..., None]], axis=4
            )

    else:
        allErprAmp2DMapPerTimeBinPerTarLoc = erprAmp2DMapPerTimeBinPerTarLoc_ave


newPickleFileName = (
    "data/ave"
    + "_perXYTimeBinPerTarLoc"
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
        [allErprAmp2DMapPerTimeBinPerTarLoc, binCentersX, binCentersY, binEdgesT, tarLocNames],
        handle,
        protocol=pickle.HIGHEST_PROTOCOL,
    )
    handle.close()