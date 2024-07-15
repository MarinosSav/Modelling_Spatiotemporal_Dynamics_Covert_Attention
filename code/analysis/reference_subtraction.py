# average across observers

allErprAmpPerXPerTimeBin_ave = np.nanmean(allErprAmpPerXPerTimeBin, axis=2)
allErprAmpPerXBin_ave = np.nanmean(allErprAmpPerXPerTimeBin_ave, axis=1)

allErprAmpPerYPerTimeBin_ave = np.nanmean(allErprAmpPerYPerTimeBin, axis=2)
allErprAmpPerYBin_ave = np.nanmean(allErprAmpPerYPerTimeBin_ave, axis=1)

allErprAmp2DMapPerTimeBin_ave = np.nanmean(allErprAmp2DMapPerTimeBin, axis=3)
allErprAmp2DMap_ave = np.nanmean(allErprAmp2DMapPerTimeBin_ave, axis=2)

allErprAmp2DMapPerTimeBinPerTarLoc_ave = np.nanmean(allErprAmp2DMapPerTimeBinPerTarLoc, axis=4)
allErprAmp2DMapPerTarLoc_ave = np.nanmean(allErprAmp2DMapPerTimeBinPerTarLoc_ave, axis=2)
    
allErprAmpPerXPerTimeBin_ref = allErprAmpPerXPerTimeBin.copy()
allErprAmpPerYPerTimeBin_ref = allErprAmpPerYPerTimeBin.copy()
allErprAmp2DMapPerTimeBin_ref = allErprAmp2DMapPerTimeBin.copy()
allErprAmp2DMapPerTimeBinPerTarLoc_ref = allErprAmp2DMapPerTimeBinPerTarLoc.copy()

if refMap == "Obs":
    # use average across time PER OBSERVER as a reference (subtraction)

    for s in range(np.shape(allErprAmp2DMapPerTimeBin)[3]):
        allErprAmpPerXPerTimeBin_ref[:, :, s] = (
            allErprAmpPerXPerTimeBin_ref[:, :, s]
            - np.tile(
                np.nanmean(allErprAmpPerXPerTimeBin_ref[:, :, s], axis=1),
                (np.shape(allErprAmpPerXPerTimeBin_ref)[1], 1),
            ).T
        )
        allErprAmpPerYPerTimeBin_ref[:, :, s] = (
            allErprAmpPerYPerTimeBin_ref[:, :, s]
            - np.tile(
                np.nanmean(allErprAmpPerYPerTimeBin_ref[:, :, s], axis=1),
                (np.shape(allErprAmpPerYPerTimeBin_ref)[1], 1),
            ).T
        )

        allErprAmp2DMapPerTimeBin_ref[:, :, :, s] = allErprAmp2DMapPerTimeBin_ref[
            :, :, :, s
        ] - np.tile(
            np.nanmean(allErprAmp2DMapPerTimeBin[:, :, :, s], axis=2)[:, :, np.newaxis],
            (1, 1, np.shape(allErprAmp2DMapPerTimeBin)[2]),
        )

        for TL in range(len(tarLocNames)):
            # subtract average reference map per direction with each reference map calculated per direction
            allErprAmp2DMapPerTimeBinPerTarLoc_ref[:, :, :, TL, s] = allErprAmp2DMapPerTimeBinPerTarLoc_ref[
                :, :, :, TL, s
            ] - np.tile(
                np.nanmean(allErprAmp2DMapPerTimeBinPerTarLoc[:, :, :, TL, s], axis=2)[:, :, np.newaxis],
                (1, 1, np.shape(allErprAmp2DMapPerTimeBinPerTarLoc)[2]),
            )


            # # subtract average reference map per direction but with only one reference map pooled across all directions
            # allErprAmp2DMapPerTimeBinPerTarLoc_ref[:, :, :, TL, s] = allErprAmp2DMapPerTimeBinPerTarLoc_ref[
            #     :, :, :, TL, s
            # ] - np.tile(
            #     np.nanmean(np.nanmean(allErprAmp2DMapPerTimeBinPerTarLoc[:, :, :, :, s],axis=3), axis=2)[:, :, np.newaxis],
            #     (1, 1, np.shape(allErprAmp2DMapPerTimeBinPerTarLoc)[2]),
            # )

    allErprAmpPerXPerTimeBin_ave_ref = np.nanmean(allErprAmpPerXPerTimeBin_ref, axis=2)
    allErprAmpPerYPerTimeBin_ave_ref = np.nanmean(allErprAmpPerYPerTimeBin_ref, axis=2)
    allErprAmp2DMapPerTimeBin_ave_ref = np.nanmean(allErprAmp2DMapPerTimeBin_ref, axis=3)
    allErprAmp2DMapPerTimeBinPerTarLoc_ave_ref = np.nanmean(allErprAmp2DMapPerTimeBinPerTarLoc_ref, axis=4)

    newPickleFileName = (
        "data/ave_ref_"
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
            [allErprAmp2DMapPerTimeBin_ref, binCentersX, binCentersY, binEdgesT],
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
        handle.close()


    newPickleFileName = (
        "data/ave_ref_"
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
            [allErprAmp2DMapPerTimeBinPerTarLoc_ref, binCentersX, binCentersY, binEdgesT, tarLocNames],
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
        handle.close()


elif refMap == "Pool":
    # use average across time and across observers as a reference (subtraction)

    allErprAmpPerXPerTimeBin_ave_ref = (
        allErprAmpPerXPerTimeBin_ave
        - np.tile(allErprAmpPerXBin_ave, (np.shape(allErprAmpPerXPerTimeBin_ave)[1], 1)).T
    )
    allErprAmpPerYPerTimeBin_ave_ref = (
        allErprAmpPerYPerTimeBin_ave
        - np.tile(allErprAmpPerYBin_ave, (np.shape(allErprAmpPerYPerTimeBin_ave)[1], 1)).T
    )
    allErprAmp2DMapPerTimeBin_ave_ref = allErprAmp2DMapPerTimeBin_ave - np.tile(
        allErprAmp2DMap_ave[:, :, np.newaxis],
        (1, 1, np.shape(allErprAmp2DMapPerTimeBin_ave)[2]),
    )
    
    for TL in range(len(tarLocNames)):
        allErprAmp2DMapPerTimeBinPerTarLoc_ave_ref[:,:,:,TL] = allErprAmp2DMapPerTimeBinPerTarLoc_ave[:,:,:,TL] - np.tile(
            allErprAmp2DMapPerTarLoc_ave[:, :, TL, np.newaxis],
            (1, 1, np.shape(allErprAmp2DMapPerTimeBinPerTarLoc_ave)[2]),
        )
elif refMap == "None":
    allErprAmpPerXPerTimeBin_ave_ref = np.nanmean(allErprAmpPerXPerTimeBin_ref, axis=2)
    allErprAmpPerYPerTimeBin_ave_ref = np.nanmean(allErprAmpPerYPerTimeBin_ref, axis=2)
    allErprAmp2DMapPerTimeBin_ave_ref = np.nanmean(allErprAmp2DMapPerTimeBin_ref, axis=3)
    allErprAmp2DMapPerTimeBinPerTarLoc_ave_ref = np.nanmean(allErprAmp2DMapPerTimeBinPerTarLoc_ref, axis=4)
else:
    print("Type of reference map calculation not recognized: " + refMap)
    print('Should be "Obs" or "Pool" ')
