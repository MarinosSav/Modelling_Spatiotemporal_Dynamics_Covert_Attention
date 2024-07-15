from scipy.stats import sem, ttest_rel
from matplotlib.patches import Polygon

# allErprAmp2DMapPerTimeBinPerTarLoc_ref = allErprAmp2DMapPerTimeBinPerTarLoc.copy()

nTimeBins = np.shape(allErprAmp2DMapPerTimeBinPerTarLoc_ref)[2]
nDirBins = np.shape(allErprAmp2DMapPerTimeBinPerTarLoc_ref)[3]

nPP = len(allPPName)

allErprAmpPerXBinPerTarLoc = np.zeros((nTimeBins, nXBins, nDirBins, nPP))
allErprAmpPerXBinPerTarLoc[:] = np.nan
allErprAmpPerYBinPerTarLoc = np.zeros((nTimeBins, nYBins, nDirBins, nPP))
allErprAmpPerYBinPerTarLoc[:] = np.nan


countPP = 0
for ppName in allPPName:

    for TL in range(nDirBins):
        # tempAmp2DMap_pp = allErprAmp2DMapPerTimeBinPerTarLoc[:, :, :, TL, countPP]

        # refAmp2DMap_pp = np.nanmean(tempAmp2DMap_pp, axis=2)

        for bT in range(nTimeBins):
            # allErprAmp2DMapPerTimeBinPerTarLoc_ref[:, :, bT, TL, countPP] = (
                # allErprAmp2DMapPerTimeBinPerTarLoc_ref[:, :, bT, TL, countPP] - refAmp2DMap_pp
            # )

            allErprAmpPerXBinPerTarLoc[bT, :, TL, countPP] = np.nanmean(
                allErprAmp2DMapPerTimeBinPerTarLoc_ref[:, :, bT, TL, countPP], axis=0
            )
            allErprAmpPerYBinPerTarLoc[bT, :, TL, countPP] = np.nanmean(
                allErprAmp2DMapPerTimeBinPerTarLoc_ref[:, :, bT, TL, countPP], axis=1
            )

    countPP += 1


###### plot results across time bins BASED ON 2D MAP
colors = plt.cm.jet(np.linspace(0, 1, nTimeBins))

plt.figure(figsize=(12, 24))

for TL in range(nDirBins):
    countBT = -1
    for bT in range(nTimeBins):
        countBT = countBT + 1
        tempDataX_ave = np.nanmean(allErprAmpPerXBinPerTarLoc[bT, :, TL, :], axis=1)
        tempDataY_ave = np.nanmean(allErprAmpPerYBinPerTarLoc[bT, :, TL, :], axis=1)

        tempDataX_sem = sem(allErprAmpPerXBinPerTarLoc[bT, :, TL, :], axis=1)
        tempDataY_sem = sem(allErprAmpPerYBinPerTarLoc[bT, :, TL, :], axis=1)

        tempPatchDataXX = np.hstack((np.round(binCentersX,0).astype(int), np.flip(np.round(binCentersX,0).astype(int))))
        tempPatchDataYX = np.hstack((np.round(binCentersY,0).astype(int), np.flip(np.round(binCentersY,0).astype(int))))
        tempPatchDataX = np.hstack(
            (tempDataX_ave - tempDataX_sem, np.flip(tempDataX_ave + tempDataX_sem))
        )
        tempPatchDataY = np.hstack(
            (tempDataY_ave - tempDataY_sem, np.flip(tempDataY_ave + tempDataY_sem))
        )

        polygonX_coor = []
        polygonY_coor = []
        for i in range(len(tempPatchDataX)):
            polygonX_coor.append((tempPatchDataXX[i], tempPatchDataX[i]))

        for i in range(len(tempPatchDataY)):
            polygonY_coor.append((tempPatchDataYX[i], tempPatchDataY[i]))

        polygonX = Polygon(polygonX_coor, alpha=0.2, color=colors[countBT, :])
        polygonY = Polygon(polygonY_coor, alpha=0.2, color=colors[countBT, :])

        ax = plt.subplot(4, 2, 1+TL*2)
        ax.plot(
            np.round(binCentersX,0).astype(int),
            tempDataX_ave,
            color=colors[countBT, :],
        )
        ax.add_patch(polygonX)

        ax = plt.subplot(4, 2, 2+TL*2)
        ax.plot(
            np.round(binCentersY,0).astype(int),
            tempDataY_ave,
            color=colors[countBT, :],
        )

        ax.add_patch(polygonY)
        plt.title(tarLocNames[TL])


ax = plt.subplot(4, 2, 1)
plt.xlabel("Horizontal position")
plt.ylabel("Pupil response amplitude " + pupilUnit)
plt.title("Per binned horizontal position")
cbar = plt.colorbar(plt.cm.ScalarMappable(cmap="jet"), ax=ax, ticks=[0, 0.5, 1], label="SOA [ms]")
cbar.ax.set_yticklabels(["0", "200", "400"])

ax = plt.subplot(4, 2, 2)
plt.xlabel("Vertical position")
plt.ylabel("Pupil response amplitude " + pupilUnit)
plt.title("Per binned vertical position")
cbar = plt.colorbar(plt.cm.ScalarMappable(cmap="jet"), ax=ax, ticks=[0, 0.5, 1], label="SOA [ms]")
cbar.ax.set_yticklabels(["0", "200", "400"])

plt.tight_layout()
# plt.legend()
plt.show()
