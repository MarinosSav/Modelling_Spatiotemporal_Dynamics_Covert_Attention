from scipy.stats import sem, ttest_rel
from matplotlib.patches import Polygon

allErprAmp2DMapPerTimeBin_ref = allErprAmp2DMapPerTimeBin.copy()

nPP = len(allPPName)

allErprAmpPerXBin = np.zeros((nTimeBins, nXBins, nPP))
allErprAmpPerYBin = np.zeros((nTimeBins, nYBins, nPP))


countPP = 0
for ppName in allPPName:
    for bT in range(nTimeBins):
        allErprAmpPerXBin[bT, :, countPP] = allErprAmpPerXPerTimeBin_ref[:, bT, countPP]
        allErprAmpPerYBin[bT, :, countPP] = allErprAmpPerYPerTimeBin_ref[:, bT, countPP]
    countPP += 1


###### plot results across time bins
colors = plt.cm.jet(np.linspace(0, 1, nTimeBins))

plt.figure(figsize=(12, 6))

countBT = -1
for bT in range(nTimeBins):
    countBT = countBT + 1
    tempDataX_ave = np.nanmean(allErprAmpPerXBin[bT, :, :], axis=1)
    tempDataY_ave = np.nanmean(allErprAmpPerYBin[bT, :, :], axis=1)

    tempDataX_sem = sem(allErprAmpPerXBin[bT, :, :], axis=1)
    tempDataY_sem = sem(allErprAmpPerYBin[bT, :, :], axis=1)

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

    ax = plt.subplot(1, 2, 1)
    ax.plot(
        np.round(binCentersX,0).astype(int),
        tempDataX_ave,
        color=colors[countBT, :],
    )
    ax.add_patch(polygonX)

    ax = plt.subplot(1, 2, 2)
    ax.plot(
        np.round(binCentersY,0).astype(int),
        tempDataY_ave,
        color=colors[countBT, :],
    )

    ax.add_patch(polygonY)


ax = plt.subplot(1, 2, 1)
plt.xlabel("Horizontal position")
plt.ylabel("Pupil response amplitude " + pupilUnit)
plt.title("Per binned horizontal position")
cbar = plt.colorbar(plt.cm.ScalarMappable(cmap="jet"), ax=ax, ticks=[0, 0.5, 1], label="SOA [ms]")
cbar.ax.set_yticklabels(["0", "200", "400"])

ax = plt.subplot(1, 2, 2)
plt.xlabel("Vertical position")
plt.ylabel("Pupil response amplitude " + pupilUnit)
plt.title("Per binned vertical position")
cbar = plt.colorbar(plt.cm.ScalarMappable(cmap="jet"), ax=ax, ticks=[0, 0.5, 1], label="SOA [ms]")
cbar.ax.set_yticklabels(["0", "200", "400"])

plt.tight_layout()
# plt.legend()
plt.show()


# post-hoc testing
firstTimeBinX = allErprAmpPerXBin[0, :, :]
lastTimeBinX = allErprAmpPerXBin[-1, :, :]
firstTimeBinY = allErprAmpPerYBin[0, :, :]
lastTimeBinY = allErprAmpPerYBin[-1, :, :]

allTX = []
allPX = []
for i in range(np.shape(firstTimeBinX)[0]):
    tstat_result = ttest_rel(firstTimeBinX[i, :], lastTimeBinX[i, :])
    allTX.append(tstat_result.statistic)
    allPX.append(tstat_result.pvalue)
allTY = []
allPY = []
for i in range(np.shape(firstTimeBinY)[0]):
    tstat_result = ttest_rel(firstTimeBinY[i, :], lastTimeBinY[i, :])
    allTY.append(tstat_result.statistic)
    allPY.append(tstat_result.pvalue)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(binCentersX, allPX)
plt.plot([binCentersX[0], binCentersX[-1]], [0.05, 0.05], "k:")
plt.xlabel("Horizontal position")
plt.ylabel("p-value")
plt.title("First vs. last bin")
plt.subplot(1, 2, 2)
plt.plot(binCentersY, allPY)
plt.plot([binCentersY[0], binCentersY[-1]], [0.05, 0.05], "k:")
plt.xlabel("Vertical position")
plt.ylabel("p-value")
plt.title("First vs. last bin")
plt.tight_layout()
