from scipy.stats import sem, ttest_rel
from matplotlib.patches import Polygon

# TO DO::::: with ref subtractioN!
allErprAmp2DMapPerTimeBin_highRes_ref = allErprAmp2DMapPerTimeBin_highRes.copy()

nTimeBins = np.shape(allErprAmp2DMapPerTimeBin_highRes_ref)[2]

nPP = len(allPPName)

allErprAmpPerXBin = np.zeros((nTimeBins, nXBins, nPP))
allErprAmpPerXBin[:] = np.nan
allErprAmpPerYBin = np.zeros((nTimeBins, nYBins, nPP))
allErprAmpPerYBin[:] = np.nan


countPP = 0
for ppName in allPPName:

    # tempAmp2DMap_pp = allErprAmp2DMapPerTimeBin_highRes_ref[:, :, :, countPP]

    # refAmp2DMap_pp = np.mean(tempAmp2DMap_pp, axis=2)

    for bT in range(nTimeBins):
        # allErprAmp2DMapPerTimeBin_highRes_ref[:, :, bT, countPP] = (
        #     allErprAmp2DMapPerTimeBin_highRes_ref[:, :, bT, countPP] - refAmp2DMap_pp
        # )

        allErprAmpPerXBin[bT, :, countPP] = np.mean(
            allErprAmp2DMapPerTimeBin_highRes_ref[:, :, bT, countPP], axis=0
        )
        allErprAmpPerYBin[bT, :, countPP] = np.mean(
            allErprAmp2DMapPerTimeBin_highRes_ref[:, :, bT, countPP], axis=1
        )

    countPP += 1


###### plot results across time bins BASED ON 2D MAP
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
secondlastTimeBinX = allErprAmpPerXBin[-2, :, :]
lastTimeBinX = allErprAmpPerXBin[-1, :, :]
firstTimeBinY = allErprAmpPerYBin[0, :, :]
secondlastTimeBinY = allErprAmpPerYBin[-2, :, :]
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


allTX = []
allPX = []
for i in range(np.shape(firstTimeBinX)[0]):
    tstat_result = ttest_rel(firstTimeBinX[i, :], secondlastTimeBinX[i, :])
    allTX.append(tstat_result.statistic)
    allPX.append(tstat_result.pvalue)
allTY = []
allPY = []
for i in range(np.shape(firstTimeBinY)[0]):
    tstat_result = ttest_rel(firstTimeBinY[i, :], secondlastTimeBinY[i, :])
    allTY.append(tstat_result.statistic)
    allPY.append(tstat_result.pvalue)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(binCentersX, allPX)
plt.plot([binCentersX[0], binCentersX[-1]], [0.05, 0.05], "k:")
plt.xlabel("Horizontal position")
plt.ylabel("p-value")
plt.title("First vs. 2nd last bin")
plt.subplot(1, 2, 2)
plt.plot(binCentersY, allPY)
plt.plot([binCentersY[0], binCentersY[-1]], [0.05, 0.05], "k:")
plt.xlabel("Vertical position")
plt.ylabel("p-value")
plt.title("First vs. 2nd last bin")
plt.tight_layout()


# fitting slope to data across horizontal X/Y space
slopesX = np.zeros((np.shape(allErprAmpPerXBin)[2], np.shape(allErprAmpPerXBin)[0]))
intercX = np.zeros((np.shape(allErprAmpPerXBin)[2], np.shape(allErprAmpPerXBin)[0]))
slopesY = np.zeros((np.shape(allErprAmpPerYBin)[2], np.shape(allErprAmpPerYBin)[0]))
intercY = np.zeros((np.shape(allErprAmpPerYBin)[2], np.shape(allErprAmpPerYBin)[0]))

nFitOrd = 3
quadX = np.zeros((nFitOrd, np.shape(allErprAmpPerXBin)[2], np.shape(allErprAmpPerXBin)[0]))
quadY = np.zeros((nFitOrd, np.shape(allErprAmpPerYBin)[2], np.shape(allErprAmpPerYBin)[0]))

for t in range(np.shape(allErprAmpPerXBin)[0]):
    for s in range(np.shape(allErprAmpPerXBin)[2]):
        pfX = np.polyfit(binCentersX, allErprAmpPerXBin[t, :, s], 1)
        intercX[s, t] = pfX[1]
        slopesX[s, t] = pfX[0]
        pfY = np.polyfit(binCentersY, allErprAmpPerYBin[t, :, s], 1)
        intercY[s, t] = pfY[1]
        slopesY[s, t] = pfY[0]

        quadX[:, s, t] = np.polyfit(binCentersX, allErprAmpPerXBin[t, :, s], nFitOrd - 1)
        quadY[:, s, t] = np.polyfit(binCentersY, allErprAmpPerYBin[t, :, s], nFitOrd - 1)


binCentersT_highRes = np.zeros((nTimeBins))
for bT in range(nTimeBins):
    binCentersT_highRes[bT] = np.mean([binEdgesT_highRes[bT], binEdgesT_highRes[bT + 1]])

plt.figure()
plt.subplot(2, 2, 1)
plt.plot(binCentersT_highRes, np.mean(slopesX, axis=0), "ko-")
plt.plot(binCentersT_highRes, np.mean(slopesX, axis=0) + sem(slopesX, axis=0), "k:")
plt.plot(binCentersT_highRes, np.mean(slopesX, axis=0) - sem(slopesX, axis=0), "k:")
plt.xlabel("Time bins")
plt.ylabel("Fitted slopes")
plt.title("Horizontal")

plt.subplot(2, 2, 2)
plt.plot(binCentersT_highRes, np.mean(slopesY, axis=0), "ko-")
plt.plot(binCentersT_highRes, np.mean(slopesY, axis=0) + sem(slopesY, axis=0), "k:")
plt.plot(binCentersT_highRes, np.mean(slopesY, axis=0) - sem(slopesY, axis=0), "k:")
plt.xlabel("Time bins")
plt.ylabel("Fitted slopes")
plt.title("Vertical")

plt.subplot(2, 2, 3)
plt.plot(binCentersT_highRes, np.mean(intercX, axis=0), "ko-")
plt.plot(binCentersT_highRes, np.mean(intercX, axis=0) + sem(intercX, axis=0), "k:")
plt.plot(binCentersT_highRes, np.mean(intercX, axis=0) - sem(intercX, axis=0), "k:")
plt.xlabel("Time bins")
plt.ylabel("Fitted intercepts")
plt.title("Horizontal")

plt.subplot(2, 2, 4)
plt.plot(binCentersT_highRes, np.mean(intercY, axis=0), "ko-")
plt.plot(binCentersT_highRes, np.mean(intercY, axis=0) + sem(intercY, axis=0), "k:")
plt.plot(binCentersT_highRes, np.mean(intercY, axis=0) - sem(intercY, axis=0), "k:")
plt.xlabel("Time bins")
plt.ylabel("Fitted intercepts")
plt.title("Vertical")
plt.tight_layout()

from statsmodels.stats.anova import AnovaRM
import pandas as pd

X_list = {
    "subjects": np.tile(range(np.shape(allErprAmpPerXBin)[2]), np.shape(allErprAmpPerXBin)[0]),
    "timeBin": np.repeat(range(np.shape(allErprAmpPerXBin)[0]), np.shape(allErprAmpPerXBin)[2]),
    "slopesX": np.hstack(slopesX.T),
    "intercX": np.hstack(intercX.T),
}
Y_list = {
    "subjects": np.tile(range(np.shape(allErprAmpPerYBin)[2]), np.shape(allErprAmpPerYBin)[0]),
    "timeBin": np.repeat(range(np.shape(allErprAmpPerYBin)[0]), np.shape(allErprAmpPerYBin)[2]),
    "slopesY": np.hstack(slopesY.T),
    "intercY": np.hstack(intercY.T),
}

X_df = pd.DataFrame(X_list)
Y_df = pd.DataFrame(Y_list)
print("Repeated measures ANOVA on fitted slopes of horizontal X bins")
print(AnovaRM(data=X_df, depvar="slopesX", subject="subjects", within=["timeBin"]).fit())

print("Repeated measures ANOVA on fitted slopes of vertical Y bins")
print(AnovaRM(data=Y_df, depvar="slopesY", subject="subjects", within=["timeBin"]).fit())

print("Repeated measures ANOVA on fitted intercept of horizontal X bins")
print(AnovaRM(data=X_df, depvar="intercX", subject="subjects", within=["timeBin"]).fit())

print("Repeated measures ANOVA on fitted intercept of vertical Y bins")
print(AnovaRM(data=Y_df, depvar="intercY", subject="subjects", within=["timeBin"]).fit())


## t-tests to determine when the shift occurs
for t1 in range(len(binCentersT_highRes)):
    t1data = X_df['slopesX'][X_df['timeBin']==t1]
    for t2 in range(t1+1,len(binCentersT_highRes)):
        t2data = X_df['slopesX'][X_df['timeBin']==t2]
        tresults = ttest_rel(t1data,t2data)
        print('Slopes ' + str(binCentersT_highRes[t1]) + ' vs ' + str(binCentersT_highRes[t2]))
        print(tresults) 



###### plot results across time bins BASED ON 2D MAP - SHOW FITS
colors = plt.cm.jet(np.linspace(0, 1, nTimeBins))

plt.figure(figsize=(12, 12))

countBT = -1
for bT in range(nTimeBins):
    countBT = countBT + 1
    tempDataX_ave = np.nanmean(allErprAmpPerXBin[bT, :, :], axis=1)
    tempDataY_ave = np.nanmean(allErprAmpPerYBin[bT, :, :], axis=1)

    ax = plt.subplot(2, 2, 1)
    ax.plot(
        np.round(binCentersX,0).astype(int),
        tempDataX_ave,
        color=colors[countBT, :],
        alpha=0.2,
        linestyle=":",
    )

    ax.plot(
        binCentersX,
        np.polyval([np.mean(slopesX[:, bT]), np.mean(intercX[:, bT])], binCentersX),
        color=colors[countBT, :],
        alpha=1.0,
    )

    ax = plt.subplot(2, 2, 2)
    ax.plot(
        np.round(binCentersY,0).astype(int),
        tempDataY_ave,
        color=colors[countBT, :],
        alpha=0.2,
        linestyle=":",
    )

    ax.plot(
        binCentersY,
        np.polyval([np.mean(slopesY[:, bT]), np.mean(intercY[:, bT])], binCentersY),
        color=colors[countBT, :],
        alpha=1.0,
    )

    ax = plt.subplot(2, 2, 3)
    ax.plot(
        np.round(binCentersX,0).astype(int),
        tempDataX_ave,
        color=colors[countBT, :],
        alpha=0.2,
        linestyle=":",
    )

    ax.plot(
        binCentersX,
        np.polyval(np.mean(quadX[:, :, bT], axis=1), binCentersX),
        color=colors[countBT, :],
        alpha=1.0,
    )

    ax = plt.subplot(2, 2, 4)
    ax.plot(
        np.round(binCentersY,0).astype(int),
        tempDataY_ave,
        color=colors[countBT, :],
        alpha=0.2,
        linestyle=":",
    )

    ax.plot(
        binCentersY,
        np.polyval(np.mean(quadY[:, :, bT], axis=1), binCentersY),
        color=colors[countBT, :],
        alpha=1.0,
    )


ax = plt.subplot(2, 2, 1)
plt.xlabel("Horizontal position")
plt.ylabel("Pupil response amplitude " + pupilUnit)
plt.title("Per binned horizontal position - Linear")
cbar = plt.colorbar(plt.cm.ScalarMappable(cmap="jet"), ax=ax, ticks=[0, 0.5, 1], label="SOA [ms]")
cbar.ax.set_yticklabels(["0", "200", "400"])

ax = plt.subplot(2, 2, 2)
plt.xlabel("Vertical position")
plt.ylabel("Pupil response amplitude " + pupilUnit)
plt.title("Per binned vertical position - Linear")
cbar = plt.colorbar(plt.cm.ScalarMappable(cmap="jet"), ax=ax, ticks=[0, 0.5, 1], label="SOA [ms]")
cbar.ax.set_yticklabels(["0", "200", "400"])

ax = plt.subplot(2, 2, 3)
plt.xlabel("Horizontal position")
plt.ylabel("Pupil response amplitude " + pupilUnit)
plt.title("Per binned horizontal position - Quadratic")
cbar = plt.colorbar(plt.cm.ScalarMappable(cmap="jet"), ax=ax, ticks=[0, 0.5, 1], label="SOA [ms]")
cbar.ax.set_yticklabels(["0", "200", "400"])

ax = plt.subplot(2, 2, 4)
plt.xlabel("Vertical position")
plt.ylabel("Pupil response amplitude " + pupilUnit)
plt.title("Per binned vertical position - Quadratic")
cbar = plt.colorbar(plt.cm.ScalarMappable(cmap="jet"), ax=ax, ticks=[0, 0.5, 1], label="SOA [ms]")
cbar.ax.set_yticklabels(["0", "200", "400"])

plt.tight_layout()
# plt.legend()
plt.show()
