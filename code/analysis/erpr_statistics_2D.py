from scipy.stats import sem, ttest_rel
from matplotlib.patches import Polygon

# allErprAmp2DMapPerTimeBin_ref = allErprAmp2DMapPerTimeBin.copy()

nTimeBins = np.shape(allErprAmp2DMapPerTimeBin_ref)[2]

nPP = len(allPPName)

allErprAmpPerXBin = np.zeros((nTimeBins, nXBins, nPP))
allErprAmpPerXBin[:] = np.nan
allErprAmpPerYBin = np.zeros((nTimeBins, nYBins, nPP))
allErprAmpPerYBin[:] = np.nan


countPP = 0
for ppName in allPPName:

    # tempAmp2DMap_pp = allErprAmp2DMapPerTimeBin[:, :, :, countPP]

    # refAmp2DMap_pp = np.nanmean(tempAmp2DMap_pp, axis=2)

    for bT in range(nTimeBins):
        # allErprAmp2DMapPerTimeBin_ref[:, :, bT, countPP] = (
            # allErprAmp2DMapPerTimeBin_ref[:, :, bT, countPP] - refAmp2DMap_pp
        # )

        allErprAmpPerXBin[bT, :, countPP] = np.nanmean(
            allErprAmp2DMapPerTimeBin_ref[:, :, bT, countPP], axis=0
        )
        allErprAmpPerYBin[bT, :, countPP] = np.nanmean(
            allErprAmp2DMapPerTimeBin_ref[:, :, bT, countPP], axis=1
        )

    countPP += 1


###### plot results across time bins BASED ON 2D MAP
# colors = plt.cm.jet(np.linspace(0, 1, nTimeBins))
colors = plt.cm.rainbow(np.linspace(0, 1, nTimeBins))
# colors = plt.cm.coolwarm(np.linspace(0, 1, nTimeBins))

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
# cbar = plt.colorbar(plt.cm.ScalarMappable(cmap="jet"), ax=ax, ticks=[0, 0.5, 1], label="SOA [ms]")
cbar = plt.colorbar(plt.cm.ScalarMappable(cmap="rainbow"), ax=ax, ticks=[0, 0.5, 1], label="SOA [ms]")

cbar.ax.set_yticklabels(["0", "200", "400"])

ax = plt.subplot(1, 2, 2)
plt.xlabel("Vertical position")
plt.ylabel("Pupil response amplitude " + pupilUnit)
plt.title("Per binned vertical position")
cbar = plt.colorbar(plt.cm.ScalarMappable(cmap="rainbow"), ax=ax, ticks=[0, 0.5, 1], label="SOA [ms]")
cbar.ax.set_yticklabels(["0", "200", "400"])

plt.tight_layout()
plt.savefig("Figures/Pupil_horizontal_perPOA.svg")

# plt.legend()
plt.show()


# # post-hoc testing
# firstTimeBinX = allErprAmpPerXBin[0, :, :]
# secondlastTimeBinX = allErprAmpPerXBin[-2, :, :]
# lastTimeBinX = allErprAmpPerXBin[-1, :, :]
# firstTimeBinY = allErprAmpPerYBin[0, :, :]
# secondlastTimeBinY = allErprAmpPerYBin[-2, :, :]
# lastTimeBinY = allErprAmpPerYBin[-1, :, :]

# allTX = []
# allPX = []
# for i in range(np.shape(firstTimeBinX)[0]):
#     tstat_result = ttest_rel(firstTimeBinX[i, :], lastTimeBinX[i, :])
#     allTX.append(tstat_result.statistic)
#     allPX.append(tstat_result.pvalue)
# allTY = []
# allPY = []
# for i in range(np.shape(firstTimeBinY)[0]):
#     tstat_result = ttest_rel(firstTimeBinY[i, :], lastTimeBinY[i, :])
#     allTY.append(tstat_result.statistic)
#     allPY.append(tstat_result.pvalue)

# plt.figure(figsize=(12, 6))
# plt.subplot(1, 2, 1)
# plt.plot(binCentersX, allPX)
# plt.plot([binCentersX[0], binCentersX[-1]], [0.05, 0.05], "k:")
# plt.xlabel("Horizontal position")
# plt.ylabel("p-value")
# plt.title("First vs. last bin")
# plt.subplot(1, 2, 2)
# plt.plot(binCentersY, allPY)
# plt.plot([binCentersY[0], binCentersY[-1]], [0.05, 0.05], "k:")
# plt.xlabel("Vertical position")
# plt.ylabel("p-value")
# plt.title("First vs. last bin")
# plt.tight_layout()


# allTX = []
# allPX = []
# for i in range(np.shape(firstTimeBinX)[0]):
#     tstat_result = ttest_rel(firstTimeBinX[i, :], secondlastTimeBinX[i, :])
#     allTX.append(tstat_result.statistic)
#     allPX.append(tstat_result.pvalue)
# allTY = []
# allPY = []
# for i in range(np.shape(firstTimeBinY)[0]):
#     tstat_result = ttest_rel(firstTimeBinY[i, :], secondlastTimeBinY[i, :])
#     allTY.append(tstat_result.statistic)
#     allPY.append(tstat_result.pvalue)

# plt.figure(figsize=(12, 6))
# plt.subplot(1, 2, 1)
# plt.plot(binCentersX, allPX)
# plt.plot([binCentersX[0], binCentersX[-1]], [0.05, 0.05], "k:")
# plt.xlabel("Horizontal position")
# plt.ylabel("p-value")
# plt.title("First vs. 2nd last bin")
# plt.subplot(1, 2, 2)
# plt.plot(binCentersY, allPY)
# plt.plot([binCentersY[0], binCentersY[-1]], [0.05, 0.05], "k:")
# plt.xlabel("Vertical position")
# plt.ylabel("p-value")
# plt.title("First vs. 2nd last bin")
# plt.tight_layout()


# fitting slope to data across horizontal X/Y space
slopesX = np.zeros((np.shape(allErprAmpPerXBin)[2], np.shape(allErprAmpPerXBin)[0]))
intercX = np.zeros((np.shape(allErprAmpPerXBin)[2], np.shape(allErprAmpPerXBin)[0]))
slopesY = np.zeros((np.shape(allErprAmpPerYBin)[2], np.shape(allErprAmpPerYBin)[0]))
intercY = np.zeros((np.shape(allErprAmpPerYBin)[2], np.shape(allErprAmpPerYBin)[0]))

nFitOrd = 3
quadX = np.zeros((nFitOrd, np.shape(allErprAmpPerXBin)[2], np.shape(allErprAmpPerXBin)[0]))
quadY = np.zeros((nFitOrd, np.shape(allErprAmpPerYBin)[2], np.shape(allErprAmpPerYBin)[0]))
cubicX = np.zeros((nFitOrd+1, np.shape(allErprAmpPerXBin)[2], np.shape(allErprAmpPerXBin)[0]))
cubicY = np.zeros((nFitOrd+1, np.shape(allErprAmpPerYBin)[2], np.shape(allErprAmpPerYBin)[0]))

for t in range(np.shape(allErprAmpPerXBin)[0]):
    for s in range(np.shape(allErprAmpPerXBin)[2]):
        pfX = np.polyfit(binCentersX, allErprAmpPerXBin[t, :, s], 1)
        intercX[s, t] = pfX[1]
        slopesX[s, t] = pfX[0]
        pfY = np.polyfit(binCentersY, allErprAmpPerYBin[t, :, s], 1)
        intercY[s, t] = pfY[1]
        slopesY[s, t] = pfY[0]

        quadX[:, s, t] = np.polyfit(binCentersX, allErprAmpPerXBin[t, :, s], nFitOrd-1)
        quadY[:, s, t] = np.polyfit(binCentersY, allErprAmpPerYBin[t, :, s], nFitOrd-1)


        cubicX[:, s, t] = np.polyfit(binCentersX, allErprAmpPerXBin[t, :, s], nFitOrd)
        cubicY[:, s, t] = np.polyfit(binCentersY, allErprAmpPerYBin[t, :, s], nFitOrd)

binCentersT = np.zeros((nTimeBins))
for bT in range(nTimeBins):
    binCentersT[bT] = np.mean([binEdgesT[bT], binEdgesT[bT + 1]])

plt.figure()
plt.subplot(2, 2, 1)
plt.plot(binCentersT, np.mean(slopesX, axis=0), "ko-")
plt.plot(binCentersT, np.mean(slopesX, axis=0) + sem(slopesX, axis=0), "k:")
plt.plot(binCentersT, np.mean(slopesX, axis=0) - sem(slopesX, axis=0), "k:")
plt.xticks(binCentersT)
plt.xlabel("Time bins")
plt.ylabel("Fitted slopes")
plt.title("Horizontal")

plt.subplot(2, 2, 2)
plt.plot(binCentersT, np.mean(slopesY, axis=0), "ko-")
plt.plot(binCentersT, np.mean(slopesY, axis=0) + sem(slopesY, axis=0), "k:")
plt.plot(binCentersT, np.mean(slopesY, axis=0) - sem(slopesY, axis=0), "k:")
plt.xticks(binCentersT)
plt.xlabel("Time bins")
plt.ylabel("Fitted slopes")
plt.title("Vertical")

plt.subplot(2, 2, 3)
plt.plot(binCentersT, np.mean(intercX, axis=0), "ko-")
plt.plot(binCentersT, np.mean(intercX, axis=0) + sem(intercX, axis=0), "k:")
plt.plot(binCentersT, np.mean(intercX, axis=0) - sem(intercX, axis=0), "k:")
plt.xticks(binCentersT)
plt.xlabel("Time bins")
plt.ylabel("Fitted intercepts")
plt.title("Horizontal")

plt.subplot(2, 2, 4)
plt.plot(binCentersT, np.mean(intercY, axis=0), "ko-")
plt.plot(binCentersT, np.mean(intercY, axis=0) + sem(intercY, axis=0), "k:")
plt.plot(binCentersT, np.mean(intercY, axis=0) - sem(intercY, axis=0), "k:")
plt.xticks(binCentersT)
plt.xlabel("Time bins")
plt.ylabel("Fitted intercepts")
plt.title("Vertical")
plt.tight_layout()

plt.savefig("Figures/Pupil_horizontal_slopes_perPOA.svg")


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
for t1 in range(len(binCentersT)):
    t1data = X_df['slopesX'][X_df['timeBin']==t1]
    for t2 in range(t1+1,len(binCentersT)):
        t2data = X_df['slopesX'][X_df['timeBin']==t2]
        tresults = ttest_rel(t1data,t2data)
        print('Slopes ' + str(binCentersT[t1]) + ' vs ' + str(binCentersT[t2]))
        print(tresults) 







# ###### plot results across time bins BASED ON 2D MAP - SHOW FITS
# colors = plt.cm.rainbow(np.linspace(0, 1, nTimeBins))

# plt.figure(figsize=(12, 12))

# countBT = -1
# for bT in range(nTimeBins):
#     countBT = countBT + 1
#     tempDataX_ave = np.nanmean(allErprAmpPerXBin[bT, :, :], axis=1)
#     tempDataY_ave = np.nanmean(allErprAmpPerYBin[bT, :, :], axis=1)

#     ax = plt.subplot(3, 2, 1)
#     ax.plot(
#         np.round(binCentersX,0).astype(int),
#         tempDataX_ave,
#         color=colors[countBT, :],
#         alpha=0.4,
#         linestyle=":",
#     )

#     ax.plot(
#         binCentersX,
#         np.polyval([np.mean(slopesX[:, bT]), np.mean(intercX[:, bT])], binCentersX),
#         color=colors[countBT, :],
#         alpha=1.0,
#     )

#     ax = plt.subplot(3, 2, 2)
#     ax.plot(
#         np.round(binCentersY,0).astype(int),
#         tempDataY_ave,
#         color=colors[countBT, :],
#         alpha=0.4,
#         linestyle=":",
#     )

#     ax.plot(
#         binCentersY,
#         np.polyval([np.mean(slopesY[:, bT]), np.mean(intercY[:, bT])], binCentersY),
#         color=colors[countBT, :],
#         alpha=1.0,
#     )

#     ax = plt.subplot(3, 2, 3)
#     ax.plot(
#         np.round(binCentersX,0).astype(int),
#         tempDataX_ave,
#         color=colors[countBT, :],
#         alpha=0.4,
#         linestyle=":",
#     )

#     ax.plot(
#         binCentersX,
#         np.polyval(np.mean(quadX[:, :, bT], axis=1), binCentersX),
#         color=colors[countBT, :],
#         alpha=1.0,
#     )

#     ax = plt.subplot(3, 2, 4)
#     ax.plot(
#         np.round(binCentersY,0).astype(int),
#         tempDataY_ave,
#         color=colors[countBT, :],
#         alpha=0.4,
#         linestyle=":",
#     )

#     ax.plot(
#         binCentersY,
#         np.polyval(np.mean(quadY[:, :, bT], axis=1), binCentersY),
#         color=colors[countBT, :],
#         alpha=1.0,
#     )


#     ax = plt.subplot(3, 2, 5)
#     ax.plot(
#         np.round(binCentersX,0).astype(int),
#         tempDataX_ave,
#         color=colors[countBT, :],
#         alpha=0.4,
#         linestyle=":",
#     )

#     ax.plot(
#         binCentersX,
#         np.polyval(np.mean(cubicX[:, :, bT], axis=1), binCentersX),
#         color=colors[countBT, :],
#         alpha=1.0,
#     )

#     ax = plt.subplot(3, 2, 6)
#     ax.plot(
#         np.round(binCentersY,0).astype(int),
#         tempDataY_ave,
#         color=colors[countBT, :],
#         alpha=0.4,
#         linestyle=":",
#     )

#     ax.plot(
#         binCentersY,
#         np.polyval(np.mean(cubicY[:, :, bT], axis=1), binCentersY),
#         color=colors[countBT, :],
#         alpha=1.0,
#     )


# ax = plt.subplot(3, 2, 1)
# plt.xlabel("Horizontal position")
# plt.ylabel("Pupil response amplitude " + pupilUnit)
# plt.title("Per binned horizontal position - Linear")
# cbar = plt.colorbar(plt.cm.ScalarMappable(cmap="rainbow"), ax=ax, ticks=[0, 0.5, 1], label="SOA [ms]")
# cbar.ax.set_yticklabels(["0", "200", "400"])

# ax = plt.subplot(3, 2, 2)
# plt.xlabel("Vertical position")
# plt.ylabel("Pupil response amplitude " + pupilUnit)
# plt.title("Per binned vertical position - Linear")
# cbar = plt.colorbar(plt.cm.ScalarMappable(cmap="rainbow"), ax=ax, ticks=[0, 0.5, 1], label="SOA [ms]")
# cbar.ax.set_yticklabels(["0", "200", "400"])

# ax = plt.subplot(3, 2, 3)
# plt.xlabel("Horizontal position")
# plt.ylabel("Pupil response amplitude " + pupilUnit)
# plt.title("Per binned horizontal position - Quadratic")
# cbar = plt.colorbar(plt.cm.ScalarMappable(cmap="rainbow"), ax=ax, ticks=[0, 0.5, 1], label="SOA [ms]")
# cbar.ax.set_yticklabels(["0", "200", "400"])

# ax = plt.subplot(3, 2, 4)
# plt.xlabel("Vertical position")
# plt.ylabel("Pupil response amplitude " + pupilUnit)
# plt.title("Per binned vertical position - Quadratic")
# cbar = plt.colorbar(plt.cm.ScalarMappable(cmap="rainbow"), ax=ax, ticks=[0, 0.5, 1], label="SOA [ms]")
# cbar.ax.set_yticklabels(["0", "200", "400"])

# ax = plt.subplot(3, 2, 5)
# plt.xlabel("Horizontal position")
# plt.ylabel("Pupil response amplitude " + pupilUnit)
# plt.title("Per binned horizontal position - Cubic")
# cbar = plt.colorbar(plt.cm.ScalarMappable(cmap="rainbow"), ax=ax, ticks=[0, 0.5, 1], label="SOA [ms]")
# cbar.ax.set_yticklabels(["0", "200", "400"])

# ax = plt.subplot(3, 2, 6)
# plt.xlabel("Vertical position")
# plt.ylabel("Pupil response amplitude " + pupilUnit)
# plt.title("Per binned vertical position - Cubic")
# cbar = plt.colorbar(plt.cm.ScalarMappable(cmap="rainbow"), ax=ax, ticks=[0, 0.5, 1], label="SOA [ms]")
# cbar.ax.set_yticklabels(["0", "200", "400"])

# plt.tight_layout()
# # plt.legend()
# plt.show()



### fit normal distributions

from scipy.optimize import curve_fit
from scipy.stats import norm

def fitTruncNorm(x,loc,scale,amplitude,intercept):
    yfit = norm.pdf(x,loc=loc,scale=scale)
    yfit = yfit*amplitude+intercept
    return yfit


# plt.plot(x,y,'k-o', lw=2, alpha=0.6, label='Actual')
# plt.plot(x,fitY,'b:x', lw=2, alpha=0.6, label='Fitted')



fitParamLabels = ['Mean','Var.','Amp','Intercept']

###### plot results across time bins BASED ON 2D MAP - POOLED
colors = plt.cm.rainbow(np.linspace(0, 1, nTimeBins))

plt.figure(figsize=(12, 12))

allPopt = np.zeros((4,nTimeBins))
countBT = -1
for bT in range(nTimeBins):
    countBT = countBT + 1

    y = np.nanmean(allErprAmpPerXBin[bT, :, :], axis=1)
    x = np.linspace(-2,2,len(y))
    popt, pcov = curve_fit(fitTruncNorm,x,y,bounds=([-2,0.1,0,-0.1],[2,1.2,2,0.1]))
    fitY = fitTruncNorm(x,popt[0],popt[1],popt[2],popt[3])

    allPopt[:,bT] = popt

    ax = plt.subplot(3, 2, 1)
    ax.plot(
        np.round(binCentersX,0).astype(int),
        y,
        color=colors[countBT, :],
        alpha=0.4,
        linestyle=":",
    )

    ax.plot(
        binCentersX,
        fitY,
        color=colors[countBT, :],
        alpha=1.0,
    )

ax = plt.subplot(3, 2, 1)
plt.xlabel("Horizontal position")
plt.ylabel("Pupil response amplitude " + pupilUnit)
plt.title("Normal distribution fit - pooled")

for k in range(4):
    ax = plt.subplot(3, 2, 3+k)
    ax.plot(np.round(binCentersT).astype('int'),allPopt[k,:],'ko-')

    plt.xlabel("Time interval (ms)")
    plt.ylabel(fitParamLabels[k])


plt.savefig("Figures/Pupil_normFits_perPOA_pooled.svg")



##############

###### plot results across time bins BASED ON 2D MAP - PER SUBJECT
colors = plt.cm.rainbow(np.linspace(0, 1, nTimeBins))

plt.figure(figsize=(12, 12))

allPopt = np.zeros((4,nTimeBins,nSubs))
countBT = -1
for bT in range(nTimeBins):
    countBT = countBT + 1
    for s in range(nSubs):
        y = allErprAmpPerXBin[bT, :, s]
        x = np.linspace(-2,2,len(y))
        popt, pcov = curve_fit(fitTruncNorm,x,y,bounds=([-np.inf,0.1,0.1,-0.1],[np.inf,2.0,0.5,0.05]))
        # popt, pcov = curve_fit(fitTruncNorm,x,y,bounds=([-np.inf,-np.inf,0,-0.3],[np.inf,np.inf,0.5,0.3]))
        # popt, pcov = curve_fit(fitTruncNorm,x,y)
        # fitY = fitTruncNorm(x,popt[0],popt[1],popt[2],popt[3])

        allPopt[:,bT,s] = popt

    allPopt_ave = np.median(allPopt,axis=2)

    fitY = fitTruncNorm(x,allPopt_ave[0,bT],allPopt_ave[1,bT],allPopt_ave[2,bT],allPopt_ave[3,bT])

    ax = plt.subplot(3, 2, 1)
    ax.plot(
        np.round(binCentersX,0).astype(int),
        np.mean(allErprAmpPerXBin[bT, :, ],axis=1),
        color=colors[countBT, :],
        alpha=0.4,
        linestyle=":",
    )

    ax.plot(
        binCentersX,
        fitY,
        color=colors[countBT, :],
        alpha=1.0,
    )

ax = plt.subplot(3, 2, 1)
plt.xlabel("Horizontal position")
plt.ylabel("Pupil response amplitude " + pupilUnit)
plt.title("Normal distribution fit - pooled")

for k in range(4):
    ax = plt.subplot(3, 2, 3+k)
    ax.plot(np.round(binCentersT).astype('int'),allPopt_ave[k,:],'ko-')

    plt.xlabel("Time interval (ms)")
    plt.ylabel(fitParamLabels[k])


plt.savefig("Figures/Pupil_normFits_perPOA_averaged.svg")



# from statsmodels.stats.anova import AnovaRM
# import pandas as pd

# X_list = {
#     "subjects": np.tile(range(np.shape(allErprAmpPerXBin)[2]), np.shape(allErprAmpPerXBin)[0]),
#     "timeBin": np.repeat(range(np.shape(allErprAmpPerXBin)[0]), np.shape(allErprAmpPerXBin)[2]),
#     "meanX": np.hstack(allPopt[0,:,:].T),
#     "ampX": np.hstack(allPopt[2,:,:].T),
# }
# # Y_list = {
# #     "subjects": np.tile(range(np.shape(allErprAmpPerYBin)[2]), np.shape(allErprAmpPerYBin)[0]),
# #     "timeBin": np.repeat(range(np.shape(allErprAmpPerYBin)[0]), np.shape(allErprAmpPerYBin)[2]),
# #     "slopesY": np.hstack(slopesY.T),
# #     "intercY": np.hstack(intercY.T),
# # }

# X_df = pd.DataFrame(X_list)
# # Y_df = pd.DataFrame(Y_list)
# print("Repeated measures ANOVA on fitted slopes of horizontal X bins")
# print(AnovaRM(data=X_df, depvar="meanX", subject="subjects", within=["timeBin"]).fit())

# # print("Repeated measures ANOVA on fitted slopes of vertical Y bins")
# # print(AnovaRM(data=Y_df, depvar="slopesY", subject="subjects", within=["timeBin"]).fit())

# print("Repeated measures ANOVA on fitted intercept of horizontal X bins")
# print(AnovaRM(data=X_df, depvar="ampX", subject="subjects", within=["timeBin"]).fit())

# # print("Repeated measures ANOVA on fitted intercept of vertical Y bins")
# # print(AnovaRM(data=Y_df, depvar="intercY", subject="subjects", within=["timeBin"]).fit())

# ## t-tests to determine when the shift occurs
# for t1 in range(len(binCentersT)):
#     t1data = X_df['slopesX'][X_df['timeBin']==t1]
#     for t2 in range(t1+1,len(binCentersT)):
#         t2data = X_df['slopesX'][X_df['timeBin']==t2]
#         tresults = ttest_rel(t1data,t2data)
#         print('Slopes ' + str(binCentersT[t1]) + ' vs ' + str(binCentersT[t2]))
#         print(tresults) 



# ### fit skewed normal distributions
# from scipy.optimize import curve_fit
# from scipy.stats import skewnorm

# def fitTruncSkewNorm(x,skewness,loc,scale,amplitude,intercept):
#     yfit = skewnorm.pdf(x,skewness, loc=loc,scale=scale)
#     yfit = yfit*amplitude+intercept
#     return yfit


# # plt.plot(x,y,'k-o', lw=2, alpha=0.6, label='Actual')
# # plt.plot(x,fitY,'b:x', lw=2, alpha=0.6, label='Fitted')


# ###### plot results across time bins BASED ON 2D MAP - SHOW FITS


# fitParamLabels = ['Skewness','Mean','Var.','Amp','Intercept']

# colors = plt.cm.rainbow(np.linspace(0, 1, nTimeBins))

# plt.figure(figsize=(12, 12))

# allPopt = np.zeros((5,nTimeBins))
# countBT = -1
# for bT in range(nTimeBins):
#     countBT = countBT + 1

#     y = np.nanmean(allErprAmpPerXBin[bT, :, :], axis=1)
#     x = np.linspace(-2,2,len(y))
#     popt, pcov = curve_fit(fitTruncSkewNorm,x,y,bounds=([-10,-2,0.1,0,-0.1],[10,2,1.2,2,0.1]))
#     # popt, pcov = curve_fit(fitTruncSkewNorm,x,y)
#     fitY = fitTruncSkewNorm(x,popt[0],popt[1],popt[2],popt[3],popt[4])

#     allPopt[:,bT] = popt

#     ax = plt.subplot(3, 2, 1)
#     ax.plot(
#         np.round(binCentersX,0).astype(int),
#         y,
#         color=colors[countBT, :],
#         alpha=0.4,
#         linestyle=":",
#     )

#     ax.plot(
#         binCentersX,
#         fitY,
#         color=colors[countBT, :],
#         alpha=1.0,
#     )

# ax = plt.subplot(3, 2, 1)
# plt.xlabel("Horizontal position")
# plt.ylabel("Pupil response amplitude " + pupilUnit)
# plt.title("Skewed normal distribution fit")

# for k in range(5):
#     ax = plt.subplot(3, 2, 2+k)
#     ax.plot(np.round(binCentersT).astype('int'),allPopt[k,:],'ko-')

#     plt.xlabel("Time interval (ms)")
#     plt.ylabel(fitParamLabels[k])