# ########### BASED ON 2D MAPS
# nTimeBins = np.shape(allErprAmp2DMapPerTimeBin_ave_ref)[2]
# nXBins = np.shape(allErprAmp2DMapPerTimeBin_ave_ref)[1]
# nYBins = np.shape(allErprAmp2DMapPerTimeBin_ave_ref)[0]

# colors = plt.cm.jet(np.linspace(0, 1, nTimeBins))

# plt.figure(figsize=(12, 6))

# allPerXBin = np.zeros((nTimeBins, nXBins))
# allPerXBin[:] = np.nan
# allPerYBin = np.zeros((nTimeBins, nYBins))
# allPerYBin[:] = np.nan

# countBT = -1
# for bT in range(nTimeBins):
#     countBT = countBT + 1
#     tempMap = allErprAmp2DMapPerTimeBin_ave_ref[:, :, countBT]

#     allPerXBin[bT, :] = np.mean(tempMap, axis=0)
#     allPerYBin[bT, :] = np.mean(tempMap, axis=1)

#     ax = plt.subplot(1, 2, 1)
#     ax.plot(round(binCentersX,1), allPerXBin[bT, :], color=colors[countBT, :])
#     ax = plt.subplot(1, 2, 2)
#     ax.plot(round(binCentersY,1), allPerYBin[bT, :], color=colors[countBT, :])

# ax = plt.subplot(1, 2, 1)
# plt.xlabel("Horizontal position")
# plt.ylabel("Pupil response amplitude " + pupilUnit)
# plt.title("Per binned horizontal position")
# cbar = plt.colorbar(plt.cm.ScalarMappable(cmap="jet"), ax=ax, ticks=[0, 0.5, 1], label="SOA [ms]")
# cbar.ax.set_yticklabels(["0", "200", "400"])

# ax = plt.subplot(1, 2, 2)
# plt.xlabel("Vertical position")
# plt.ylabel("Pupil response amplitude " + pupilUnit)
# plt.title("Per binned vertical position")
# cbar = plt.colorbar(plt.cm.ScalarMappable(cmap="jet"), ax=ax, ticks=[0, 0.5, 1], label="SOA [ms]")
# cbar.ax.set_yticklabels(["0", "200", "400"])
# # plt.legend()
# plt.show()


########### BASED ON JUST X/YBINS
nTimeBins = np.shape(allErprAmpPerXPerTimeBin_ave_ref)[1]
nXBins = np.shape(allErprAmpPerXPerTimeBin_ave_ref)[0]
nYBins = np.shape(allErprAmpPerYPerTimeBin_ave_ref)[0]

colors = plt.cm.jet(np.linspace(0, 1, nTimeBins))

plt.figure(figsize=(12, 6))

allPerXBin = allErprAmpPerXPerTimeBin_ave_ref.T
allPerYBin = allErprAmpPerYPerTimeBin_ave_ref.T

countBT = -1
for bT in range(nTimeBins):
    countBT = countBT + 1

    ax = plt.subplot(1, 2, 1)
    ax.plot(round(binCentersX,1), allPerXBin[bT, :], color=colors[countBT, :])
    ax = plt.subplot(1, 2, 2)
    ax.plot(round(binCentersY,1), allPerYBin[bT, :], color=colors[countBT, :])

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
# plt.legend()
plt.show()
