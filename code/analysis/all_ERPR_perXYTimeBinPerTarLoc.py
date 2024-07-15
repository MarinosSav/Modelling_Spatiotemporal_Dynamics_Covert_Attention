nXBins = np.shape(allErprAmp2DMapPerTimeBinPerTarLoc_ave_ref)[1]
nYBins = np.shape(allErprAmp2DMapPerTimeBinPerTarLoc_ave_ref)[0]


varCBarRange = [
    np.percentile(allErprAmp2DMapPerTimeBinPerTarLoc_ave_ref[:], 1),
    np.percentile(allErprAmp2DMapPerTimeBinPerTarLoc_ave_ref[:], 99),
]


nTimeBins = np.shape(allErprAmp2DMapPerTimeBinPerTarLoc_ave_ref)[2]
nTarLocs = np.shape(allErprAmp2DMapPerTimeBinPerTarLoc_ave_ref)[3]

# print(nTimeBins)
# print(nTarLocs)
plt.figure(figsize=(10, 8))
countBT = -1
for bT in range(nTimeBins):
    countBT = countBT + 1
    for TL in range(nTarLocs):

        plt.subplot(nTimeBins, nTarLocs, (countBT*nTarLocs) + 1 + TL)
        plt.imshow(
            allErprAmp2DMapPerTimeBinPerTarLoc_ave_ref[:, :, countBT,TL],
            cmap=colormapper,
            vmin=varCBarRange[0],
            vmax=varCBarRange[1],
        )
        plt.xticks(range(0, nXBins), labels=binCentersX[0:nXBins].astype(int))
        plt.yticks(range(0, nYBins), labels=binCentersY[0:nYBins].astype(int))
        plt.xlim([-1, nXBins])
        plt.ylim([-1, nYBins])

        plt.xlabel("Horizontal position")
        plt.ylabel("Vertical position")
        plt.title("SOA: " + str(int(binEdgesT[bT])) + "-" + str(int(binEdgesT[bT + 1])) + "-" + tarLocNames[TL])

        # plt.colorbar(label="AUC Pupil Response")

plt.tight_layout()
plt.show()
