
# colormapper = 'hot'

nXBins = np.shape(allErprAmp2DMapPerTimeBin_ave_ref)[1]
nYBins = np.shape(allErprAmp2DMapPerTimeBin_ave_ref)[0]


varCBarRange = [
    np.percentile(allErprAmp2DMapPerTimeBin_ave_ref[:], 1),
    np.percentile(allErprAmp2DMapPerTimeBin_ave_ref[:], 99),
]

# varCBarRange = [
#     np.percentile(allErprAmp2DMapPerTimeBin_ave_ref[:], 0),
#     np.percentile(allErprAmp2DMapPerTimeBin_ave_ref[:], 100),
# ]

nTimeBins = np.shape(allErprAmp2DMapPerTimeBin_ave_ref)[2]

fig = plt.figure(figsize=(7, 2.2 * nTimeBins))
countBT = -1
for bT in range(nTimeBins):
    countBT = countBT + 1

    plt.subplot(nTimeBins, 1, countBT + 1)
    plt.imshow(
        allErprAmp2DMapPerTimeBin_ave_ref[:, :, countBT],
        cmap=colormapper,
        vmin=varCBarRange[0],
        vmax=varCBarRange[1],
    )
    plt.xticks(range(0, nXBins), labels=[]*nXBins)
    plt.yticks(range(0, nYBins), labels=np.round(binCentersY[0:nYBins],0).astype(int))
    plt.ylim([-1, nYBins])
    plt.xlim([-1, nXBins])

    if bT == nTimeBins-1:
        plt.xlabel("Horizontal position")
        plt.xticks(range(0, nXBins), labels=np.round(binCentersX[0:nXBins], 0).astype(int))
    plt.colorbar(label="Pupil amplitude (z)")

    plt.ylabel("Vertical position")
    plt.title("SOA: " + str(int(binEdgesT[bT])) + "-" + str(int(binEdgesT[bT + 1])))

#plt.colorbar(label="Pupil amplitude (z)")
plt.tight_layout()
plt.subplots_adjust(hspace=-0.2, wspace=0)

plt.savefig("Figures/PupilImaging_perPOA.svg")

plt.show()