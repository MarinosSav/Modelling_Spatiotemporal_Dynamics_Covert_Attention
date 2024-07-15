
varCBarRange = [
    np.percentile(allErprAmp2DMap_ave[:], 1),
    np.percentile(allErprAmp2DMap_ave[:], 99),
]

plt.figure()
plt.imshow(
    allErprAmp2DMap_ave,
    cmap=colormapper,
    vmin=varCBarRange[0],
    vmax=varCBarRange[1],
)
plt.xticks(range(0, nXBins), labels=np.round(binCentersX[0:nXBins],0).astype(int))
plt.yticks(range(0, nYBins), labels=np.round(binCentersY[0:nYBins],0).astype(int))
plt.xlim([-1, nXBins])
plt.ylim([-1, nYBins])

plt.xlabel("Horizontal position")
plt.ylabel("Vertical position")
plt.title("Averaged across all time bins (reference)")
plt.colorbar(label="AUC Pupil Response")

plt.savefig("Figures/PupilImaging_average.svg")


varCBarRange = [
    np.percentile(allErprAmp2DMapPerTarLoc_ave[:], 1),
    np.percentile(allErprAmp2DMapPerTarLoc_ave[:], 99),
]

plt.figure()
for TL in range(nTarLocs):
    plt.subplot(2,2,TL+1)
    plt.imshow(
        allErprAmp2DMapPerTarLoc_ave[:,:,TL],
        cmap=colormapper,
        vmin=varCBarRange[0],
        vmax=varCBarRange[1],
    )
    plt.xticks(range(0, nXBins), labels=np.round(binCentersX[0:nXBins],0).astype(int))
    plt.yticks(range(0, nYBins), labels=np.round(binCentersY[0:nYBins],0).astype(int))
    plt.xlim([-1, nXBins])
    plt.ylim([-1, nYBins])

    plt.xlabel("Horizontal position")
    plt.ylabel("Vertical position")
    plt.title("Averaged (reference) - " + tarLocNames[TL])
    # plt.colorbar(label="AUC Pupil Response")
plt.tight_layout()

plt.savefig("Figures/PupilImaging_average_perTarLoc.svg")