nTimeBins = np.shape(allErprAmpPerTimeBin)[0]
nSubs = np.shape(allErprAmpPerTimeBin)[1]

# colors = plt.cm.jet(np.linspace(0, 1, nTimeBins))
# colors = plt.cm.nipy_spectral(np.linspace(0, 1, nTimeBins))
colors = plt.cm.rainbow(np.linspace(0, 1, nTimeBins))

plt.figure(figsize=(12, 6))

countBT = -1
for bT in range(nTimeBins):
    countBT = countBT + 1

    ax = plt.subplot(1, 2, 1)
    ax.plot(
        np.repeat(binCentersT[countBT].astype(int), nSubs),
        allErprAmpPerTimeBin[countBT, :],
        color=colors[countBT, :],
        linestyle="",
        marker="d",
        alpha=0.2,
    )
    ax.plot(
        binCentersT[countBT].astype(int),
        np.mean(allErprAmpPerTimeBin[countBT, :]),
        color=colors[countBT, :],
        linestyle="",
        marker="o",
        alpha=1.0,
        markersize=10,
    )

ax = plt.subplot(1, 2, 1)
plt.xlabel("Time bin")
plt.ylabel("Pupil response amplitude " + pupilUnit)
plt.title("Per binned probe time")
cbar = plt.colorbar(plt.cm.ScalarMappable(cmap="jet"), ax=ax, ticks=[0, 0.5, 1], label="SOA [ms]")
cbar.ax.set_yticklabels(["0", "200", "400"])


countBT = -1
for bT in range(nTimeBins):
    countBT = countBT + 1

    ax = plt.subplot(1, 2, 2)
    ax.plot(
        np.linspace(0, 2, 2000),
        np.nanmean(allErprMatPerTimeBin[countBT, :, :], axis=1),
        color=colors[countBT, :],
        alpha=1.0,
    )

ax = plt.subplot(1, 2, 2)
plt.xlabel("Time [s]")
plt.ylabel("Pupil size " + pupilUnit)
plt.title("Per binned probe time")
cbar = plt.colorbar(plt.cm.ScalarMappable(cmap="jet"), ax=ax, ticks=[0, 0.5, 1], label="SOA [ms]")
cbar.ax.set_yticklabels(["0", "200", "400"])

plt.savefig("Figures/Pupil_responses_perSOA.svg")

plt.show()