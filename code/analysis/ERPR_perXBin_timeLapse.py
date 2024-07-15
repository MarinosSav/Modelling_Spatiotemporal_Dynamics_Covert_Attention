# colors = plt.cm.coolwarm(np.linspace(0, 1, np.max(nTimeBinsArr)))
colors = plt.cm.jet(np.linspace(0, 1, np.max(nTimeBinsArr)))
# plt.figure()

if showPlotsPerObs:

    plt.figure(figsize=(12, 6))
    ax = plt.subplot(1, 2, 1)

    countBT = -1
    for bT in range(np.max(nTimeBinsArr)):
        countBT = countBT + 1
        tempMap = erprAmp2DMapPerTimeBin_ave[:, :, countBT]

        ax.plot(binCentersX.astype(int), np.mean(tempMap, axis=0), color=colors[countBT, :])

    plt.xlabel("Horizontal position")
    plt.ylabel("Pupil response amplitude " + pupilUnit)
    plt.title("Per binned horizontal position")
    cbar = plt.colorbar(
        plt.cm.ScalarMappable(cmap="jet"), ax=ax, ticks=[0, 0.5, 1], label="SOA [ms]"
    )
    cbar.ax.set_yticklabels(["0", "200", "400"])
    # plt.legend()
    plt.show()
