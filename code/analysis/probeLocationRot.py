probeLocationRot = np.zeros((len(probeLocation), 2))
probeLocationCoor = np.zeros((len(probeLocation), 2))

# calculate probeLocationRot
# plot all probe locations
cueLocCoor = {}
cueLocCoor["DOWN"] = [0, -8]
cueLocCoor["UP"] = [0, 8]
cueLocCoor["LEFT"] = [-8, 0]
cueLocCoor["RIGHT"] = [8, 0]


for probeLocationInd in range(len(probeLocation)):
    tempCoor = probeLocation[probeLocationInd].split(",")
    tempCoor[0] = float(tempCoor[0][1:])
    tempCoor[1] = float(tempCoor[1][1:-1])

    rho, phi = cart2pol(tempCoor[0], tempCoor[1])
    rhoCue, phiCue = cart2pol(
        cueLocCoor[cueLocation[probeLocationInd]][0],
        cueLocCoor[cueLocation[probeLocationInd]][1],
    )

    if targetDirection[probeLocationInd] == "CCW":
        if cueLocation[probeLocationInd] == "DOWN":
            phi = phi - 0.25 * np.pi
            phiCue = phiCue - 0.25 * np.pi
        elif cueLocation[probeLocationInd] == "LEFT":
            phi = phi - 0.25 * np.pi + 0.5 * np.pi
            phiCue = phiCue - 0.25 * np.pi + 0.5 * np.pi
        elif cueLocation[probeLocationInd] == "UP":
            phi = phi - 0.25 * np.pi + 1.0 * np.pi
            phiCue = phiCue - 0.25 * np.pi + 1.0 * np.pi
        elif cueLocation[probeLocationInd] == "RIGHT":
            phi = phi - 0.25 * np.pi - 0.5 * np.pi
            phiCue = phiCue - 0.25 * np.pi - 0.5 * np.pi
    elif targetDirection[probeLocationInd] == "CW":
        if cueLocation[probeLocationInd] == "DOWN":
            phi = -1 * phi - 0.25 * np.pi - 1.0 * np.pi
            phiCue = -1 * phiCue - 0.25 * np.pi - 1.0 * np.pi
        elif cueLocation[probeLocationInd] == "LEFT":
            phi = -1 * phi + 0.25 * np.pi
            phiCue = -1 * phiCue + 0.25 * np.pi
        elif cueLocation[probeLocationInd] == "UP":
            phi = -1 * phi - 0.25 * np.pi
            phiCue = -1 * phiCue - 0.25 * np.pi
        elif cueLocation[probeLocationInd] == "RIGHT":
            phi = -1 * phi - 0.25 * np.pi - 0.5 * np.pi
            phiCue = -1 * phiCue - 0.25 * np.pi - 0.5 * np.pi

    # plt.subplot(1, 2, 1)
    # plt.plot(
    #     cueLocCoor[cueLocation[probeLocationInd]][0],
    #     cueLocCoor[cueLocation[probeLocationInd]][1],
    #     "ko",
    #     alpha=0.1,
    # )
    # plt.plot(tempCoor[0], tempCoor[1], "k.", alpha=0.1)

    # cueLocationRot[probeLocationInd, 0] = cueLocCoor[cueLocation[probeLocationInd]][0]
    # cueLocationRot[probeLocationInd, 1] = cueLocCoor[cueLocation[probeLocationInd]][1]

    probeLocationCoor[probeLocationInd, 0] = tempCoor[0]
    probeLocationCoor[probeLocationInd, 1] = tempCoor[1]

    # plt.text(tempCoor[0], tempCoor[1], str(probeLocationInd))

    x, y = pol2cart(rho, phi)
    xCue, yCue = pol2cart(rhoCue, phiCue)
    probeLocationRot[probeLocationInd, 0] = x
    probeLocationRot[probeLocationInd, 1] = y

    # plt.subplot(1, 2, 2)
    # plt.plot(xCue, yCue, "ro", alpha=0.1)
    # plt.plot(x, y, "r.", alpha=0.1)
    # plt.text(x, y, str(probeLocationInd))


if showPlotsPerObs:
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.plot(
        probeLocationCoor[:, 0],
        probeLocationCoor[:, 1],
        "k.",
        alpha=0.1,
        markersize=15,
        markeredgewidth=0,
    )
    # for k in range(10):
    # plt.text(probeLocationCoor[k, 0], probeLocationCoor[k, 1],str(k))
    plt.axis("square")
    plt.title("Raw coordinates")

    plt.subplot(1, 2, 2)
    plt.plot(xCue, yCue, "ro", alpha=0.1)
    plt.plot(probeLocationRot[:, 0], probeLocationRot[:, 1], "r.", alpha=0.1)
    # for k in range(10):
    # plt.text(probeLocationRot[k, 0], probeLocationRot[k, 1],str(k))
    plt.axis("square")
    plt.title("Transformed coordinates")

    plt.figure()
    countPlot = 0
    for dir in ["CW", "CCW"]:
        dirVect = targetDirection == dir
        for loc in ["UP", "RIGHT", "DOWN", "LEFT"]:
            locVect = cueLocation == loc
            countPlot = countPlot + 1
            plt.subplot(2, 4, countPlot)

            plt.plot(
                probeLocationCoor[(locVect) & (dirVect), 0],
                probeLocationCoor[(locVect) & (dirVect), 1],
                "k.",
                alpha=0.1,
                markersize=15,
                markeredgewidth=0,
            )
            # for k in range(10):
            # plt.text(probeLocationCoor[k, 0], probeLocationCoor[k, 1],str(k))
            plt.axis("square")
            plt.title("Raw coordinates")
    plt.tight_layout()
