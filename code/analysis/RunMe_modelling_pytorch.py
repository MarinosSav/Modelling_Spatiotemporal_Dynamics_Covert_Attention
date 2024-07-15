import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats, io
import random
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, log_loss, r2_score
from sklearn import decomposition
from sklearn.preprocessing import MinMaxScaler
import multiprocessing
import time
from keras.layers import Input, Dense, Layer, InputSpec, Dropout, LeakyReLU
from keras.models import Sequential, Model
from keras.regularizers import L1, L2
from tensorflow.keras.optimizers import Adam
from keras.losses import KLDivergence
from keras.callbacks import learning_rate_schedule
from keras import backend
import torch
from torchvision import datasets
from torchvision import transforms

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler


"""
from keras.models import Sequential
from keras.layers import Dense, Input
from keras.datasets import mnist
from keras.regularizers import l1
#from keras.optimizers import Adam"""
counter = 0

def loadData():

    normTypeName = "z"  # "z" or "None"
    varOfInterestAbr = "amp"  # "amp" or "area"
    onsetEvent = "cue"  # "cue" or "probe"
    pupilUnit = "z"  # "z" or "au" (None)

    pickleFileName = (
        "data/ave_ref_"
        + "_perXYTimeBin"
        + "_norm-"
        + normTypeName
        + "_mes-"
        + varOfInterestAbr
        + "_onset-"
        + onsetEvent
        + ".pickle"
    )

    with open(pickleFileName, "rb") as handle:
        allErprAmp2DMapPerTimeBin_ave_ref, binCentersX, binCentersY, binEdgesT = pickle.load(handle)
        handle.close()

    return allErprAmp2DMapPerTimeBin_ave_ref, binCentersX, binCentersY, binEdgesT


def adjust_distribution(flat_space, real_space, centers, center_dispersion=[], barbapapa=False, alt_distribution=False):

    if len(centers) == 0:
        temp_space = np.copy(real_space).flatten()
        np.random.shuffle(temp_space)
        return np.reshape(temp_space, real_space.shape)

    assert np.shape(flat_space) == np.shape(real_space)

    value_distribution = -np.sort(-real_space.flatten(),)
    adjusted_space = np.zeros(np.shape(flat_space))
    priority_mask = np.zeros(np.shape(adjusted_space))

    max_value = 40

    for i in range(np.shape(priority_mask)[0]):
        for j in range(np.shape(priority_mask)[1]):
            if not alt_distribution:
                if len(centers) == 1:
                    priority_mask[i][j] = max_value - max(abs(i - centers[0][0]), abs(j - centers[0][1]))
                elif len(centers) == 2:
                    priority_mask[i][j] = max_value - min(max(abs(i - centers[0][0]), abs(j - centers[0][1])),
                                                          max(abs(i - centers[1][0]), abs(j - centers[1][1])))
            else:
                if len(centers) == 1:
                    priority_mask[i][j] = max_value - abs(j - centers[0][1])
                elif len(centers) == 2:
                    priority_mask[i][j] = max_value - min(abs(j - centers[0][1]), abs(j - centers[1][1]))

    if len(center_dispersion) == 1:
        for _ in range(center_dispersion[0]):

            priority_mask[priority_mask == np.max(priority_mask)] -= 1
    elif len(center_dispersion) == 2:

        mid = int(np.shape(priority_mask)[1]/2)

        temp_mask = priority_mask[:, :mid]
        for _ in range(center_dispersion[0]):
            temp_mask[temp_mask == np.max(temp_mask)] -= 1
        priority_mask[:, :mid] = temp_mask

        temp_mask = priority_mask[:, mid:]
        for _ in range(center_dispersion[1]):
            temp_mask[temp_mask == np.max(temp_mask)] -= 1
        priority_mask[:, mid:] = temp_mask

        if barbapapa:
            temp_mask = priority_mask[centers[0][0] - 2:centers[0][0] + 3, centers[0][1] + 1:centers[1][1]]
            temp_mask.fill(round(np.mean(temp_mask)))
            priority_mask[centers[0][0] - 2:centers[0][0] + 3, centers[0][1] + 1:centers[1][1]] = temp_mask

    pointer = 0
    for n in range(max_value, int(np.min(priority_mask))-1, -1):
        replace_x, replace_y = np.where(priority_mask == n)
        shuffled_value_distribution = np.random.permutation(value_distribution[pointer:pointer+len(replace_x)])
        value_pointer = 0
        for i in range(len(replace_x)):
            adjusted_space[replace_x[i], replace_y[i]] = shuffled_value_distribution[value_pointer]
            value_pointer += 1
            pointer += 1

    assert (-np.sort(-real_space.flatten(),) == value_distribution).all()

    return adjusted_space


def train_ml_models():

    global lr_model, autoencoders

    train_features = []
    train_labels = []
    #x_data = []
    for i in range(np.shape(allErprAmp2DMapPerTimeBin_ave_ref)[3]):
        for j in range(np.shape(allErprAmp2DMapPerTimeBin_ave_ref)[2]):
            train_labels.append([j])
            train_features.append(allErprAmp2DMapPerTimeBin_ave_ref[:, :, j, i].ravel())
            # x_data.append(allErprAmp2DMapPerTimeBin_ave_ref[:, :, j, i])

    lr_model = LinearRegression()
    lr_model.fit(train_labels, train_features)

    mlp_model = MLPRegressor(activation='relu', solver='adam', alpha=1e-4, hidden_layer_sizes=(100,),
                             learning_rate='adaptive', learning_rate_init=0.001, max_iter=1000)
    mlp_model.fit(train_labels, train_features)

    autoencoders = []
    train_features = torch.Tensor(train_features)
    train_labels = torch.Tensor(train_labels)

    for bT in range(nTimeBins):

        scaler1 = MinMaxScaler()

        X_train = train_features[train_labels[:, 0] == bT][:20]
        X_train_scaled = scaler1.fit_transform(X_train)
        X_train_scaled = torch.Tensor(X_train_scaled)

        X_val = train_features[train_labels[:, 0] == bT][20:]
        X_val_scaled = MinMaxScaler().fit_transform(X_val)
        X_val_scaled = torch.Tensor(X_val_scaled)

        nb_epoch = 100
        batch_size = 16
        input_dim = X_train_scaled.shape[1]
        hidden_dim = int(input_dim / 4)
        encoding_dim = 2

        class Autoencoder(nn.Module):
            def __init__(self):
                super(Autoencoder, self).__init__()
                self.encoder = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.LeakyReLU(),
                    nn.Linear(hidden_dim, encoding_dim),
                    nn.LeakyReLU()
                )
                self.decoder = nn.Sequential(
                    nn.Linear(encoding_dim, hidden_dim),
                    nn.LeakyReLU(),
                    nn.Linear(hidden_dim, input_dim),
                    nn.LeakyReLU()
                )

            def forward(self, x):
                x = self.encoder(x)
                x = self.decoder(x)
                return x

        autoencoder = Autoencoder()
        criterion = nn.MSELoss()
        optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)

        for epoch in range(nb_epoch):
            running_loss = 0.0

            # Mini-batch training
            for i in range(0, X_train_scaled.size(0), batch_size):
                inputs = X_train_scaled[i:i + batch_size]
                optimizer.zero_grad()
                outputs = autoencoder(inputs)
                loss = criterion(outputs, inputs)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            epoch_loss = running_loss / (X_train_scaled.size(0) / batch_size)
            print(f"Epoch {epoch + 1}/{nb_epoch} Loss: {epoch_loss:.4f}")

        # Store the trained autoencoder
        autoencoders.append(autoencoder)

    #interpretModel(autoencoders[2], allErprAmp2DMapPerTimeBin_ave_ref[:, :, 2, 2])

    return


def interpretModel(model, input_map):

    print(len(model.layers))

    scaler = MinMaxScaler()

    reshaped_data = np.reshape(input_map.flatten(), (1, -1))
    reshaped_data_scaled = scaler.fit_transform(reshaped_data)

    predict_data = model.predict(reshaped_data_scaled)
    modelSpace_unscaled = scaler.inverse_transform(predict_data)
    modelSpace = np.reshape(modelSpace_unscaled, (nYBins, nXBins))

    interpret_model_encoder = Model(inputs=model.input, outputs=model.layers[2].output)
    interpret_model_decoder = Model(inputs=model.layers[3].input, outputs=model.output).compile()
    #test = backend.function([model.layers[3].input], [model.output])
    #print(model.layers[2].output.get_weights())
    print(model.summary())
    exit()
    middle_layer_outputs = interpret_model_encoder.predict(reshaped_data_scaled)[0]
    #decoder_layer_outputs = interpret_model_decoder.predict(middle_layer_outputs)
    print(interpret_model_encoder.summary())
    exit()
    #print(decoder_layer_outputs)
    #middle_layer_outputs += 100
    #print(type(interpret_model))
    #output_model = Model(inputs=interpret_model, outputs=model.output)

    fig = plt.figure(figsize=(15, 8))
    plt.scatter(middle_layer_outputs[0], middle_layer_outputs[1],)

    fig = plt.figure(figsize=(15, 8))
    for i in range(1, 5):
        fig.add_subplot(2, 2, i)
        plt.imshow(input_map, cmap='coolwarm')

        if i == 1:
            continue
        elif i == 2:
            pass


        """
        plt.imshow(allErprAmp2DMapPerTimeBin_ave_ref[:, :, bT, s],
                   cmap='coolwarm',
                   vmin=varCBarRange[0],
                   vmax=varCBarRange[1],
                   )
        plt.xticks(range(0, nXBins), labels=binCentersX[0:nXBins].astype(int))
        plt.yticks(range(0, nYBins), labels=binCentersY[0:nYBins].astype(int))
        plt.xlim([-1, nXBins])
        plt.ylim([-1, nYBins])
        plt.title("SOA: " + str(int(binEdgesT[bT])) + "-" + str(int(binEdgesT[bT + 1])))"""
    #plt.xlabel("Horizontal position")

    plt.tight_layout()
    plt.subplots_adjust(hspace=-0.2, wspace=0)
    #plt.savefig("Figures/IndividualModels/s" + str(s) + ".svg")
    plt.show()

    exit()


def createPrecodedModels(s, m, bT):

    barbapapa = False
    spotlight_centers = [(4, 3)]
    center_dispersion = [0]
    modelSpace = np.zeros((nYBins, nXBins))
    alt_distribution = False

    if m == 0:  # random
        modelSpace = np.copy(allErprAmp2DMapPerTimeBin_ave_ref[:, :, bT, s])
        spotlight_centers = []
    elif m == 1:  # stationary
        pass
    elif m == 2:  # jump
        if bT == 0 or bT == 1:
            pass
        else:
            spotlight_centers = [(4, 13)]
    elif m == 3:  # move
        shift_rate = 2 * bT
        spotlight_centers = [(4, 2 + shift_rate + 1)]
    elif m == 4:  # gradual shift + shrink
        shift_rate = 1.5 * bT
        if bT == 0 or bT == 1:
            shrink = 0
        elif bT == 2 or bT == 3:
            shift_rate += 1
            shrink = 1
        else:
            shift_rate += 2
            shrink = 2
        spotlight_centers = [(4, int(2 + shift_rate + 1))]
        center_dispersion = [2 - shrink]
    elif m == 5:  # fading
        spotlight_centers.append((4, 13))
        phases = np.arange(nTimeBins)
        center_dispersion = [phases[bT], phases[nTimeBins - 1 - bT]]
    elif m == 6:  # barbapapa
        if bT == 0:
            pass
        elif bT == 5:
            spotlight_centers = [(4, 13)]
            barbapapa = False
        else:
            spotlight_centers.append((4, 13))
            if bT == 1:
                center_dispersion = [1, 3]
            elif bT == 4:
                center_dispersion = [3, 1]
                barbapapa = True
            elif bT == 2 or bT == 3:
                center_dispersion = [2, 2]
                barbapapa = True
    elif m == 7:  # Jump (No vert)
        alt_distribution = True
        if bT == 0 or bT == 1:
            pass
        else:
            spotlight_centers = [(4, 13)]
    elif m == 8:  #
        alt_distribution = True
        spotlight_centers.append((4, 13))
        phases = np.arange(nTimeBins)
        center_dispersion = [phases[bT], phases[nTimeBins - 1 - bT]]
    elif m == 9:  # linear regressor
        modelSpace = np.reshape(lr_model.predict([[bT]]), (nYBins, nXBins))
    elif m == 10:  # autoencoder
        scaler = MinMaxScaler()

        reshaped_data = np.reshape(allErprAmp2DMapPerTimeBin_ave_ref[:, :, bT, s].flatten(), (1, -1))
        reshaped_data_scaled = scaler.fit_transform(reshaped_data)

        predict_data = autoencoders[bT].forward(torch.Tensor(reshaped_data_scaled))
        predict_data = predict_data.detach().numpy()
        modelSpace_unscaled = scaler.inverse_transform(predict_data)
        modelSpace = np.reshape(modelSpace_unscaled, (nYBins, nXBins))

        #interpret_model = Model(inputs=autoencoders[bT].input, outputs=autoencoders[bT].layers[2].output)
        #middle_layer_outputs = interpret_model.predict(reshaped_data_scaled)[0]

        #autoencoder_middle_layer[s, bT, 0] = middle_layer_outputs[0]
        #autoencoder_middle_layer[s, bT, 1] = middle_layer_outputs[1]

    if m != 9 and m != 10:
        modelSpace = adjust_distribution(flat_space=modelSpace,
                                         real_space=allErprAmp2DMapPerTimeBin_ave_ref[:, :, bT, s],
                                         centers=spotlight_centers,
                                         center_dispersion=center_dispersion,
                                         barbapapa=barbapapa,
                                         alt_distribution=alt_distribution)

    rhos = stats.spearmanr(modelSpace, allErprAmp2DMapPerTimeBin_ave_ref[:, :, bT, s], axis=None).correlation
    mses = mean_squared_error(modelSpace, allErprAmp2DMapPerTimeBin_ave_ref[:, :, bT, s])

    return modelSpace, rhos, mses


def processSessions(nSubs, nModels, nIterations=1, showModelledImagePerSub=False, mlModels=True):

    """
    if showModelledImagePerSub:
        nIterations = 1"""

    if mlModels:
        train_ml_models()

    nTimeBins = np.shape(allErprAmp2DMapPerTimeBin_ave_ref)[2]
    allRhos = np.zeros((nSubs, nTimeBins, nModels))
    allMSEs = np.zeros((nSubs, nTimeBins, nModels))
    allModelSpace = np.zeros((nSubs, nTimeBins, nModels, nIterations, nYBins, nXBins))

    global autoencoder_middle_layer
    autoencoder_middle_layer = np.zeros((nSubs, nTimeBins, 2))

    for s in range(nSubs):

        print("> sess:", s, " started")
        start_time = time.perf_counter()

        varCBarRange = [
            np.percentile(allErprAmp2DMapPerTimeBin_ave_ref[:,:,:,s], 1),
            np.percentile(allErprAmp2DMapPerTimeBin_ave_ref[:,:,:,s], 99),
        ]

        if showModelledImagePerSub:
            fig = plt.figure(figsize=(15, 8))

        for m in range(nModels):

            for bT in range(nTimeBins):

                total_rhos = []
                total_mses = []

                for i, _ in enumerate(range(nIterations)):
                    if i == 0:
                        print('>m:', m, ', bt:', bT)
                    modelSpace, rhos, mses = createPrecodedModels(s, m, bT)
                    total_rhos.append(rhos)
                    total_mses.append(mses)
                    allModelSpace[s, bT, m, i] = modelSpace
                allRhos[s, bT, m] = np.mean(total_rhos)
                allMSEs[s, bT, m] = np.mean(total_mses)

                if showModelledImagePerSub:

                    if nIterations > 1:
                        modelSpace = np.mean(allModelSpace[s, bT, m, :, :, :], axis=0)

                    fig.add_subplot(nTimeBins, nModels+1, (m+1)+bT*(nModels+1))
                    if bT == 0:
                        plt.text(6-(int(len(modelNames[m])/3)), 12, modelNames[m])
                    plt.imshow(modelSpace,
                               cmap='coolwarm',
                               vmin=varCBarRange[0],
                               vmax=varCBarRange[1])
                    plt.xticks(range(0, nXBins), labels=binCentersX[0:nXBins].astype(int))
                    plt.yticks(range(0, nYBins), labels=binCentersY[0:nYBins].astype(int))
                    plt.xlim([-1, nXBins])
                    plt.ylim([-1, nYBins])

                    if bT == 4:
                        plt.xlabel("Horizontal position")
                    if m == 0:
                        plt.ylabel("Vertical position")
                    plt.title("SOA: " + str(int(binEdgesT[bT])) + "-" + str(int(binEdgesT[bT + 1])))

        if showModelledImagePerSub:
            for bT in range(nTimeBins):
                fig.add_subplot(nTimeBins, nModels+1, (bT+1)*(nModels+1))
                plt.imshow(allErprAmp2DMapPerTimeBin_ave_ref[:, :, bT, s],
                           cmap='coolwarm',
                           vmin=varCBarRange[0],
                           vmax=varCBarRange[1],
                           )

                if bT == 0:
                    plt.text(4, 12, "Real data")
                plt.xticks(range(0, nXBins), labels=binCentersX[0:nXBins].astype(int))
                plt.yticks(range(0, nYBins), labels=binCentersY[0:nYBins].astype(int))
                plt.xlim([-1, nXBins])
                plt.ylim([-1, nYBins])
                plt.title("SOA: " + str(int(binEdgesT[bT])) + "-" + str(int(binEdgesT[bT + 1])))
            plt.xlabel("Horizontal position")

            plt.tight_layout()
            plt.subplots_adjust(hspace=-0.2, wspace=0)
            plt.savefig("Figures/IndividualModels/s" + str(s) + ".svg")
            plt.show()

        print("sess:", s, " done, time:", time.perf_counter()-start_time)

    return allRhos, allMSEs, allModelSpace


def plotSummary(allRhos, allMSEs, allModelSpace):

    fig = plt.figure()

    fig.add_subplot(121)
    plt.plot(np.mean(allRhos, axis=0))
    plt.xlabel('Time bin')
    plt.xticks(np.arange(nTimeBins))
    plt.ylabel('Correlaton')
    plt.legend(modelNames, loc='upper left')
    plt.savefig("Figures/ModelCorr.svg")

    fig.add_subplot(122)
    plt.plot(np.mean(allMSEs, axis=0))
    plt.xlabel('Time bin')
    plt.xticks(np.arange(nTimeBins))
    plt.ylabel('MSE')
    plt.legend(modelNames, loc='lower right')
    plt.savefig("Figures/ModelMSE.svg")

    plot_pos = 0
    fig = plt.figure()
    for bT in range(nTimeBins):
        t_table = np.zeros((nYBins, nXBins))
        p_table = np.zeros((nYBins, nXBins))
        for i in range(nYBins):
            for j in range(nXBins):
                t, p = stats.ttest_1samp(a=allErprAmp2DMapPerTimeBin_ave_ref[i, j, bT, :], popmean=0)
                t_table[i][j] = t
                p_table[i][j] = p

        plot_pos += 1

        fig.add_subplot(nTimeBins, 2, plot_pos)
        plt.imshow(t_table,
                   cmap='bwr',
                   vmin=np.min(t_table),
                   vmax=np.max(t_table))

        plt.xticks(range(0, nXBins), labels=binCentersX[0:nXBins].astype(int))
        plt.yticks(range(0, nYBins), labels=binCentersY[0:nYBins].astype(int))
        plt.xlim([-1, nXBins])
        plt.ylim([-1, nYBins])

        if plot_pos == 9:
            plt.xlabel("Horizontal position")
        plt.ylabel("Vertical position")
        plt.title("SOA: " + str(int(binEdgesT[bT])) + "-" + str(int(binEdgesT[bT + 1])))
        plt.colorbar()

        if bT == 0:
            plt.text(4, 12, "t-stat")

        plot_pos += 1

        fig.add_subplot(nTimeBins, 2, plot_pos)
        plt.imshow(p_table,
                   cmap='bwr',
                   vmin=0,
                   vmax=1)

        plt.xticks(range(0, nXBins), labels=binCentersX[0:nXBins].astype(int))
        plt.yticks(range(0, nYBins), labels=binCentersY[0:nYBins].astype(int))
        plt.xlim([-1, nXBins])
        plt.ylim([-1, nYBins])

        if plot_pos == 10:
            plt.xlabel("Horizontal position")
        plt.ylabel("Vertical position")
        plt.title("SOA: " + str(int(binEdgesT[bT])) + "-" + str(int(binEdgesT[bT + 1])))
        plt.colorbar()

        if bT == 0:
            plt.text(4, 12, "p-stat")


    fig = plt.figure(figsize=(15, 8))

    varCBarRange = [
        np.percentile(allErprAmp2DMapPerTimeBin_ave_ref[:, :, :, :], 1),
        np.percentile(allErprAmp2DMapPerTimeBin_ave_ref[:, :, :, :], 99),
    ]

    for m in range(nModels):
        for bT in range(nTimeBins):

            modelSpace = np.mean(allModelSpace[:, bT, m, :, :, :], axis=(0,1))

            fig.add_subplot(nTimeBins, nModels+1, (m+1)+bT*(nModels+1))
            if bT == 0:
                plt.text(6-(int(len(modelNames[m])/3)), 12, modelNames[m])
            plt.imshow(modelSpace,
                       cmap='coolwarm',
                       vmin=varCBarRange[0],
                       vmax=varCBarRange[1])
            plt.xticks(range(0, nXBins), labels=binCentersX[0:nXBins].astype(int))
            plt.yticks(range(0, nYBins), labels=binCentersY[0:nYBins].astype(int))
            plt.xlim([-1, nXBins])
            plt.ylim([-1, nYBins])

            if bT == 4:
                plt.xlabel("Horizontal position")
            if m == 0:
                plt.ylabel("Vertical position")
            plt.title("SOA: " + str(int(binEdgesT[bT])) + "-" + str(int(binEdgesT[bT + 1])))

    for bT in range(nTimeBins):
        fig.add_subplot(nTimeBins, nModels+1, (bT+1)*(nModels+1))
        plt.imshow(np.mean(allErprAmp2DMapPerTimeBin_ave_ref[:, :, bT, :], axis=2),
                   cmap='coolwarm',
                   vmin=varCBarRange[0],
                   vmax=varCBarRange[1],
                   )

        if bT == 0:
            plt.text(4, 12, "Real data")
        plt.xticks(range(0, nXBins), labels=binCentersX[0:nXBins].astype(int))
        plt.yticks(range(0, nYBins), labels=binCentersY[0:nYBins].astype(int))
        plt.xlim([-1, nXBins])
        plt.ylim([-1, nYBins])
        plt.title("SOA: " + str(int(binEdgesT[bT])) + "-" + str(int(binEdgesT[bT + 1])))
    plt.xlabel("Horizontal position")
    plt.savefig("Figures/ModelAvg.svg")

    fig = plt.figure()
    fig.add_subplot(1,1,1)

    cs = ['red', 'blue', 'purple', 'green', 'orange']
    for bT in range(nTimeBins):

        #m, b = np.polyfit(autoencoder_middle_layer[:, bT, 0], autoencoder_middle_layer[:, bT, 1], deg=1)
        plt.scatter(autoencoder_middle_layer[:, bT, 1], autoencoder_middle_layer[:, bT, 0], c=cs[bT], s=10)
        #plt.scatter(np.mean(autoencoder_middle_layer[:, bT, 0]), np.mean(autoencoder_middle_layer[:, bT, 1]), c=cs[bT], s=30, marker=',')

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.4, wspace=0.05)
    plt.savefig("Figures/LatentSpace.svg")
    plt.show()

    return


def saveMatlab():

    data = {'allErprAmp2DMapPerTimeBin_ave_ref': allErprAmp2DMapPerTimeBin_ave_ref}
    io.savemat('test.mat', data)

    return


if __name__ == "__main__":

    global modelNames, nModels, nIterations, nXBins, nYbins, nTimeBins, allErprAmp2DMapPerTimeBin_ave_ref

    modelNames = ['Random', 'Stationary', 'Jump', 'Move', 'Move + Shrink', 'Fading', 'Morph', 'Jump (No vert.)', 'Fading (No vert.)', 'Linear Regression', 'Autoencoder']
    nModels = 11
    nIterations = 1

    allErprAmp2DMapPerTimeBin_ave_ref, binCentersX, binCentersY, binEdgesT = loadData()
    nYBins, nXBins, nTimeBins, nSubs = np.shape(allErprAmp2DMapPerTimeBin_ave_ref)

    print("Number of sessions in data:", nSubs)
    nSubs = 1

    allRhos, allMSEs, allModelSpace = processSessions(nSubs=nSubs, showModelledImagePerSub=True, nModels=nModels, nIterations=nIterations)
    plotSummary(allRhos, allMSEs, allModelSpace)
    saveMatlab()