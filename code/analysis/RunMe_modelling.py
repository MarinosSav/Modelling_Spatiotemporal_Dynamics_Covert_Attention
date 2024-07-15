import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy import stats, io
import random
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, log_loss, r2_score, pairwise_distances_argmin_min
from sklearn import decomposition
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import DBSCAN
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
from scipy.spatial import distance
import cv2
import math
from sklearn.cluster import DBSCAN
from sklearn.metrics import adjusted_rand_score
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from matplotlib.gridspec import GridSpec

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
            temp_mask = priority_mask[centers[0][0] - 4:centers[0][0] + 4, centers[0][1] + 2:centers[1][1] - 1]
            temp_mask.fill(round(np.mean(temp_mask)))
            priority_mask[centers[0][0] - 4:centers[0][0] + 4, centers[0][1] + 2:centers[1][1] - 1] = temp_mask

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
    train_features = np.array(train_features)
    train_labels = np.array(train_labels)


    for bT in range(nTimeBins):

        scaler1 = MinMaxScaler()

        X_train = train_features[train_labels[:, 0] == bT][:20]
        X_train_scaled = scaler1.fit_transform(X_train)

        X_val = train_features[train_labels[:, 0] == bT][20:]
        X_val_scaled = MinMaxScaler().fit_transform(X_val)

        nb_epoch = 100 #100
        batch_size = 16 #16
        input_dim = X_train_scaled.shape[1]  # num of predictor variables,
        hidden_dim = int(input_dim / 4)
        encoding_dim = 2

        encoder = Dense(hidden_dim, input_shape=(input_dim,), use_bias=True, name='encoder')
        hidden1 = Dense(encoding_dim, input_shape=(hidden_dim,), use_bias=True, name='code')
        hidden2 = Dense(hidden_dim, input_shape=(encoding_dim,), use_bias=True)
        decoder = Dense(input_dim, input_shape=(hidden_dim,), use_bias=True, name='decoder')

        autoencoder = Sequential()
        autoencoder.add(encoder)
        autoencoder.add(LeakyReLU())
        autoencoder.add(hidden1)
        autoencoder.add(LeakyReLU())
        autoencoder.add(hidden2)
        autoencoder.add(LeakyReLU())
        autoencoder.add(decoder)
        autoencoder.add(LeakyReLU())

        autoencoder.compile(metrics=['accuracy'],
                            loss='mean_squared_error',
                            optimizer=Adam(learning_rate=0.001))
        #autoencoder.summary()
        history = autoencoder.fit(X_train_scaled, X_train_scaled,
                                  epochs=nb_epoch,
                                  batch_size=batch_size,
                                  validation_data=(X_val_scaled, X_val_scaled),
                                  shuffle=True,
                                  verbose=0)

        """
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        #plt.ylabel('accuracy')
        #plt.xlabel('epoch')
        #plt.legend(['train'], loc='upper left')
        plt.show()
        #print(mean_squared_error(autoencoder.predict(X_val_scaled), X_val_scaled))

        #print(np.round(autoencoder.layers[0].get_weights()[0], 2).T)
        #print(np.round(autoencoder.layers[1].get_weights()[0], 2))"""

        autoencoders.append(autoencoder)
        """
        res = scaler1.inverse_transform(autoencoder.predict(X_train_scaled))
        res = np.reshape(autoencoder.predict(res), (np.shape(X_train_scaled)[0], 8, 16))
        real = np.reshape(X_train, (np.shape(X_train)[0], 8, 16))

        print(np.shape(res), np.shape(real))

        res = np.mean(res, axis=0)
        real = np.mean(real, axis=0)

        fig = plt.figure()
        fig.add_subplot(211)
        plt.imshow(res, cmap='coolwarm')
        fig.add_subplot(212)
        plt.imshow(real, cmap='coolwarm')
        plt.show()"""

        #test = np.mean(np.array(autoencoder.layers[0].get_weights()[0]), axis=1)
        #modelSpace = np.reshape(test, (nYBins, nXBins))
        #plt.imshow(modelSpace, cmap='coolwarm')
        #plt.show()


        #interpretModel(autoencoders[bT], allErprAmp2DMapPerTimeBin_ave_ref[:, :, bT, 2])


    """
    dataset = datasets.MNIST(root="./data",
                             train=True,
                             download=True,
                             transform=transforms.tensor_transform)

    # DataLoader is used to load the dataset
    # for training
    loader = torch.utils.data.DataLoader(dataset=dataset,
                                         batch_size=32,
                                         shuffle=True)

    for i, (images, labels) in enumerate(loader):
        print(type(images))

    exit()

    print(loader)"""
    #exit()

    return


def interpretModel(model, input_map):

    scaler = MinMaxScaler()

    reshaped_data = np.reshape(input_map.flatten(), (1, -1))
    reshaped_data_scaled = scaler.fit_transform(reshaped_data)

    predict_data = model.predict(reshaped_data_scaled)
    modelSpace_unscaled = scaler.inverse_transform(predict_data)
    modelSpace = np.reshape(modelSpace_unscaled, (nYBins, nXBins))

    interpret_model_encoder = Model(inputs=model.input, outputs=model.layers[2].output)

    interpret_model_decoder = Sequential()
    interpret_model_decoder.add(model.layers[4])
    interpret_model_decoder.add(LeakyReLU())
    interpret_model_decoder.add(model.layers[6])
    interpret_model_decoder.add(LeakyReLU())

    middle_layer_outputs = interpret_model_encoder.predict(reshaped_data_scaled)[0]

    fig = plt.figure(figsize=(15, 8))
    plt.scatter(middle_layer_outputs[0], middle_layer_outputs[1], )
    plt.show()

    test = np.reshape([0, 1], (1, -1))
    decoder_layer_outputs = interpret_model_decoder.predict(test)

    modelSpace_unscaled = scaler.inverse_transform(decoder_layer_outputs)
    modelSpace = np.reshape(modelSpace_unscaled, (nYBins, nXBins))

    fig = plt.figure(figsize=(15, 8))
    multiplier = 1
    inputs = [[-2, -2], [-1, -2], [0, -2], [1, -2], [2, -2],
              [-2, -1], [-1, -1], [0, -1], [1, -1], [2, -1],
              [-2, 0], [-1, 0], [0, 0], [1, 0], [2, 0],
              [-2, 1], [-1, 1], [0, 1], [1, 1], [2, 1],
              [-2, 2], [-1, 2], [0, 2], [1, 2], [2, 2]]


    grid_rows, grid_cols = 5, 5
    for i in range(1, ((grid_rows+1)*grid_cols)+1):
        print(i)
        fig.add_subplot(grid_rows+1, grid_cols, i)
        if i == 1:
            plt.imshow(input_map, cmap='coolwarm')
            plt.title(str(middle_layer_outputs))
        elif i < grid_cols+1:
            print('NO')
            plt.axis('off')
        else:
            print('YES')
            id = i-grid_cols-1
            test = np.reshape(inputs[id], (1, -1))

            decoder_layer_outputs = interpret_model_decoder.predict(test)
            modelSpace_unscaled = scaler.inverse_transform(decoder_layer_outputs)
            modelSpace = np.reshape(modelSpace_unscaled, (nYBins, nXBins))

            plt.imshow(modelSpace, cmap='coolwarm')
            plt.title(str(inputs[id]))


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
        if bT >= 2:
            spotlight_centers = [(4, 13)]
    elif m == 4:  # move
        if bT == 0:
            pass
        elif bT == 1:
            spotlight_centers = [(4, 6)]
        elif bT == 2:
            spotlight_centers = [(4, 10)]
        elif bT == 3:
            spotlight_centers = [(4, 13)]
        elif bT == 4:
            spotlight_centers = [(4, 13)]
    elif m == 5:  # gradual shift + shrink
        if bT == 0:
            shrink = 0
        elif bT == 1:
            spotlight_centers = [(4, 6)]
            shrink = 0
        elif bT == 2:
            spotlight_centers = [(4, 10)]
            shrink = 1
        elif bT == 3:
            spotlight_centers = [(4, 13)]
            shrink = 2
        elif bT == 4:
            spotlight_centers = [(4, 13)]
            shrink = 2
        center_dispersion = [2 - shrink]
    elif m == 6:  # fading
        spotlight_centers.append((4, 13))
        phases = np.arange(nTimeBins)
        center_dispersion = [phases[bT], phases[nTimeBins - 1 - bT]]
    elif m == 8:  # barbapapa
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
    elif m == 3:  # Jump (No vert)
        alt_distribution = True
        if bT >= 2:
            spotlight_centers = [(4, 13)]
    elif m == 7:  #
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

        predict_data = autoencoders[bT].predict(reshaped_data_scaled)
        modelSpace_unscaled = scaler.inverse_transform(predict_data)
        modelSpace = np.reshape(modelSpace_unscaled, (nYBins, nXBins))

        interpret_model = Model(inputs=autoencoders[bT].input, outputs=autoencoders[bT].layers[2].output)
        middle_layer_outputs = interpret_model.predict(reshaped_data_scaled)[0]

        autoencoder_middle_layer[s, bT, 0] = middle_layer_outputs[0]
        autoencoder_middle_layer[s, bT, 1] = middle_layer_outputs[1]

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
    allCosines = np.zeros((nSubs, nTimeBins, nModels))
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
                total_cosines = []

                for i, _ in enumerate(range(nIterations)):
                    if i == 0:
                        print('>m:', m, ', bt:', bT)
                    modelSpace, rhos, mses = createPrecodedModels(s, m, bT)

                    total_rhos.append(rhos)
                    total_mses.append(mses)

                    Aflat = np.hstack(modelSpace)
                    Bflat = np.hstack(allErprAmp2DMapPerTimeBin_ave_ref[:, :, bT, s])
                    dist = distance.cosine(Aflat, Bflat)

                    total_cosines.append(dist)
                    allModelSpace[s, bT, m, i] = modelSpace
                allRhos[s, bT, m] = np.mean(total_rhos)
                allMSEs[s, bT, m] = np.mean(total_mses)
                allCosines[s, bT, m] = np.mean(total_cosines)

                allMSEs[s, bT, m] = 1-np.mean(total_cosines)

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
    plt.xlabel('time bin')
    plt.xticks(np.arange(nTimeBins), labels=np.arange(1, nTimeBins+1))
    plt.ylabel("correlation")
    plt.legend(modelNames, loc='upper left')
    plt.ylim(np.min(np.mean(allRhos, axis=0)-0.2), np.max(np.mean(allRhos, axis=0)+0.2))
    plt.savefig("Figures/PrecodedModelCorr.svg")
    print(np.mean(np.mean(allRhos, axis=1), axis=0))

    fig.add_subplot(122)
    plt.plot(np.mean(allRhos[:,:,-2:], axis=0))
    plt.xlabel('time bin')
    plt.xticks(np.arange(nTimeBins), labels=np.arange(1, nTimeBins+1))
    plt.ylabel('correlaton')
    plt.legend(modelNames[-2:], loc='lower right')
    plt.ylim(np.min(np.mean(allRhos[:, :, -2:], axis=0) - 0.2), np.max(np.mean(allRhos[:, :, -2:], axis=0) + 0.2))
    plt.savefig("Figures/TrainedModelCorr.svg")

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

    modelNames = ['Random', 'Stationary', 'Jump', 'Jump (No vert.)', 'Move', 'Move + Shrink', 'Fading', 'Fading (No vert.)', 'Morph', 'Linear Regression', 'Autoencoder']
    nModels = 11
    nIterations = 1

    allErprAmp2DMapPerTimeBin_ave_ref, binCentersX, binCentersY, binEdgesT = loadData()
    nYBins, nXBins, nTimeBins, nSubs = np.shape(allErprAmp2DMapPerTimeBin_ave_ref)

    print("Number of sessions in data:", nSubs)
    #nSubs = 3

    ##################################
    centers = np.zeros((nSubs, nTimeBins, 2))
    stds = np.zeros((nSubs, nTimeBins, 2))
    showGraphs = True
    method = 2
    for s in range(nSubs):
        print("Session:", s)

        if showGraphs:
            fig = plt.figure(figsize=(15, 8))
            pos = 1
        for t in range(nTimeBins):

            modelSpace = allErprAmp2DMapPerTimeBin_ave_ref[:, :, t, s]

            if showGraphs:
                fig.add_subplot(nTimeBins, 2, pos)
                pos+=1
                plt.imshow(modelSpace.reshape([8, 16]), cmap='coolwarm')

            clusterSpace = modelSpace.copy()

            if method == 1:
                clusterSpace[clusterSpace < np.mean(modelSpace)-np.std(modelSpace)] = -1
                clusterSpace[clusterSpace > np.mean(modelSpace)+np.std(modelSpace)] = 1
                clusterSpace = np.array([clusterSpace[i, j] if (clusterSpace[i, j] == -1 or clusterSpace[i, j] == 1) else 0 for i in range(nYBins) for j in range(nXBins)])
                clusterSpace = clusterSpace.reshape([8, 16])
                temp = np.array([[i, j, clusterSpace[i, j]] for i in range(nYBins) for j in range(nXBins)])

                center = np.mean(temp[temp[:, 2] == 1], axis=0)[0:2]
            elif method == 2:
                temp = np.array([[i, j, clusterSpace[i, j]] for i in range(nYBins) for j in range(nXBins)])
                #print(np.std(temp))

                xScaler = MinMaxScaler((0, 1))
                yScaler = MinMaxScaler((0, 1))
                spaceScaler = MinMaxScaler((-3, 3))

                a = xScaler.fit_transform(temp[:, 0].reshape(-1,1))
                b = yScaler.fit_transform(temp[:, 1].reshape(-1,1))
                c = spaceScaler.fit_transform(temp[:, 2].reshape(-1,1))
                d = np.transpose([a, b, c])[0]
                #print(np.std(d))

                kmeans = KMeans(n_clusters=3, max_iter=200, n_init=50).fit(d)
                clusterSpace = np.array(kmeans.labels_)
                clusterSpace_red = d[clusterSpace==1]

                temp_centers = kmeans.cluster_centers_
                center = temp_centers[0,:2] if temp_centers[0, 2] > temp_centers[1, 2] else temp_centers[1,:2]
                center[0] = xScaler.inverse_transform(center[0].reshape(1,-1))
                center[1] = yScaler.inverse_transform(center[1].reshape(1,-1))

                clusterSpace_red[:, 0] = xScaler.inverse_transform(clusterSpace_red[:, 0].reshape(-1, 1))[:, 0]
                clusterSpace_red[:, 1] = yScaler.inverse_transform(clusterSpace_red[:, 1].reshape(-1, 1))[:, 0]

                y_std = np.std(clusterSpace_red[:, 0])
                x_std = np.std(clusterSpace_red[:, 1])
                #print(clusterSpace_red)
                #x_ci = (1.96 * x_std) / math.sqrt(len(clusterSpace_red[:, 0]))
                #y_ci = (1.96 * y_std) / math.sqrt(len(clusterSpace_red[:, 1]))
                #print(kmeans.inertia_)

            centers[s, t, :] = center
            stds[s, t, :] = [x_std, y_std]

            if showGraphs:
                fig.add_subplot(nTimeBins, 2, pos)
                pos+=1
                ellipse = patches.Ellipse(
                    (center[1], center[0]),
                    width=2*x_std,
                    height=2*y_std,
                    angle=0,
                    fill=False,  # Set to True if you want a filled ellipse
                    color='red',  # Set the color of the ellipse
                    linestyle='--',  # Set the linestyle (optional),
                    linewidth=1
                )
                plt.imshow(clusterSpace.reshape([8, 16]), cmap='Greys')
                plt.scatter(center[1], center[0], s=100, c='red', marker='o')
                ax = plt.gca()
                ax.add_patch(ellipse)

        if showGraphs:
            plt.tight_layout()
            plt.subplots_adjust(hspace=0.4, wspace=0.05)
            plt.show()

    fig = plt.figure(constrained_layout=True, figsize=(12, 8))

    fig.add_subplot(1, 2, 1)
    y = np.arange(1, 6, 1)
    for s in range(nSubs):
        plt.plot(y, centers[s, :, 1])
    mean = np.mean(centers[:, :, 1], axis=0)
    std = np.std(centers[:, :, 1], axis=0)
    plt.plot(y, mean, linewidth=5, color='red')
    plt.fill_between(y, mean-std, mean+std, alpha=0.2, color='red')
    plt.xticks(y)
    plt.ylim([0, 16])
    plt.xlabel('time bin')
    plt.ylabel('X coord')

    fig.add_subplot(1, 2, 2)
    for s in range(nSubs):
        plt.plot(y, centers[s, :, 0])
    mean = np.mean(centers[:, :, 0], axis=0)
    std = np.std(centers[:, :, 0], axis=0)
    plt.plot(y, mean, linewidth=5, color='red')
    plt.fill_between(y, mean - std, mean + std, alpha=0.2, color='red')
    plt.xticks(y)
    plt.ylim([0, 8])
    plt.xlabel('time bin')
    plt.ylabel('Y coord')
    plt.show()

    fig = plt.figure(constrained_layout=True, figsize=(12, 8))
    for t in range(nTimeBins):
        center_x = np.mean(centers[:, t, 0])
        center_y = np.mean(centers[:, t, 1])
        print(center_x, center_y)
        std_x = np.mean(stds[:, t, 0])
        std_y = np.mean(stds[:, t, 1])
        fig.add_subplot(nTimeBins, 1, t+1)
        plt.imshow(np.ones((8, 16)), cmap='Greys')
        plt.scatter(center_y, center_x, s=100, c='red', marker='o')
        min_error_x = center_x-np.min(centers[:, t, 0])
        max_error_x = np.max(centers[:, t, 0])
        min_error_y = -np.min(centers[:, t, 1])
        max_error_y = np.max(centers[:, t, 1])
        print(min_error_x, max_error_x, min_error_y, max_error_y)
        plt.errorbar(center_y, center_x) #yerr=np.array([[min_error_x, max_error_x]]).T,
        #xerr=np.array([[min_error_y, max_error_y]]).T, capsize=2, ecolor='red')
        ellipse = patches.Ellipse(
            (center_y, center_x),
            width=2 * std_x,
            height=2 * std_y,
            angle=0,
            fill=False,  # Set to True if you want a filled ellipse
            color='red',  # Set the color of the ellipse
            linestyle='--',  # Set the linestyle (optional),
            linewidth=1
        )
        ax = plt.gca()
        ax.add_patch(ellipse)

    plt.show()

    ##################################
    exit()
    #allRhos, allMSEs, allModelSpace = processSessions(nSubs=nSubs, showModelledImagePerSub=False, nModels=nModels, nIterations=nIterations)
    plotSummary(allRhos, allMSEs, allModelSpace)
    saveMatlab()