import glob
import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm
from Features import Stats as st
from Features import Games as gm
from sklearn.model_selection import KFold
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
from sklearn import metrics


SEASON_AVG = "season_averages/"
SEASON_GAMES = "season_games/"
BALANCED_DATASETS = True


def balancedDatasets(x, y):
    # There are more local wins (0), so look for visitor (1)
    x_home = np.argwhere(np.array(y) == 0)  # Get local win elements
    x_home = np.hstack(x_home)
    x_away = np.argwhere(np.array(y) == 1)  # Get away win elements
    x_away = np.hstack(x_away)
    x_home_indices = np.random.choice(x_home, size=len(x_away), replace=False)
    total_indices = np.sort(np.hstack((x_home_indices, x_away)))

    return [x[i] for i in total_indices], [y[i] for i in total_indices]

def teamLoader(local, visitor, season, features, averaged=True):
    local_team = np.zeros((12, 23))
    visitor_team = np.ones((12, 23))
    local_id_norm = float("{:.2f}".format(local / 30))
    visitor_id_norm = float("{:.2f}".format(visitor / 30))
    id_1 = str(local)
    id_2 = str(visitor)
    if averaged:
        id_1 += "-36"
        id_2 += "-36"
    files = glob.glob(SEASON_AVG + str(season) + "/" + id_1 + '/*.json', recursive=False)  # Local team
    for file in files:
        json_file = pd.read_json(file)
        for idx, row in json_file.iterrows():
            if np.array(list(row['data'].values())).shape[0] != 22:
                return None, None
            local_team[idx] = np.hstack((np.array(list(row['data'].values())), local_id_norm))

    files = glob.glob(SEASON_AVG + str(season) + "/" + id_2 + '/*.json', recursive=False)  # Visitor team
    for file in files:
        json_file = pd.read_json(file)
        for idx, row in json_file.iterrows():
            if np.array(list(row['data'].values())).shape[0] != 22:
                return None, None
            visitor_team[idx] = np.hstack((np.array(list(row['data'].values())), visitor_id_norm))

    if len(features) > 0:
        local_team = local_team[:, features]
        visitor_team = visitor_team[:, features]
    # local_team = np.append(local_team, np.zeros((12, 1)), axis=1)
    # visitor_team = np.append(visitor_team, np.ones((12, 1)), axis=1)

    return local_team, visitor_team


def gameLoader(season):
    file = glob.glob(SEASON_GAMES + str(season) + '/*.json', recursive=False)  # Season games
    json_file = pd.read_json(file[0])
    games = np.zeros((json_file.shape[0], len(json_file.iloc[0][0]) - 1))  # Remove the date
    for idx, row in json_file.iterrows():
        games[idx] = np.array(list(row['data'].values())[1:])

    return games


def dataLoader(season_start, season_end, features):
    x = list()
    y = list()
    seasons = list()
    for season in range(season_start, season_end + 1):
        games = gameLoader(season)
        seasons.append(season)
        for game in tqdm(games):
            local_team, visitor_team = teamLoader(int(game[gm.home_team_id.value]),
                                                  int(game[gm.visitor_team_id.value]),
                                                  season,
                                                  features,
                                                  True)
            if local_team is not None and visitor_team is not None:
                x.append(np.vstack((local_team, visitor_team)))
                y.append(np.argmax([game[gm.home_team_score.value], game[gm.visitor_team_score.value]]))
    print("Seasons obtained " + str(seasons))

    return x, y


def main():
    # Training data
    features = [
        st.ast.value,
        st.blk.value,
        st.dreb.value,
        # st.fg3_pct.value,
        st.fg3a.value,
        st.fg3m.value,
        # st.fg_pct.value,
        st.fga.value,
        st.fgm.value,
        # st.ft_pct.value,
        st.fta.value,
        st.ftm.value,
        st.games_played.value,
        # st.seconds.value,
        st.oreb.value,
        st.pf.value,
        # st.player_id.value,
        st.pts.value,
        # st.reb.value,
        # st.season.value,
        st.stl.value,
        st.turnover.value
    ]
    print("Loading training data...")
    if os.path.exists("train.npz"):
        npz_file = np.load("train.npz")
        x_train = npz_file['x']
        y_train = npz_file['y']
    else:
        x_train, y_train = dataLoader(2002, 2002, features)  # Both initial and end are included
        if BALANCED_DATASETS:
            x_train, y_train = balancedDatasets(x_train, y_train)
        np.savez("train.npz", x=x_train, y=y_train)
    training_samples = len(x_train)
    # Testing data
    print("Loading testing data...")
    if os.path.exists("test.npz"):
        npz_file = np.load("test.npz")
        x_test = npz_file['x']
        y_test = npz_file['y']
    else:
        x_test, y_test = dataLoader(2015, 2015, features)  # Both initial and end are included
        if BALANCED_DATASETS:
            x_test, y_test = balancedDatasets(x_test, y_test)
        np.savez("test.npz", x=x_test, y=y_test)
    testing_samples = len(x_test)
    # Scale training data
    x_to_scale = np.vstack(x_train)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_to_scale[:, :len(features)])
    x_train = np.hstack((x_train, x_to_scale[:, -1].reshape((-1, 1))))
    x_train = np.hstack((x_train, np.ones((x_train.shape[0], 1))))
    x_train = [x_train[i * 24:24 * (i + 1), :] for i in range(training_samples)]
    x_train = np.vstack([x_train[i].flatten() for i in range(len(x_train))])
    # Scale testing data
    x_to_scale = np.vstack(x_test)
    x_test = scaler.transform(x_to_scale[:, :len(features)])
    x_test = np.hstack((x_test, x_to_scale[:, -1].reshape((-1, 1))))
    x_test = np.hstack((x_test, np.ones((x_test.shape[0], 1))))
    x_test = [x_test[i * 24:24 * (i + 1), :] for i in range(testing_samples)]
    x_test = np.vstack([x_test[i].flatten() for i in range(len(x_test))])

    # clf = MLPClassifier(hidden_layer_sizes=(100, 100, 100), early_stopping=True, max_iter=250,
    #                     solver="adam",
    #                     activation="logistic"
    #                     ).fit(x_train, y_train)
    # print("MLP: Accuracy on testing: " + str(clf.score(x_test, y_test)))


    print("Starting K folds...")
    X_kfolds = np.vstack((x_train, x_test))
    Y_kfolds = np.hstack((y_train, y_test))
    kf = KFold(n_splits=10, shuffle=False)
    result_MLP_kf = []
    result_NB_kf = []
    result_RF_kf = []
    result_LR_kf = []
    y_pred_label_MLP = []
    y_pred_label_NB = []
    y_pred_label_RF = []
    y_pred_label_LR = []
    y_real_list = []
    counter = 0

    for train_index, test_index in kf.split(X_kfolds):
        X_ktrain, X_ktest = X_kfolds[train_index], X_kfolds[test_index]
        Y_ktrain, Y_ktest = Y_kfolds[train_index], Y_kfolds[test_index]

        # MLP
        print("MLP...")
        clf_MLP = MLPClassifier(hidden_layer_sizes=(100, 100, 100), early_stopping=True, max_iter=100000,
                            solver="adam",
                            activation="relu"
                            ).fit(X_ktrain, Y_ktrain)
        y_pred_MLP = clf_MLP.predict_proba(X_ktest)
        y_pred_label_MLP.append(np.mean(clf_MLP.predict(X_ktest)))
        result_MLP_kf.append(clf_MLP.score(X_ktest, Y_ktest))
        fpr, tpr, _ = metrics.roc_curve(Y_ktest, [y_pred_MLP[i, Y_ktest[i]] for i in range(y_pred_MLP.shape[0])])
        roc_auc = auc(fpr, tpr)
        plt.figure(1)
        plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)


        # NB
        print("Naive Bayes...")
        clf_gnb = GaussianNB().fit(X_ktrain, Y_ktrain)
        result_NB_kf.append(clf_gnb.score(X_ktest, Y_ktest))
        y_pred_label_NB.append(np.mean(clf_gnb.predict(X_ktest)))
        y_pred_NB = clf_gnb.predict_proba(X_ktest)
        result_NB_kf.append(clf_gnb.score(X_ktest, Y_ktest))
        fpr, tpr, _ = metrics.roc_curve(Y_ktest, [y_pred_NB[i, Y_ktest[i]] for i in range(y_pred_NB.shape[0])])
        roc_auc = auc(fpr, tpr)
        plt.figure(2)
        plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)

        # Random Forest
        print("Random Forest...")
        clf_rf = RandomForestClassifier().fit(X_ktrain, Y_ktrain)
        result_RF_kf.append(clf_rf.score(X_ktest, Y_ktest))
        y_pred_label_RF.append(np.mean(np.array(clf_rf.predict(X_ktest))))
        y_pred_RF = clf_rf.predict_proba(X_ktest)
        result_RF_kf.append(clf_rf.score(X_ktest, Y_ktest))
        fpr, tpr, _ = metrics.roc_curve(Y_ktest, [y_pred_RF[i, Y_ktest[i]] for i in range(y_pred_RF.shape[0])])
        roc_auc = auc(fpr, tpr)
        plt.figure(3)
        plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)

        # Logistic Regression
        print("Logistic Regression...")
        clf_LR = LogisticRegression(max_iter=100000).fit(X_ktrain, Y_ktrain)
        result_LR_kf.append(clf_LR.score(X_ktest, Y_ktest))
        y_pred_LR = clf_LR.predict_proba(X_ktest)
        y_pred_label_LR.append(np.mean(np.array(clf_LR.predict(X_ktest))))
        result_LR_kf.append(clf_LR.score(X_ktest, Y_ktest))
        fpr, tpr, _ = metrics.roc_curve(Y_ktest, [y_pred_LR[i, Y_ktest[i]] for i in range(y_pred_LR.shape[0])])
        roc_auc = auc(fpr, tpr)
        plt.figure(4)
        plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)

        y_real_list.append(np.mean(Y_ktest))
        counter += 1
        print("Fold " + str(counter) + " performed.")




    # MLP results
    kf_MLP_acc_mean = np.mean(result_MLP_kf)
    kf_MLP_acc_std = np.std(result_MLP_kf)
    print("MLP: Mean Accuracy on testing: " + str(kf_MLP_acc_mean) + " +- " + str(kf_MLP_acc_std))
    print("MLP: Mean prediction: " + str(np.mean(y_pred_label_MLP)) + "+-" + str(np.std(y_pred_label_MLP)))


    # NB results
    kf_NB_acc_mean = np.mean(result_NB_kf)
    kf_NB_acc_std = np.std(result_NB_kf)
    print("Naive Bayes: Mean Accuracy on testing: " + str(kf_NB_acc_mean) + " +- " + str(kf_NB_acc_std))
    print("Naive Bayes: Mean prediction: " + str(np.mean(y_pred_label_NB)) + "+-" + str(np.std(y_pred_label_NB)))

    # RF Results
    kf_RF_acc_mean = np.mean(result_RF_kf)
    kf_RF_acc_std = np.std(result_RF_kf)
    print("Random Forest: Mean Accuracy on testing: " + str(kf_RF_acc_mean) + " +- " + str(kf_RF_acc_std))
    print("Random Forest: Mean prediction: " + str(np.mean(y_pred_label_RF)) + "+-" + str(np.std(y_pred_label_RF)))

    # Logistic Regression
    kf_LR_acc_mean = np.mean(result_LR_kf)
    kf_LR_acc_std = np.std(result_LR_kf)
    print("Logistic Regression: Mean Accuracy on testing: " + str(kf_LR_acc_mean) + " +- " + str(kf_LR_acc_std))
    print("Logistic Regression: Mean prediction: " + str(np.mean(y_pred_label_LR)) + "+-" + str(np.std(y_pred_label_LR)))

    print("Total percentage: " + str(np.mean(y_real_list)) + "+-" + str(np.std(y_real_list)))


    #Compute ROC - AUC

    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('MLP ROC')
    plt.legend(loc="lower right")

    plt.figure(2)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Naive Bayes ROC')
    plt.legend(loc="lower right")

    plt.figure(3)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Random Forest ROC')
    plt.legend(loc="lower right")

    plt.figure(4)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Logistic Regression ROC')
    plt.legend(loc="lower right")

    plt.show()



if __name__ == "__main__":
    main()
