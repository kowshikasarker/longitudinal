# imports
import argparse
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import csv, os, warnings, sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate, ParameterGrid
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score
import shutil
import json
from pathlib import Path

# turn off spammy warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def random_forest_classifier(in_path):
    treatment = 'Recur'
    control = 'No' + treatment
    classes = {treatment: 1, control: 0}
    
    df = pd.read_csv(in_path, sep='\t')
    if(df.shape[1] == 1):
        print('No features')
        return    
    df['class'] = df['Key'].apply(lambda x: 0 if control in x else 1)
    print(list(zip(df['Key'], df['class'])))
    y_true = np.array(df['class'].values)
    X = df.set_index('Key').drop(columns=['class'])
    print(X.shape, y_true.shape)
    
    # Param grid to search for each food
    param_grid = {
        "n_estimators": [500],
        "oob_score": [True],
        "n_jobs": [-1],
        "random_state": [1],
        "max_features": [None],
        "min_samples_leaf": [1],
    }

    # Grid search
    best_rf = None
    best_params = None
    for params in ParameterGrid(param_grid):
        rfc = RandomForestClassifier()
        rfc.set_params(**params)

        # Perform LOO evaluation for this parameter set
        cv_result = cross_validate(
            rfc,
            X.values,
            y_true,
            scoring=None,
            cv=StratifiedKFold(n_splits=5),
            n_jobs=-1,
            return_estimator=True,
        )

        # Update the best parameters
        estimators = cv_result["estimator"]
        print('estimator_count', len(estimators))
        estimator_count = 0
        for estimator in estimators:
            estimator_count += 1
            if best_rf is None or estimator.oob_score_ > best_rf.oob_score_:
                best_rf = estimator
                best_params = params
        print('estimator_count', estimator_count)

    print(treatment, "-> Best parameters from grid search:", best_params)

    rfc = RandomForestClassifier()
    rfc.set_params(**best_params)
    y_proba = cross_val_predict(
        rfc,
        X.values,
        y_true,
        cv=StratifiedKFold(n_splits=5),
        n_jobs=-1,
        method="predict_proba",
    )

    np.savetxt(fname=treatment + '.' + control + '.classifier_prediction.tsv', header=treatment + '\t' + control, X=y_proba, delimiter='\t')
    
    y_pred = [score.argmax() for score in y_proba]
    
    metric = classification_report(y_true, y_pred, target_names=[control, treatment], output_dict=True)
    metric['auroc'] = round(roc_auc_score(y_true, y_proba[:, 1]), 2)
    metric['auprc'] = round(average_precision_score(y_true, y_proba[:, 1]), 2)
    
    metric_file = open(treatment + '.' + control + '.classifier_performance.json', 'w')
    
    json.dump(metric, metric_file)
    
    print('Accuracy', metric['accuracy'])
    print('AUROC', metric['auroc'])
    print('AUPRC', metric['auprc'])

    # Plot feature importance graph
    feature_idxs = np.argsort(best_rf.feature_importances_)[::-1]
    plt.figure(figsize=(5, 5))
    plt.title(treatment + 'Feature Importances')
    plt.xlabel("Feature #")
    plt.ylabel("Feature Importance")
    plt.plot(sorted(best_rf.feature_importances_, reverse=True))
    plt.savefig(treatment + '.' + control + '.feature-importance.png')
    plt.close()
    best_features = X.columns[feature_idxs[:10]]
    print('Top-10 features for', treatment, ':', best_features)

    # feature means per group write-out
    classes = {0: 'Control', 1: treatment}
    X_gb = X.copy().iloc[:, feature_idxs]
    X_gb["group"] = list(map(lambda i: classes[i], y_true))
    print("group", X_gb["group"])
    X_gb.groupby("group").mean().to_csv(treatment + '.' + control + '.feature-mean.csv')
    X_gb.groupby("group").std().to_csv(treatment + '.' + control + '.feature-std.csv')

    # Feautre importances write out + figure generation
    best_features_list = list(
        zip(
            [X.columns[idx] for idx in feature_idxs],
            [best_rf.feature_importances_[idx] for idx in feature_idxs],
        )
    )
    #best_features_per_food[food] = best_features_list
    with open(treatment + '.' + control + '.feature-importance.csv', "w") as f:
        w = csv.writer(f)
        w.writerow(["Feature", "Importance"])
        for idx in feature_idxs:
            w.writerow([X.columns[idx], best_rf.feature_importances_[idx]])

    # plot features
    for feature in best_features:
        fig, ax = plt.subplots(figsize=(5, 5))
        sns.boxplot(
            data=X_gb, x="group", y=feature, linewidth=2.5, width=0.4, ax=ax
        )
        ax.set_xlabel("Treatment group")
        ax.set_ylabel("Relative concentration")
        fig.suptitle(feature, size=22)
        fig.savefig(treatment + '.' + control + '.' + feature + '.boxplot.png', bbox_inches="tight")
        plt.close(fig)

    
    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--in_path", type=str,
                        help="path to input features in .tsv format",
                        required=True, default=None)
    parser.add_argument("-o", "--out_dir", type=str,
                        help="path to output dir",
                        required=True, default=None)
    parser.add_argument("-l", "--log_path", type=str,
                        help="path to log file",
                        required=True, default=None)
    args = parser.parse_args()
    return args

def main(args):
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    os.chdir(args.out_dir)
    
    original_stdout = sys.stdout
    log_file = open(args.log_path, 'w')
    sys.stdout = log_file

    random_forest_classifier(args.in_path)
    
    sys.stdout = original_stdout
    log_file.flush()
    log_file.close()
    
if __name__ == "__main__":
    main(parse_args())