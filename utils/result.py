
import os
import torch
import copy

import pandas as pd

from typing import List

from .metric import accuracy, classwise_accuracy


def load_prediction(exp_folder : str, pth_files : List[str] = []):
    """Method to load predictions in a given experimet folder

    Args:
        exp_folder (str): path to the experiment folder
        pth_files (List[str], optional): List of pth files to load. Defaults to None.

    Returns:
        dict: predictions dataframe and additional pth files data
    """

    prediction_file = os.path.join(exp_folder, "prediction.csv")
    if os.path.exists(prediction_file):
        prediction_df = pd.read_csv(prediction_file)

        prediction_dict = {}
        for pth_file in pth_files:
            pth_filename = os.path.join(exp_folder, f"{pth_file}.pth")
            content = torch.load(pth_filename)
            prediction_dict[pth_file] = content

        return prediction_df, prediction_dict
    else:
        print(f"{exp_folder} does not exist !")
        return None, None


def get_results_df(root_dir : str, dataset : str, fields_dict : dict, 
                  fields_order : list = ["method", "arch", "n_shots", "seed"], epoch : str = "last"):

    all_folders = [os.path.join(root_dir, dataset)]
    n_fields = len(fields_order)

    depth = 0
    while depth < n_fields:
        all_pred_files = []
        field = fields_order[depth]
        for pred_folder in all_folders:
            all_pred_files += [os.path.join(pred_folder, x) for x in os.listdir(pred_folder) if x in fields_dict[field]]
        depth += 1

        if depth < n_fields:
            all_folders = copy.deepcopy(all_pred_files)
        else:
            all_pred_files = [os.path.join(x, f"prediction_{epoch}.csv") for x in all_pred_files]

    # get all prediction files and filter them
    all_pred_files = [x for x in all_pred_files if os.path.exists(x)]

    # Load all results
    rows_list = []
    for prediction_file in all_pred_files:
        prediction_df = pd.read_csv(prediction_file)[["label", "pred", "wsi"]]
        fields = prediction_file.split("/", n_fields+3)[3:-1]
        results = classification_results(prediction_df["pred"].values, prediction_df["label"].values)
        for field_name, field_value in zip(fields_order, fields):
            results[field_name] = field_value
        rows_list.append(results)

    df = pd.DataFrame(rows_list)
    df.drop("seed", axis=1)
    df = df.groupby(fields_order[:-1])

    return df.mean(numeric_only=True), df.std(numeric_only=True)


def classification_results(preds, targets, prob=None):

    results = {}

    # Accuracy
    if preds is not None:
        results["ACC"] = accuracy(preds, targets).item()
        results["AVG"], class_acc = classwise_accuracy(preds, targets)

        if len(class_acc) == 2 and prob is not None:
            from sklearn.metrics import roc_auc_score
            results["AUC"] = roc_auc_score(targets, prob)

        for i, acc in enumerate(class_acc):
            results[str(i)] = acc.item()

    return results

    