import time
import torch
import numpy as np
import pandas as pd

from torch import nn
from sklearn import metrics
import torch.optim as optim


def dataframe_to_torch_dataset(dataframe, using_ce_loss=False, class_label=None):
    """Convert a one-hot pandas dataframe to a PyTorch Dataset of Tensor objects"""

    new = dataframe.copy()
    if class_label:
        label = class_label
    else:
        label = list(new.columns)[-1]
        # print(f"Inferred that class label is '{label}' while creating dataloader")
    labels = torch.Tensor(pd.DataFrame(new[label]).values)
    del new[label]
    data = torch.Tensor(new.values)

    if using_ce_loss:
        # Fixes tensor dimension and float -> int if using cross entropy loss
        return torch.utils.data.TensorDataset(
            data, labels.squeeze().type(torch.LongTensor)
        )
    else:
        return torch.utils.data.TensorDataset(data, labels)


def dataset_to_dataloader(
    dataset, batch_size=256, num_workers=4, shuffle=True, persistent_workers=False
):
    """Wrap PyTorch dataset in a Dataloader (to allow batch computations)"""
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        persistent_workers=persistent_workers,
    )
    return loader


def dataframe_to_dataloader(
    dataframe,
    batch_size=256,
    num_workers=4,
    shuffle=True,
    persistent_workers=False,
    using_ce_loss=False,
    class_label=None,
):
    """Convert a pandas dataframe to a PyTorch Dataloader"""

    dataset = dataframe_to_torch_dataset(
        dataframe, using_ce_loss=using_ce_loss, class_label=class_label
    )
    return dataset_to_dataloader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        persistent_workers=persistent_workers,
    )


def get_metrics(y_true, y_pred):
    """Takes in a test dataloader + a trained model and returns a numpy array of predictions
    ...
    Parameters
    ----------
        y_true: Ground Truth Predictions

        y_pred: Model Predictions

    ...
    Returns
    -------
        Accuracy, Precision, Recall, F1 score
    """
    acc = metrics.accuracy_score(y_true, y_pred)
    # precision = metrics.precision_score(y_true, y_pred)
    # recall = metrics.recall_score(y_true, y_pred)
    # f1 = metrics.f1_score(y_true, y_pred)

    # return acc, precision, recall, f1
    return acc

def get_prediction(test_loader, model, one_hot=False, ground_truth=False, device="cpu"):
    """Takes in a test dataloader + a trained model and returns a numpy array of predictions
    ...
    Parameters
    ----------
        test_loader : PyTorch Dataloader
            The Pytorch Dataloader for Dtest

        model : torch.nn.Module (PyTorch Neural Network Model)
            A trained model to be queried on the test data

        one_hot: bool
            If true, returns predictions in one-hot format
            else, returns predictions in integer format

        device : str
            The processing device for training computations.
            Ex: cuda, cpu, cuda:0

    ...
    Returns
    -------
        attackdata_arr : np.array
            Numpy array of predictions
    """
    model = model.to(device)
    y_pred_torch = torch.Tensor([])
    y_true_torch = torch.Tensor([])

    for d, l in test_loader:
        d = d.to(device)
        l = l.squeeze()
        model.eval()
        # with torch.no_grad():
        out = nn.Sigmoid()(model(d))
        # out_np = out.cpu().detach().numpy()
        # y_pred = np.r_[y_pred,out_np]
        y_pred_torch = torch.concat([torch.argmax(out, dim=1).cpu(), y_pred_torch])
        y_true_torch = torch.concat([l.cpu(), y_true_torch])

    y_pred = y_pred_torch.cpu().detach().numpy()
    y_true = y_true_torch.cpu().detach().numpy()

    # y_pred = np.argmax(y_pred,axis=1)

    if one_hot:
        y_pred = np.eye(model._num_classes)[y_pred]

    if ground_truth == True:
        return y_pred, y_true

    return y_pred

def get_logits_torch(
    test_loader, model, device="cpu", middle_measure="mean", variance_adjustment=1, max_conf = 1 - 1e-16, min_conf = 0 + 1e-16 , label = None
):
    """Takes in a test dataloader + a trained model and returns the scaled logit values

    ...
    Parameters
    ----------
        test_loader : PyTorch Dataloader
            The Pytorch Dataloader for Dtest
        model : torch.nn.Module (PyTorch Neural Network Model)
            A trained model to be queried on the test data
        device : str
            The processing device for training computations.
            Ex: cuda, cpu, cuda:0
        middle_measure : str
            When removing outliters from the data, this is the
            "center" of the distribution that will be used.
            Options are ["mean", "median"]
        variance_adjustment : float
            The number of standard deviations away from the "center"
            we want to keep.
    ...
    Returns
    -------
        logits_arr : np.array
            An array containing the scaled model confidence values on the query set
    """

    n_samples = len(test_loader.dataset)
    logit_arr = np.zeros((n_samples, 1))
    # activation_dict = {}
        
    model = model.to(device)

    y_prob = torch.Tensor([])
    y_test = torch.Tensor([])
    for d, l in test_loader:
        d = d.to(device)
        model.eval()
        with torch.no_grad():
            out = model(d)
            # Get class probabilities
            out = nn.functional.softmax(out, dim=1).cpu()
            y_prob = torch.concat([y_prob, out])
            y_test = torch.concat([y_test, l])

    y_prob, y_test = np.array(y_prob), np.array(y_test, dtype=np.uint8)

    # print(y_prob.shape)

    if np.sum(y_prob > max_conf):
        indices = np.argwhere(y_prob > max_conf)
        #             print(indices)
        for idx in indices:
            r, c = idx[0], idx[1]
            y_prob[r][c] = y_prob[r][c] - 1e-50

    if np.sum(y_prob < min_conf):
        indices = np.argwhere(y_prob < min_conf)
        for idx in indices:
            r, c = idx[0], idx[1]
            y_prob[r][c] = y_prob[r][c] + 1e-50

    possible_labels = len(y_prob[0])
    for sample_idx, sample in enumerate(zip(y_prob, y_test)):

        conf, og_label = sample
        if(label == None):
            label = og_label
        selector = [True for _ in range(possible_labels)]
        selector[label] = False

        first_term = np.log(conf[label])
        second_term = np.log(np.sum(conf[selector]))

        logit_arr[sample_idx, 0] = first_term - second_term

    # print(logit_arr.shape)

    logit_arr = logit_arr.reshape(-1)

    if middle_measure == "mean":
        middle = logit_arr.mean()
    elif middle_measure == "median":
        middle = np.median(logit_arr)

    # if(distinguish_type == 'global_threshold'):
    logit_arr_filtered = logit_arr[
        logit_arr > middle - variance_adjustment * logit_arr.std()
    ]  # Remove observations below the min_range
    logit_arr_filtered = logit_arr_filtered[
        logit_arr_filtered < middle + variance_adjustment * logit_arr_filtered.std()
    ]  # Remove observations above max range

    return logit_arr_filtered

def fit(
    dataloaders,
    model,
    epochs=100,
    optim_init=optim.Adam,
    optim_kwargs={"lr": 0.03, "weight_decay": 0.0001},
    criterion=nn.CrossEntropyLoss(),
    device="cpu",
    verbose=True,
    mini_verbose=True,
    train_only=True,
    early_stopping=False,
    tol=10e-6,
):
    """Fits a PyTorch model to any given dataset

    ...
    Parameters
    ----------
        dataloaders : dict
            Dictionary containing 2 PyTorch DataLoaders with keys "train" and
            "test" corresponding to the two dataloaders as values
        model : torch.nn.Module (PyTorch Neural Network Model)
            The desired model to fit to the data
        epochs : int
            Training epochs for shadow models
        optim_init : torch.optim init object
            The init function (as an object) for a PyTorch optimizer.
            Note: Pass in the function without ()
        optim_kwargs : dict
            Dictionary of keyword arguments for the optimizer
            init function
        criterion : torch.nn Loss Function
            Loss function used to train model
        device : str
            The processing device for training computations.
            Ex: cuda, cpu, cuda:1
        verbose : bool
            If True, prints running loss and accuracy values
        train_only : bool
            If True, only uses "train" dataloader

    ...
    Returns
    -------
        model : torch.nn.Module (PyTorch Neural Network Model)
            The trained model
        train_error : list
            List of average training errors at each training epoch
        test_acc : list
            List of average test accuracy at each training epoch
    """

    model = model.to(device)
    optimizer = optim_init(model.parameters(), **optim_kwargs)
    train_error = []
    test_loss = []
    test_acc = []
    # if train_only:
    #     phases = ["train"]
    # else:
    #     phases = ["train", "test"]
    if mini_verbose:
        print("Training...")
    if verbose:
        print("-" * 8)

    try:

        for epoch in range(1, epochs + 1):
            if verbose:
                a = time.time()
                print(f"Epoch {epoch}")

            running_train_loss = 0
            running_test_loss = 0
            running_test_acc = 0

            for (inputs, labels) in dataloaders['train']:
                optimizer.zero_grad()
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model.forward(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_train_loss += loss.item() * inputs.size(0)


            train_error.append(running_train_loss / len(dataloaders['train'].dataset))

            if len(train_error) > 1 and early_stopping:
                if abs(train_error[-1] - train_error[-2]) < tol:
                    print(f"Loss did not decrease by more than {tol}")
                    if mini_verbose:
                        print(f"Final Train Error: {train_error[-1]:.6}")
                    if not train_only:
                        return model, train_error, test_loss, test_acc
                    else:
                        return model, train_error

            if not train_only:
                test_loss.append(running_test_loss / len(dataloaders["test"].dataset))
                test_acc.append(running_test_acc / len(dataloaders["test"].dataset))
            if verbose:
                b = time.time()
                print(f"Train Error: {train_error[-1]:.6}")
                if not train_only:
                    print(f"Test Error: {test_loss[-1]:.6}")
                    print(f"Test Accuracy: {test_acc[-1]*100:.4}%")
                print(f"Time Elapsed: {b - a:.4} seconds")
                print("-" * 8)
    except KeyboardInterrupt:
        if mini_verbose:
            print(f"Final Train Error: {train_error[-1]:.6}")
        if not train_only:
            return model, train_error, test_loss, test_acc
        else:
            return model, train_error

    if mini_verbose:
        print(f"Final Train Error: {train_error[-1]:.6}")
    if not train_only:
        return model, train_error, test_loss, test_acc
    else:
        return model, train_error
