import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def get_adult_columns():

    column_names = [
        "age",
        "workclass",
        "education-num",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
        "native-country",
    ]

    cont_columns = [
        "age",
        "education-num",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
    ]

    cat_columns = sorted(list(set(column_names).difference(cont_columns)))

    return cat_columns, cont_columns


def get_census_columns():
    """Returns names of categorical and continuous columns for census dataset"""

    column_names = [
        "age",
        "class-of-worker",
        "detailed-industry-recode",
        "detailed-occupation-recode",
        "education",
        "wage-per-hour",
        "enroll-in-edu-inst-last-wk",
        "marital-stat",
        "major-industry-code",
        "major-occupation-code",
        "race",
        "hispanic-origin",
        "sex",
        "member-of-a-labor-union",
        "reason-for-unemployment",
        "full-or-part-time-employment-stat",
        "capital-gains",
        "capital-losses",
        "dividends-from-stocks",
        "tax-filer-stat",
        "region-of-previous-residence",
        "state-of-previous-residence",
        "detailed-household-and-family-stat",
        "detailed-household-summary-in-household",
        "instance-weight",
        "migration-code-change-in-msa",
        "migration-code-change-in-reg",
        "migration-code-move-within-reg",
        "live-in-this-house-1-year-ago",
        "migration-prev-res-in-sunbelt",
        "num-persons-worked-for-employer",
        "family-members-under-18",
        "country-of-birth-father",
        "country-of-birth-mother",
        "country-of-birth-self",
        "citizenship",
        "own-business-or-self-employed",
        "fill-inc-questionnaire-for-veterans-admin",
        "veterans-benefits",
        "weeks-worked-in-year",
        "year",
    ]

    cont_columns = [
        "age",
        "wage-per-hour",
        "capital-gains",
        "capital-losses",
        "dividends-from-stocks",
        "instance-weight",
        "num-persons-worked-for-employer",
        "weeks-worked-in-year",
    ]
    cat_columns = sorted(list(set(column_names).difference(cont_columns)))

    return cat_columns, cont_columns


def generate_class_imbalance(data, target_class, target_ratio):
    Nt = sum(data["class"] == target_class)
    No = (1 - target_ratio) * Nt / target_ratio

    tgt_idx = data["class"] == target_class
    tgt_data = data[tgt_idx]
    other_data = data[~tgt_idx]
    other_data, _ = train_test_split(
        other_data, train_size=No / other_data.shape[0], random_state=21
    )

    data = pd.concat([tgt_data, other_data])
    data = data.sample(frac=1).reset_index(drop=True)

    return data


def v2_fix_imbalance(
    df,
    target_split=0.4,
    categories=["sex"],
    target_attributes=[" Female"],
    random_seed=21,
    return_indices=False,
):
    """Corrects a data set to have a target_split percentage of the target attributes

    ...
    Parameters
    ----------
        df : Pandas Dataframe
            The dataset
        target_split : float
            The desired proportion of the subpopulation within the dataset
        categories : list
            Column names for each attribute
        target_attributes : list
            Labels to create subpopulation from.
            Ex. The subpopulation will be df[categories] == attributes
        return_indices : bool
            If True, returns the indices to drop instead of the modified dataframe

    ...
    Returns
    -------

    df : Pandas Dataframe
        The dataset with a target_split/(1-target_split) proportion of the subpopulation
    """

    assert target_split >= 0 and target_split <= 1, "target_split must be in (0,1)"

    if len(categories) == 0 or len(target_attributes) == 0:
        return df

    df = df.copy()

    np.random.seed(random_seed)

    indices_with_each_target_prop = []

    for category, target in zip(categories, target_attributes):
        indices_with_each_target_prop.append(df[category] == target)

    indices_with_all_targets = np.array(
        [all(l) for l in zip(*indices_with_each_target_prop)]
    )

    subpop_df = df[indices_with_all_targets]
    remaining_df = df[~indices_with_all_targets]

    rem_samples = remaining_df.shape[0]
    subpop_samples = int(target_split * rem_samples / (1 - target_split))

    if subpop_samples <= subpop_df.shape[0]:
        df = pd.concat(
            [remaining_df, subpop_df.sample(n=subpop_samples, random_state=random_seed)]
        )
    else:
        # print("oversampling")
        df = pd.concat(
            [
                remaining_df,
                subpop_df.sample(
                    n=subpop_samples, replace=True, random_state=random_seed
                ),
            ]
        )

    df = df.sample(frac=1).reset_index(drop=True)

    return df


def generate_subpopulation(
    df, categories=[], target_attributes=[], return_not_subpop=False
):
    """Given a list of categories and target attributes, generate a dataframe with only those targets
    ...
    Parameters
    ----------
        df : Pandas Dataframe
            A pandas dataframe
        categories : list
            Column names for each attribute
        target_attributes : list
            Labels to create subpopulation from.
            Ex. The subpopulation will be df[categories] == attributes
        return_not_subpop : bool
            If True, also return df/subpopulation

    ...
    Returns
    -------
        subpop : Pandas Dataframe
            The dataframe containing the target subpopulation
        not_subpop : Pandas Dataframe (optional)
            df/subpopulation
    """

    indices_with_each_target_prop = []

    for category, target in zip(categories, target_attributes):

        indices_with_each_target_prop.append(df[category] == target)

    indices_with_all_targets = np.array(
        [all(l) for l in zip(*indices_with_each_target_prop)]
    )

    if return_not_subpop:
        return df[indices_with_all_targets].copy(), df[~indices_with_all_targets].copy()
    else:
        return df[indices_with_all_targets].copy()


def generate_all_datasets(
    train_df,
    test_df,
    t0=0.1,
    t1=0.5,
    categories=["race"],
    target_attributes=[" White"],
    sub_categories=["occupation"],
    sub_attributes=[" Sales"],
    poison_class=1,
    k=None,
    poison_percent=None,
    verbose=True,
    allow_custom_freq=False,
    label_frequency=0.5,
    subproperty_sampling=False,
    restrict_sampling=False,
    random_state=21,
):
    """Generates the model owner's dataset (D_mo), the adversary's datasets (D0, D1), and
    the poisoned dataset (Dp)

        ...
        Parameters
        ----------
            train_df : Pandas Dataframe
                The train set
            test_df : Pandas Dataframe
                The validation set or some dataset that is disjoint from train_df,
                but drawn from the same distribution
            mo_frac: float
                Setting the proportion of of the subpopulation model owner's  to mo_frac
            t0 : float
                Lower estimate for proportion of the subpopulation in model owner's
                dataset
            t1 : float
                Upper estimate for proportion of the subpopulation in model owner's
                dataset
            categories : list
                Column names for each attribute
            target_attributes : list
                Labels to create subpopulation from.
                Ex. The subpopulation will be df[categories] == attributes
            poison_class : int
                The label we want our poisoned examples to have
            k : int
                The number of points in the poison set
            poison_percent : float
                [0,1] value that determines what percentage of the
                total dataset (D0) size we will make our poisoning set
                Note: Must use either k or poison_percent
            verbose : bool
                If True, reports dataset statistics via a print-out
            return_one_hot : bool
                If True returns dataframes with one-hot-encodings
            cat_columns : list
                The list of columns with categorical features (only used if
                return_one_hot = True)
            cont_columns : list
                The list of columns with continuous features (only used if
                return_one_hot = True)

        ...
        Returns
        -------
            D0_mo : Pandas Dataframe
                The model owner's dataset with t0 fraction of the target subpopulation
            D1_mo : Pandas Dataframe
                The model owner's dataset with t1 fraction of the target subpopulation
            D0 : Pandas Dataframe
                The adversary's dataset with t0 fraction of the target subpopulation
            D1 : Pandas Dataframe
                The adversary's dataset with t0 fraction of the target subpopulation
            Dp : Pandas Dataframe
                The adversary's poisoned set
            Dtest : Pandas Dataframe
                The adversary's query set
    """

    assert t0 >= 0 and t0 < 1, "t0 must be in [0,1)"
    assert t1 >= 0 and t1 < 1, "t1 must be in [0,1)"

    np.random.seed(random_state)
    all_indices = np.arange(0, len(train_df), dtype=np.uint64)
    np.random.shuffle(all_indices)

    D_mo, D_adv = train_test_split(train_df, test_size=0.5, random_state=random_state)

    D_mo = D_mo.reset_index(drop=True)
    D_adv = D_adv.reset_index(drop=True)

    if allow_custom_freq == True:

        D_mo = v2_fix_imbalance(
            D_mo,
            target_split=label_frequency,
            categories=["class"],
            target_attributes=[poison_class],
            random_seed=random_state,
        )

        D_adv = v2_fix_imbalance(
            D_adv,
            target_split=label_frequency,
            categories=["class"],
            target_attributes=[poison_class],
            random_seed=random_state,
        )

    if verbose:
        label_split_mo = sum(D_mo[D_mo.columns[-1]]) / len(D_mo)
        label_split_adv = sum(D_adv[D_adv.columns[-1]]) / len(D_adv)
        print(
            f"The model owner has {len(D_mo)} total points with {label_split_mo*100:.4}% class 1"
        )
        print(
            f"The adversary has {len(D_adv)} total points with {label_split_adv*100:.4}% class 1"
        )

    D0_mo = v2_fix_imbalance(
        D_mo,
        target_split=t0,
        categories=categories,
        target_attributes=target_attributes,
        random_seed=random_state,
    )

    D1_mo = v2_fix_imbalance(
        D_mo,
        target_split=t1,
        categories=categories,
        target_attributes=target_attributes,
        random_seed=random_state,
    )

    D0 = v2_fix_imbalance(
        D_adv,
        target_split=t0,
        categories=categories,
        target_attributes=target_attributes,
        random_seed=random_state,
    )

    D1 = v2_fix_imbalance(
        D_adv,
        target_split=t1,
        categories=categories,
        target_attributes=target_attributes,
        random_seed=random_state,
    )

    if len(D0) > len(D1):
        D0 = D0.sample(n=len(D1), random_state=random_state)
    elif len(D1) > len(D0):
        D1 = D1.sample(n=len(D0), random_state=random_state)

    if len(D0_mo) > len(D1_mo):
        D0_mo = D0_mo.sample(n=len(D1_mo), random_state=random_state)

    elif len(D1_mo) > len(D0_mo):
        D1_mo = D1_mo.sample(n=len(D0_mo), random_state=random_state)

    if len(D0_mo) > len(D0):
        D0_mo = D0_mo.sample(n=len(D0), random_state=random_state)
        D1_mo = D1_mo.sample(n=len(D0), random_state=random_state)

    elif len(D0) > len(D0_mo):
        D0 = D0.sample(n=len(D0_mo), random_state=random_state)
        D1 = D1.sample(n=len(D0_mo), random_state=random_state)

    if verbose:
        print(f"The model owner's dataset has been downsampled to {len(D0_mo)} points")

    if verbose:
        label_split_d0 = sum(D0[D0.columns[-1]]) / len(D0)
        label_split_d1 = sum(D1[D1.columns[-1]]) / len(D1)
        D0_subpop = generate_subpopulation(
            D0, categories=categories, target_attributes=target_attributes
        )
        D1_subpop = generate_subpopulation(
            D1, categories=categories, target_attributes=target_attributes
        )
        print(
            f"D0 has {len(D0)} points with {len(D0_subpop)} members from the target subpopulation and {label_split_d0*100:.4}% class 1"
        )
        print(
            f"D1 has {len(D1)} points with {len(D1_subpop)} members from the target subpopulation and {label_split_d1*100:.4}% class 1"
        )

    if poison_percent is not None:
        k = int(poison_percent * len(D0_mo))

    if subproperty_sampling == True:
        Dp, Dtest = generate_Dp(
            test_df,
            categories=categories + sub_categories,
            target_attributes=target_attributes + sub_attributes,
            k=k,
            poison_class=poison_class,
            random_state=random_state,
        )
    else:
        Dp, Dtest = generate_Dp(
            test_df,
            categories=categories,
            target_attributes=target_attributes,
            k=k,
            poison_class=poison_class,
            random_state=random_state,
        )

    if len(Dtest) == 0:
        Dtest = Dp.copy()

    if verbose:
        subpop = generate_subpopulation(
            test_df, categories=categories, target_attributes=target_attributes
        )
        print(
            f"The poisoned set has {k} points sampled uniformly from {len(subpop)} total points in the subpopulation"
        )

    if restrict_sampling == True:
        D0_subpop = generate_subpopulation(
            D0, categories=categories, target_attributes=target_attributes
        )
        D1_subpop = generate_subpopulation(
            D1, categories=categories, target_attributes=target_attributes
        )
        print(
            f"D0 has {len(D0)} points with {len(D0_subpop)} members from the target subpopulation"
        )
        print(
            f"D1 has {len(D1)} points with {len(D1_subpop)} members from the target subpopulation"
        )

        return D0_mo, D1_mo, D0, D1, Dp, Dtest, D0_subpop, D1_subpop

    return D0_mo, D1_mo, D0, D1, Dp, Dtest


def generate_Dp(
    test_df,
    categories=["race"],
    target_attributes=[" White"],
    poison_class=1,
    k=None,
    random_state=21,
):
    """Generate Dp, the poisoned dataset
    ...
    Parameters
    ----------
        test_df : Pandas Dataframe
            The validation set or some dataset that is disjoint from train set,
            but drawn from the same distribution
        categories : list
            Column names for each attribute
        target_attributes : list
            Labels to create subpopulation from.
            Ex. The subpopulation will be df[categories] == attributes
        poison_class : int
            The label we want our poisoned examples to have
        k : int
            The number of points in the poison set
        verbose : bool
            If True, reports dataset statistics via a print-out

    ...
    Returns
    -------
        Dp : Pandas Dataframe
            The adversary's poisoned set
        remaining_indices : np.array
            The remaining indices from test_df that we can use to query the target model.
            The indices correspond to points in the subpopulation that are *not* in the
            poisoned set
    """

    if k is None:
        raise NotImplementedError("Poison set size estimation not implemented")

    subpop = generate_subpopulation(
        test_df, categories=categories, target_attributes=target_attributes
    ).copy()
    label = subpop.columns[-1]
    np.random.seed(random_state)

    subpop_without_poison_label = subpop[subpop[label] != poison_class]

    all_indices = np.arange(0, len(subpop_without_poison_label), dtype=np.uint64)
    np.random.shuffle(all_indices)

    Dp = subpop_without_poison_label.iloc[all_indices[:k]]

    Dp.loc[:, label] = poison_class
    remaining_indices = all_indices[k:]

    Dtest = subpop_without_poison_label.iloc[remaining_indices]

    return Dp, Dtest


def all_dfs_to_one_hot(dataframes, cat_columns=[], class_label=None):
    """Transform multiple dataframes to one-hot concurrently so that
    they maintain consistency in their columns

        ...
        Parameters
        ----------
            dataframes : list
                A list of pandas dataframes to convert to one-hot
            cat_columns : list
                A list containing all the categorical column names for the list of
                dataframes
            class_label : str
                The column label for the training label column

        ...
        Returns
        -------
            dataframes_OH : list
                A list of one-hot encoded dataframes. The output is ordered in the
                same way the list of dataframes was input
    """

    keys = list(range(len(dataframes)))

    # Make copies of dataframes to not accidentally modify them
    dataframes = [df.copy() for df in dataframes]
    cont_columns = sorted(
        list(set(dataframes[0].columns).difference(cat_columns + [class_label]))
    )

    # Get dummy variables over union of all columns.
    # Keys keep track of individual dataframes to
    # split later
    temp = pd.get_dummies(pd.concat(dataframes, keys=keys), columns=cat_columns)
    temp.replace({False: 0, True: 1}, inplace=True)
    
    # Normalize continuous values
    temp[cont_columns] = temp[cont_columns] / temp[cont_columns].max()
    
    if class_label:
        temp["label"] = temp[class_label]
        temp = temp.drop([class_label], axis=1)

    # Return the dataframes as one-hot encoded in the same order
    # they were given
    return [temp.xs(i) for i in keys]


# --------------------------------------------------------------------------------------


def load_adult(one_hot=True, custom_balance=False, target_class=1, target_ratio=0.3):
    """Load the Adult dataset."""

    filename_train = "dataset/adult.data"
    filename_test = "dataset/adult.test"
    names = [
        "age",
        "workclass",
        "fnlwgt",
        "education",
        "education-num",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
        "native-country",
        "class",
    ]
    df_tr = pd.read_csv(filename_train, names=names)
    df_ts = pd.read_csv(filename_test, names=names, skiprows=1)

    df_tr.drop(["fnlwgt", "education"], axis=1, inplace=True)
    df_ts.drop(["fnlwgt", "education"], axis=1, inplace=True)

    # Separate Labels from inputs
    df_tr["class"] = df_tr["class"].astype("category")
    cat_columns = df_tr.select_dtypes(["category"]).columns
    df_tr[cat_columns] = df_tr[cat_columns].apply(lambda x: x.cat.codes)
    df_ts["class"] = df_ts["class"].astype("category")
    cat_columns = df_ts.select_dtypes(["category"]).columns
    df_ts[cat_columns] = df_ts[cat_columns].apply(lambda x: x.cat.codes)

    cont_cols = [
        "age",
        "education-num",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
    ]

    cat_cols = [
        "workclass",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]  # Indexs: {1,3,5,6,7,8,9,13}
    df = pd.concat([df_tr, df_ts])
    for col in cat_cols:
        df[col] = df[col].astype("category")

    if custom_balance == True:
        df_tr = generate_class_imbalance(
            data=df_tr, target_class=target_class, target_ratio=target_ratio
        )
        df_ts = generate_class_imbalance(
            data=df_ts, target_class=target_class, target_ratio=target_ratio
        )

    if one_hot == False:
        return df_tr, df_ts

    else:

        df_one_hot = pd.get_dummies(df)
        df_one_hot["labels"] = df_one_hot["class"]
        df_one_hot = df_one_hot.drop(["class"], axis=1)

        # Normalizing continuous coloumns between 0 and 1
        df_one_hot[cont_cols] = df_one_hot[cont_cols] / (df_one_hot[cont_cols].max())
        df_one_hot[cont_cols] = df_one_hot[cont_cols].round(3)
        #         df_one_hot.loc[:, df_one_hot.columns != cont_cols] = df_one_hot.loc[:, df_one_hot.columns != cont_cols].astype(int)

        df_tr_one_hot = df_one_hot[: len(df_tr)]
        df_ts_one_hot = df_one_hot[len(df_tr) :]

        return df_tr, df_ts, df_tr_one_hot, df_ts_one_hot


def load_census_data(
    one_hot=True, custom_balance=False, target_class=1, target_ratio=0.1
):
    """Load the data from the census income (KDD) dataset

    ...
    Parameters
    ----------
        one_hot : bool
            Indicates whether one-hot versions of the data should be loaded.
            The one-hot dataframes also have normalized continuous values

    Returns
    -------
        dataframes : tuple
            A tuple of dataframes that contain the census income dataset.
            They are in the following order [train, test, one-hot train, one-hot test]
    """

    filename_train = "dataset/census-income.data"
    filename_test = "dataset/census-income.test"

    column_names = [
        "age",
        "class-of-worker",
        "detailed-industry-recode",
        "detailed-occupation-recode",
        "education",
        "wage-per-hour",
        "enroll-in-edu-inst-last-wk",
        "marital-stat",
        "major-industry-code",
        "major-occupation-code",
        "race",
        "hispanic-origin",
        "sex",
        "member-of-a-labor-union",
        "reason-for-unemployment",
        "full-or-part-time-employment-stat",
        "capital-gains",
        "capital-losses",
        "dividends-from-stocks",
        "tax-filer-stat",
        "region-of-previous-residence",
        "state-of-previous-residence",
        "detailed-household-and-family-stat",
        "detailed-household-summary-in-household",
        "instance-weight",
        "migration-code-change-in-msa",
        "migration-code-change-in-reg",
        "migration-code-move-within-reg",
        "live-in-this-house-1-year-ago",
        "migration-prev-res-in-sunbelt",
        "num-persons-worked-for-employer",
        "family-members-under-18",
        "country-of-birth-father",
        "country-of-birth-mother",
        "country-of-birth-self",
        "citizenship",
        "own-business-or-self-employed",
        "fill-inc-questionnaire-for-veterans-admin",
        "veterans-benefits",
        "weeks-worked-in-year",
        "year",
    ]

    uncleaned_df_train = pd.read_csv(filename_train, header=None)
    uncleaned_df_test = pd.read_csv(filename_test, header=None)

    mapping = {i: column_names[i] for i in range(len(column_names))}
    mapping[len(column_names)] = "class"
    uncleaned_df_train = uncleaned_df_train.rename(columns=mapping)
    uncleaned_df_test = uncleaned_df_test.rename(columns=mapping)

    cont_columns = [
        "age",
        "wage-per-hour",
        "capital-gains",
        "capital-losses",
        "dividends-from-stocks",
        "instance-weight",
        "num-persons-worked-for-employer",
        "weeks-worked-in-year",
    ]
    cat_columns = sorted(list(set(column_names).difference(cont_columns)))

    encoder = LabelEncoder()
    uncleaned_df_train["class"] = encoder.fit_transform(uncleaned_df_train["class"])

    encoder = LabelEncoder()
    uncleaned_df_test["class"] = encoder.fit_transform(uncleaned_df_test["class"])

    uncleaned_df_train = uncleaned_df_train.drop(
        uncleaned_df_train[uncleaned_df_train["class"] == 2].index
    )
    uncleaned_df_test = uncleaned_df_test.drop(
        uncleaned_df_test[uncleaned_df_test["class"] == 2].index
    )

    if custom_balance == True:
        uncleaned_df_train = generate_class_imbalance(
            data=uncleaned_df_train,
            target_class=target_class,
            target_ratio=target_ratio,
        )
        uncleaned_df_test = generate_class_imbalance(
            data=uncleaned_df_test, target_class=target_class, target_ratio=target_ratio
        )

    if one_hot:
        # Normalize continous values
        uncleaned_df_train[cont_columns] = (
            uncleaned_df_train[cont_columns] / uncleaned_df_train[cont_columns].max()
        )
        uncleaned_df_test[cont_columns] = (
            uncleaned_df_test[cont_columns] / uncleaned_df_test[cont_columns].max()
        )

        uncleaned_df = pd.concat([uncleaned_df_train, uncleaned_df_test])

        dummy_tables = [
            pd.get_dummies(uncleaned_df[column], prefix=column)
            for column in cat_columns
        ]
        dummy_tables.append(uncleaned_df.drop(labels=cat_columns, axis=1))
        one_hot_df = pd.concat(dummy_tables, axis=1)

        one_hot_df["labels"] = one_hot_df["class"]
        one_hot_df = one_hot_df.drop(["class"], axis=1)

        one_hot_df_train = one_hot_df[: len(uncleaned_df_train)]
        one_hot_df_test = one_hot_df[len(uncleaned_df_train) :]

        return uncleaned_df_train, uncleaned_df_test, one_hot_df_train, one_hot_df_test

    return uncleaned_df_train, uncleaned_df_test


def load_data(data_string, one_hot=False):
    """Load data given the name of the dataset

    ...

    Parameters
    ----------
        data_string : str
            The string that corresponds to the desired dataset.
            Options are {mnist, fashion, adult, census}
        one_hot : bool
            Indicates whether one-hot versions of the data should be loaded.
            The one-hot dataframes also have normalized continuous values
    """

    if data_string.lower() == "adult":
        return load_adult(one_hot)

    elif data_string.lower() == "census":
        return load_census_data(one_hot)

    else:
        print("Enter valid data_string")
