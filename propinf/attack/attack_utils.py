import pandas as pd
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import propinf.data.ModifiedDatasets as data
from propinf.training import training_utils, models


class AttackUtil:
    def __init__(self, target_model_layers, df_train, df_test, cat_columns=None, verbose=True):
        if verbose:
            message = """Before attempting to run the property inference attack, set hyperparameters using
            1. set_attack_hyperparameters()
            2. set_model_hyperparameters()"""
            print(message)
        self.target_model_layers = target_model_layers
        self.df_train = df_train
        self.df_test = df_test
        self.cat_columns = cat_columns

        # Attack Hyperparams
        self._categories = None
        self._target_attributes = None
        self._sub_categories = None
        self._sub_attributes = None
        self._poison_class = None
        self._poison_percent = None
        self._k = None
        self._t0 = None
        self._t1 = None
        self._middle = None
        self._variance_adjustment = None
        self._num_queries = None
        self._nsub_samples = None
        self._ntarget_samples = None
        self._subproperty_sampling = False
        self._allow_subsampling = False
        self._allow_target_subsampling = False
        self._restrict_sampling = False
        self._pois_rs = None
        self._model_metrics = None

        # Model + Data Hyperparams
        self._layer_sizes = None
        self._num_classes = None
        self._epochs = None
        self._optim_init = None
        self._optim_kwargs = None
        self._criterion = None
        self._device = None
        self._tol = None
        self._verbose = None
        self._early_stopping = None
        self._dropout = None
        self._shuffle = None
        self._using_ce_loss = None
        self._batch_size = None
        self._num_workers = None
        self._persistent_workers = None

    def set_attack_hyperparameters(
        self,
        categories=["race"],
        target_attributes=[" Black"],
        sub_categories=["occupation"],
        sub_attributes=[" Sales"],
        subproperty_sampling=False,
        restrict_sampling=False,
        poison_class=1,
        poison_percent=0.03,
        k=None,
        t0=0.1,
        t1=0.25,
        middle="median",
        variance_adjustment=1,
        nsub_samples=1000,
        allow_subsampling=False,
        ntarget_samples=1000,
        num_target_models=25,
        allow_target_subsampling=False,
        pois_random_seed=21,
        num_queries=1000,
    ):

        self._categories = categories
        self._target_attributes = target_attributes
        self._sub_categories = sub_categories
        self._sub_attributes = sub_attributes
        self._subproperty_sampling = subproperty_sampling
        self._restrict_sampling = restrict_sampling
        self._poison_class = poison_class
        self._poison_percent = poison_percent
        self._k = k
        self._t0 = t0
        self._t1 = t1
        self._middle = middle
        self._variance_adjustment = variance_adjustment
        self._num_queries = num_queries
        self._nsub_samples = nsub_samples
        self._allow_subsampling = allow_subsampling
        self._num_target_models = num_target_models
        self._ntarget_samples = ntarget_samples
        self._allow_target_subsampling = allow_target_subsampling
        self._pois_rs = pois_random_seed

    def set_shadow_model_hyperparameters(
        self,
        layer_sizes=[64],
        num_classes=2,
        epochs=10,
        optim_init=optim.Adam,
        optim_kwargs={"lr": 0.03, "weight_decay": 0.0001},
        criterion=nn.CrossEntropyLoss(),
        device="cpu",
        tol=10e-7,
        verbose=True,
        mini_verbose=True,
        early_stopping=True,
        dropout=False,
        shuffle=True,
        using_ce_loss=True,
        batch_size=1024,
        num_workers=8,
        persistent_workers=True,
    ):

        self._layer_sizes = layer_sizes
        self._num_classes = num_classes
        self._epochs = epochs
        self._optim_init = optim_init
        self._optim_kwargs = optim_kwargs
        self._criterion = criterion
        self._device = device
        self._tol = tol
        self._verbose = verbose
        self._mini_verbose = mini_verbose
        self._early_stopping = early_stopping
        self._dropout = dropout
        self._shuffle = shuffle
        self._using_ce_loss = using_ce_loss
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._persistent_workers = persistent_workers
        self._activation_val = {}

    def generate_datasets(self):
        """Generate all datasets required for the property inference attack"""

        self._target_worlds = np.append(
            np.zeros(self._num_target_models), np.ones(self._num_target_models)
        )

        # print(self._target_worlds)

        # Generate Datasets
        if self._mini_verbose:
            print("Generating all datasets...")

        (
            self._D0_mo,
            self._D1_mo,
            self._D0,
            self._D1,
            self._Dp,
            self._Dtest,
        ) = data.generate_all_datasets(
            self.df_train,
            self.df_test,
            t0=self._t0,
            t1=self._t1,
            categories=self._categories,
            target_attributes=self._target_attributes,
            sub_categories=self._sub_categories,
            sub_attributes=self._sub_attributes,
            poison_class=self._poison_class,
            poison_percent=self._poison_percent,
            subproperty_sampling=self._subproperty_sampling,
            restrict_sampling=self._restrict_sampling,
            verbose=self._verbose,
        )

        if self._t0 == 0:
            self._Dtest = pd.concat([self._Dp, self._Dtest])

        #Changes-Harsh
        # Converting to poisoned class
        # self._Dtest["class"] = self._poison_class

        if self._k is None:
            self._k = int(self._poison_percent * len(self._D0_mo))
        else:
            self._poison_percent = self._k / len(self._D0_mo)

        if len(self._Dp) == 0:
            (
                _,
                self._D0_mo_OH,
                self._D1_mo_OH,
                self._D0_OH,
                self._D1_OH,
                self._Dtest_OH,
                self._test_set,
            ) = data.all_dfs_to_one_hot(
                [
                    self.df_train,
                    self._D0_mo,
                    self._D1_mo,
                    self._D0,
                    self._D1,
                    self._Dtest,
                    self.df_test,
                ],
                cat_columns=self.cat_columns,
                class_label="class",
            )

        else:
            (
                _,
                self._D0_mo_OH,
                self._D1_mo_OH,
                self._D0_OH,
                self._D1_OH,
                self._Dp_OH,
                self._Dtest_OH,
                self._test_set,
            ) = data.all_dfs_to_one_hot(
                [
                    self.df_train,
                    self._D0_mo,
                    self._D1_mo,
                    self._D0,
                    self._D1,
                    self._Dp,
                    self._Dtest,
                    self.df_test,
                ],
                cat_columns=self.cat_columns,
                class_label="class",
            )

    def train_and_poison_target(self, need_metrics=False, df_cv=None):
        """Train target model with poisoned set if poisoning > 0"""

        owner_loaders = {}
        self._poisoned_target_models = [None] * self._num_target_models * 2
        input_dim = len(self._D0_mo_OH.columns) - 1

        if self._allow_target_subsampling == False:

            self._ntarget_samples = self._D0_mo_OH.shape[0]

            if len(self._Dp) == 0:
                poisoned_D0_MO = self._D0_mo_OH.sample(
                    n=self._ntarget_samples, random_state=21
                )

            else:
                # Changes
                clean_D0_MO = self._D0_mo_OH.sample(
                    n=int((1 - self._poison_percent) * self._ntarget_samples),
                    random_state=21,
                )

                if (
                    int(self._poison_percent * self._ntarget_samples) <= self._Dp_OH.shape[0]
                ):

                    poisoned_D0_MO = pd.concat(
                        [
                            clean_D0_MO,
                            self._Dp_OH.sample(
                                n=int(self._poison_percent * self._ntarget_samples),
                                random_state=self._pois_rs,
                                replace=False,
                            ),
                        ]
                    )
                else:

                    poisoned_D0_MO = pd.concat(
                        [
                            clean_D0_MO,
                            self._Dp_OH.sample(
                                n=int(self._poison_percent * self._ntarget_samples),
                                random_state=self._pois_rs,
                                replace=True,
                            ),
                        ]
                    )

        """Trains half the target models on t0 fraction"""
        # for i in tqdm(range(self._num_target_models), desc = "Training target models with frac t0"):
        for i in tqdm(range(self._num_target_models), desc=f"Training Target Models with {self._poison_percent*100:.2f}% poisoning"):

            if self._allow_target_subsampling == True:
                if len(self._Dp) == 0:
                    poisoned_D0_MO = self._D0_mo_OH.sample(
                        n=self._ntarget_samples, random_state=i + 1
                    )

                else:

                    poisoned_D0_MO = pd.concat(
                        [
                            self._D0_mo_OH.sample(
                                n=int(
                                    (1 - self._poison_percent) * self._ntarget_samples
                                ),
                                random_state=i + 1,
                            ),
                            self._Dp_OH.sample(
                                n=int(self._poison_percent * self._ntarget_samples),
                                random_state=self._pois_rs,
                            ),
                        ]
                    )

            owner_loaders["train"] = training_utils.dataframe_to_dataloader(
                poisoned_D0_MO,
                batch_size=self._batch_size,
                using_ce_loss=self._using_ce_loss,
                num_workers=self._num_workers,
                persistent_workers=self._persistent_workers,
            )

            if len(self.target_model_layers) != 0:
                target_model = models.NeuralNet(
                    input_dim=input_dim,
                    layer_sizes=self.target_model_layers,
                    num_classes=self._num_classes,
                    dropout=self._dropout,
                )

            else:
                target_model = models.LogisticRegression(
                    input_dim=input_dim,
                    num_classes=self._num_classes,
                    using_ce_loss=self._using_ce_loss,
                )

            if self._mini_verbose:
                print("-" * 10, "\nTarget Model")

            self._poisoned_target_models[i], _ = training_utils.fit(
                dataloaders=owner_loaders,
                model=target_model,
                # alterdata_list=self._alterdata_list,
                epochs=self._epochs,
                optim_init=self._optim_init,
                optim_kwargs=self._optim_kwargs,
                criterion=self._criterion,
                device=self._device,
                verbose=self._verbose,
                mini_verbose=self._mini_verbose,
                early_stopping=self._early_stopping,
                tol=self._tol,
                train_only=True,
            )

        if self._allow_target_subsampling == False:
            self._ntarget_samples = self._D0_mo_OH.shape[0]
            if len(self._Dp) == 0:
                # poisoned_D1_MO = self._D1_mo_OH.copy()
                poisoned_D1_MO = self._D1_mo_OH.sample(
                    n=self._ntarget_samples, random_state=21
                )

            else:
                clean_D1_MO = self._D1_mo_OH.sample(
                    n=int((1 - self._poison_percent) * self._ntarget_samples),
                    random_state=21,
                )
                # Changes
                if (
                    int(self._poison_percent * self._ntarget_samples)
                    <= self._Dp_OH.shape[0]
                ):
                    poisoned_D1_MO = pd.concat(
                        [
                            clean_D1_MO,
                            self._Dp_OH.sample(
                                n=int(self._poison_percent * self._ntarget_samples),
                                random_state=self._pois_rs,
                                replace=False,
                            ),
                        ]
                    )
                else:
                    poisoned_D1_MO = pd.concat(
                        [
                            clean_D1_MO,
                            self._Dp_OH.sample(
                                n=int(self._poison_percent * self._ntarget_samples),
                                random_state=self._pois_rs,
                                replace=True,
                            ),
                        ]
                    )

        """Trains half the target models on t1 fraction"""
        for i in range(self._num_target_models, 2 * self._num_target_models):
            # for i in tqdm(range(self._num_target_models, 2*self._num_target_models), desc = "Training target models with frac t1"):

            if self._allow_target_subsampling == True:
                if len(self._Dp) == 0:
                    poisoned_D1_MO = self._D1_mo_OH.sample(
                        n=self._ntarget_samples, random_state=i + 1
                    )

                else:
                    poisoned_D1_MO = pd.concat(
                        [
                            self._D1_mo_OH.sample(
                                n=int(
                                    (1 - self._poison_percent) * self._ntarget_samples
                                ),
                                random_state=i + 1,
                            ),
                            self._Dp_OH.sample(
                                n=int(self._poison_percent * self._ntarget_samples),
                                random_state=self._pois_rs,
                            ),
                        ]
                    )

            owner_loaders["train"] = training_utils.dataframe_to_dataloader(
                poisoned_D1_MO,
                batch_size=self._batch_size,
                using_ce_loss=self._using_ce_loss,
                num_workers=self._num_workers,
                persistent_workers=self._persistent_workers,
            )

            if len(self.target_model_layers) != 0:
                target_model = models.NeuralNet(
                    input_dim=input_dim,
                    layer_sizes=self.target_model_layers,
                    num_classes=self._num_classes,
                    dropout=self._dropout,
                )

            else:
                target_model = models.LogisticRegression(
                    input_dim=input_dim,
                    num_classes=self._num_classes,
                    using_ce_loss=self._using_ce_loss,
                )

            if self._mini_verbose:
                print("-" * 10, "\nTarget Model")

            self._poisoned_target_models[i], _ = training_utils.fit(
                dataloaders=owner_loaders,
                model=target_model,
                # alterdata_list=self._alterdata_list,
                epochs=self._epochs,
                optim_init=self._optim_init,
                optim_kwargs=self._optim_kwargs,
                criterion=self._criterion,
                device=self._device,
                verbose=self._verbose,
                mini_verbose=self._mini_verbose,
                early_stopping=self._early_stopping,
                tol=self._tol,
                train_only=True,
            )


    def property_inference_categorical(
        self,
        num_shadow_models=1,
        query_trials=1,
        query_selection="random",
        distinguishing_test="median",
    ):
        """Property inference attack for categorical data. (e.g. Census, Adults)

        ...
        Parameters
        ----------
            num_shadow_models : int
                The number of shadow models per "world" to use in the attack
            query_trials : int
                Number of times we want to query num_queries points on the target
            distinguishing_test : str
                The distinguishing test to use on logit distributions
                Options are the following:

                "median" : uses the middle of medians as a threshold and checks on which side of the threshold
                           the majority of target model prediction confidences are on
                "divergence" : uses KL divergence to measure the similarity between the target model
                               prediction scores and the

        ...
        Returns
        ----------
            out_M0 : np.array
                Array of scaled logit values for M0
            out_M1 : np.array
                Array of scaled logit values for M1
            logits_each_trial : list of np.arrays
                Arrays of scaled logit values for target model.
                Each index is the output of a single query to the
                target model
            predictions : list
                Distinguishing test predictions; 0 if prediction
                is t0, 1 if prediction is t1
            correct_trials : list
                List of booleans dentoting whether query trial i had
                a correct prediction
        """

        # Train multiple shadow models to reduce variance
        if self._mini_verbose:
            print("-" * 10, "\nTraining Shadow Models...")

        D0_loaders = {}
        D1_loaders = {}

        if self._allow_subsampling == False:

            self._nsub_samples = self._D0_OH.shape[0]

            # print("Size of Shadow model dataset: ", self._nsub_samples)

            if len(self._Dp) == 0:
                poisoned_D0 = self._D0_OH.sample(n=self._nsub_samples, random_state=21)
                poisoned_D1 = self._D1_OH.sample(n=self._nsub_samples, random_state=21)

            else:
                clean_D0 = self._D0_OH.sample(
                    n=int((1 - self._poison_percent) * self._nsub_samples),
                    random_state=21,
                )
                clean_D1 = self._D1_OH.sample(
                    n=int((1 - self._poison_percent) * self._nsub_samples),
                    random_state=21,
                )

                if (
                    int(self._poison_percent * self._nsub_samples)
                    <= self._Dp_OH.shape[0]
                ):
                    # Changes
                    poisoned_D0 = pd.concat(
                        [
                            clean_D0,
                            self._Dp_OH.sample(
                                n=int(self._poison_percent * self._nsub_samples),
                                random_state=self._pois_rs,
                                replace=False,
                            ),
                        ]
                    )
                    poisoned_D1 = pd.concat(
                        [
                            clean_D1,
                            self._Dp_OH.sample(
                                n=int(self._poison_percent * self._nsub_samples),
                                random_state=self._pois_rs,
                                replace=False,
                            ),
                        ]
                    )
                else:
                    poisoned_D0 = pd.concat(
                        [
                            clean_D0,
                            self._Dp_OH.sample(
                                n=int(self._poison_percent * self._nsub_samples),
                                random_state=self._pois_rs,
                                replace=True,
                            ),
                        ]
                    )
                    poisoned_D1 = pd.concat(
                        [
                            clean_D1,
                            self._Dp_OH.sample(
                                n=int(self._poison_percent * self._nsub_samples),
                                random_state=self._pois_rs,
                                replace=True,
                            ),
                        ]
                    )

            D0_loaders["train"] = training_utils.dataframe_to_dataloader(
                poisoned_D0,
                batch_size=self._batch_size,
                using_ce_loss=self._using_ce_loss,
                num_workers=self._num_workers,
                persistent_workers=self._persistent_workers,
            )

            D1_loaders["train"] = training_utils.dataframe_to_dataloader(
                poisoned_D1,
                batch_size=self._batch_size,
                using_ce_loss=self._using_ce_loss,
                num_workers=self._num_workers,
                persistent_workers=self._persistent_workers,
            )

        Dtest_OH_loader = training_utils.dataframe_to_dataloader(
            self._Dtest_OH,
            batch_size=self._batch_size,
            num_workers=self._num_workers,
            using_ce_loss=self._using_ce_loss,
        )

        input_dim = len(self._D0_mo_OH.columns) - 1

        out_M0 = np.array([])
        out_M1 = np.array([])

        for i in tqdm(range(num_shadow_models), desc=f"Training {num_shadow_models} Shadow Models with {self._poison_percent*100:.2f}% Poisoning"):

            if self._mini_verbose:
                print("-" * 10, f"\nModels {i+1}")

            if len(self._layer_sizes) != 0:
                M0_model = models.NeuralNet(
                    input_dim=input_dim,
                    layer_sizes=self._layer_sizes,
                    num_classes=self._num_classes,
                    dropout=self._dropout,
                )

                M1_model = models.NeuralNet(
                    input_dim=input_dim,
                    layer_sizes=self._layer_sizes,
                    num_classes=self._num_classes,
                    dropout=self._dropout,
                )
            else:
                M0_model = models.LogisticRegression(
                    input_dim=input_dim,
                    num_classes=self._num_classes,
                    using_ce_loss=self._using_ce_loss,
                )

                M1_model = models.LogisticRegression(
                    input_dim=input_dim,
                    num_classes=self._num_classes,
                    using_ce_loss=self._using_ce_loss,
                )

            if self._allow_subsampling == True:

                if len(self._Dp) == 0:
                    poisoned_D0 = self._D0_OH.sample(n=self._nsub_samples)
                    poisoned_D1 = self._D1_OH.sample(n=self._nsub_samples)

                else:

                    if self._allow_target_subsampling == True:

                        poisoned_D0 = pd.concat(
                            [
                                self._D0_OH.sample(
                                    n=int(
                                        (1 - self._poison_percent) * self._nsub_samples
                                    )
                                ),
                                self._Dp_OH.sample(
                                    n=int(self._poison_percent * self._nsub_samples),
                                    random_state=self._pois_rs,
                                ),
                            ]
                        )

                        poisoned_D1 = pd.concat(
                            [
                                self._D1_OH.sample(
                                    n=int(
                                        (1 - self._poison_percent) * self._nsub_samples
                                    )
                                ),
                                self._Dp_OH.sample(
                                    n=int(self._poison_percent * self._nsub_samples),
                                    random_state=self._pois_rs,
                                ),
                            ]
                        )
                    else:
                        poisoned_D0 = pd.concat(
                            [
                                self._D0_OH.sample(
                                    n=int(
                                        (1 - self._poison_percent) * self._nsub_samples
                                    )
                                ),
                                self._Dp_OH.sample(
                                    n=int(self._poison_percent * self._nsub_samples)
                                ),
                            ]
                        )

                        poisoned_D1 = pd.concat(
                            [
                                self._D1_OH.sample(
                                    n=int(
                                        (1 - self._poison_percent) * self._nsub_samples
                                    )
                                ),
                                self._Dp_OH.sample(
                                    n=int(self._poison_percent * self._nsub_samples)
                                ),
                            ]
                        )

                D0_loaders["train"] = training_utils.dataframe_to_dataloader(
                    poisoned_D0,
                    batch_size=self._batch_size,
                    using_ce_loss=self._using_ce_loss,
                    num_workers=self._num_workers,
                    persistent_workers=self._persistent_workers,
                )

                D1_loaders["train"] = training_utils.dataframe_to_dataloader(
                    poisoned_D1,
                    batch_size=self._batch_size,
                    using_ce_loss=self._using_ce_loss,
                    num_workers=self._num_workers,
                    persistent_workers=self._persistent_workers,
                )

            M0_trained, _ = training_utils.fit(
                dataloaders=D0_loaders,
                model=M0_model,
                epochs=self._epochs,
                optim_init=self._optim_init,
                optim_kwargs=self._optim_kwargs,
                criterion=self._criterion,
                device=self._device,
                verbose=self._verbose,
                mini_verbose=self._mini_verbose,
                early_stopping=self._early_stopping,
                tol=self._tol,
                train_only=True,
            )

            M1_trained, _ = training_utils.fit(
                dataloaders=D1_loaders,
                model=M1_model,
                epochs=self._epochs,
                optim_init=self._optim_init,
                optim_kwargs=self._optim_kwargs,
                criterion=self._criterion,
                device=self._device,
                verbose=self._verbose,
                mini_verbose=self._mini_verbose,
                early_stopping=self._early_stopping,
                tol=self._tol,
                train_only=True,
            )

            out_M0_temp = training_utils.get_logits_torch(
                Dtest_OH_loader,
                M0_trained,
                device=self._device,
                middle_measure=self._middle,
                variance_adjustment=self._variance_adjustment,
                label = self._poison_class
            )

            out_M0 = np.append(out_M0, out_M0_temp)

            out_M1_temp = training_utils.get_logits_torch(
                Dtest_OH_loader,
                M1_trained,
                device=self._device,
                middle_measure=self._middle,
                variance_adjustment=self._variance_adjustment,
                label = self._poison_class
            )
            out_M1 = np.append(out_M1, out_M1_temp)

            if self._verbose:
                print(
                    f"M0 Mean: {out_M0.mean():.5}, Variance: {out_M0.var():.5}, StDev: {out_M0.std():.5}, Median: {np.median(out_M0):.5}"
                )
                print(
                    f"M1 Mean: {out_M1.mean():.5}, Variance: {out_M1.var():.5}, StDev: {out_M1.std():.5}, Median: {np.median(out_M1):.5}"
                )

        if distinguishing_test == "median":
            midpoint_of_medians = (np.median(out_M0) + np.median(out_M1)) / 2
            thresh = midpoint_of_medians

        if self._verbose:
            print(f"Threshold: {thresh:.5}")

        # Query the target model and determine
        correct_trials = 0

        if self._mini_verbose:
            print(
                "-" * 10,
                f"\nQuerying target model {query_trials} times with {self._num_queries} query samples",
            )

        oversample_flag = False
        if self._num_queries > self._Dtest_OH.shape[0]:
            oversample_flag = True
            print("Oversampling test queries")

        for i, poisoned_target_model in enumerate(tqdm(self._poisoned_target_models, desc=f"Querying Models and Running Distinguishing Test")):
            for query_trial in range(query_trials):

                if query_selection.lower() == "random":
                    Dtest_OH_sample_loader = training_utils.dataframe_to_dataloader(
                        self._Dtest_OH.sample(
                            n=self._num_queries, replace=oversample_flag, random_state = i+1
                        ),
                        batch_size=self._batch_size,
                        num_workers=self._num_workers,
                        using_ce_loss=self._using_ce_loss,
                    )
                else:
                    print("Incorrect Query selection type")

                out_target = training_utils.get_logits_torch(
                    Dtest_OH_sample_loader,
                    poisoned_target_model,
                    device=self._device,
                    middle_measure=self._middle,
                    variance_adjustment=self._variance_adjustment,
                    label = self._poison_class
                )

                if self._verbose:
                    print("-" * 10)
                    print(
                        f"Target Mean: {out_target.mean():.5}, Variance: {out_target.var():.5}, StDev: {out_target.std():.5}, Median: {np.median(out_target):.5}\n"
                    )

                """ Perform distinguishing test"""
                if distinguishing_test == "median":
                    M0_score = len(out_target[out_target > thresh])
                    M1_score = len(out_target[out_target < thresh])
                    if self._verbose:
                        print(f"M0 Score: {M0_score}\nM1 Score: {M1_score}")

                    if M0_score >= M1_score:
                        if self._mini_verbose:
                            print(
                                f"Target is in t0 world with {M0_score/len(out_target)*100:.4}% confidence"
                            )

                        correct_trials = correct_trials + int(
                            self._target_worlds[i] == 0
                        )

                    elif M0_score < M1_score:
                        if self._mini_verbose:
                            print(
                                f"Target is in t1 world {M1_score/len(out_target)*100:.4}% confidence"
                            )

                        correct_trials = correct_trials + int(
                            self._target_worlds[i] == 1
                        )

        if distinguishing_test == "median":
            return (
                out_M0,
                out_M1,
                thresh,
                correct_trials / (len(self._target_worlds) * query_trials),
            )