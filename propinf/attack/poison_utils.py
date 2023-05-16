import pandas as pd
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import propinf.data.ModifiedDatasets as data
from propinf.training import training_utils, models


class PoisonUtil:
    def __init__(self, df_train, df_test, cat_columns=None, verbose=False):
        if verbose:
            message = """Before attempting to run the property inference attack, set hyperparameters using
            1. set_attack_hyperparameters()
            2. set_model_hyperparameters()"""
            print(message)
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
        verbose=False,
        mini_verbose=False,
        early_stopping=True,
        dropout=False,
        shuffle=True,
        using_ce_loss=True,
        batch_size=256,
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

    def compute_poison_rate(
        self,
        num_shadow_models=1,
        sub_ratio = 1
    ):
        """Property inference attack for categorical data. (e.g. Census, Adults)

        ...
        Parameters
        ----------
            num_shadow_models : int
                The number of shadow models per "world" to use in the attack
        ...
        Returns
        ----------
            pois_th: Poison Rate to use for Label-flipping attack to work.
        """

        # Train multiple shadow models to reduce variance
        if self._mini_verbose:
            print("-" * 10, "\nTraining Shadow Models...")

        D0_loaders = {}

        if self._allow_subsampling == False:

            self._nsub_samples = self._D0_OH.shape[0]

            # print("Size of Shadow model dataset: ", self._nsub_samples)

            if len(self._Dp) == 0:
                poisoned_D0 = self._D0_OH.sample(n=self._nsub_samples, random_state=21)

            else:
                clean_D0 = self._D0_OH.sample(
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

            D0_loaders["train"] = training_utils.dataframe_to_dataloader(
                poisoned_D0,
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

        for i in tqdm(range(num_shadow_models), desc=f"Training {num_shadow_models} Models to compute optimal Poisoning Rate"):

            if self._mini_verbose:
                print("-" * 10, f"\nModels {i+1}")

            if len(self._layer_sizes) != 0:
                M0_model = models.NeuralNet(
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

                D0_loaders["train"] = training_utils.dataframe_to_dataloader(
                    poisoned_D0,
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

            out_M0_temp = training_utils.get_logits_torch(
                Dtest_OH_loader,
                M0_trained,
                device=self._device,
                middle_measure=self._middle,
                variance_adjustment=self._variance_adjustment,
                label = self._poison_class
            )

            out_M0 = np.append(out_M0, out_M0_temp)

        
        #Computing theoretical poisoning rate. Find the poisoning rate that makes the mean of the distribution for t0 fraction > 0.
        mu_hat = np.mean(out_M0)
        var_hat = np.var(out_M0)
        pi_v = 1.0
        th_mean = 0.4
        t = self._t0
        pois_th = -1.0
        alpha = np.exp(mu_hat+var_hat/2)
        beta = np.sqrt(np.exp(var_hat)-1)
        gamma = np.exp(th_mean)
        coeffs = [(1+alpha)**2, 0, -(gamma**2)*((1+alpha)**2 + (alpha*beta)**2),-2*(alpha*beta*gamma)**2, -(alpha*beta*gamma)**2]
        Mvals = np.roots(coeffs)
        valid_Mvals = Mvals.real[abs(Mvals.imag)<1e-5]
        for val in valid_Mvals:
            # p_th = (t*pi_v*(val-alpha))/(1+alpha+pi_v*t*(val-alpha))
            p_th = (sub_ratio*t*pi_v*(val-alpha))/(1+alpha+pi_v*sub_ratio*t*(val-alpha))
            if(p_th>0):
                pois_th = p_th
                
        if(pois_th < 0):
            print("Poisoning Rate is set to negative, check input parameters.")
            
        else:
            print("Theoretical Poisoning Rate: {:0.3f}".format(pois_th))
        
        return (
            pois_th
        )