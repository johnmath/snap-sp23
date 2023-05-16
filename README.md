# SNAP: Efficient Extraction of Private Properties with Poisoning
**Authors: Harsh Chaudhari, John Abascal, Alina Oprea, Matthew Jagielski, Florian Tramèr, Jonathan Ullman.**

Code for our [SNAP: Efficient Extraction of Private Properties with Poisoning](https://arxiv.org/pdf/2208.12348.pdf) paper that will appear at IEEE S&P 2023.  

## Running the Label-Only attack
The following script computes the theoretical poisoning rate for the distinguishing test, trains target and shadow models, runs the attack, and prints the results.
```shell
python run_attacks.py -dat [--dataset] -tp [--targetproperties] -t0 [--t0frac] -t1 [--t1frac] \
                      -sm [--shadowmodels] -d [--device] -fsub [--flagsub] \
                      -subcat [--subcategories] -q [--nqueries] -nt [--ntrials]

```

Each of the arguments can be set to one of the following:

```shell
dataset (string): "adult" -- Adult dataset
                  "census" -- Census-Income (KDD) dataset. (Link provided at the end to download the dataset).

targetproperties (string): An array representation of the list of target properties. 
                           e.g. '[(race, White), (sex, Male)]'
                    
t0frac (float): value between [0, 1] for t0 fraction of target property.

t1frac (float): value between [0, 1] for t1 fraction of target property. (t0 < t1)

shadowmodels (int): Number of shadow models per fraction. Default: 4.
                     

device (string): PyTorch device
                 e.g. "mps" (for Apple Silicon), "cpu", "cuda"

flagsub (bool): If True, runs the optimized version of SNAP that poisons a subproperty of the target property.
                Make sure the original target property is large-sized (t0 > 0.1) for the optimized version.

subcategories (string): An array representation of the list of subproperties for the optimized version of SNAP.
                        e.g. '[(marital-status, Never-married)]'

nquereis (int): Number of black-box queries made to a target model. Default: 1000.

ntrials (int): The number of experimental trials to run. Default: 1.
```

An example to run **SNAP** attack on a **medium-sized** property :

```shell
python run_attack.py -tp="[(sex, Female),(occupation, Sales)]" -t0=0.01 -t1=0.035
```
An example to run the optimized version of **SNAP** attack on **large-sized** property. We recommend using the optimized version for large properties so that the theoretical poisoning rate is low.

```shell
python run_attack.py -fsub=True -tp="[(race, White),(sex, Male)]" -subcat="[(marital-status, Never-married)]" -t0=0.15 -t1=0.30
```
Link to Download Census: https://archive.ics.uci.edu/ml/datasets/Census-Income+(KDD). Download the dataset and place it in the 'dataset' folder.



