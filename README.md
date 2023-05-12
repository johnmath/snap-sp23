# SNAP: Efficient Extraction of Private Properties with Poisoning

Code for the aforementioned paper that appeared at IEEE S&P 2023.  

## Running the attack
The following script modifies the training dataset, trains target and shadow models, runs the attack, and prints the results.
```shell
python run_attacks.py -dat [--dataset] -tp [--targetproperties] -t0 [--t0frac] -t1 [--t1frac] \
                      -sm [--shadowmodels] -p [--poisonlist] -d [--device] -fsub [--flagsub] \
                      -subcat [--subcategories] -nt [--ntrials]

```

Each of the arguments can be set to one of the following:

```shell
dataset (string): "adult" -- Adult dataset
                  "census" -- Census-Income (KDD) Data Set.

targetproperties (string): An array representation of the list of target properties. 
                           e.g. '[(race, White), (sex, Male)]'
                    
t0frac (float): Fraction in [0, 1] for t0 faction of target property

t1frac (float): Fraction in [0, 1] for t1 faction of target property. (t0 < t1)

shadowmodels (int): Number of shadow models per fraction.
                     
poisonlist (string): An array representation of the list of poisoning amounts as decimals.
                     e.g. '[0.03, 0.05]'

device (string): PyTorch device
                 e.g. "mps" (for Apple Silicon), "cpu", "cuda"

flagsub (bool): If True, runs the optimized version of SNAP that poisons a subproperty of the target property

subcategories (string): An array representation of the list of subproperties for the optimized version of SNAP.
                        e.g. '[(marital-status, Never-married)]'

ntrials (int): The number of experimental trials to run
```

An example to run **SNAP** using the attack script:

```shell
python run_attack.py -fsub=False -tp="[(sex, Male)]" -p="[0.015]" -t0=0.3 -t1=0.5
```

Link to Download Census: https://archive.ics.uci.edu/ml/datasets/Census-Income+(KDD)


