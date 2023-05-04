import numpy as np
from tqdm import tqdm
import argparse
from propinf.attack.attack_utils import AttackUtil
import propinf.data.ModifiedDatasets as data
from sklearn.model_selection import train_test_split


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        '-dat',
        '--dataset',
        help='dataset name',
        type=str,
        default='adult'
    )
    
    parser.add_argument(
        '-cat',
        '--categories',
        help='list of catogories',
        type=str,
        default='(sex, Female), (occupation, Sales)'
    )
    
    parser.add_argument(
        '-t0',
        '--t0frac',
        help='t0 fraction of target property',
        type=int,
        default=0.01
    )
    
    parser.add_argument(
        '-t1',
        '--t1frac',
        help='t1 fraction of target property',
        type=int,
        default=0.035
    )
    
     parser.add_argument(
        '-listp',
        '--psnlist',
        help='list of poison percent',
        type=str,
        default= '[0, 0.005]'
    )
    
    parser.add_argument(
        '-fsub',
        '--flagsub',
        help='set to True if want to use the optimized attack for large properties',
        type=bool,
        default= False
    )
    
    parser.add_argument(
        '-subcat',
        '--subcategories',
        help='list of sub-catogories',
        type=str,
        default='(occupation, Transportation)'
    )
    
    
    
    
    