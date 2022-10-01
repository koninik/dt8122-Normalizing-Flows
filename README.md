# dt8122 - Normalizing-Flows

Assignment report and code for ProbAI 2022 Summer School.

# conda environment

To create the conda environment with the necessary packages please use the following command
  ```bash
  conda env create --file nf_env.txt
  ```

# Planar Flows
  ```bash
  python main_planar.py -- dataset_idx 0 --K 32
  ```

# Real-NVP

To train the Real-NVP flow use the following command
  ```bash
  python main_realnvp.py --dataset_idx 0 --train_flag True
  ```
After traing, to execute the inference and sampling that also create the wanted plots use the following:
  ```bash
  python main_realnvp.py --dataset_idx --train_flag False
  ```

# Continuous Normalizing Flows (CNF)

To run the code for the CNF please use the following command. With the following arguments all asked plots and gifs are generated.
  ```bash
  python main_cnf.py —train_flag True --inference True —viz True —gif True --niters 2000 --dataset_idx 0
  ```

The gifs (prior to posterior) can also be found in the folder gifs or below:

## Two Moons
![](https://github.com/koninik/dt8122-Normalizing-Flows/blob/main/gifs/cnf-Two_Moons.gif)

## Two Blobs
![](https://github.com/koninik/dt8122-Normalizing-Flows/blob/main/gifs/cnf-Two_Blobs.gif)

## Boomerang
![](https://github.com/koninik/dt8122-Normalizing-Flows/blob/main/gifs/cnf-Boomerang.gif)
 
