# dt8122 - Normalizing-Flows

Assignment report and code for ProbAI 2022 Summer School.

# Planar Flows
  ```bash
  python main_planar.py 
  ```

# Real-NVP
  ```bash
  python main_realnvp.py 
  ```

# Continuous Normalizing Flows (CNF)

To run the code for the CNF please use the following command. With the following arguments all asked plots and gifs are generated.
  ```bash
  python main_cnf.py —train_flag True --inference True —viz True —gif True --niters 2000 --dataset_idx 0
  ```

The gifs can also be found in the folder gifs or below:

## Two Moons
![](https://github.com/koninik/dt8122-Normalizing-Flows/blob/main/gifs/cnf-Two_Moons.gif)

## Two Blobs
![](https://github.com/koninik/dt8122-Normalizing-Flows/blob/main/gifs/cnf-Two_Blobs.gif)

## Boomerang
![](https://github.com/koninik/dt8122-Normalizing-Flows/blob/main/gifs/cnf-Boomerang.gif)
 
