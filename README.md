# An API to use tensorflow with slurm

Originally inspired by Tensorspark

- Works for 2.7 <= version(python) < 3 , with tensorflow tested as 0.8 and 0.9
- Build your model that extends ParameterServerModel.
  - An example of small model is given in model.py
- Select your constants in constants.py
- Launch using slurm.sh and slurm.conf with number in accordance to constants.py
- Your data should be given in files in csv format where the first number is the classification