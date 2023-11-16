# Predictive Portfolio Optimization Group Work

This is the course project of AI3607 Deep Learning and its Applications. In this projects, we integrated prediction and optimization while incorporated the constraints into the network architecture and computational operators.

If you want to get more detail about our project, please refer to our [report](./DL_final_project.pdf).

## Setup

1. clone the repsitory
```
git clone https://github.com/LJY-XCX/AI3607-Deep-Learning-and-its-Application-Course-Project.git
```
2. create a virtual environment 
```
conda create -n portfolio python=3.8
conda activate portfolio
```
3. install the dependencies
```
pip install -r requirements.txt
```

## Training Steps
### Run the Training Script
```
python portfolio_opt_experiment.py
```
If you want to change the evaluation matrix or optimization methods, please modify the parameters of the function train_test_portfolio.


