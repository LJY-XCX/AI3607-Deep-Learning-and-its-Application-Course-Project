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

## Training steps
### Run the Training Script
```
python portfolio_opt_experiment.py
```
If you want to change the evaluation matrix or optimization methods, please modify the parameters of the function train_test_portfolio.

### Mathematical Theory

1. Denote the index of "now" as $t=0$. $\{p_t | t<0\}$ means the percentage change of prices of each day in history, $\{p_t | t\geq 0\}$ means the percentage change of prices of each day in future.
2. An encoder-decoder LSTM module predicts the prices in the future:
    $$\{p_t | t\geq 0\}, \mathbf{h} =\text{LSTM}(\{p_t | t<0\}),$$
    wehre $\mathbf{h}$ denotes the hidden state of LSTM.
3. Compute risk and return for the future:
    $$\mathbf{\mu} = \text{mean}(\{p_t | t\geq 0\}), \mathbf{\Sigma} = \text{cov}(\{p_t | t\geq 0\}).$$
4. In the CardNN-GS module, predict $\mathbf{s}$ (the probability of selected each asset) from $\mathbf{h}$:
    $$\mathbf{s} = \text{fully-connected}(\mathbf{h}).$$
5. Enforce the cardinality constraint by Gumbel-Sinkhorn layer introduced in Sec 3.2, whereby there are \#G\ Gumbel samples:
    $$\{\widetilde{\mathbf{T}}_{i} | i=1,2,...,\#G\} = \text{Gumbel-Sinkhorn}(\mathbf{s})$$
6. Compute the weights of each assets based on the second row of $\widetilde{\mathbf{T}}_{i}$ ($r_f$ is risk-free return, set as 3\%):
    $$\mathbf{x}_i = \mathbf{\Sigma}^{-1} (\mu - r_f), \mathbf{x}_i = \mathrm{relu}(\mathbf{x}_i \odot \widetilde{\mathbf{T}}_i[2,:]), \mathbf{x}_i = \mathbf{x}_i / \mathrm{sum}(\mathbf{x})$$
7. Based on the ground-truth prices in the future $\{p_t^{gt} | t\geq 0\}$, compute the ground truth risk and return:
    $$\mathbf{\mu}^{gt} = \text{mean}(\{p_t^{gt} | t\geq 0\}), \mathbf{\Sigma}^{gt} = \text{cov}(\{p_t^{gt} | t\geq 0\}).$$
8. Estimate the ground-truth Sharpe ratio in the future, if we invest based on $\mathbf{x}_i$:
    $$\widetilde{J}_i = \frac{(\mathbf{\mu}^{gt} - r_f)^\top \mathbf{x}_i}{\sqrt{\mathbf{x}_i^\top \mathbf{\Sigma}^{gt} \mathbf{x}_i}}.$$
9. The self-supervised loss is the average over all Gumbel samples:
    $$Loss = -\text{mean}(\widetilde{J}_1, \widetilde{J}_2, \widetilde{J}_3, ..., \widetilde{J}_{\#G})$$
