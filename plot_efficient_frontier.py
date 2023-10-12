import matplotlib
import math

matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = 12
from matplotlib.patches import Ellipse

from portfolio_opt_methods import *
from portfolio_opt_data import PortDataset


######################################
#           Hyperparameters          #
######################################

START_EPOCH = 0
NUM_EPOCHS = 10000
HISTORY_LEN = 120
FUTURE_LEN = 120
NUM_FEATURE = 32
DEVICE = torch.device('cuda:0')
LR = 1e-3
LR_STEPS = [1500]
RF = 0.03 # risk-free return to compute the Sharpe ratio
K = 20 # the cardinality (topK) constraint


def efficient_frontier(mu, cov, num_samples=50):
    min_return = torch.mean(mu) #torch.tensor(0.25, dtype=mu.dtype, device=mu.device) #
    max_return = torch.max(mu)
    step_size = (max_return - min_return) / (num_samples - 1)
    expected_return = min_return
    return_list = []
    risk_list = []
    while expected_return <= max_return:
        print(f'{expected_return:.2f}/{max_return:.2f}')
        _, opt_weight, __ = gurobi_portfolio_opt(mu, cov, expected_return, obj='MinRisk', timeout_sec=120)
        risk = torch.sqrt(torch.chain_matmul(opt_weight.unsqueeze(0), cov, opt_weight.unsqueeze(1)).squeeze()).item()
        return_list.append(expected_return.item())
        risk_list.append(risk)
        expected_return += step_size
    return return_list, risk_list


def plot_portfolio(model_jpo, model_pto, test_set, verbose=True):
    with torch.set_grad_enabled(False):
        for iter_idx, test_data in enumerate(test_set):
            if iter_idx % 10 != 7:
                continue

            history = torch.tensor(test_data['history'].values).to(device=DEVICE, dtype=torch.double)
            future = torch.tensor(test_data['future'].values).to(device=DEVICE, dtype=torch.double)
            mu, cov = compute_measures(future)

            #x_lims = torch.sqrt(torch.min(torch.diag(cov))).item(), torch.sqrt(torch.max(torch.diag(cov))).item()
            #y_lims = torch.min(mu).item(), torch.max(mu).item()
            #x_lims = (0., 0.5)
            #y_lims = (-0.4, 0.8)
            #x_len = x_lims[1] - x_lims[0]
            #y_len = y_lims[1] - y_lims[0]

            plt.figure()
            #f, (ax0, ax1, ax2) = plt.subplots(3, 1, gridspec_kw={'height_ratios': [4, 1, 1]})
            ax = plt.gca()
            ax.set_xlabel('Risk')
            #ax.set_xlim(x_lims)
            ax.set_ylabel('Return')
            #ax.set_ylim(y_lims)

            # efficient frontier
            ef_return, ef_risk = efficient_frontier(mu, cov)
            plt.plot(ef_risk, ef_return, 'k-', label='efficient frontier')

            def plot_a_portfolio(mu, cov, weight, color, method_name):
                risk = torch.sqrt(torch.chain_matmul(weight.unsqueeze(0), cov, weight.unsqueeze(1)).squeeze()).item()
                return_v = torch.sum(mu * weight).item()
                sharpe = compute_sharpe_ratio(mu, cov, weight, RF)

                for rank, idx in enumerate(torch.topk(weight, K).indices):
                    asset_return = mu[idx].item()
                    asset_risk = torch.sqrt(cov[idx, idx]).item()
                    asset_weight = weight[idx].item()
                    if rank == 0:
                        ax.plot(asset_risk, asset_return, 'o', ms=15 * math.sqrt(asset_weight), color=color, alpha=0.5,
                                 label=f'{method_name} assets')
                    else:
                        ax.plot(asset_risk, asset_return, 'o', ms=15 * math.sqrt(asset_weight), color=color, alpha=0.5)
                ax.plot(risk, return_v, '*', ms=15, color=color, label=f'{method_name} portfolio')

                #ax1 = plt.gca()
                #x0, y0 = ax1.transAxes.transform((0, 0))  # lower left in pixels
                #x1, y1 = ax1.transAxes.transform((1, 1))  # upper right in pixes
                #dx = x1 - x0
                #dy = y1 - y0
                #plt.text(risk + 0.03 * dx, return_v - 0.03 * dy, f'Sharpe ratio={sharpe:.3f}')

            # joint predict and optimize
            _, jpo_weight = model_jpo(history, FUTURE_LEN, 'jpo', RF, K, gumbel_sample_num=1000, gumbel_noise_fact=0.1, return_best_weight=True)
            plot_a_portfolio(mu, cov, jpo_weight, 'blue', 'CardNN')

            # history
            _, hist_weight = model_jpo(history, FUTURE_LEN, 'history', RF, K)
            plot_a_portfolio(mu, cov, hist_weight, 'purple', 'history-opt')

            # predict-then-optimize
            _, pto_weight = model_pto(history, FUTURE_LEN, 'pto', RF, K)
            plot_a_portfolio(mu, cov, pto_weight, 'orange', 'pred-then-opt')

            plt.legend(loc='lower right')

            #plt.subplot(3, 1, 2, gridspec_kw={'height_ratios': [4, 1, 1]})
            #topkargs = torch.argsort(jpo_weight, descending=True)[:K]
            #topklabels = test_data['future'].columns[topkargs]
            #ax1.stem(topkargs, label=topklabels)

            #plt.subplot(3, 1, 3, gridspec_kw={'height_ratios': [4, 1, 1]})
            #topkargs = torch.argsort(pto_weight, descending=True)[:K]
            #topklabels = test_data['future'].columns[topkargs]
            #ax2.stem(topkargs, label=topklabels)

            plt.savefig(f'portfolio-{test_data["real_date"]}.pdf')
            print(f'Plot id={iter_idx} complete.')

        print('Plot complete.')


if __name__ == '__main__':
    dataset = PortDataset('snp500', HISTORY_LEN, FUTURE_LEN, train_test_split=0.75)
    model_jpo = LSTMModel(dataset.num_assets, NUM_FEATURE, 1).to(device=DEVICE)
    model_jpo.double()
    model_pto = LSTMModel(dataset.num_assets, NUM_FEATURE, 1).to(device=DEVICE)
    model_pto.double()

    JPO_WEIGHT = 'output/portfolio_lstm_epoch55.pt.jpo'
    PTO_WEIGHT = 'output/portfolio_lstm_epoch1500.pt.pto'
    print(f'Loading model weights from {JPO_WEIGHT}...')
    model_jpo.load_state_dict(torch.load(JPO_WEIGHT))
    print(f'Loading model weights from {PTO_WEIGHT}...')
    model_pto.load_state_dict(torch.load(PTO_WEIGHT))

    plot_portfolio(model_jpo, model_pto, dataset.test_set)
