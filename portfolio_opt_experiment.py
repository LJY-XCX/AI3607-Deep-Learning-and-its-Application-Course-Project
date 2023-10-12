import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from torch.utils.tensorboard import SummaryWriter
from portfolio_opt_methods import *
from portfolio_opt_data import PortDataset


######################################
#           Hyperparameters          #
######################################

START_EPOCH = 0
NUM_EPOCHS = 100
HISTORY_LEN = 120
FUTURE_LEN = 120
NUM_FEATURE = 32
DEVICE = torch.device('cuda:0')
LR = 1e-3
LR_STEPS = [1500]
RF = 0.03 # risk-free return to compute the Sharpe ratio
K = 20 # the cardinality (topK) constraint


######################################
#              training              #
######################################

def train_test_portfolio(model, train_set, mse_weight=1, sharpe_weight=1, opt_method='jpo', test_set=None, test_items='all'):
    assert mse_weight > 0 or sharpe_weight > 0

    working_lr_steps = []
    for step in LR_STEPS:
        if step - START_EPOCH < 0:
            continue
        working_lr_steps.append(step - START_EPOCH)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, working_lr_steps)
    gradient_list = []

    writer = SummaryWriter()
    for epoch in range(START_EPOCH, NUM_EPOCHS):
        train_num = len(train_set)
        gradient_one_epoch = 0
        epoch_mse = 0
        epoch_sharpe = 0
        for iter_idx, train_data in enumerate(np.random.permutation(train_set)):
            # prepare data
            history = torch.tensor(train_data['history'].values, requires_grad=True).to(device=DEVICE, dtype=torch.double)
            future = torch.tensor(train_data['future'].values).to(device=DEVICE, dtype=torch.double)

            # forward pass
            if sharpe_weight > 0:
                pred_seq, weight = model(history, FUTURE_LEN, opt_method, RF, K,
                                         gumbel_sample_num=10 if opt_method == 'jpo-old' else 1000)
            else:
                pred_seq = model(history, FUTURE_LEN)

            # compute loss
            loss = 0
            mse = torch.sum((pred_seq - future) ** 2) / FUTURE_LEN
            writer.add_scalar('train_mse', mse.detach(), iter_idx + epoch * train_num)
            epoch_mse += mse.detach() / train_num
            if mse_weight > 0:
                loss += mse_weight * mse
            if sharpe_weight > 0:
                mu, cov = compute_measures(future)
                sharpe = compute_sharpe_ratio(mu, cov, weight, RF)
                writer.add_scalars(
                    'train_sharpe',
                    {'mean': sharpe.mean().detach(), 'min': sharpe.min().detach(), 'max': sharpe.max().detach()},
                    iter_idx + epoch * train_num
                )
                loss += - sharpe_weight * sharpe.mean()
                epoch_sharpe += sharpe.mean().detach() / train_num

            history.retain_grad()
            loss.backward()
            gradient_one_epoch += history.grad.sum()
            optimizer.step()
            optimizer.zero_grad()
        lr_scheduler.step()
        writer.add_scalar('epoch_mse', epoch_mse, (epoch + 1) * train_num)
        if sharpe_weight > 0:
            writer.add_scalar('epoch_sharpe', epoch_sharpe, (epoch + 1) * train_num)
        print(f'e={epoch}/{NUM_EPOCHS}, loss={mse_weight * epoch_mse - sharpe_weight * epoch_sharpe:.6f}')

        if epoch % 1 == 0:
            torch.save(model.state_dict(), f'output/portfolio_lstm_epoch{epoch}.pt')
        if epoch % 1 == 0 and test_set is not None:
            test_result = test_portfolio(model, test_set, test_items, verbose=True)
            writer.add_scalars('test',
                               {k: sum(v) / len(v) for k, v in test_result.items()},
                               (epoch + 1) * train_num)
        gradient_list.append(gradient_one_epoch)
    gradient_list = torch.tensor(gradient_list, device='cpu')
    gradient_list = np.array(gradient_list)
    epochs = np.arange(START_EPOCH, NUM_EPOCHS)
    if opt_method == 'jpo':
        plt.savefig("./figure/jpo_gradient.png")
    if opt_method == 'jpo-old':
        plt.savefig("./figure/jpo_old_gradient.png")
        
    plt.plot(epochs, gradient_list, color='tab:blue')


######################################
#              testing               #
######################################

def test_portfolio(model, test_set, test_items='all', verbose=True):
    with torch.set_grad_enabled(False):
        if test_items == 'all':
            test_items = ['mse', 'jpo', 'uncstr_jpo', 'pto', 'uncstr_pto', 'opt']
        return_dict = {k: [] for k in test_items}

        for iter_idx, test_data in enumerate(test_set):
            history = torch.tensor(test_data['history'].values).to(device=DEVICE, dtype=torch.double)
            future = torch.tensor(test_data['future'].values).to(device=DEVICE, dtype=torch.double)
            mu, cov = compute_measures(future)

            test_print = [f'test id={iter_idx}, date={test_data["real_date"]}']

            # simple prediction
            if 'mse' in test_items:
                pred_seq = model(history, FUTURE_LEN)
                mse = torch.sum((pred_seq - future) ** 2) / FUTURE_LEN
                test_print.append(f'pred_mse={mse:.4f}')
                return_dict['mse'].append(mse)

            if 'jpo-old' in test_items:
                _, jpo_weight = model(history, FUTURE_LEN, 'jpo-old', RF, K, gumbel_sample_num=10, gumbel_noise_fact=0.1, return_best_weight=True)
                jpo_sharpe = compute_sharpe_ratio(mu, cov, jpo_weight, RF)
                #for s in jpo_sharpe:
                test_print.append(f'joint_pred_opt_old={jpo_sharpe:.4f}')
                return_dict['jpo-old'].append(jpo_sharpe.detach())

            # topK constrained joint predict and optimize
            if 'jpo' in test_items:
                _, jpo_weight = model(history, FUTURE_LEN, 'jpo', RF, K, gumbel_sample_num=1000, gumbel_noise_fact=0.1, return_best_weight=True)
                jpo_sharpe, jpo_risk, jpo_return = compute_sharpe_ratio(mu, cov, jpo_weight, RF, return_details=True)
                #for s in jpo_sharpe:
                test_print.append(f'joint_pred_opt: sharpe={jpo_sharpe:.4f}, return={jpo_return:.4f}, risk={jpo_risk:.4f}')
                return_dict['jpo'].append(jpo_sharpe)

            # unconstrained joint predict and optimize
            if 'uncstr_jpo' in test_items:
                _, uncstr_jpo_weight = model(history, FUTURE_LEN, 'jpo', RF, -1, return_best_weight=True)
                uncstr_jpo_sharpe = compute_sharpe_ratio(mu, cov, uncstr_jpo_weight, RF)
                #for s in uncstr_jpo_sharpe:
                test_print.append(f'joint_pred_opt_unconstr={uncstr_jpo_sharpe:.4f}')
                return_dict['uncstr_jpo'].append(uncstr_jpo_sharpe)

            # find and return the best portfolio in history
            if 'history' in test_items:
                _, history_weight = model(history, FUTURE_LEN, 'history', RF, K)
                history_sharpe, history_risk, history_return = compute_sharpe_ratio(mu, cov, history_weight, RF, return_details=True)
                test_print.append(f'history: sharpe={history_sharpe:.4f}, return={history_return:.4f}, risk={history_risk:.4f}')
                return_dict['history'].append(history_sharpe.detach())

            # topK constrained predict-then-optimize
            if 'pto' in test_items:
                _, pto_weight = model(history, FUTURE_LEN, 'pto', RF, K)
                pto_sharpe, pto_risk, pto_return = compute_sharpe_ratio(mu, cov, pto_weight, RF, return_details=True)
                test_print.append(f'pred_then_opt: sharpe={pto_sharpe:.4f}, return={pto_return:.4f}, risk={pto_risk:.4f}')
                return_dict['pto'].append(pto_sharpe)

            # unconstrained predict-then-optimization
            if 'uncstr_pto' in test_items:
                _, uncstr_pto_weight = model(history, FUTURE_LEN, 'pto', RF, -1)
                uncstr_pto_sharpe = compute_sharpe_ratio(mu, cov, uncstr_pto_weight, RF)
                test_print.append(f'pred_then_opt_unconstr={uncstr_pto_sharpe:.4f}')
                return_dict['uncstr_pto'].append(uncstr_pto_sharpe)

            # optimal sharpe ratio on future data
            #if 'opt' in test_items:
            #    _, opt_weight, __ = gurobi_portfolio_opt(mu, cov, rf=RF, obj='Sharpe', card_constr=K, linear_relaxation=False, timeout_sec=120)
            #    opt_sharpe = compute_sharpe_ratio(mu, cov, opt_weight, RF)
            #    test_print.append(f'opt={opt_sharpe:.4f}')
            #    return_dict['opt'].append(opt_sharpe)

            if verbose:
                print(', '.join(test_print))
        print('Evaluation complete.')
        for k, v in return_dict.items():
            print(f'{k}: {sum(v) / len(v):.4f}')

    return return_dict


if __name__ == '__main__':
    dataset = PortDataset('snp500', HISTORY_LEN, FUTURE_LEN, train_test_split=0.75)
    model = LSTMModel(dataset.num_assets, NUM_FEATURE, 1).to(device=DEVICE)
    model.double()

    START_EPOCH = 0
    if START_EPOCH > 0:
        pretrained_path = f'output_100/portfolio_lstm_epoch{START_EPOCH}.pt'
        print(f'Loading model weights from {pretrained_path}...')
        model.load_state_dict(torch.load(pretrained_path))


    #test_portfolio(model, dataset.test_set, test_items=['jpo-old'], verbose=True)
    train_test_portfolio(model, dataset.train_set, mse_weight=0, sharpe_weight=1, opt_method='jpo-old',
                        test_set=dataset.test_set, test_items=['mse', 'jpo-old'])