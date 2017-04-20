import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from scipy.stats import binom, poisson
import argparse

parser = argparse.ArgumentParser(description="Run Good-Toulmin Estimation")
parser.add_argument('--mode', '-m', type=str, default='none',
    help="The mode of the smoothing distribution.")
parser.add_argument('--upper-bound', '-u', type=float, default=2.1,
    help="The upper bound.")

args = parser.parse_args()

MAX_ZIPF = 10000
DATA_SIZE = 15000
SEEN_DATA_SIZE = 5000

def generate_data(max_zipf=MAX_ZIPF, size=DATA_SIZE):
    np.random.seed(0)
    prob = np.zeros(max_zipf)
    norm = 0.0
    for i in range(max_zipf):
        prob[i] = 1.0 / (i+1+10)
        norm += prob[i]
    prob /= norm
    datarange = np.arange(1, max_zipf+1)
    dataset = np.random.choice(datarange, size=size, p=prob)
    return dataset

def calc_combination_frequency(dataset, seen_size=SEEN_DATA_SIZE):
    seen_data = dataset[:seen_size]
    freq_dist = defaultdict(int)
    phi = defaultdict(int)
    for data in seen_data:
        freq_dist[data] += 1

    for v in freq_dist.values():
        phi[v] += 1

    return phi

def poisson_smoothing_distn(i, t, n):
    mu = 1.0/(2*t)*np.log(n*(t+1)**2/(t-1))
    return 1-poisson.cdf(i, mu)

def binomial_smoothing_distn(i, t, n):
    num_test = 0.5*np.log2(n*t**2/(t-1))
    prob = 1.0/(t+1)
    return 1-binom.cdf(i, num_test, prob)


def calc_good_toulmin(t, phi, mode='none', seen_size=SEEN_DATA_SIZE):
    Ugt = 0.0

    used_prob_model = lambda i, t, n: 1.0
    if mode=='poisson':
        used_prob_model = poisson_smoothing_distn
    elif mode=='binom':
        used_prob_model = binomial_smoothing_distn

    for i, phi_i in phi.items():
        Ugt_i = ((-t)**(i))*phi_i
        if t>1.0:
            Ugt_i *= used_prob_model(i, t, seen_size)
        Ugt += Ugt_i
    return -Ugt

def calc_ground_truth_unseen(dataset, t, seen_size=SEEN_DATA_SIZE):
    U = 0
    counting_size = seen_size*(1+t)
    freq_dist = defaultdict(int)
    for i in range(int(counting_size)):
        if i >= seen_size and freq_dist[dataset[i]] == 0:
            U += 1
        freq_dist[dataset[i]] += 1
    return U

def main():
    dataset = generate_data()
    phi = calc_combination_frequency(dataset)
    t_range = np.arange(0, args.upper_bound, 0.1)
    Ugt_array = np.zeros(len(t_range))
    U_group_truth_array = np.zeros(len(t_range))
    for i, t in enumerate(t_range):
        Ugt_array[i] = calc_good_toulmin(t, phi, args.mode)
        U_group_truth_array[i] = calc_ground_truth_unseen(dataset, t)

    plt.plot(t_range, Ugt_array, 'r', t_range, U_group_truth_array, 'b')

    title_str = "without smoothing"
    if args.mode == 'binom':
        title_str = "with binomial smoothing"
    elif args.mode == 'poisson':
        title_str = "with poisson smoothing"
    plt.title('Good-Toulmin Estimation {}'.format(title_str))
    plt.xlabel('t')
    plt.ylabel('# of new items')
    plt.show()

if __name__ == '__main__':
    main()
