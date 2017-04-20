import numpy as np

import matplotlib.pyplot as plt

def create_histogram (
        list_to_draw, x_label='X Axis',
        hist_title=r'$\mathrm{Histogram}$', hist_color='green'):
    hist_bins = int(len(list_to_draw)/3.5)
    plt.hist(list_to_draw, bins=hist_bins, normed=True,
             color=hist_color)

    plt.xlabel(x_label)
    plt.ylabel('Frequency')
    plt.title(hist_title)
    plt.grid(True)

    plt.show()

def generate_data():
    mu, sigma = 5,1
    num_of_dataset  = 100
    dataset = np.random.normal(mu, sigma, num_of_dataset)
    return dataset


def main():
    dataset = generate_data()
    num_of_experiment = 100

    bootstrap_estimator = np.zeros(num_of_experiment)
    for i in range(num_of_experiment):
        bootstrapped_data = np.random.choice(dataset, len(dataset))
        bootstrap_estimator[i] = np.mean(bootstrapped_data)

    random_sample_estimator = np.zeros(num_of_experiment)
    for i in range(num_of_experiment):
        random_sample_data = generate_data()
        random_sample_estimator[i] = np.mean(random_sample_data)

    started = 5
    sorted_boostrap_est = sorted(bootstrap_estimator)
    sorted_random_sample_est = sorted(random_sample_estimator)

    create_histogram(sorted_boostrap_est[started:num_of_experiment-started])
    create_histogram(sorted_random_sample_est[started:num_of_experiment-started])


if __name__ == '__main__':
    main()
