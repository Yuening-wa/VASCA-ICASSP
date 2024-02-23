from matplotlib import pyplot as plt


def plot_spectra(A):
    # Plot the columns of A
    plt.figure()
    for i in range(A.shape[1]):
        plt.plot(A.numpy()[:, i], label=f'line{i+1}')
    # plt.legend()
    # plt.show()
    return plt

def compare_spectra(A, A_est):
    fig, ax = plt.subplots(1, 2)
    for i in range(A.shape[1]):
        ax[0].plot(A.numpy()[:, i], label=f'a_{i+1}')
        ax[1].plot(A_est.numpy()[:, i])
    fig.tight_layout()
    ax[0].legend()

