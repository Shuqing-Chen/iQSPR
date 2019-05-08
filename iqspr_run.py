import pandas as pd
from . import iQSPR_util
import numpy as np
import os


def main():
    data = pd.read_csv('temp.csv')
    smiles = data['smiles']
    energy = data['E']
    gap = data['HOMO-LUMO gap']

    # learn ngram
    n_gram = iQSPR_util.make_ngram_model(smiles)

    fingerprints_generator = iQSPR_util.make_fingerprints()
    e_model = iQSPR_util.make_forward_model(smiles, energy, fingerprints_generator)
    gap_model = iQSPR_util.make_forward_model(smiles, gap, fingerprints_generator)
    models = {'E': e_model, 'HOMO-LUMO gap': gap_model}

    # settings for iQSPR
    seed = iQSPR_util.sample_seeds(smiles)  # 25 seeds are randomly sampled, for visualising reason
    beta = iQSPR_util.make_beta_from_sigmoid(50)  # run 50 cycle of iqspr
    desired_values = {'E': (0, 200), 'HOMO-LUMO gap': (-np.inf, 3)}

    iqspr = iQSPR_util.IqsprWrapper(models=models, desired_values=desired_values, beta=beta,
                                    ngram_model=n_gram, seed_structure=seed)
    iqspr.run()

    out_dir = 'retults/'
    if os.path.exists(out_dir):
        os.mkdir(out_dir)

    iqspr.plot_likelihood(out_dir=out_dir)
    iqspr.visualize(out_dir=out_dir)


if __name__ == '__main__':
    main()
