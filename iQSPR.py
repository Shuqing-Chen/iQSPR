# XenonPy descriptor calculation library
from xenonpy.descriptor.base import BaseDescriptor
from xenonpy.descriptor import ECFP, MACCS
import warnings
import os

# basic uses
import numpy as np
import pandas as pd
import pickle as pk

# plotting figures
import matplotlib.pyplot as plt

# RDKit molecule conversion and drawing
from rdkit import Chem
from rdkit.Chem import Draw

# Forward model template in XenonPy-iQSPR
from xenonpy.inverse.iqspr import BayesianRidgeEstimator

# N-gram library in XenonPy-iQSPR
from xenonpy.inverse.iqspr import NGram

warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

np.random.seed(201903)


class RDKitDesc(BaseDescriptor):
    def __init__(self, n_jobs=-1):
        super().__init__()
        self.n_jobs = n_jobs

        self.rdkit_fp = ECFP(n_jobs, on_errors='nan', input_type='smiles')
        self.rdkit_fp = MACCS(n_jobs, on_errors='nan', input_type='smiles')


def make_forward_model_with_iqspr_tools(data_ss, RDKit_FPs):
    # write down list of property name(s) for forward models
    # (they will be used as a key in whole iQSPR run)
    prop = ['E', 'HOMO-LUMO gap']

    # import descriptor class to iQSPR
    prd_mdls = BayesianRidgeEstimator(descriptor=RDKit_FPs)

    # train forward models inside iQSPR
    prd_mdls.fit(data_ss['SMILES'], data_ss[prop])
    return prd_mdls


def make_forward_model(data_ss, RDKit_FPs):
    # forward model library from scikit-learn
    from sklearn.linear_model import BayesianRidge
    # xenonpy library for data splitting (cross-validation)
    from xenonpy.datatools import Splitter

    # property name will be used as a reference for calling models
    prop = ['E', 'HOMO-LUMO gap']

    # prepare indices for cross-validation data sets
    sp = Splitter(data_ss.shape[0], test_size=0, cv=5)

    # initialize output variables
    y_trues, y_preds = [[] for i in range(len(prop))], [[] for i in range(len(prop))]
    y_trues_fit, y_preds_fit = [[] for i in range(len(prop))], [[] for i in range(len(prop))]
    y_preds_std, y_preds_std_fit = [[] for i in range(len(prop))], [[] for i in range(len(prop))]

    # cross-validation test
    for iTr, iTe in sp.cv():
        x_train = data_ss['SMILES'].iloc[iTr]
        x_test = data_ss['SMILES'].iloc[iTe]

        fps_train = RDKit_FPs.transform(x_train)
        fps_test = RDKit_FPs.transform(x_test)

        y_train = data_ss[prop].iloc[iTr]
        y_test = data_ss[prop].iloc[iTe]
        for i in range(len(prop)):
            mdl = BayesianRidge(compute_score=True)
            mdl.fit(fps_train, y_train.iloc[:, i])
            prd_train, std_train = mdl.predict(fps_train, return_std=True)
            prd_test, std_test = mdl.predict(fps_test, return_std=True)

            y_trues[i].append(y_test.iloc[:, i].values)
            y_trues_fit[i].append(y_train.iloc[:, i].values)
            y_preds[i].append(prd_test)
            y_preds_fit[i].append(prd_train)
            y_preds_std[i].append(std_test)
            y_preds_std_fit[i].append(std_train)

    # write down list of property name(s) for forward models
    prop = ['E', 'HOMO-LUMO gap']  # match with data table for convenience

    # calculate descriptor values for all SMILES in the data subset
    fps_train = RDKit_FPs.transform(data_ss['SMILES'])

    # initialize a dictionary for model storage
    mdls = {}

    # fill in and train the models
    for x in prop:
        mdls[x] = BayesianRidge()
        mdls[x].fit(fps_train, data_ss[x])

    # import descriptor calculator and forward model to iQSPR
    prd_mdls = BayesianRidgeEstimator(descriptor=RDKit_FPs, **mdls)
    return prd_mdls, mdls


def learn_n_gram0(smiles):
    # initialize a new n-gram
    n_gram = NGram()

    # train the n-gram with SMILES of available molecules
    n_gram.fit(smiles, train_order=5)
    return n_gram


def learn_n_gram1(smiles):
    # Method 1: use canonical SMILES in RDKit with no reordering
    cans = [Chem.MolToSmiles(Chem.MolFromSmiles(smi)) for smi in smiles]
    n_gram_cans = NGram(reorder_prob=0)
    n_gram_cans.fit(cans)

    # save results
    # with open('ngram_cans.obj', 'wb') as f:
    #     pk.dump(n_gram_cans, f)
    return n_gram_cans


def learn_n_gram2(smiles):
    # Method 2: expand n-gram training set with randomly reordered SMILES
    # (we show one of the many possible ways of doing it)
    n_reorder = 10  # pick a fixed number of re-ordering

    # convert the SMILES to canonical SMILES in RDKit (not necessary in general)
    cans = []
    for smi in smiles:
        # remove some molecules in the full SMILES list that may lead to error
        try:
            cans.append(Chem.MolToSmiles(Chem.MolFromSmiles(smi)))
        except:
            print(smi)
            pass

    mols = [Chem.MolFromSmiles(smi) for smi in cans]
    smi_reorder = []
    for mol in mols:
        idx = list(range(mol.GetNumAtoms()))
        np.random.shuffle(idx)
        tmp = [Chem.MolToSmiles(mol, rootedAtAtom=x) for x in range(min(len(idx), n_reorder))]
        smi_reorder.append(list(set(tmp)))

    # flatten out the list and train the N-gram
    flat_list = [item for sublist in smi_reorder for item in sublist]
    n_gram_reorder = NGram(reorder_prob=0.5)
    n_gram_reorder.fit(flat_list)

    # save results
    # with open('ngram_reorder_full.obj', 'wb') as f:
    #     pk.dump(n_gram_reorder, f)
    return n_gram_reorder


def run_iqspr(prd_mdls, n_gram, init_samples, beta):
    # library for running iQSPR in XenonPy-iQSPR
    from xenonpy.inverse.iqspr import IQSPR

    # set up likelihood and n-gram models in iQSPR
    iqspr = IQSPR(estimator=prd_mdls, modifier=n_gram)

    np.random.seed(201903)  # fix the random seed
    # main loop of iQSPR
    samples, loglike, prob, freq = [], [], [], []
    for s, ll, p, freq in iqspr(init_samples, beta, yield_lpf=True,
                                **{'E': (0, 200), 'HOMO-LUMO gap': (-np.inf, 3)}):
        samples.append(s)
        loglike.append(ll)
        prob.append(p)
        freq.append(freq)

    # record all outputs
    iqspr_results_reorder = {
        "samples": samples,
        "loglike": loglike,
        "prob": prob,
        "freq": freq,
        "beta": np.hstack([0, beta])  # include the step of initial samples
    }

    # save results
    with open('iQSPR_results_reorder.obj', 'wb') as f:
        pk.dump(iqspr_results_reorder, f)

    with open('iQSPR_results_reorder.obj', 'rb') as f:
        iqspr_results_reorder = pk.load(f)

    return iqspr_results_reorder


def visualize(iqspr_results, RDKit_FPs, mdls, data_ss, beta):
    # re-calculate the property values for the proposed molecules
    x_mean, x_std, y_mean, y_std = [], [], [], []
    r_std = []
    FPs_samples = []
    for i, smis in enumerate(iqspr_results["samples"]):
        tmp_fps = RDKit_FPs.transform(smis)
        FPs_samples.append(tmp_fps)

        tmp1, tmp2 = mdls["E"].predict(tmp_fps, return_std=True)
        x_mean.append(tmp1)
        x_std.append(tmp2)

        tmp1, tmp2 = mdls["HOMO-LUMO gap"].predict(tmp_fps, return_std=True)
        y_mean.append(tmp1)
        y_std.append(tmp2)

        r_std.append([np.sqrt(x_std[-1][i] ** 2 + y_std[-1][i] ** 2) for i in range(len(x_std[-1]))])

    # flatten the list for max/min calculation
    flat_list = [item for sublist in r_std for item in sublist]
    print('Range of std. dev.: (%.4f,%.4f)' % (min(flat_list), max(flat_list)))

    # prepare a folder to save all the figures
    ini_dir = './iQSPR_tutorial_prd/'
    if not os.path.exists(ini_dir):
        os.makedirs(ini_dir)

    flat_list = np.asarray([item for sublist in r_std for item in sublist])
    s_max, s_min = max(flat_list), min(flat_list)
    flat_list = np.concatenate((data_ss["E"],
                                np.asarray([item for sublist in x_mean for item in sublist])))
    x_max, x_min = max(flat_list), min(flat_list)
    flat_list = np.concatenate((data_ss["HOMO-LUMO gap"],
                                np.asarray([item for sublist in y_mean for item in sublist])))
    y_max, y_min = max(flat_list), min(flat_list)
    tmp_beta = np.hstack([0, beta])

    for i in range(len(r_std)):
        dot_size = 45 * ((np.asarray(r_std[i]) - s_min) / (s_max - s_min)) + 5

        plt.figure(figsize=(5, 5))
        rectangle = plt.Rectangle((0, 0), 200, 3, fc='y', alpha=0.1)
        plt.gca().add_patch(rectangle)
        plt.scatter(data_ss["E"], data_ss["HOMO-LUMO gap"], s=3, c='b', alpha=0.2)
        plt.scatter(x_mean[i], y_mean[i], s=dot_size, c='r', alpha=0.5)
        plt.title('Step: %i (beta = %.3f)' % (i, tmp_beta[i]))
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.xlabel('Internal energy')
        plt.ylabel('HOMO-LUMO gap')
        # plt.show()
        plt.savefig(ini_dir + 'Step_%02i.png' % i, dpi=500)
        plt.close()

    # prepare a folder to save all the figures
    ini_dir = './iQSPR_tutorial_smiles/'
    if not os.path.exists(ini_dir):
        os.makedirs(ini_dir)

    n_S = 25
    for i, smis in enumerate(iqspr_results['samples']):
        tmp_smis = iqspr_results['samples'][i][
            np.argsort(iqspr_results['loglike'][i])[::-1]]
        fig, ax = plt.subplots(5, 5)
        fig.set_size_inches(20, 20)
        fig.set_tight_layout(True)
        for j in range(n_S):
            xaxis = j // 5
            yaxis = j % 5
            try:
                img = Draw.MolToImage(Chem.MolFromSmiles(tmp_smis[j]))
                ax[xaxis, yaxis].clear()
                ax[xaxis, yaxis].set_frame_on(False)
                ax[xaxis, yaxis].imshow(img)
            except:
                pass
            ax[xaxis, yaxis].set_axis_off()
        fig.savefig(ini_dir + 'Step_%02i.png' % i, dpi=500)
        plt.close()

    target_smis = [Chem.MolToSmiles(Chem.MolFromSmiles(smi)) for i, smi in enumerate(data_ss['SMILES'])
                   if ((data_ss['HOMO-LUMO gap'].iloc[i] <= 3) and (data_ss['E'].iloc[i] <= 200))]

    # prepare a folder to save all the figures
    ini_dir = './iQSPR_tutorial_target_smiles/'
    if not os.path.exists(ini_dir):
        os.makedirs(ini_dir)

    n_S = 25

    fig, ax = plt.subplots(5, 5)
    fig.set_size_inches(20, 20)
    fig.set_tight_layout(True)
    for j in range(n_S):
        xaxis = j // 5
        yaxis = j % 5
        try:
            img = Draw.MolToImage(Chem.MolFromSmiles(target_smis[j]))
            ax[xaxis, yaxis].clear()
            ax[xaxis, yaxis].set_frame_on(False)
            ax[xaxis, yaxis].imshow(img)
        except:
            pass
        ax[xaxis, yaxis].set_axis_off()
    fig.savefig(ini_dir + 'target_region.png', dpi=500)
    plt.close()


def main():
    # load in-house data from csv file
    data = pd.read_csv("./iQSPR_sample_data.csv")
    data_ss = data.sample(3000).reset_index()

    from xenonpy.descriptor import Fingerprints

    RDKit_FPs = Fingerprints(featurizers=['ECFP'], input_type='smiles')

    # make models
    # prd_mdls = make_forward_model_with_iqspr_tools(data_ss, RDKit_FPs)
    prd_mdls, mdls = make_forward_model(data_ss, RDKit_FPs)

    # learn NGrams by method 0 (not recommended)
    # n_gram = learn_n_gram0(data_ss['SMILES'])

    # learn NGrams by method 1 (recommended)
    # n_gram = learn_n_gram1(data_ss['SMILES'])

    # learn NGrams by method2 (recommended)
    n_gram = learn_n_gram2(data_ss['SMILES'])
    n_gram.set_params(del_range=[1, 20], max_len=500, reorder_prob=0.5)

    # set up initial molecules for iQSPR
    np.random.seed(201903)  # fix the random seed
    cans = [Chem.MolToSmiles(Chem.MolFromSmiles(smi)) for i, smi in enumerate(data_ss['SMILES'])
            if (data_ss['HOMO-LUMO gap'].iloc[i] > 4)]
    init_samples = np.random.choice(cans, 25)

    # set up annealing schedule in iQSPR
    beta = np.hstack(
        [np.linspace(0.01, 0.2, 20), np.linspace(0.21, 0.4, 10), np.linspace(0.4, 1, 10), np.linspace(1, 1, 10)])

    iqspr_results = run_iqspr(prd_mdls, n_gram, init_samples, beta)

    visualize(iqspr_results, RDKit_FPs, mdls, data_ss, beta)
