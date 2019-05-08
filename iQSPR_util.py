from xenonpy.inverse.iqspr import NGram
from rdkit import Chem
import numpy as np
from xenonpy.descriptor.base import BaseDescriptor


def make_beta_from_sigmoid(n, start=-6, end=8):
    x = np.linspace(start, end, n)
    return 1 / (1 + np.exp(-x))


def make_ngram_model(smiles):
    return _learn_n_gram2(smiles)


def make_forward_model(smiles, values, fingerprints_generator):
    from sklearn.linear_model import BayesianRidge

    x = fingerprints_generator.transform(smiles)
    model = BayesianRidge(compute_score=True)
    model.fit(x, values)

    return model


def make_fingerprints(kwds=('ECFP', 'TopologicalTorsionFP'), **kwargs):
    """
    Chose kwds from below.
    kwds = ['RDKitFP',
            'AtomPairFP',
            'TopologicalTorsionFP',
            'MACCS',
            'ECFP',
            'FCFP',
            'DescriptorFeature']
            or 'all'

    :param kwds: list of fingerprint name
    :param kwargs: parameter of Fingerprints
    :return:
    """

    if kwds == 'all' or (len(kwds) == 1 and kwds[0] == 'all'):
        kwds = ('RDKitFP', 'AtomPairFP', 'TopologicalTorsionFP',
                'MACCS', 'ECFP', 'FCFP', 'DescriptorFeature')

    return Fingerprints(kwds=kwds, **kwargs)


class IqsprUtil:

    def __init__(self, models=None, desired_values=None, beta=None, ngram_model=None, seed_structure=None):
        self.models = models
        self.desired_values = desired_values
        self.beta = beta
        self.ngram_model = ngram_model
        self.seed_structure = seed_structure
        self.iqspr_results = None
        return

    @property
    def models(self):
        return self.models

    @models.setter
    def models(self, dic):
        if isinstance(dic, dict):
            self.models = dic
        else:
            raise TypeError

    @property
    def desired_values(self):
        return self.desired_values

    @desired_values.setter
    def desired_values(self, dic):
        if isinstance(dic, dict):
            self.desired_values = dic
        else:
            raise TypeError

    def check_properties(self):
        pass

    def run(self):
        from xenonpy.inverse.iqspr import IQSPR
        self.check_properties()
        iqspr = IQSPR(estimator=self.models, modifier=self.ngram_model)
        proposed_structures, log_likelihood, probability_score, iqspr_freq = [], [], [], []
        for s, ll, p, freq in iqspr(self.seed_structure, self.beta, yield_lpf=True, **self.desired_values):
            proposed_structures.append(s)
            log_likelihood.append(ll)
            probability_score.append(p)
            iqspr_freq.append(freq)

        # record all outputs
        iqspr_results = {
            "samples": proposed_structures,
            "loglike": log_likelihood,
            "prob": probability_score,
            "freq": iqspr_freq,
            "beta": np.hstack([0, self.beta])  # include the step of initial samples
        }

        self.iqspr_results = iqspr_results
        return iqspr, iqspr_results

    def plot_likelihood(self, how='log', out_dir='./', file_name='likelihood.png'):
        import matplotlib.pyplot as plt

        flat_list = np.asarray([item for sublist in self.iqspr_results["loglike"] for item in sublist])
        y_max, y_min = max(flat_list), min(flat_list)

        plt.figure(figsize=(10, 5))
        plt.xlim(0, len(self.iqspr_results["loglike"]))
        plt.xlabel('Step')

        if how == 'log':
            plt.ylim(y_min, y_max)
            plt.ylabel('Log-likelihood')
            for i, ll in enumerate(self.iqspr_results["loglike"]):
                plt.scatter([i] * len(ll), ll, s=10, c='b', alpha=0.2)
        else:
            y_max, y_min = np.exp(y_max), np.exp(y_min)
            plt.ylim(y_min, y_max)
            plt.ylabel('Likelihood')
            for i, ll in enumerate(self.iqspr_results["loglike"]):
                plt.scatter([i] * len(ll), np.exp(ll), s=10, c='b', alpha=0.2)

        plt.savefig(out_dir + file_name, dpi=500)
        plt.close()
        return

    def visualize(self, data_ss, outdir='./', file_name='temp.png'):
        import matplotlib.pyplot as plt
        from rdkit.Chem import Draw

        n_S = 25
        for i, smis in enumerate(self.iqspr_results['samples']):
            tmp_smis = self.iqspr_results['samples'][i][np.argsort(self.iqspr_results['loglike'][i])[::-1]]
            fig, ax = plt.subplots(5, 5)
            fig.set_size_inches(20, 20)
            fig.set_tight_layout(True)
            for j in range(n_S):
                x_axis = j // 5
                y_axis = j % 5
                try:
                    img = Draw.MolToImage(Chem.MolFromSmiles(tmp_smis[j]))
                    ax[x_axis, y_axis].clear()
                    ax[x_axis, y_axis].set_frame_on(False)
                    ax[x_axis, y_axis].imshow(img)
                except:
                    pass
                ax[x_axis, y_axis].set_axis_off()
            fig.savefig(outdir + 'Step_%02i.png' % i, dpi=500)
            plt.close()


def _learn_n_gram0(smiles):
    # initialize a new n-gram
    n_gram = NGram()

    # train the n-gram with SMILES of available molecules
    n_gram.fit(smiles, train_order=5)
    return n_gram


def _learn_n_gram1(smiles):
    # Method 1: use canonical SMILES in RDKit with no reordering
    cans = _canonicalize_smiles(smiles)
    n_gram = NGram(reorder_prob=0)
    n_gram.fit(cans)

    return n_gram


def _learn_n_gram2(smiles, reorder_prob=0.5, paraphrased_smiles_number=10):
    # Method 2: expand n-gram training set with randomly reordered SMILES
    # (we show one of the many possible ways of doing it)

    mols = _smiles_to_mol(smiles)

    generated_smiles = []
    gen_smi_append = generated_smiles.append
    for mol in mols:
        number_of_atom = mol.GetNumAtoms()
        sample_number = min(number_of_atom, paraphrased_smiles_number)

        shuffled_index = np.random.permutation(number_of_atom)
        tmp = [Chem.MolToSmiles(mol, rootedAtAtom=x) for x in shuffled_index[:sample_number]]
        gen_smi_append(list(set(tmp)))

    flat_list = [item for sublist in generated_smiles for item in sublist]

    n_gram = NGram(reorder_prob=reorder_prob)
    n_gram.fit(flat_list)

    return n_gram


def _smiles_to_mol(smiles):
    mols = _convert_safety(smiles, Chem.MolFromSmiles)
    return mols


def _canonicalize_smiles(smiles):
    mols = _convert_safety(smiles, Chem.MolFromSmiles)
    cans = _convert_safety(mols, Chem.MolToSmiles)
    return cans


def _convert_safety(target, function):
    ret = []
    append = ret.append  # use variable for execution speed
    for i in target:
        try:
            append(function(i))
        except Exception as e:
            print(i, e)
            pass
    return ret


class Fingerprints(BaseDescriptor):
    """
    Calculate fingerprints or descriptors of organic molecules.
    """

    def __init__(self, kwds=None, n_jobs=-1, *, radius=3, n_bits=2048, fp_size=2048, input_type='mol',
                 featurizers='all',
                 on_errors='raise'):
        """

        Parameters
        ----------
        n_jobs: int
            The number of jobs to run in parallel for both fit and predict. Set -1 to use all cpu cores (default).
        radius: int
            The radius parameter in the Morgan fingerprints,
            which is roughly half of the diameter parameter in ECFP/FCFP,
            i.e., radius=2 is roughly equivalent to ECFP4/FCFP4.
        n_bits: int
            Fixed bit length based on folding.
        featurizers: list[str] or 'all'
            Featurizers that will be used.
            Default is 'all'.
        input_type: string
            Set the specific type of transform input.
            Set to ``mol`` (default) to ``rdkit.Chem.rdchem.Mol`` objects as input.
            When set to ``smlies``, ``transform`` method can use a SMILES list as input.
            Set to ``any`` to use both.
            If input is SMILES, ``Chem.MolFromSmiles`` function will be used inside.
            for ``None`` returns, a ``ValueError`` exception will be raised.
        on_errors: string
            How to handle exceptions in feature calculations. Can be 'nan', 'keep', 'raise'.
            When 'nan', return a column with ``np.nan``.
            The length of column corresponding to the number of feature labs.
            When 'keep', return a column with exception objects.
            The default is 'raise' which will raise up the exception.
        """

        super().__init__(featurizers=featurizers)
        self.n_jobs = n_jobs

        if 'RDKitFP' in kwds:
            from xenonpy.descriptor.fingerprint import RDKitFP
            self.mol = RDKitFP(n_jobs, fp_size=fp_size, input_type=input_type, on_errors=on_errors)
        if 'AtomPairFP' in kwds:
            from xenonpy.descriptor.fingerprint import AtomPairFP
            self.mol = AtomPairFP(n_jobs, n_bits=n_bits, input_type=input_type, on_errors=on_errors)
        if 'TopologicalTorsionFP' in kwds:
            from xenonpy.descriptor.fingerprint import TopologicalTorsionFP
            self.mol = TopologicalTorsionFP(n_jobs, n_bits=n_bits, input_type=input_type, on_errors=on_errors)
        if 'MACCS' in kwds:
            from xenonpy.descriptor.fingerprint import MACCS
            self.mol = MACCS(n_jobs, input_type=input_type, on_errors=on_errors)
        if 'ECFP' in kwds:
            from xenonpy.descriptor.fingerprint import ECFP
            self.mol = ECFP(n_jobs, radius=radius, n_bits=n_bits, input_type=input_type, on_errors=on_errors)
        if 'FCFP' in kwds:
            from xenonpy.descriptor.fingerprint import FCFP
            self.mol = FCFP(n_jobs, radius=radius, n_bits=n_bits, input_type=input_type, on_errors=on_errors)
        if 'DescriptorFeature' in kwds:
            from xenonpy.descriptor.fingerprint import DescriptorFeature
            self.mol = DescriptorFeature(n_jobs, input_type=input_type, on_errors=on_errors)
