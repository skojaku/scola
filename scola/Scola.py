import numpy as np
from scipy import linalg
from scipy import sparse
from scipy import stats
import pickle
import os
import tqdm
import sys
import configcorr as cg
from functools import partial


class Scola():
    """
    This is a class object for the Scola algorithm.

    Parameters
    ----------
    disp : bool 
        Set disp=True to disply the progress. Otherwise set disp=False.
    """

    def __init__(self):
        pass

    def run(self, C_samp, L, null_model="config", disp=True, gamma=0.5, working_directory = None):
        """
        Construct a network from a correlation matrices 
        using the Scola algorithm
   
        Parameters
        ----------
        C_samp : numpy.matrix
            Sample correlation matrix. 
            
        L: int 
            Number of samples

        null_model : string, default 'config'
            Name of the null model. Available null models are;
                * White noise model (null_model='white-noise')
                * HQS model (null_model='hqs')
                * Configuration model (null_model='config')

        disp : bool, default True 
            Set disp=True to disply the progress. 
            Otherwise set disp=False.
            
        gamma : float, default 0.5 
            Hyperparameter for the extended Baysian information criterion.

        working_directory : string, default None.
            If working_directory is given, the program will create a directry of the name.
            Then, the program saves its progress in the working directory.
            If the directory already exists, then the program will read the results and resume the computation.
    
        Returns
        -------
        W : numpy.matrix 
            Weighted adjacency matrix of the generated network.  

        EBIC : float 
            Value of the extended BIC for the generated network.
                
        Note
        ----
        This function saves the tentative results in networks-<name of the null model>.pickle in the working directory.
        """
            
        # Check input types
        if type(C_samp) is np.ndarray:
                C_samp = np.matrix(C_samp)

        if type(C_samp) is not np.matrix:
            raise TypeError("C_samp must be a numpy.matrix")

        if (type(L) is not int) and (type(L) is not float):
            raise TypeError("L must be an integer")

        if type(null_model) is not str:
            raise TypeError("null_model must be string")

        if null_model not in ['white-noise', 'hqs', 'config']:
            raise ValueError(
                'Null model %s is unknown. See the Readme for the available null models' % null_model)

        if type(disp) is not bool:
            raise TypeError("disp must be a bool")

        if type(gamma) is not float:
            raise TypeError("gamma must be a float")


        # Create working directory. The tentative results will be saved 
        # in networks-<name of the null model>.pickle in the working directory
        self.null_model = null_model
        if not os.path.exists(self.working_directory):
            os.mkdir(self.working_directory)
            if disp:
                print("Directory ", self.working_directory,  "is created.")
        file_tentative_results = '%s/networks-%s.pickle' % (
            self.working_directory, null_model)
        Logs = self._load_logs(file_tentative_results)

        # Compute the null correlation matrices
        C_null, K_null = self._compute_null_model(
            C_samp, null_model, disp)

        # Optimise the lasso penalty using the golden section search
        lam_l = 0  # Lower bound for lambda
        lam_u = 1  # Upper bound for lambda
        invphi = (np.sqrt(5) - 1) / 2  # 1/phi
        invphi2 = (3 - np.sqrt(5)) / 2  # 1/phi^2
        h = lam_u-lam_l
        lam_1 = lam_l + invphi2 * h
        lam_2 = lam_l + invphi * h
        n = int(np.ceil(np.log(0.01/h)/np.log(invphi)))
        N = C_samp.shape[0]

        if disp:
            print("")
            print('Constructing networks...')

        for k in tqdm.tqdm(range(n), disable=(disp is False)):
            # Compute the EBIC values for the initial lambda values
            if k == 0:
                # Construct networks for lam_l, lam_1, lam_2 and lam_u
                W_l = C_samp - C_null
                W_u = self._construct_network(C_samp, C_null, lam_u, Logs)
                W_1 = self._construct_network(C_samp, C_null, lam_1, Logs)
                W_2 = self._construct_network(C_samp, C_null, lam_2, Logs)

                # Compute the extended BIC
                EBIC_l = self._calc_EBIC(
                    W_l, C_samp, C_null, L, gamma, K_null)
                EBIC_u = self._calc_EBIC(
                    W_u, C_samp, C_null, L, gamma, K_null)
                EBIC_1 = self._calc_EBIC(
                    W_1, C_samp, C_null, L, gamma, K_null)
                EBIC_2 = self._calc_EBIC(
                    W_2, C_samp, C_null, L, gamma, K_null)

                # Save tentative results
                Logs[lam_l] = {'W': W_l, 'EBIC': EBIC_l}
                Logs[lam_u] = {'W': W_u, 'EBIC': EBIC_u}
                Logs[lam_1] = {'W': W_1, 'EBIC': EBIC_1}
                Logs[lam_2] = {'W': W_2, 'EBIC': EBIC_2}
                self._save_results(file_tentative_results, Logs)
                continue

            if (EBIC_1 < EBIC_2) |\
                    ((EBIC_1 == EBIC_2) & (np.random.rand() > 0.5)):
                # Update the range for lambda
                lam_u = lam_2
                lam_2 = lam_1
                EBIC_u = EBIC_2
                EBIC_2 = EBIC_1
                h = invphi * h

                # Construct a networkf for a new lambda value
                lam_1 = lam_l + invphi2 * h
                W_1 = self._construct_network(C_samp, C_null, lam_1, Logs)
                EBIC_1 = self._calc_EBIC(
                    W_1, C_samp, C_null, L, gamma, K_null)

                # Save results
                Logs[lam_1] = {'W': W_1, 'EBIC': EBIC_1}
                self._save_results(file_tentative_results, Logs)

            else:  # If EBIC_2 < EBIC_1 or EBIC_2 = EBIC_1
                # Update the range for lambda
                lam_l = lam_1
                lam_1 = lam_2
                EBIC_l = EBIC_1
                EBIC_1 = EBIC_2
                h = invphi * h

                # Construct a networkf for a new lambda value
                lam_2 = lam_l + invphi * h

                W_2 = self._construct_network(C_samp, C_null, lam_2, Logs)
                EBIC_2 = self._calc_EBIC(
                    W_2, C_samp, C_null, L, gamma, K_null)

                # Save results
                Logs[lam_2] = {'W': W_2, 'EBIC': EBIC_2}
                self._save_results(file_tentative_results, Logs)

        return self._find_best_model(Logs)

    def get_network(self, null_model=None):
        if null_model is None:
            null_model = self.null_model
        file_tentative_results = '%s/networks-%s.pickle' % (
            self.working_directory, null_model)
        Logs = self._load_logs(file_tentative_results)
        W, EBIC = self._find_best_model(Logs)
        return W

    def get_null_corr_matrix(self, null_model=None):
        if null_model is None:
            null_model = self.null_model
        file_null_model = '%s/null-model-%s.pickle' % (
            self.working_directory, null_model)
        with open(file_null_model, 'rb') as f:
            res = pickle.load(f)
        C_null = res["C_null"]
        K_null = res["K_null"]
        return C_null, K_null

    def get_corr_matrix(self, null_model=None):
        W = self.get_network(null_model)
        C_null, _ = self.get_null_corr_matrix(null_model)
        return C_null + W

    def _find_best_model(self, Logs):
        W = None
        EBIC = 1e+30
        for lam_, net in Logs.items():
            if EBIC > net["EBIC"]:
                W = net["W"]
                EBIC = net["EBIC"]
        return W, EBIC

    def _compute_null_model(self, C_samp, null_model, disp):
        file_null_model = '%s/null-model-%s.pickle' % (
            self.working_directory, null_model)
        C_null = []
        K_null = -1

        if disp:
            print('Computing null correlation matrix.')

        if os.path.exists(file_null_model):
            with open(file_null_model, 'rb') as f:
                res = pickle.load(f)
                C_null = res["C_null"]
                K_null = res["K_null"]
        else:
            if null_model == "white-noise":
                C_null = np.matrix(np.eye(C_samp.shape[0]))
                K_null = 0
            elif null_model == "hqs":
                C_null = np.mean(np.triu(C_samp, 1)) * np.ones(C_samp.shape)
                np.fill_diagonal(C_null, 1)
                K_null = 1
            elif null_model == "config":
                C_null = cg.max_ent_config_dmcc(C_samp, 1e-4, disp)
                std_ = np.sqrt(np.diag(C_null))
                C_null = C_null / np.outer(std_, std_)
                K_null = C_samp.shape[0]
            else:
                raise ValueError(
                    'Null model %s is unknown.' % null_model)

            with open(file_null_model, 'wb') as f:
                pickle.dump(
                    {'C_null': C_null, 'K_null': K_null}, f)

        return C_null, K_null

    def _construct_network(self, C_samp, C_null, lam, Logs=None):
        if Logs is not None:
            if lam in Logs:
                return Logs[lam]["W"]

        N = C_samp.shape[0]
        Lambda = 1/(np.power(np.abs(C_samp-C_null), 2)+1e-20)
        W = self._prox(C_samp - C_null, lam * Lambda)
        score_prev = -1.7976931348623157e+308
        while True:
            _W = self._maximisation_step(C_samp, C_null, W, lam)
            score = self._penalized_likelihood(_W, C_samp, C_null, lam, Lambda)
            if score <= score_prev:
                break
            W = _W
            score_prev = score

        return W

    def _maximisation_step(self, C_samp, C_null, W_base, lam):
        # Initialise the variables for the ADAM algorithm
        N = C_samp.shape[0]
        mt = np.matrix(np.zeros((N, N)))
        vt = np.matrix(np.zeros((N, N)))
        t = 0
        eps = 1e-8
        b1 = 0.9
        b2 = 0.99
        maxscore = -1.7976931348623157e+308
        t_best = 0
        eta = 0.001
        maxIteration = 1e+7
        maxLocalSearch = 300
        quality_assessment_interval = 100
        W_best = W_base
        W = W_base

        # Compute the penalty strength using the adaptive lasso
        Lambda = 1/(np.power(np.abs(C_samp-C_null), 2)+1e-20)

        inv_C_base = self._fast_inv_mat_lapack(C_null + W_base)
        while (t < maxIteration) & (t <= (t_best + maxLocalSearch+1)):
            t = t + 1
            # Calculate the gradient
            # Call LAPAC for fast matrix inversions
            inv_C = self._fast_inv_mat_lapack(C_null+W)
            gt = inv_C_base - inv_C @ C_samp@ inv_C
            gt = (gt + gt.T)/2
            gt = np.nan_to_num(gt)
            np.fill_diagonal(gt, 0)
            mt = b1 * mt + (1-b1)*gt
            vt = b2 * vt + (1-b2)*np.power(gt, 2)
            mthat = mt / (1-np.power(b1, t))
            vthat = vt / (1-np.power(b2, t))
            dtheta = np.divide(mthat, (np.sqrt(vthat) + eps))

            # Update the network
            W = self._prox(W - eta * dtheta, eta * lam * Lambda)
            np.fill_diagonal(W, 0)

            # Compute the likelihood
            if (t % quality_assessment_interval) == 0:
                s = self._penalized_likelihood(W, C_samp, C_null, lam, Lambda)
                if s > maxscore:
                    W_best = W
                    maxscore = s
                    t_best = t

        return W_best

    def _fast_inv_mat_lapack(self, M):
        zz, _ = linalg.lapack.dpotrf(M, False, False)
        inv_M, info = linalg.lapack.dpotri(zz)
        inv_M = np.triu(inv_M) + np.triu(inv_M, k=1).T
        return inv_M

    def _loglikelihood(self, W, C_samp, C_null):
        Cov = W + C_null
        w, v = np.linalg.eig(Cov)
        iCov = np.real(v @ np.diag(1/w) @ v.T)
        return -0.5 * np.sum(np.log(w)) - 0.5 * np.trace(C_samp @ iCov)\
            - 0.5 * Cov.shape[0] * np.log(2 * np.pi)

    def _calc_EBIC(self, W, C_samp, C_null, L, gamma, null_model_parameters):
        k = null_model_parameters + np.count_nonzero(W)/2
        EBIC = np.log(L) * k - 2 * L * self._loglikelihood(W, C_samp, C_null)
        EBIC += 4 * gamma * k * np.log(W.shape[0])
        return EBIC

    # soft thresholding method
    def _prox(self, y, lam):
        return np.multiply(np.sign(y),
                           np.maximum(np.abs(y)
                                      - lam, np.matrix(np.zeros(y.shape))))

    def _penalized_likelihood(self, W, C_samp, C_null, lam, Lambda):
        return self._loglikelihood(W, C_samp, C_null)\
                        - lam * np.sum(np.dot(Lambda, np.abs(W)))

    def _save_results(self, filename, networks):
        with open(filename, 'wb') as f:
            pickle.dump(networks, f)

    def _load_logs(self, filename):
        if os.path.exists(filename):  # if tentative-results.pickle exists
            with open(filename, 'rb') as f:
                Logs = pickle.load(f)
        else:
            # Otherwise start from scratch
            Logs = {}

        return Logs
