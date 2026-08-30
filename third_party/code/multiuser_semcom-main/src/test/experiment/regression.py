from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from typing import *
from pathlib import Path
import torchvision
import numpy as np
from tqdm import tqdm
import re
import pickle
import matplotlib.pyplot as plt
import itertools
import argparse
import pandas as pd
import functools
import joblib
import random
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.metrics import mean_squared_error
import scipy
from scipy.optimize import minimize, Bounds, LinearConstraint, shgo

from ...channel import AWGNMultiUplinkChannel, RayleighFadingMultiUplinkChannel
from ...utils import str_type, DeviceParallel, dp_delayed, MonitorFileActivityWatcher
from .method import (
    UITestSuite, UplinkInference, UplinkInference_SHAPbasedSelection, FSM_Msa, FSM_Ave,
    UplinkInference_RandomSelection, FSM_Msa_random
)

from .modality import (
    UplinkExperiment20250911, test_general, TestHelper, UplinkExperiment20250914, plot_general
)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def _fsm_for(model_type: str):
    return FSM_Msa if model_type.endswith('_msa') else FSM_Ave

def _model_infos_from_config(config: dict) -> list[tuple]:
    mi = config['model_info']
    ids = config['model_nums']['nosic']
    base = (mi['channel_type'], mi['snr_db'], mi['model_type'], mi['encoder_out_dims'], mi['dataset_type'])
    
    if ids > 1:
        print(type(base))
        return [(*base, i) for i in range(1, ids + 1)]
    elif ids == 1:
        return [(*base, 1)]
    else:
        raise ValueError(f"Value of \"model_nums\" in config requires at least 1, but got {ids}")

def _model_info_noid_from_config(config: dict) -> tuple:
    mi = config['model_info']
    return (mi['channel_type'], mi['snr_db'], mi['model_type'], mi['encoder_out_dims'], mi['dataset_type'])

def _resolve_shap_path(config: dict, ue_cls, method: str) -> Path:
    """
    If config['shap_file_path'] is a directory, we assume:
        <dir>/<model_name>/feat_contribs.npz
    Otherwise treat it as a direct .npz file path.
    """
    p = Path(config['shap_file_path'])
    # model_name without model_id
    model_name = ue_cls._get_model_name(*_model_info_noid_from_config(config))
    # return p / model_name / 'feat_contribs.npz' if p.is_dir() else p
    return p / 'feat_contribs.npz' if p.is_dir() else p

def _get_shap_paths(config: dict, ue_cls) -> list[Path]:
    """
    Given the config, return the list of SHAP file paths for all model ids.
    We assume config['shap_file_path'] is a directory to the SHAP files of different model ids
    """
    shap_dir = Path(config['shap_file_path'])
    
    model_infos = _model_infos_from_config(config)
    model_names = [ue_cls._get_model_name(*info) for info in model_infos]
    paths = [shap_dir / m_name / 'feat_contribs.npz' for m_name in model_names]
    
    return paths

def parse_args() -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='configuration file for test and method settings')
    
    args = parser.parse_args()
    
    return args

def calc_rmse(y_true: np.ndarray, y_pred:np.ndarray) -> float:
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

class RegressionMethod:
    """
    for different regression models
    """
    ue_cls = UplinkExperiment20250914
    # result_main_folder = ue_cls.result_main_folder / 'ui_reg'
    
    @classmethod
    def _result_root(cls) -> Path:
        return cls.ue_cls.result_main_folder / 'ui_reg'
    
    class RegressionAcc():
        def __init__(self):
           pass
    
        def fitting(self, data):
            raise NotImplementedError()
        
        @classmethod
        def _get(cls, rga_spec, name):
            return rga_spec[name] if isinstance(rga_spec, dict) else getattr(rga_spec, name, None)
        
        @classmethod
        def _get_all(cls, rga_spec, *names):
            return [rga_spec[name] if isinstance(rga_spec, dict) else getattr(rga_spec, name, None)
                    for name in names]
    
    class PolynomialRegression(RegressionAcc):
        def __init__(self, degree=2, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.degree = degree
    
        def fitting(self, data, acc):
            # Polynomial regression (degree=2 or 3 depending on smoothness)
            # Prediction
            self.poly_model = Pipeline([
                ("poly", PolynomialFeatures(degree=self.degree, include_bias=False)),
                ("lin", LinearRegression())
            ])

            # logit transform of accuracy (avoid 0/1)
            eps = 1e-6
            y_logit = np.log((acc+eps) / (1 - acc + eps))

            self.poly_model.fit(data, y_logit)
            
            return self.poly_model
        
        @classmethod
        def predict_acc(self, X_new, poly_model):
            if poly_model is not None:
                logit_pred = poly_model.predict(X_new)
                return 1 / (1 + np.exp(-logit_pred))  # sigmoid
            else:
                raise ValueError("Model not fitted yet.")

        @classmethod
        def rga_folder_name(cls, rga_spec) -> str:
            # we do not encode the number of users nor the sample dataset used btw
            d = cls._get(rga_spec, 'degree')
            return f'rga_poly_{d}'

        @classmethod
        def parse_rga_folder_name(self, folder_name: str) -> dict | None:
            # use re to match
            pattern = r'rga_poly(\d+)'
            match = re.match(pattern, folder_name)
            # if match:
                
                
    @classmethod
    def _get_stuff(cls, ue_cls: type[UplinkExperiment20250911], batch_size: int, model_info: dict | tuple, shuffle: bool, device: str):
        """
        use ue_cls to get the related model
        returns:
            th: TestHelper20250426 instance
            edp: EDP instance for the model, made from th
            dataloader: InfiniteDataLoader instance for the model, made from th
        """
        
        model_info = ue_cls._normalize_model_info(model_info, tuple)
        channel_type, snr_db, model_type, encoder_out_dims, dataset_type, model_id = model_info
        
        th = ue_cls.TestHelper20250911(
            device=device,
            path=ue_cls._get_model_path(channel_type, snr_db, model_type, encoder_out_dims, dataset_type, model_id), 
            model_type=model_type, encoder_out_dims=encoder_out_dims, dataset_type=dataset_type,
            use_latest_checkpoint=False
        )
        sc = th.get_multimodal_sc()
        dataloader = th.get_val_dataloader(1, batch_size, False, shuffle=shuffle)

        return th, sc, dataloader
    
    @classmethod
    def _test(cls, ui_cls: UplinkInference, ui_kwargs: dict,
             ue_cls: UplinkExperiment20250911, model_infos: list[tuple] | list[dict],
             uplink_method: UplinkInference.SupportedUplinkMethods, n_user: int, snr_range: Optional[list[int]] = None,
             batch_size: int = 50, n_batch: int = 20, device: Optional[str] = 'cuda:0'
             ):
        """
        Test the given UplinkInference class with the given model info and method.

        Args:
            ui_cls: UplinkInference class to test
            ui_kwargs: kwargs to pass to the ui_cls constructor
                except the ones from UplinkInference: ['edps', 'power_constraint', 'channel', 'device']
            ue_cls: UplinkExperiment20250911 class to use for getting the models
            model_info: tuple or dict, model info for the first model, should be
                ('channel_type', snr_db, 'model_type', encoder_out_channels, 'dataset_type', model_id)
            uplink_method: the uplink method to use
                one of UplinkInference.SupportedUplinkMethods
            batch_size: batch size to use for the dataloaders, default is 50
            n_batch: number of batches to test, default is 20
            get_data_batch: the batch index to get the data from, default is 0
            device: the device to use for the ui_cls, default is 'cuda:0'
        
        Returns:
            test results of given snr_range, a dict with shape:
            {snr_db_1: {'awgn': accuracy, 'rayleigh': accuracy}, ...}
        
        Note:
            For model commutative, we assume that any uplink method is commutative, so this commutative property is only for the HUI
        """
        th_ls = []
        for model_info in model_infos:
            model_info = ue_cls._normalize_model_info(model_info, tuple)

            # get the necessary stuff
            th, sc, dataloader = cls._get_stuff(ue_cls, batch_size, model_info, shuffle=True, device=device)
            
            th_ls.append(th)

        partial_ui_cls = functools.partial(ui_cls, **ui_kwargs)
        
        md_if = list(model_infos[0])
        md_if.pop()
        channel_type = md_if[0]

        # check if the result file already exists
        result_path = (
            cls._result_root() / 
            ui_cls.ui_folder_name(ui_kwargs) / 
            uplink_method / 
            ue_cls._get_model_name(*md_if) /  
            'result.pkl'
        )
        
        all_snr_results = test_general(
            th_ls, result_path, uplink_method, 'val',
            batch_size, n_batch, snr_range,
            ui_cls=partial_ui_cls,
            n_user=n_user,
            channel_type=channel_type,
            divide_gain=True,
            save_results=False
        )
        
        return all_snr_results
        
    @classmethod
    def test_samples_fitting(cls, 
                             rg_cls:'RegressionMethod.RegressionAcc', rg_kwargs: dict,
                             ui_cls: UplinkInference, 
                             ui_kwargs: dict, 
                             uplink_method: str,
                             model_infos: list[dict] | list[tuple],  
                             m_ratios_ls: list[list[float]], 
                             snr_range: list[int],
                             device: Optional[str] = 'cuda:0'
                             ) -> Path:
        """
        Test the given UplinkInference class with the given model info and method.

        Args:
            rg_cls: RegressionMethod.RegressionAcc class to use for fitting model
            rg_kwargs: kwargs to pass to the rg_cls constructor
            ui_cls: UplinkInference class to test
            fsm_class: FSM class to use for feature selection
            shap_file_path: Path to the SHAP values file
            modalities: list of modalities to use, e.g., ['text', 'image', 'speech']
            model_info: tuple or dict, model info for the first model, should be
                ('channel_type', snr_db, 'model_type', encoder_out_channels, 'dataset_type', model_id)
            m_ratios_ls: list of list of float, each inner list is the m
                ratios to test for the corresponding modality in `modalities`
                e.g., if modalities = ['text', 'image'], m_ratios_ls could be
                [[0.1, 0.2, 0.4], [0.1, 0.3, 0.5]]
            snr_range: list of int, the snr range to test
            
        Return:
            regression result path: path of the regression model
        
        """
        
        test_settings = list(itertools.product(*m_ratios_ls))
        
        md_if = list(model_infos[0])
        md_if.pop()
        channel_type = md_if[0]
        
        reg_result_path = (
            cls._result_root() / 
            ui_cls.ui_folder_name(ui_kwargs) /
            uplink_method /  
            rg_cls.rga_folder_name(rg_kwargs) / 
            cls.ue_cls._get_model_name(*md_if) /
            'reg_model.pkl'
        )
        
        print(f"Regression Model path: {reg_result_path}")
        if reg_result_path.exists():
            print(f"[*] Result file {reg_result_path} already exists, skipping test.")
            return reg_result_path
        
        awgn_res_all_ls = []
        progress_bar = tqdm(enumerate(test_settings), desc="Sample fitting data")
        for idx, m_ratios in progress_bar:
            ui_kwargs.update({'m_ratios': m_ratios})
            results = cls._test(
                ui_cls, 
                ui_kwargs,
                cls.ue_cls, model_infos, uplink_method, n_user=len(m_ratios), batch_size=40, n_batch=20, device=device, snr_range=snr_range
            )
            awgn_res_ls = [v[channel_type] for k, v in results.items()]
            awgn_res_all_ls.append(awgn_res_ls)
        
        # print(str_type(awgn_res_all_ls))
        # reshape the data
        # res_all_ls: [n_ratio_combinations, n_snr]
        acc_ls = np.array(awgn_res_all_ls)  # shape (n_ratio_combinations, N_snr)
        # print(str_type(acc_ls))
        acc_ls = acc_ls / 100 # => test_uplink return in %, change to between (0, 1)
        rho_comb_ls = np.array(test_settings)  # shape (n_ratio_combinations, 3)
        snr_range = np.array(snr_range)  # shape (N_snr,)
        X, y = cls._pre_process_fitting_data(acc_ls, rho_comb_ls, snr_range)
        
        reg_result_path.parent.mkdir(parents=True, exist_ok=True)
        
        partial_rg_cls = functools.partial(rg_cls, **rg_kwargs)
        rg = partial_rg_cls()
        
        # perform fitting
        reg_model = rg.fitting(X, y)
        
        cls.save_model(reg_result_path, reg_model)
        
        return reg_result_path
        
        
    @classmethod
    def _pre_process_fitting_data(cls, acc_ls: np.ndarray, m_ratios_ls: np.ndarray, snr_range: np.ndarray):
        """
        pre-process the data for fitting
        Args:
            res_all_ls: list of list of float, shape [n_modal_combinations, n_snr]
            m_ratios_ls: list of list of float, shape [3, n_modalities]
            snr_range: list of int, the snr range used in the test. shape [n_snr,]
            
        Returns:
            X: np.ndarray, shape [n_samples, n_modalities + 1]
            y: np.ndarray, shape [n_samples,]
        """        
        
        N_rho, N_snr = acc_ls.shape
        print(f"[*] Pre-process fitting data: {N_rho} modality combinations, {N_snr} SNR values each.")

        rho_expanded = np.repeat(m_ratios_ls, N_snr, axis=0)          # (N_rho*N_snr, 3)
        snr_expanded = np.tile(snr_range, N_rho).reshape(-1,1)      # (N_rho*N_snr, 1)
        
        # combine features
        X = np.hstack([rho_expanded, snr_expanded])     # (N_rho*N_snr, 4)
        y = acc_ls.flatten()                            # (N_rho*N_snr,) 
        
        return X, y
    
    @classmethod
    def save_model(cls, path: Path, poly_model):
        """
        save the model to the given path
        """
        # save
        joblib.dump(poly_model, path)
    
    @classmethod
    def load_model(cls, path: Path):
        # load
        poly_model_loaded = joblib.load(path)
        return poly_model_loaded

class RegTestSuite:
    """
    Given a UI and RegressionAcc class, this class will test the class with different models and such

    we will use the following file tree:
    - cls._result_root()/
        - ui_reg/
            - ui_folder_name/uplink_method/rga_folder_name/model_name/reg_model.pkl
    """
    rm_cls = RegressionMethod
    rg_cls = RegressionMethod.PolynomialRegression
    ue_cls = rm_cls.ue_cls
    result_main_folder = rm_cls.ue_cls.result_main_folder
    
    @classmethod
    def test_rg(cls):
        """
            Set the test things here
        """
        
        sample_dataloader_batch_size = 100
        sample_channel_snr_dbs = np.arange(-6, 17, 2).tolist()  # [-6, -4, -2, 0, 2, 4, 6, 8, 10, 12, 14, 16]
        model_info = ('awgn', 12, 'udeepsc_msa', 48, 'cmu-mosei')
        shap_file_path = Path('./tmp/20250914/kernel_shap_5000_feature/awgn_12_udeepsc_msa_symbols_24_cmu-mosei/feat_contribs.npz')
        modalities = ['text', 'image', 'speech']
        m_ratios_ls = [[0.1, 0.2, 0.4, 1.0] for _ in range(len(modalities))] # [0.1, 0.2, 0.4, 0.6, 0.8, 1.0] for each modality
        # [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
        cls.rm_cls.test_samples_fitting(
            RegressionMethod.PolynomialRegression,
            {'degree': 2},
            UplinkInference_SHAPbasedSelection, 
            {'power_constraint': [1] * len(modalities)},
            FSM_Msa, shap_file_path,
            model_info, modalities, m_ratios_ls, sample_channel_snr_dbs
        )
    
    @classmethod
    def _test_reg_model(cls, ui_cls, ui_kwargs, 
                        model_info, uplink_method,
                        test_ratio_comb:list[list[float]],
                        snr_db:int = 0):
        reg = cls.rg_cls(degree=2)
        # method = 'signals'
        reg_model_path = (
            cls.rm_cls.result_main_folder / 
            ui_cls.ui_folder_name(ui_kwargs) /
            uplink_method /  
            cls.rg_cls.rga_folder_name({'degree': 2}) / 
            cls.ue_cls._get_model_name(*model_info) /
            'reg_model.pkl'
        )
        
        print(reg_model_path)
        
        result_path = (
            cls.rm_cls.result_main_folder / 
            ui_cls.ui_folder_name(ui_kwargs) /
            uplink_method /  
            cls.rg_cls.rga_folder_name({'degree': 2}) / 
            cls.ue_cls._get_model_name(*model_info) /
            f'reg_model_snr{snr_db}_acc.pkl'
        )
        
        print(f"Regression Model test path: {result_path}")
        if result_path.exists():
            print(f"[*] Result file {result_path} already exists, skipping test.")
            return result_path
        
        reg_model = cls.rm_cls.load_model(reg_model_path)
         
        acc_ls = []
        for m_ratios in test_ratio_comb:
            X = np.array([[*m_ratios, snr_db]])
            acc_pred = reg.predict_acc(X, reg_model)
            acc_ls.append(acc_pred * 100)
        # Example: rho = (0.5, 0.7, 0.3), snr = 10
        # X_new = np.array([[0.5, 0.7, 0.3, 10]])
        # acc_pred = reg.predict_acc(X_new, reg_model)

        # print(f'[*] Test Model fit accuracy: {acc_pred}')
        def save_metrics(metrics, path):
            import pickle
            with open(path, 'wb') as f:
                pickle.dump(metrics, f)
        
        save_metrics(acc_ls, result_path)
        
    
    @classmethod
    def test(cls):
        # config must be injected in __main__ (see step 4)
        config = getattr(cls, 'config', None)
        if config is None:
            raise RuntimeError("RegTestSuite.config was not set. Set it before calling test().")

        mi = config['model_info']
        model_type = mi['model_type']
        modalities = config['modalities']
        fsm_cls = _fsm_for(model_type)
            
        ui_cls = UplinkInference_SHAPbasedSelection
        snr = 10
        method = 'features'
        test_ratios = [np.arange(0.1, 1.1, 0.1).round(1).tolist() for _ in range(len(modalities))]
        test_ratio_comb = list(itertools.product(*test_ratios))
        
        # use the config model with a single model_id for this check
        model_info = (*_model_info_noid_from_config(config), 1)
        
        def test_reg_model():
            # model_info = ('awgn', 12, 'udeepsc_msa', 48, 'msa', 1)
            ui_kwargs = {
                'modalities': modalities,
                # no shap path needed for folder naming;
                'fsm_class': fsm_cls
            }
            cls._test_reg_model(
                ui_cls, 
                ui_kwargs, 
                model_info, method, test_ratio_comb, snr
            )
            
        test_reg_model()

class Optimization:
    class Optimization_Masking():
        def __init__(self, model, n_mods: int, rho_bounds: list[Tuple], rho_budget=None):
            """
            Args:
                model: Fitted regression model. It maps [rho1, rho2, rho3, snr]-> logit(accuracy). 
                rho_bounds: list of (low, high) per modality; default (0,1). Must have length n_mods
                rho_budget (optional, default = None): A scalar upper 
                        bound for the sum of masking ratios 

            Note:
                The masking ratio rho will be bounded [0, 1]           
                
            """
            if len(rho_bounds) != n_mods:
                raise ValueError(f"Number of rho_bounds should be the same as n_mods, but got {rho_bounds=}, {n_mods=}")
            
            self.model = model
            self.n_mods = n_mods
            self.rho_bounds = rho_bounds
            self.rho_budget = rho_budget
        
        def _logit_pred(self, rho, snr_val):
            raise NotImplementedError

        def _acc_pred(self,rho, snr_val):
            raise NotImplementedError

        def _optimize_rho(self, snr_db: float):
            raise NotImplementedError
        
        def get_optimize_rho(snr_db: float):
            raise NotImplementedError
        
        @classmethod
        def opt_folder_name(cls, opt_spec: dict=None) -> str:
            """
            Return the name for this HeteroUplinkInference instance, which will be used as a folder name later
            to store related results of the Optimization method implemented by this OPT
            this will act as a folder name
            """
            return f'opt_mask'
        
        @classmethod
        def parse_opt_folder_name(self, folder_name: str) -> dict | None:
            """
            Parse the folder name
            if this matches with the folder name of this class, return a dict with the specific settings of this IM method
            if this folder name isn't generated by this class ('s opt_folder_name), return None
            """
            return {} if folder_name == 'opt_mask' else None
    
    class Optimization_PolyRegression(Optimization_Masking):
        # poly_model: X -> y_logit (PolynomialFeatures + LinearRegression)
        # X = [rho1, rho2, rho3, snr]
        
        # result
        
        def __init__(self, n_starts=20, random_state=0, *args, **kwargs):
            """
            Args:
                n_starts (default = 20): Number of random initial 
                    guesses for the optimizer (SLSQP).
                        Since the fitted polynomial can be non-convex, starting from multiple points reduces the chance of getting stuck in a bad local optimum.
                random_state: (default = 0) Seed for the random number 
                        generator to make the random initial guesses reproducible.
            """
            super().__init__(*args, **kwargs)
            self.n_starts = n_starts
            self.random_state = random_state
        
        def _logit_pred(self,  rho, snr_val):
            # X = np.array([[rho[0], rho[1], rho[2], snr_val]])
            X = np.hstack([rho, [snr_val]]).reshape(1, -1)
            return self.model.predict(X)[0]  # scalar logit

        def _acc_pred(self, rho, snr_val):
            z = self._logit_pred(self.model, rho, snr_val)
            return 1.0 / (1.0 + np.exp(-z))

        def _optimize_rho(self, snr_db: float):
            """
            Args:
                poly_model: Fitted regression model (the Pipeline we   
                        built: PolynomialFeatures + LinearRegression). It maps [rho1, rho2, rho3, snr]-> logit(accuracy). ref. PolynomialRegression.fitting()
                snr_db: A scalar (float). The fixed SNR value 
                        Example: snr_db = 10.0.
                rho_bounds: list of (low, high) per modality; default (0,1). Must have length n_mods
                
                rho_budget (optional, default = None): A scalar upper 
                        bound for the sum of masking ratios
                n_starts (default = 20): Number of random initial 
                    guesses for the optimizer (SLSQP).
                        Since the fitted polynomial can be non-convex, starting from multiple points reduces the chance of getting stuck in a bad local optimum.
                random_state: (default = 0) Seed for the random number 
                        generator to make the random initial guesses reproducible. 
                        
            Note:
                The masking ratio rho will be bounded [0, 1]           
                
            """
            
            rng = np.random.default_rng(self.random_state)
            # masking ratio bounded
            lb = np.array([b[0] for b in self.rho_bounds], dtype=float)
            ub = np.array([b[1] for b in self.rho_bounds], dtype=float)
            bounds = Bounds(lb, ub) 

            # Optional linear budget: rho1 + rho2 + rho3 <= rho_budget
            lin_constr = None
            if self.rho_budget is not None:
                A = np.array([[1.0, 1.0, 1.0]])
                lb = -np.inf
                ub = self.rho_budget
                lin_constr = LinearConstraint(A, lb, ub)

            def obj(rho):
                # Minimize negative logit to maximize logit
                return -self._logit_pred(rho, snr_db)
            
            res = shgo(
                obj,
                bounds=bounds,
                sampling_method="sobol"   # deterministic
            )

            if res.success and res.x is not None:
                best_x = res.x
                best_f = res.fun
            else:
                best_x = None

            # ---------- Fallback: coarse global grid ----------
            if best_x is None:
                grid = np.linspace(lb, ub, 11)
                best_val, best_rho = -np.inf, None

                for indices in np.ndindex(*(11,) * self.n_mods):
                    rho = np.array([grid[d][i] for d, i in enumerate(indices)], float)
                    z = self._logit_pred(rho, snr_db)

                    if z > best_val:
                        best_val, best_rho = z, rho

                best_x = best_rho

            # ---------- Final prediction ----------
            z_star = self._logit_pred(best_x, snr_db)
            acc_star = 1.0 / (1.0 + np.exp(-z_star))

            return best_x, acc_star, z_star
        
        def get_optimize_rho(self, snr):
            opt_rho, opt_acc, _ = self._optimize_rho(snr)
            return opt_rho
        
        @classmethod
        def opt_folder_name(cls, opt_spec: dict=None) -> str:
            return f'opt_mask_poly'
        
        @classmethod
        def parse_opt_folder_name(self, folder_name: str) -> dict | None:
            return {} if folder_name == 'opt_mask' else None
        
def test_general_opt(
        th_ls: list[TestHelper],
        opt_m: list[Optimization.Optimization_Masking],
        result_path: Path,
        uplink_method: UplinkInference.SupportedUplinkMethods = 'features',
        test_datatype: Literal['test', 'val'] = 'test',
        batch_size: int = 20, n_batch: int = 50,
        snr_range: list[int] = np.linspace(-5, 25, 13).tolist(),
        ui_cls: Callable = UplinkInference,
        n_user: int = 3,
        channel_type: Literal['awgn', 'rayleigh'] = 
        None,
        save_results: bool = True
        ):
    """

        Args:
            th: TestHelper
            result_path: the path to save the metrics, in pickle format
            uplink_method: the uplink method for UplinkInference, default is 'features'
            batch_size: the batch size for dataloader
            n_batch: the number of batch for the tests
            snr_range: the channel SNR range to test, in dB, should be increasing
            ui_cls: the class to use for UplinkInference, default is UplinkInference
                (technically we only use this as a constructor, so you can pass a factory function in here or something
                 we will pass in arguments ['edps', 'power_constraint', 'channel', 'device'])
        Returns:
            None, but the metrics will be saved to result_path in pickle format
            the metrics will be indexed like this:
                metrics[snr][channel][user][metric_type] = list of metric_values for
    """
    
    n_th = len(th_ls)
    
    if test_datatype == 'test':
        th_dataloader = th_ls[0].get_dataloader(1, batch_size, True, shuffle=False)
    elif test_datatype == 'val':
        th_dataloader = th_ls[0].get_val_dataloader(1, batch_size, False, shuffle=True)
    else:
        raise ValueError(f"Unknown test_datatype {test_datatype}, should be 'test' or 'val'")

    def avg_metric(metrics: list[dict]) -> dict:
        """
        Average the results of multiple models with different seeds
            Args:
                metrics: list of metric, e.g., from test_all_snr
                and is indexed like this
                metrics[i][snr][channel] = accuracy
        """
        
        n = len(metrics)
        snr_range = list(sorted(metrics[0].keys()))
        
        stat = {
            snr:{
                channel: sum(mt[snr][channel] for mt in metrics) / n
                for channel in metrics[0][snr].keys()
            }
            for snr in snr_range
        }
        
        return stat
    
    def test_snr(n_user, sc, dataloader, n_batch, th, snr_db, rhos, ch_type):
        """
            test the snr for the given edp and dataloader
            n_user: number of users
            edp_ls: list of edp
            dataloader: a multiuser dataloader
            snr: snr in dB
            rhos: masking ratio for ui_cls
        """

        ret = {}

        if ch_type == 'awgn':
            awgn_channel = AWGNMultiUplinkChannel(n_user=n_user, snr_db=[snr_db], interfere_mode='all')
            # ui_fs_cls = functools.partial(ui_cls, m_ratios=rhos)
            ui = ui_cls(m_ratios=rhos, model=sc, channel=awgn_channel, device=th.device)
            ret['awgn'] = ui.test_uplink(dataloader, uplink_method, n_batch=n_batch)
            # print(f"Feature Mask : {ui.get_masks()}")
        elif ch_type == 'rayleigh': 
            rayleigh_channel = RayleighFadingMultiUplinkChannel(n_user=n_user, snr_db=[snr_db], channel_gain_var=[1], 
                divide_gain=True,
                noise_power_density_dBm=-90,    # ref. ISSNOMATrainer's note
                reference_distance=1,
                reference_path_loss=pow(10, -30/10),
                path_loss_exponent=4,
                distance=torch.Tensor([33, 83, 133]).reshape(3, 1),
                fading_mode='slow')
            ui = ui_cls(m_ratios=rhos, model=sc, channel=rayleigh_channel, device=th.device)
            ret['rayleigh'] = ui.test_uplink(dataloader, uplink_method, n_batch=n_batch)
        else:
            raise ValueError(f"Unknown channel type{ch_type}, should be 'awgn' or 'rayleigh")
        
        return ret

    def test_all_snr(*args, **kwargs):
        metric_ls = []
        ratios_all = {}
        for i, th in enumerate(th_ls):
            th_sc = th.get_multimodal_sc() 
            metric = {}
            ratios_all[i] = {}
            for snr in tqdm(snr_range, desc="Testing SNRs", leave=False):
                # make the MultimodalSC
                mk_rhos = opt_m.get_optimize_rho(snr)
                acc = test_snr(*args, **kwargs, sc=th_sc, th = th, snr_db=snr, rhos=mk_rhos)
                metric[snr] = acc
                ratios_all[i][snr] = list(mk_rhos)
            
            metric_ls.append(metric)
            
        metrics = avg_metric(metric_ls)
         
        return metrics, ratios_all

    def save_metrics(metrics, path):
        import pickle
        with open(path, 'wb') as f:
            pickle.dump(metrics, f)
            
    def save_json(metrics, path):
        import json
        # Save to file
        with open(path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=4, ensure_ascii=False)

    metrics, ratios_all = test_all_snr(n_user=n_user, dataloader=th_dataloader, n_batch=n_batch, ch_type=channel_type)
    
    if save_results:
        save_metrics(metrics, result_path)
        save_json(ratios_all, result_path.parent / 'opt_ratios.json')
    
    return metrics


        
class OptimizeTestSuit:
    """
    Given a UplinkInference class and feature importance path, this class will do regression then optimization.
    
    Note: Before do test here, ensure feature importance has been evaluated, e.g., Shapley values have been computed and saved.

    we will use the following file tree:
    - cls._result_root()/
        - ui_reg/
            - ui_folder_name/uplink_method/rga_folder_name/model_name/reg_model.pkl
        - opt_test/
            - ui_folder_name/uplink_method/opt_folder_name/model_name/result.pkl
        - result/
            - ui_folder_name/uplink_method/model_name/<PLOT_RELATED_FILES>
    """
    rm_cls = RegressionMethod
    ue_cls = UplinkExperiment20250914

    @classmethod
    def _result_root(cls) -> Path:
        return cls.ue_cls.result_main_folder / 'opt_test'
    
    @classmethod
    def _get_stuff(cls, ue_cls: type[UplinkExperiment20250911], batch_size: int, model_info: dict | tuple, shuffle: bool, device: str):
        """
        use ue_cls to get the related model
        returns:
            th: TestHelper20250426 instance
            edp: EDP instance for the model, made from th
            dataloader: InfiniteDataLoader instance for the model, made from th
        """
        
        model_info = ue_cls._normalize_model_info(model_info, tuple)
        channel_type, snr_db, model_type, encoder_out_dims, dataset_type, model_id = model_info
        
        th = ue_cls.TestHelper20250911(
            device=device,
            path=ue_cls._get_model_path(channel_type, snr_db, model_type, encoder_out_dims, dataset_type, model_id), 
            model_type=model_type, encoder_out_dims=encoder_out_dims, channel_type=channel_type, dataset_type=dataset_type,
            use_latest_checkpoint=False
        )
        sc = th.get_multimodal_sc()
        dataloader = th.get_dataloader(1, batch_size, False, shuffle=shuffle)

        return th, sc, dataloader

    @classmethod
    def _test(cls, ui_cls: UplinkInference, ui_kwargs: dict,
             opt_cls: Optimization, opt_kwargs: dict,
             ue_cls: UplinkExperiment20250911, 
             model_infos: list[tuple | dict],
             reg_model,
             uplink_method: UplinkInference.SupportedUplinkMethods, snr_range: Optional[list[int]] = None,
             batch_size: int = 50, n_batch: int = 20, device: Optional[str] = 'cuda:0'
             ):
        """
        Test the given UplinkInference class with the given model info and method.

        Args:
            ui_cls: UplinkInference class to test
            ui_kwargs: kwargs to pass to the ui_cls constructor
                except the ones from UplinkInference: ['edps', 'power_constraint', 'channel', 'device']
            ue_cls: UplinkExperiment20250911 class to use for getting the models
            model_info: tuple or dict, model info for the first model, should be
                ('channel_type', snr_db, 'model_type', encoder_out_channels, 'dataset_type', model_id)
            uplink_method: the uplink method to use
                one of UplinkInference.SupportedUplinkMethods
            batch_size: batch size to use for the dataloaders, default is 50
            n_batch: number of batches to test, default is 20
            get_data_batch: the batch index to get the data from, default is 0
            device: the device to use for the ui_cls, default is 'cuda:0'
        
        Note:
            For model commutative, we assume that any uplink method is commutative, so this commutative property is only for the HUI
        """
        th_ls = []
        md_if = list(model_infos[0])
        md_if.pop()
        channel_type = md_if[0]
        
        for model_info in model_infos:
            model_info = ue_cls._normalize_model_info(model_info, tuple)

            # get the necessary stuff
            th, sc, dataloader = cls._get_stuff(ue_cls, batch_size, model_info, shuffle=True, device=device)
            
            th_ls.append(th)

        partial_ui_cls = functools.partial(ui_cls, **ui_kwargs)

        # check if the result file already exists
        result_path = (
            cls._result_root() / 
            ui_cls.ui_folder_name(ui_kwargs) / 
            uplink_method /
            opt_cls.opt_folder_name(opt_kwargs)/
            ue_cls._get_model_name(*md_if) /  
            'result.pkl'
        )

        if result_path.exists():
            print(f"[*] Result file {result_path} already exists, skipping test.")
            return result_path
        
        result_path.parent.mkdir(parents=True, exist_ok=True)
        print(result_path)
        
        partial_opt_cls = functools.partial(opt_cls, **opt_kwargs)

        # opt_ms = [partial_opt_cls(model=reg_model) for reg_model in reg_models]
        
        opt_ms = partial_opt_cls(model=reg_model)
        
        test_general_opt(
            th_ls, opt_ms, result_path, uplink_method,
            batch_size=batch_size, n_batch=n_batch, 
            snr_range=snr_range,
            n_user=len(ui_kwargs['modalities']),
            channel_type=channel_type,
            ui_cls=partial_ui_cls
        )
    
    @classmethod
    def _test_rg(cls, sample_snrs, model_infos, modalities, method, ui_cls, ui_kwargs, device: Optional[str] = 'cuda:0'):
        """
            Set the test things here
        """
        m_ratios_ls = [[0.1, 0.2, 0.4, 0.6, 0.8, 1.0] for _ in range(len(modalities))]
        # m_ratios_ls = [np.arange(0.1, 1.1, 0.1).round(1).tolist() for _ in range(len(modalities))]
        return cls.rm_cls.test_samples_fitting(
            RegressionMethod.PolynomialRegression,
            {'degree': 2},
            ui_cls, ui_kwargs, method, model_infos, m_ratios_ls, sample_snrs, device
        )
    
    @classmethod
    def _test_time(cls, n_user, sc, dataloader, n_batch, th, snr_db, 
        uplink_method: UplinkInference.SupportedUplinkMethods = 'features', 
        ui_cls: Callable = UplinkInference
        ):
        awgn_channel = AWGNMultiUplinkChannel(n_user=n_user, snr_db=[snr_db], interfere_mode='all')
        ui = ui_cls(model=sc, power_constraint=[1] * n_user, channel=awgn_channel, device=th.device)
        
        ret = ui.test_uplink(dataloader, uplink_method, n_batch=n_batch)
        
        return ret
    
    @classmethod
    def test(cls):
        test_settings = []
        
        
        config = getattr(cls, 'config', None)
        if config is None:
            raise RuntimeError("OptimizeTestSuit.config was not set. Set it before calling test().")
        
        device = f"cuda:{config['gpu'][0]}"

        mi = config['model_info']
        model_type = mi['model_type']
        modalities = config['modalities']
        fsm_cls = _fsm_for(model_type)
        
        method = 'features'
        # method = 'signals'
        model_infos = _model_infos_from_config(config)
        shap_file_path = _resolve_shap_path(config, cls.ue_cls, method)
        # shap_file_paths = _get_shap_paths(config, cls.ue_cls)
        
        # SNR grid for sample generation and for final test
        sample_snr_dbs = np.arange(-6, 17, 2).tolist()
        snr_range = np.linspace(-6, 12, 10).tolist()
        
        def test_opt_method_SLSQP():
            reg = RegressionMethod.PolynomialRegression(degree=2)
            reg_model_path = (
                cls.rm_cls.result_main_folder /
                reg.rga_folder_name({'degree': 2}) /
                method /
                'reg_model.pkl'
            )
            
            print(reg_model_path)
            
            reg_model = cls.rm_cls.load_model(reg_model_path)
            random_seed = 1000
            n_guess = 30
            n_mods = len(modalities)
            
            opt_m = Optimization.Optimization_PolyRegression(
                                model=reg_model, n_mods=n_mods,
                                rho_bounds=[(0.05, 1)] * n_mods,
                                n_starts=n_guess, random_state=random_seed
                                )
            
            rho_opt, acc_opt, _ = opt_m._optimize_rho(snr_db=12)

            print("Optimal rho:", rho_opt)
            print("Predicted accuracy:", acc_opt)
            
            opt_kwargs = {
                'n_mods': n_mods,
                'rho_bounds': [(0.05, 1.0)] * n_mods,
                'n_starts': n_guess,
                'random_state': random_seed,
            }
            
            print("======Test all SNRs========")
            return cls._test(
                UplinkInference_SHAPbasedSelection, 
                {
                    'shap_val_path': shap_file_path, 
                    'modalities': modalities, 
                    'fsm_class': FSM_Msa
                },
                Optimization.Optimization_PolyRegression,
                opt_kwargs,
                cls.ue_cls, model_infos, method, batch_size=50, n_batch=20, device='cuda:0', snr_range=snr_range
            )
            
        
        def test_shap_regression_and_opt_method():
            ui_cls = UplinkInference_SHAPbasedSelection
            ui_kwargs = {
                'shap_val_path': shap_file_path,
                'modalities': modalities,
                'fsm_class': fsm_cls,
                'power_constraint': [1] * len(modalities),
            }
            
            reg_ls = []
            reg_ls.extend(dp_delayed(cls._test_rg)(
                sample_snr_dbs, 
                mi, 
                modalities, method, ui_cls, 
                ui_kwargs
                ) for mi in model_infos)
            

            # aw = MonitorFileActivityWatcher(Path('./mon.txt'), show_queue=True, name=f'OptimizeTestSuit.test_shap_regression_and_opt_method()')
            # reg_model_paths = DeviceParallel(devices=['cuda:0', 'cuda:1'], activity_callback=aw, exception_blocking=False)(reg_ls)
            
            reg_model_path = cls._test_rg(sample_snr_dbs, model_infos, modalities, method, ui_cls, ui_kwargs)
              
            print(reg_model_path)
            
            # reg_models = [cls.rm_cls.load_model(p) for p in reg_model_paths]
            
            reg_model = cls.rm_cls.load_model(reg_model_path)
            
            # opt setting
            random_seed = 1000
            n_guess = 30
            n_mods = len(modalities)
            
            # optimization settings
            opt_kwargs = {
                'n_mods': n_mods,
                'rho_bounds': [(0.05, 1.0)] * n_mods,
                'n_starts': n_guess,
                'random_state': random_seed,
            }
            
            print("======Test all SNRs========")
            return cls._test(
                ui_cls, ui_kwargs,
                Optimization.Optimization_PolyRegression, opt_kwargs,
                cls.ue_cls, model_infos, reg_model,
                method, batch_size=30, n_batch=None, snr_range=snr_range, device=device
            )
        
        def test_random_regression_and_opt_method():
            ui_cls = UplinkInference_RandomSelection
            ui_kwargs = {
                'modalities': modalities,
                'fsm_class': FSM_Msa_random,
                'power_constraint': [1] * len(modalities),
            }
            
            
            reg_model_path = cls._test_rg(sample_snr_dbs, model_infos, modalities, method, ui_cls, ui_kwargs)
              
            print(reg_model_path)
            
            # reg_models = [cls.rm_cls.load_model(p) for p in reg_model_paths]
            
            reg_model = cls.rm_cls.load_model(reg_model_path)
            
            # opt setting
            random_seed = 1000
            n_guess = 30
            n_mods = len(modalities)
            
            # optimization settings
            opt_kwargs = {
                'n_mods': n_mods,
                'rho_bounds': [(0.5, 1.0)] * n_mods,
                'n_starts': n_guess,
                'random_state': random_seed,
            }
            
            print("======Test all SNRs========")
            return cls._test(
                ui_cls, ui_kwargs,
                Optimization.Optimization_PolyRegression, opt_kwargs,
                cls.ue_cls, model_infos, reg_model,
                method, batch_size=30, n_batch=None, snr_range=snr_range, device=device
            )
        
        
        def test_delay_time(): 
            model_info = ('awgn', 12, 'udeepsc_msa', 48, 'cmu-mosei', 1)
            sic_model_info = ('awgn', 12, 'udeepscSIC_msa', 48, 'cmu-mosei', 1)
            rg_cls = RegressionMethod.PolynomialRegression
            ui_cls = UplinkInference_SHAPbasedSelection
            
            th, sc, dataloader = cls._get_stuff(cls.ue_cls, model_info=model_info, shuffle=True, batch_size=10, device='cuda:1')
            
            th2, sic_sc, _ = cls._get_stuff(cls.ue_cls, model_info=sic_model_info, shuffle=True, batch_size=10, device='cuda:1')
            
            md_if = list(model_info)
            md_if.pop()
            
            reg_result_path = (
                cls.rm_cls._result_root() / 
                'ui_fs_shap_TIS' /
                method /  
                rg_cls.rga_folder_name({'degree':2}) / 
                cls.ue_cls._get_model_name(*md_if) /
                'reg_model.pkl'
            ) 
            reg_model = cls.rm_cls.load_model(reg_result_path)
            # opt setting
            random_seed = 1000
            n_guess = 30
            n_mods = len(modalities)
            opt_m = Optimization.Optimization_PolyRegression(
                model=reg_model, n_mods=n_mods, 
                random_state=random_seed, n_starts=n_guess, 
                rho_bounds=[(0.05, 1)] * n_mods, 
            )
            
            snr_db = 10
            n_user =  3
            
            mk_rhos = opt_m.get_optimize_rho(snr_db)
            opt_partial_ui_cls = functools.partial(
                ui_cls, m_ratios=mk_rhos, shap_val_path=shap_file_path,
                modalities=modalities,
                fsm_class=FSM_Msa
                )
            
            result_path = (
                cls.ue_cls.result_main_folder /
                'experiment 4' /
                'msa' /
                'E2E_delay.json'
            )
            
            if result_path.exists():
                print(f"[*] Result file {result_path} already exists, skipping test.")
                return result_path
            
            result_path.parent.mkdir(parents=True, exist_ok=True)
        
            
            from timeit import default_timer, timeit
            
            ret = {}
            partial_test_func = functools.partial(cls._test_time, n_user, sic_sc, dataloader, th=th2, snr_db=snr_db, uplink_method='no_FSM_sic', n_batch=1, ui_cls=UplinkInference)
            ret['SIC'] = timeit(partial_test_func, number=100) / 100
            
            partial_test_func2 = functools.partial(cls._test_time, n_user, sc, dataloader, th=th, snr_db=snr_db, uplink_method='no_FSM', n_batch=1, ui_cls=UplinkInference)
            ret['noSIC'] = timeit(partial_test_func2, number=100) / 100
            
            partial_test_func3 = functools.partial(cls._test_time, n_user, sc, dataloader, th=th, snr_db=snr_db, uplink_method=method, n_batch=1, ui_cls=opt_partial_ui_cls)
            ret['FSM'] = timeit(partial_test_func3, number=100) / 100
            
            
            def save_json(metrics, path):
                import json
                # Save to file
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(metrics, f, indent=4, ensure_ascii=False)
            
            save_json(ret, result_path)
                   
        # test_opt_method_SLSQP()
        test_shap_regression_and_opt_method()
        if config['task'] == 'msa':
            test_random_regression_and_opt_method()
            test_delay_time()
        
class TempPlotting:
    """
    Temporary plotting, mostly for generating images for PPTs
    """
    rm_cls = RegressionMethod
    rg_cls = RegressionMethod.PolynomialRegression
    ue_cls = UplinkExperiment20250914
    
    @classmethod
    def _result_root(cls) -> Path:
        return UITestSuite.result_main_folder

    @classmethod
    def _opt_result_root(cls) -> Path:
        return OptimizeTestSuit._result_root()
    
    # result_plot_folder = cls._result_root() / '..'
    
    @classmethod
    def plots(cls):
        config = getattr(cls, 'config', None)
        if config is None:
            raise RuntimeError("OptimizeTestSuit.config was not set. Set it before calling test().")
        
        mi = config["model_info"]
        channel = mi['channel_type']
        
        def opt_plots():
            model_name = f"{channel}_{mi['snr_db']}_udeepsc_msa_symbols_{mi['encoder_out_dims'] // 2}_{mi['dataset_type']}"
            oma_model_name = f"{channel}_{mi['snr_db']}_udeepscOMA_msa_symbols_{mi['encoder_out_dims_OMA'] // 2}_{mi['dataset_type']}"
            sic_model_name = f"{channel}_{mi['snr_db']}_udeepscSIC_msa_symbols_{mi['encoder_out_dims'] // 2}_{mi['dataset_type']}"
            uplinkMethod = config["uplinkMethod"]
            result_path = cls._result_root() / '..' / 'experiment 1' / uplinkMethod / model_name
            print(result_path)
            
            nofsm = cls._result_root() / 'ui' / 'no_FSM' / model_name / 'result.pkl'
            fsm_poly_feat = cls._opt_result_root() / 'ui_fs_shap_TIS' / uplinkMethod / 'opt_mask_poly' / model_name / 'result.pkl'
            
            fsm_poly_signal = cls._opt_result_root() / 'ui_fs_shap_TIS' / 'signals' / 'opt_mask_poly' / model_name / 'result.pkl'
            
            oma_model = cls._result_root() / 'ui' / 'no_FSM_oma' / oma_model_name / 'result.pkl'
            
            sic_model = cls._result_root() / 'ui' / 'no_FSM_sic' / sic_model_name / 'result.pkl'
            
            # print(nofsm)
            # print(fsm_poly)
            
            result_path.mkdir(parents=True, exist_ok=True)
            
            plot_general(
                result_path, 'Accuracy (%)', 'acc',
                [oma_model, sic_model, nofsm, fsm_poly_feat], 
                [
                    {'marker': 'D', 'linestyle': ':', 'alpha': 0.6, 'color': 'tab:blue', 'label': 'OMA'},
                    {'marker': '*', 'linestyle': '--', 'alpha': 0.6, 'color': 'tab:green', 'label': 'w/ Signal Detection'},
                    {'marker': 'v', 'linestyle': '--', 'alpha': 0.8, 'color': 'tab:orange', 'label': 'w/o Signal Detection'},
                    {'marker': 'o', 'linestyle': '-', 'alpha': 0.8, 'color': 'tab:red', 'label': 'w/o Signal Detection + FSM'},
                ], legend_kwargs={'loc': 'outside upper center', 'ncols': 2}, task_type='msa', channel_type=channel, y_limited=[55, 85, 7]
            )
        
        def FS_method_plots():
            model_name = f"{channel}_{mi['snr_db']}_udeepsc_msa_symbols_{mi['encoder_out_dims'] // 2}_{mi['dataset_type']}"
            uplinkMethod = config["uplinkMethod"]
            result_path = cls._result_root() / '..' / 'experiment 2' / uplinkMethod / model_name
            print(result_path)
            
            nofsm = cls._result_root() / 'ui' / 'no_FSM' / model_name / 'result.pkl'
            fs_shap = cls._opt_result_root()/ 'ui_fs_shap_TIS' / uplinkMethod / 'opt_mask_poly' / model_name / 'result.pkl'
            
            fs_random= cls._opt_result_root()/ 'ui_fs_rand_TIS' / uplinkMethod / 'opt_mask_poly' / model_name / 'result.pkl'
            
            # print(nofsm)
            # print(fsm_poly)
            
            result_path.mkdir(parents=True, exist_ok=True)
            
            plot_general(
                result_path, 'Accuracy (%)', 'acc',
                [nofsm, fs_shap, fs_random], 
                [
                    {'marker': 'v', 'linestyle': '--', 'alpha': 0.6, 'color': 'tab:orange', 'label': 'w/o Signal Detection'},
                    {'marker': 'o', 'linestyle': '-', 'alpha': 0.8, 'color': 'tab:red', 'label': 'w/o Signal Detection + FSM (SHAP selection)'},
                    {'marker': 'o', 'linestyle': '-', 'alpha': 0.8, 'color': 'tab:olive', 'label': 'w/o Signal Detection + FSM (random selection)'},
                ], legend_kwargs={'loc': 'outside upper center', 'ncols': 1}, task_type='msa', channel_type=channel, y_limited=[55, 85, 7]
            )
        
        def only2mod_opt_plots():
            model_name = f"awgn_{mi['snr_db']}_udeepsc_msa_symbols_{mi['encoder_out_dims'] // 2}_{mi['dataset_type']}"
            uplinkMethod = 'features'
            result_path = cls._result_root() / '..' / 'experiment 5' / uplinkMethod / model_name
            print(result_path)
            
            nofsm = cls._result_root() / 'ui' / 'no_FSM' / model_name / 'result.pkl'
            fsm_poly_feat = cls._opt_result_root()/ 'ui_fs_shap_TIS' / uplinkMethod / 'opt_mask_poly' / model_name / 'result.pkl'
            
            nofsm_noSpe = cls._result_root() / 'ui' / 'no_FSM' / model_name / 'result_noSpe.pkl'
            
            # print(nofsm)
            # print(fsm_poly)
            
            result_path.mkdir(parents=True, exist_ok=True)
            
            plot_general(
                result_path, 'Accuracy (%)', 'acc',
                [nofsm, fsm_poly_feat, nofsm_noSpe], 
                [                
                    {'marker': 'v', 'linestyle': '--', 'alpha': 0.6, 'color': 'tab:orange', 'label': 'w/o Signal Detection'},
                    {'marker': 'o', 'linestyle': '-', 'alpha': 0.8, 'color': 'tab:red', 'label': 'w/o Signal Detection + FSM'},
                    {'marker': '^', 'linestyle': ':', 'alpha': 0.8, 'color': 'tab:green', 'label': 'w/ Signal Detection (No Speech)'},
                ], legend_kwargs={'loc': 'outside upper center', 'ncols': 2}, task_type='msa', channel_type=channel, y_limited=[55, 85, 7]
            )
    
        def ave_opt_plots():
            model_name = f"awgn_{mi['snr_db']}_udeepsc_ave_symbols_{mi['encoder_out_dims'] // 2}_ave"
            oma_model_name = f"awgn_{mi['snr_db']}_udeepscOMA_ave_symbols_{mi['encoder_out_dims_OMA'] // 2}_ave"
            sic_model_name = f"awgn_{mi['snr_db']}_udeepscSIC_ave_symbols_{mi['encoder_out_dims'] // 2}_ave"
            
            uplinkMethod = config["uplinkMethod"]
            result_path = cls._result_root() / '..' / 'experiment 3' / uplinkMethod / model_name
            print(result_path)
            
            nofsm = cls._result_root() / 'ui' / 'no_FSM' / model_name / 'result.pkl'
            fsm_poly_feat = cls._opt_result_root() / 'ui_fs_shap_IS' / uplinkMethod / 'opt_mask_poly' / model_name / 'result.pkl'
            
            fsm_poly_signal = cls._opt_result_root() / 'ui_fs_shap_IS' / 'signals' / 'opt_mask_poly' / model_name / 'result.pkl'
            
            oma_model = cls._result_root() / 'ui' / 'no_FSM_oma' / oma_model_name / 'result.pkl'
            
            sic_model = cls._result_root() / 'ui' / 'no_FSM_sic' / sic_model_name / 'result.pkl'
            
            # print(nofsm)
            # print(fsm_poly)
            
            result_path.mkdir(parents=True, exist_ok=True)
            
            plot_general(
                result_path, 'Accuracy (%)', 'acc',
                [oma_model, sic_model, nofsm, fsm_poly_feat], 
                [
                    {'marker': 'D', 'linestyle': ':', 'alpha': 0.6, 'color': 'tab:blue', 'label': 'OMA'},
                    {'marker': '*', 'linestyle': '--', 'alpha': 0.6, 'color': 'tab:green', 'label': 'w/ Signal Detection'},
                    {'marker': 'v', 'linestyle': '--', 'alpha': 0.8, 'color': 'tab:orange', 'label': 'w/o Signal Detection'},
                    {'marker': 'o', 'linestyle': '-', 'alpha': 0.8, 'color': 'tab:red', 'label': 'w/o Signal Detection + FSM'},
                ], legend_kwargs={'loc': 'outside upper center', 'ncols': 2},
                task_type='ave', channel_type=channel
            )
        
        def plot_comm_time():
            result_path = cls._result_root() / '..' / 'experiment 4' / 'msa'
            print(result_path)
            
            def load_json(path):
                import json
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            
            test_time_path = (
                    cls.ue_cls.result_main_folder /
                    'experiment 4' /
                    'msa' /
                    'E2E_delay.json'
            )
            
            test_time = load_json(test_time_path)
            time_ls = [v for k, v in test_time.items()]
            time_ms_ls = [t * 1000 for t in time_ls]
            percent_increase = [(t - time_ls[0]) * 100 / time_ls[0] for t in time_ls]
            
            schemes = ['w/ \nSignal Detection', 'w/o \nSignal Detection', 'w/o \nSignal Detection + FSM']
            colors = ['tab:green', 'tab:blue', 'tab:red']
            hatches = ["--", "///", "xxx"]
            
            # Plot
            fig, ax = plt.subplots(figsize=(6,3))
            bars = ax.bar(schemes, time_ms_ls, color=colors, hatch=None, edgecolor="black")
            # Apply hatches
            for bar, hatch in zip(bars, hatches):
                bar.set_hatch(hatch)
                
            # Add text annotations above bars
            for i, (bar, pct) in enumerate(zip(bars, percent_increase)):
                if pct < 0:
                    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(), f"{pct:.1f}%", ha='center', va='bottom', fontsize=10, weight='bold')

            # Labels
            ax.set_ylabel("End-to-End Delay (ms)")
            ax.set_ylim(0, 20)

            plt.tight_layout()
            plt.savefig(result_path / 'time.png', dpi=300)
            plt.savefig(result_path / 'time.pdf', dpi=300)
            
        
        if config['task'] == 'msa':
            opt_plots()
            FS_method_plots()
            plot_comm_time() 
        elif config['task'] == 'ave':
            ave_opt_plots()
        # only2mod_opt_plots()
        
if __name__ == '__main__':
    args = parse_args()
    
    import json
    # load config
    with open(args.config) as f:
        config = json.load(f)
        
    seed = config.get("seed", None)
    if seed is None:
        seed = random.randint(0, 10000)
        
    set_seed(seed)
    
    UplinkExperiment20250914.result_main_folder = Path(config['result_dir'])
    UITestSuite.ue_cls.cp_main_folder = Path(config['cp_main_folder'])
    UITestSuite.result_main_folder = Path(config['result_dir']) / 'ui_test'
    UITestSuite.shap_file_path = Path(config['shap_file_path'])
    
    UITestSuite.config = config
    OptimizeTestSuit.config = config
    TempPlotting.config = config
    
    UITestSuite.test()
    OptimizeTestSuit.test()
    
    TempPlotting.plots()
        
        