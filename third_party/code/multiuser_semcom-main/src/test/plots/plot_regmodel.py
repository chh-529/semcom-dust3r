from pathlib import Path
import numpy as np
import pickle
import matplotlib.pyplot as plt
import argparse
import torch
import random

from ...utils import str_type
from ..experiment.method import (
    UplinkInference, UplinkInference_SHAPbasedSelection, FSM_Msa, FSM_Ave
)

from ..experiment.modality import (
    test_general, TestHelper, UplinkExperiment20250914, plot_general
)

from ..experiment.regression import (
    RegressionMethod, _model_infos_from_config, _fsm_for, _resolve_shap_path
)

def parse_args() -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='configuration file for test and method settings')
    
    args = parser.parse_args()
    
    return args

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class RegressionVisualization:
    """
    Visualization for regression models
    Evaluate the relation betweeen masking ratios, SNR, and accuracy
    
    For convienience, we take the ave task with 2 modalaities here as expample
        1. Fix the SNR, plot the accuracy vs. masking ratios of 2 modalities
        2. Fix masking ratios of one modality, plot the accuracy vs. SNR vs. masking ratio of the other modality
    """ 
    
    rm_cls = RegressionMethod
    rg_cls = RegressionMethod.PolynomialRegression
    ue_cls = UplinkExperiment20250914
    
    @classmethod
    def _result_root(cls) -> Path:
        return cls.ue_cls.result_main_folder / 'reg_visual_res'
    
    @classmethod
    def main(cls):
        config = getattr(cls, 'config', None)
        if config is None:
            raise RuntimeError("OptimizeTestSuit.config was not set. Set it before calling test().")

        mi = config['model_info']
        model_type = mi['model_type']
        modalities = config['modalities']
        fsm_cls = _fsm_for(model_type)
        
        method = 'features'
        # method = 'signals'
        
        device = 'cuda:0'
        
        shap_file_path = _resolve_shap_path(config, cls.ue_cls, method)
        
        # SNR grid for sample generation and for final test
        sample_snr_dbs = np.arange(-6, 17, 2).tolist()
        snrs = np.arange(-6, 13)
        
        ## TODO: RMSE calculation
        
        def _plots_fixed_snr_regModel(metric_path: Path, 
                                      save_dir: Path | None,
                                      m_ratio_pred_1: np.ndarray,
                                      m_ratio_pred_2: np.ndarray
                                      ):
            def load_metric(path):
                """
                    metric[snr] = predicted accuracy
                """
                with open(path, 'rb') as f:
                    return pickle.load(f)
                
            metric = load_metric(metric_path)
            snrs = sorted(metric.keys())  # ensure consistent order

            n_snrs = len(snrs)
            n_rows, n_cols = 1, n_snrs

            # fig = plt.figure(figsize=(7 * n_cols, 6), constrained_layout=True)

            for i, snr in enumerate(snrs, 1):
                fig = plt.figure(figsize=(7, 6))
                ax = fig.add_subplot(111, projection='3d')
                ax.plot_surface(
                    m_ratio_pred_1,
                    m_ratio_pred_2,
                    metric[snr],
                    cmap='viridis',
                    alpha=1.0
                )

                ax.set_xlabel(
                    "Feature selection ratio (I)",
                    labelpad=12,
                    fontsize=14
                )
                ax.set_ylabel(
                    "Feature selection ratio (S)",
                    labelpad=12,
                    fontsize=14
                )
                ax.set_zlabel(
                    "Predicted Accuracy",
                    labelpad=12,
                    fontsize=14
                )

                # ====== Title ======
                # ax.set_title(f"SNR = {snr} dB", fontsize=16, pad=10)

                ax.view_init(elev=25, azim=-55)
                
                ax.invert_xaxis()
                # ax.view_init(elev=25, azim=-60)
                
                # Find the maximum point
                max_idx = np.unravel_index(np.argmax(metric[snr]), metric[snr].shape)
                max_x = m_ratio_pred_1[max_idx]
                max_y = m_ratio_pred_2[max_idx]
                max_z = metric[snr][max_idx]
                
                diff = metric[snr].max() - metric[snr].min()
                
                # how far above the surface you want the line (tune this)
                offset = 0.2 * diff
                z_top = max_z + offset
                
                eps = 0.02 * diff
                
                
                # Add an arrow/line label
                ax.plot(
                    [max_x, max_x],
                    [max_y, max_y],
                    [max_z, z_top],
                    color='red', linestyle='--', linewidth=1.2,
                    zorder=100   # force draw in front
                )

                # label text ---
                ax.text(
                    max_x, max_y, z_top,
                    f"({max_x:.2f}, {max_y:.2f})\nMax Acc={max_z:.2f}",
                    color='red',
                    ha='center', va='bottom',
                    zorder=300   # force draw in front
                )

                output_path = save_dir / f"reg_model_fixed_snr_{snr}.pdf"
                plt.savefig(output_path)
                print(f"✅ Saved: {output_path}")
                plt.close()

            # if save_dir is not None:
            #     output_path = save_dir / 'reg_model_fixed_snr_all.pdf'
            #     plt.savefig(output_path, bbox_inches='tight')
            #     print(f"✅ Saved: {output_path}")
        
        def _fixed_snr_test(reg_model, result_folder: Path):
            # Prepare test snrs and masking ratios for visualization
            test_snr_dbs = [-5, 0, 5]
            test_m_ratios_1 = np.linspace(0, 1, 101)
            test_m_ratios_2 = np.linspace(0, 1, 101)
            
            m_ratio_pred_1, m_ratio_pred_2 = np.meshgrid(test_m_ratios_1, test_m_ratios_2)
            X_ratios = np.column_stack([m_ratio_pred_1.ravel(), m_ratio_pred_2.ravel()])
            
            result_folder.mkdir(parents=True, exist_ok=True)
            result_path = result_folder / f'reg_fixed_snr.pkl'
            
            if result_path.exists():
                print(f"[*] Result file {result_path} already exists, skipping test.")
                _plots_fixed_snr_regModel(result_path, result_folder, m_ratio_pred_1, m_ratio_pred_2)
                return result_path
            
            # Test the regression model on the grid of snrs and masking ratios
            ret = {}
            n_snrs = len(test_snr_dbs)
            n_rows, n_cols = 1, n_snrs
            
            for i, snr in enumerate(test_snr_dbs, 1):
                snr_col = np.full((X_ratios.shape[0], 1), snr)
                X_pred = np.hstack((X_ratios, snr_col))
                ret[snr] = cls.rg_cls.predict_acc(X_pred, reg_model).reshape(m_ratio_pred_2.shape)
                
                fig = plt.figure(figsize=(7, 6))
                ax = fig.add_subplot(111, projection='3d')
                ax.plot_surface(
                    m_ratio_pred_1,
                    m_ratio_pred_2,
                    ret[snr],
                    cmap='viridis',
                    alpha=1.0
                )

                ax.set_xlabel(
                    "Feature selection ratio (I)",
                    labelpad=12,
                    fontsize=14
                )
                ax.set_ylabel(
                    "Feature selection ratio (S)",
                    labelpad=12,
                    fontsize=14
                )
                ax.set_zlabel(
                    "Predicted Accuracy",
                    labelpad=12,
                    fontsize=14
                )

                # ====== Title ======
                # ax.set_title(f"SNR = {snr} dB", fontsize=16, pad=10)

                ax.view_init(elev=25, azim=-55)
                
                ax.invert_xaxis()
                # ax.view_init(elev=25, azim=-60)
                
                # Find the maximum point
                max_idx = np.unravel_index(np.argmax(ret[snr]), ret[snr].shape)
                max_x = m_ratio_pred_1[max_idx]
                max_y = m_ratio_pred_2[max_idx]
                max_z = ret[snr][max_idx]
                
                diff = ret[snr].max() - ret[snr].min()
                
                # how far above the surface you want the line (tune this)
                offset = 0.2 * diff
                z_top = max_z + offset
                
                eps = 0.02 * diff
                
                
                # Add an arrow/line label
                ax.plot(
                    [max_x, max_x],
                    [max_y, max_y],
                    [max_z, z_top],
                    color='red', linestyle='--', linewidth=1.2,
                    zorder=100   # force draw in front
                )

                # label text ---
                ax.text(
                    max_x, max_y, z_top,
                    f"({max_x:.2f}, {max_y:.2f})\nMax Acc={max_z:.2f}",
                    color='red',
                    ha='center', va='bottom',
                    zorder=300   # force draw in front
                )

                output_path = result_folder / f"reg_model_fixed_snr_{snr}.pdf"
                plt.savefig(output_path)
                print(f"✅ Saved: {output_path}")
                plt.close()
                
            def save_metrics(metrics, path):
                import pickle
                with open(path, 'wb') as f:
                    pickle.dump(metrics, f)
            
            save_metrics(ret, result_path)
            
            return result_path
        
        
        def _plots_fixed_masking_ratio(metric_path: Path, 
                                      save_path: Path | None,
                                      m_ratios,
                                      m_ratio_pred: np.ndarray,
                                      snr_pred: np.ndarray,
                                      x_label: str = "Feature selection ratio",
                                      add_ridge_line: bool = False,
                                      ):
            def load_metric(path):
                """
                    metric[snr] = predicted accuracy
                """
                with open(path, 'rb') as f:
                    return pickle.load(f)
                
            metric = load_metric(metric_path)
            
            # Plot 3D surface
            fig = plt.figure(figsize=(10, 7))
            ax = fig.add_subplot(111, projection='3d')
            ax.plot_surface(m_ratio_pred, snr_pred, metric, cmap='viridis')
            ax.set_xlabel(x_label, labelpad=8)
            ax.set_ylabel("SNR (dB)", labelpad=8)
            ax.set_zlabel("Predicted Accuracy")

            ax.set_yticks(np.arange(snrs[0], snrs[-1] + 1, 2)) 
            ax.invert_xaxis()
            
            # Mark the maximum point
            ## Find the maximum point
            max_idx = np.unravel_index(np.argmax(metric), metric.shape)
            max_x = m_ratio_pred[max_idx]
            max_y = snr_pred[max_idx]
            max_z = metric[max_idx]
            
            diff = metric.max() - metric.min()
            
            ## how far above the surface you want the line (tune this)
            offset = 0.2 * diff
            z_top = max_z + offset
            
            ## Add an arrow/line label
            ax.plot(
                [max_x, max_x],
                [max_y, max_y],
                [max_z, z_top],
                color='red', linewidth=1.2,
                marker='^', 
                markersize=3,
                markevery=[1],
                zorder=100   # force draw in front
            )

            ## label text
            ax.text(
                max_x, max_y, z_top + 0.02 * diff,
                f"({max_x:.2f}, {max_y:.2f})\nMax Acc={max_z:.2f}",
                color='red',
                ha='center', va='bottom',
                zorder=300   # force draw in front
            )
            
            if add_ridge_line:
                max_x_indices = np.argmax(metric, axis=1)
                x_line = m_ratios[max_x_indices]      
                y_line = snrs                    
                z_line = metric[np.arange(len(y_line)), max_x_indices]
                
                ax.plot(x_line, y_line, z_line, color='red', linewidth=0.9, linestyle='--',label='Max Accuracy Path', zorder=200)

            plt.tight_layout()
            
            plt.savefig(save_path, dpi=200)
            
        
        def _fixed_masking_ratio_test(reg_model, result_folder: Path):
            # Prepare masking ratios for visualization
            test_m_ratios_1 = np.linspace(0, 1, 101)
            test_m_ratios_2 = np.linspace(0, 1, 101)
            
            def save_metrics(metrics, path):
                import pickle
                with open(path, 'wb') as f:
                    pickle.dump(metrics, f)
            
            # Fix speech ratio equal to 1 (select all speech features)
            result_folder.mkdir(parents=True, exist_ok=True)
            result_path_1 = result_folder / f'reg_fixed_ratios_1.pkl'

            m_ratio_pred_1, snr_pred = np.meshgrid(test_m_ratios_1, snrs)
            ratio_col = np.full(m_ratio_pred_1.flatten().shape, 1)
            X_pred_1 = np.column_stack([m_ratio_pred_1.ravel(), ratio_col, snr_pred.ravel()])
        
            if result_path_1.exists():
                print(f"[*] Result file {result_path_1} already exists, skipping test.")
            else:
                y_pred_1 = cls.rg_cls.predict_acc(X_pred_1, reg_model).reshape(m_ratio_pred_1.shape)
                save_metrics(y_pred_1, result_path_1)
                
            _plots_fixed_masking_ratio(
                result_path_1,
                result_folder / 'reg_model_fixed_ratio1.png',
                test_m_ratios_1,
                m_ratio_pred_1,
                snr_pred,
                "Feature selection ratio (Image)",
                add_ridge_line=True
            )
            
            # Fix image ratio equal to 1 (select all image features)
            result_path_2 = result_folder / f'reg_fixed_ratios_2.pkl'

            m_ratio_pred_2, snr_pred = np.meshgrid(test_m_ratios_2, snrs)
            ratio_col = np.full(m_ratio_pred_2.flatten().shape, 1)
            X_pred_2 = np.column_stack([ratio_col, m_ratio_pred_2.ravel(), snr_pred.ravel()])
                
            if result_path_2.exists():
                print(f"[*] Result file {result_path_2} already exists, skipping test.")
            else:
                y_pred_2 = cls.rg_cls.predict_acc(X_pred_2, reg_model).reshape(m_ratio_pred_2.shape)
                save_metrics(y_pred_2, result_path_2)
                
            _plots_fixed_masking_ratio(
                result_path_2,
                result_folder / 'reg_model_fixed_ratio2.png',
                test_m_ratios_2,
                m_ratio_pred_2,
                snr_pred,
                "Feature selection ratio (Speech)",
                add_ridge_line=True
            )
            
            
        def regression_samples_test():
            # Load regression model or fitting a new one if not exists
            ui_cls = UplinkInference_SHAPbasedSelection
            ui_kwargs = {
                'shap_val_path': shap_file_path,
                'modalities': modalities,
                'fsm_class': fsm_cls,
                'power_constraint': [1] * len(modalities),
            }
            
            model_infos = _model_infos_from_config(config)  # adjust how many IDs you have
            m_ratios_ls = [[0.1, 0.2, 0.4, 0.6, 0.8, 1.0] for _ in range(len(modalities))]
            
            reg_model_path = cls.rm_cls.test_samples_fitting(
                            RegressionMethod.PolynomialRegression,
                            {'degree': 2},
                            ui_cls, ui_kwargs, method, model_infos, 
                            m_ratios_ls, sample_snr_dbs, device
                    )
            
            print(f"[*] Regression model path for sample test: {reg_model_path}")
            
            reg_model = cls.rm_cls.load_model(reg_model_path)   
            
            result_folder = (
                cls._result_root() / 
                ui_cls.ui_folder_name(ui_kwargs) /
                cls.rg_cls.rga_folder_name({'degree': 2}) / 
                cls.ue_cls._get_model_name(*model_infos[0])
            )
            
            # Save the results and visualizations
            _fixed_snr_test(reg_model, result_folder)
            _fixed_masking_ratio_test(reg_model, result_folder)
            
        
        regression_samples_test()
        
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
    UplinkExperiment20250914.cp_main_folder = Path(config['cp_main_folder'])
    RegressionVisualization.config = config
    RegressionVisualization.main()
    