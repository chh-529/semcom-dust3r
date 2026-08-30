import torch
import numpy as np
from pathlib import Path
from typing import *
from tqdm import tqdm
import random
import pickle
import warnings
import matplotlib.pyplot as plt

from ...log import get_logger
from ... import utils

from ...channel import *
from .cp_load import *

from ...trainer.trainer_udeepsc import UDeepSCNoSICTrainer_Msa
from ...dataset.udeepsc import MSA_dataset, make_udeepsc_msa_testdataloader, make_udeepsc_msa_dataloader
from ..shap.shap_helper import (
    KernelAccuracyWrapper_MSA, save_shap_summary, make_reference_vectors, split_by_modality, DeepExplainerWrapper_MSA, plot_mod_pos_neg, group_mod_shap, plot_group_bar, plot_group_signed_mean_shap
)

from ...channel import *
from ...utils import calc_metrics, to_device, pad_tensor_batch
import shap

def test_dataset(trainer, test_loader, n_batch, result_folder: Path, save_predict: bool = False, save_per_nbatch: int = 10):
    """
        Args:
            trainer: the trainer object
            test_loader: the dataloader to test on, n_user should be the same as trainer
            n_batch: number of batch to test 
                     (so the total signal count will be n_batch * n_user)
            result_folder: the folder to store the result
    """
    result_folder.mkdir(parents=True, exist_ok=True)
    # if save_predict:
    #     (result_folder / 'images').mkdir(parents=True, exist_ok=True)

    with open(result_folder / 'trainer.txt', 'w') as fout:
        trainer_var = vars(trainer)
        if 'metrics' in trainer_var: 
            del trainer_var['metrics']
        print(utils.str_type(trainer_var, indent=4, array_limit_items=20), file=fout)

    trainer.model.eval()
    total_acc  = 0
    y_true, y_pred = [], []
    with torch.no_grad():
        for it, (inputs, targets) in tqdm(zip(range(n_batch), test_loader)):
            inputs = to_device(inputs, trainer.device)
            targets = targets.to(trainer.device)
            results = trainer.transmit(inputs)

            targets, results = targets.to('cpu'), results.to('cpu')
            acc = calc_metrics(results, targets)
            total_acc += float(acc)

            # y_pred.append(results.detach().cpu().numpy())
            # y_true.append(targets.detach().cpu().numpy())
            trainer.logger.info('test iter %s/%s.  acc: %.4f' \
            % (it + 1, n_batch, float(acc)))

    # y_true = np.concatenate(y_true, axis=0).squeeze()
    # y_pred = np.concatenate(y_pred, axis=0).squeeze()
    
    average_acc = total_acc / n_batch
    stat = {
        'snr': trainer.channel.snr_db,
        'accuracy': float(average_acc),
    }

    with open(result_folder / 'metric.txt', 'w') as fout:
        for name in stat.keys():
            print(f"{name}:{stat[name]}", file=fout)

def get_encoded_features(trainer, dataloader, is_JSCC:bool=False):
    """
    Returns:
        features output by semanic encoders
        in order: Text, Image, Speech
    """
    trainer.model.eval()
    
    data_iter = iter(dataloader)
    M = len(next(data_iter)[0]) # get # modalities
    
    encodes = [[] for _ in range(M)]      # per-modality lists of batch tensors

    targets = []
    with torch.no_grad():
        for data, t in tqdm(dataloader):
            batch_size = t.shape[0]
            data = to_device(data, trainer.device)
            encoded_data = trainer.get_features(data, is_JSCC)
            frames = encoded_data[0].shape[1]
            for i, enc in enumerate(encoded_data):
                sb_len = enc.shape[-1]
            
                # squeeze data to (batch * frames, feature dim)
                enc = enc.reshape(-1, sb_len)
                encodes[i].append(enc)
            
            t = t.unsqueeze(-2)
            cls = t.shape[-1]
            dims = t.shape[:-2]
            t = t.expand(*dims, frames, -1).reshape(-1, cls) # expand targets to each frame (batch * frames, 1)
            targets.append(t)
    
    # encodes_t = pad_tensor_batch(encodes_t)
    # encodes_i = pad_tensor_batch(encodes_i)
    # encodes_s = pad_tensor_batch(encodes_s)
    # concatenate list of batch tensors to one tensor
    encodes = [torch.cat(encodes[i], dim=0) for i in range(M)]
    
    return tuple(encodes), targets
            
        
def test_SHAP_DeepExplainer(trainer, result_dir: str | Path, batch_size=100, n_bg = 100, n_samples:int=200, save_result: bool=False):    
    """
    Deprecated !! Use test_SHAP_KernelExplainer_Masker()
        
        Calculate SHAP value by DeepExplainer
        For DeepExplainer, The background dataset to use for integrating out features. Deep integrates over these samples. The data passed here must match the input tensors given in the first argument. 
        
        Args:
            n_bg: number of backgroud data for shap to reference
            n_samples: number of samples for shap to calulate shapley values 
    """
    result_dir = Path(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    
    # load dataset
    _, val_dataloader = make_udeepsc_msa_dataloader(batch_size=batch_size, root='data/msadata')
    
    modalities = ['image', 'text', 'speech']
    
    encodes, targets = get_encoded_features(trainer, val_dataloader)
    encodes_t, encodes_i, encodes_s = encodes
    
    all_targets = torch.cat(targets)
    
    # concatenate list of batch tensors to one tensor
    encodes_t = torch.cat(encodes_t)
    encodes_i = torch.cat(encodes_i)
    encodes_s = torch.cat(encodes_s)
    
    trainer.model.eval()
    modelWrapper = DeepExplainerWrapper_MSA(trainer, modalities, all_targets[:n_bg], all_targets)
    
    # split valid dataset to be background and test for SHAP
    bg_t, bg_i, bg_s = encodes_t[:n_bg], encodes_i[:n_bg], encodes_s[:n_bg]
    
    test_t, test_i, test_s = encodes_t[n_bg : n_samples + 1], encodes_i[n_bg : n_samples + 1], encodes_s[n_bg : n_samples + 1]
    
    # explain by SHAP
    explainer = shap.DeepExplainer(modelWrapper, [bg_t, bg_i, bg_s])
    shapley_values = explainer.shap_values([test_t, test_i, test_s])
    print(f'Shapley value shape: {str_type(shapley_values)}')
    
    # save shapley values
    if save_result:
        shap_dict = {"text": shapley_values[0], "image": shapley_values[1], "speech": shapley_values[2]}
        np.savez("shap_values.npz", **shap_dict)

        
def test_SHAP_KernelExplainer_Masker(
    trainer, 
    result_dir: str | Path,
    nsamples: str | int = "auto",  # can be integer like 2*T for speed/quality tradeoff
    bg_size: int = 64,
    n_test: int = None,
    is_JSCC: bool = False, 
    save_result: bool=False
):
    """
    Args:
        n_test: number of data to test accuracy. Need to be smaller than val dataset size

    """

    result_dir = Path(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device(trainer.device)
    
    # load dataset
    _, val_dataloader = make_udeepsc_msa_dataloader(batch_size=100, root='data/msadata')
    
    encodes, targets = get_encoded_features(trainer, val_dataloader, is_JSCC)
    encodes_t, encodes_i, encodes_s = encodes
    
    bg_text, bg_image, bg_speech = encodes_t[:bg_size], encodes_i[:bg_size], encodes_s[:bg_size] # (bg_size, dim)

    # 2) Build per-modality reference vectors (global means over valid positions)
    ref_t, ref_v, ref_s = make_reference_vectors([(bg_text, None), (bg_image, None), (bg_speech, None)], how="mean")
    ref_vectors = {'text': ref_t.unsqueeze(0), 'image': ref_v.unsqueeze(0), 'speech': ref_s.unsqueeze(0)}
    
    all_targets = torch.cat(targets)
    assert n_test <= all_targets.shape[0] - bg_size, f"Number of samples: {n_test} must not exceed the remain test samples {all_targets.shape[0] - bg_size}"
    
    
    # the single sample to explain (same shapes with background but batch=1)
    if n_test is not None:
        test_t, test_i, test_s = encodes_t[bg_size:bg_size + n_test + 1], encodes_i[bg_size: bg_size + n_test + 1], encodes_s[bg_size: bg_size + n_test + 1]
        test_targets = all_targets[bg_size: bg_size + n_test + 1]
    else:
        test_t, test_i, test_s = encodes_t[bg_size1:], encodes_i[bg_size:], encodes_s[bg_size:]
        test_targets = all_targets[bg_size:]

    # 3) Build the masker wrapper around the instance

    wrapper = KernelAccuracyWrapper_MSA(
        trainer=trainer,
        instance_feats={
            'text':  test_t,
            'image': test_i,
            'speech': test_s,
        },
        ref_vectors=ref_vectors,
        batch_size = 64,
        bg_size=bg_size,
        targets=test_targets,
        is_JSCC=is_JSCC,
        )

    # KernelExplainer uses a background in the same input space.
    #    Here the "input space" is the mask space, so we provide mask backgrounds.
    bg_masks = wrapper.background_masks(m=1)  # zeros = full reference
    
    def predict_fn(Z: np.ndarray) -> np.ndarray:
        return wrapper.predict(Z)
        
    # Initialize KernelExplainer
    explainer = shap.KernelExplainer(model=predict_fn, data=bg_masks, link="identity")

    # Explain the all-ones instance (keep all positions)
    x_ref = wrapper.ones_instance()  # shape [1, dim]
    
    # keep nsamples modest; start small and scale up if needed
    # nsamples = min(5 * wrapper.total_positions, 800)  # example cap
    
    shap_vals = explainer.shap_values(x_ref, nsamples=nsamples, l1_reg=0) # (N, dim)
    
    if isinstance(shap_vals, list):  # older SHAP versions may return a list per-output
        shap_flat = shap_vals[0]
    print(str_type(shap_vals))
    shap_flat = np.asarray(shap_vals).reshape(-1)  # [dim]

    meta = {
        'slices': wrapper.slices,                # modality -> (start, end)
        'lengths': {'text': wrapper.Tt, 'image': wrapper.Tv, 'speech': wrapper.Ts},
        'total_positions': wrapper.total_positions
    }

    # visualization
    feature_names = []
    for m, (a, b) in meta['slices'].items():
        for j in range(b - a):
            feature_names.append(f"{m}[{j}]")
            
    # Global mean absolute  (all rows at once)
    save_shap_summary(
        shap_values=shap_vals,
        X=x_ref,
        feature_names=feature_names,
        path=result_dir / "shap_summary_mean_bar_plot.png",
        plot_type="bar"
        )
    # Dot summary (shows distribution across samples)
    save_shap_summary(
        shap_values=shap_vals,
        X=x_ref,
        feature_names=feature_names,
        path=result_dir / "shap_summary_dot_plot.png",
        plot_type="dot"
        ) # default “dot”
    
    # Split to modalities
    shap_mod = split_by_modality(shap_flat, meta['slices'])
    print("SHAP per modality shapes:",
          {k: v.shape for k, v in shap_mod.items()})
    print("SHAP per modality val_avgs:",
          {k: v.mean() for k, v in shap_mod.items()})
    
    # group by modality and plot
    shap_group, X_group, group_names = group_mod_shap(shap_mod, x_ref, meta['slices'])
    # plot_group_bar(explainer, shap_group, X_group, group_names, 
    #                path= result_dir / 'shap_modality_mean_bar_plot')
    
    plot_group_bar(explainer, shap_group, X_group, group_names, 
                   path= result_dir / 'shap_modality_mean_sum_plot')
    
    plot_group_signed_mean_shap(shap_mod, x_ref, slices=meta['slices'], agg="sum", path=result_dir / 'shap_modality_sum_plot')
    
    plot_mod_pos_neg(shap_mod, title="Negative vs Positive SHAP value", result_path=result_dir  / 'shap_sum_plot.png')
    
    if save_result:
        file_name = result_dir / "feat_contribs.npz"
        # Save results
        np.savez_compressed(
            file_name,
            shap_vals=shap_vals,
            slices=np.array([meta['slices']['text'],
                             meta['slices']['image'],
                             meta['slices']['speech']], dtype=object),
            shap_mod=shap_mod,
        )
        
        print(f"Saved to {str(file_name)}")


def test_get_features():
    logger = get_logger('UdeepSCNoSIC', log_file_path=None, stdout=True, stdout_tqdm_write=True)
    trainer = get_trainer_from_checkpoint(
        logger=logger,
        checkpoint_path=Path('./checkpoint/20250915/awgn_12_udeepsc_msa_symbols_24_cmu-mosei_1/checkpoint'),
        args_path=Path('./checkpoint/20250915/awgn_12_udeepsc_msa_symbols_24_cmu-mosei_1/args.pkl'),
        gpus=[0]
    )
    
    # load dataset
    _, val_dataloader = make_udeepsc_msa_dataloader(batch_size=100, root='data/msadata')
    
    encodes, targets = get_encoded_features(trainer, val_dataloader)
    encodes_t, encodes_i, encodes_s = encodes
    # encodes_t_tensor = torch.cat(encodes_t)
    # encodes_i_tensor = torch.cat(encodes_i)
    all_tars = torch.cat(targets)
    
    # print(str_type(encodes[0]))
    print(str_type(encodes_t))
    print(str_type(encodes_i))
    print(str_type(all_tars))

def main_test_20250902():
    logger = get_logger('UdeepSCNoSIC', log_file_path=None, stdout=True, stdout_tqdm_write=True)
    trainer = get_trainer_from_checkpoint(
        logger=logger,
        checkpoint_path=Path('./checkpoint/20251105/awgn_12_udeepsc_msa_symbols_24_cmu-mosei_1/checkpoint'),
        args_path=Path('./checkpoint/20251105/awgn_12_udeepsc_msa_symbols_24_cmu-mosei_1/args.pkl'),
        gpus=[0]
    )

    n_user = 3
    trainer.channel = AWGNMultiUplinkChannel(
            n_user=n_user,
            snr_db = [12.0], # test for different snr here
            interfere_mode='all'
        )
    # trainer.channel = RayleighFadingMultiUplinkChannel(
    #         n_user, 
    #         snr_db=[0.0], 
    #         channel_gain_var=[[1.0], [1.0], [1.0]],
    #         divide_gain=True, 
    #         noise_power_density_dBm=-90,    # ref. ISSNOMATrainer's note
    #         reference_distance=1,
    #         reference_path_loss=pow(10, -30/10),
    #         path_loss_exponent=4,
    #         distance=torch.Tensor([33, 83, 133]).reshape(3, 1),
    #         interfere_mode='all', 
    #         fading_mode='slow',
    #     )

    args = pickle.load(open(Path('./checkpoint/20251105/awgn_12_udeepsc_msa_symbols_24_cmu-mosei_1/args.pkl'), 'rb'))

    # data_dir = './data/MF'
    test_dataloader = make_udeepsc_msa_testdataloader(batch_size=10, root='data/msadata')

    test_dataset(
        trainer, test_dataloader, n_batch = len(test_dataloader), result_folder = Path('./tmp/20251105/model_test/awgn_12_udeepsc_msa_symbols_24_cmu-mosei_1'), save_predict = True, save_per_nbatch = 15
    )
    
def main_test_only_2_modal_20250910():
    logger = get_logger('UdeepSCNoSIC', log_file_path=None, stdout=True, stdout_tqdm_write=True)
    trainer = get_trainer_from_checkpoint(
        logger=logger,
        checkpoint_path=Path('/home/ldap/hansliu/multiuser_semcom/checkpoint/20250909_udeepscNoSIC_msa_test/checkpoint'),
        args_path=Path('/home/ldap/hansliu/multiuser_semcom/checkpoint/20250909_udeepscNoSIC_msa_test/args.pkl'),
        gpus=[0]
    )

    n_user = 2
    trainer.channel = AWGNMultiUplinkChannel(
            n_user=n_user,
            snr_db = [12.0], # test for different snr here
            interfere_mode='all'
        )
    
    trainer.power_constraint = torch.tensor([1] * n_user)

    args = pickle.load(open(Path('/home/ldap/hansliu/multiuser_semcom/checkpoint/20250909_udeepscNoSIC_msa_test/args.pkl'), 'rb'))

    # data_dir = './data/MF'
    test_dataloader = make_udeepsc_msa_testdataloader(batch_size=10, root='data/msadata')

    test_dataset(
        trainer, test_dataloader, n_batch = len(test_dataloader), result_folder = Path('./tmp/20250910_udeepscNoSIC_msa_NoSpeech_test'), save_predict = True, save_per_nbatch = 15
    )
    
def main_test_SHAP_DeepExplainer_20250907():
    logger = get_logger('UdeepSCNoSIC', log_file_path=None, stdout=True, stdout_tqdm_write=True)
    trainer = get_trainer_from_checkpoint(
        logger=logger,
        checkpoint_path=Path('/home/ldap/hansliu/multiuser_semcom/checkpoint/20250907_udeepscNoSIC_msa_test/checkpoint'),
        args_path=Path('/home/ldap/hansliu/multiuser_semcom/checkpoint/20250907_udeepscNoSIC_msa_test/args.pkl'),
        gpus=[0]
    )

    test_SHAP_DeepExplainer(trainer, result_dir = Path('./tmp/20250907_udeepscNoSIC_msa_SHAP_test'), batch_size=100, n_samples=100, save_result=False)
    
def main_test_SHAP_KernelExplainer_20250908():
    logger = get_logger('UdeepSCNoSIC_shap_test', log_file_path=None, stdout=True, stdout_tqdm_write=True)
    trainer = get_trainer_from_checkpoint(
        logger=logger,
        checkpoint_path=Path('./checkpoint/20250915/awgn_12_udeepsc_msa_symbols_24_cmu-mosei_2/checkpoint'),
        args_path=Path('./checkpoint/20250915/awgn_12_udeepsc_msa_symbols_24_cmu-mosei_2/args.pkl'),
        gpus=[1]
    )
    
    n_tests = 3000
    method = 'features'
    is_JSCC = method == 'signals'
    
    result_main_folder = Path('./tmp/20250915')
    # result_main_folder = Path('./tmp/shap_test')
    model_name = 'awgn_12_udeepsc_msa_symbols_24_cmu-mosei'
    # model_name = 'awgn_12_udeepsc_ave_symbols_24_cmu-mosei'
    result_path = result_main_folder / f'kernel_shap_{n_tests}_{method}'/ model_name
    test_SHAP_KernelExplainer_Masker(trainer, result_dir = result_path, save_result=True, n_test=n_tests, is_JSCC=is_JSCC, bg_size=10)

def main_test_20251120():
    logger = get_logger('UdeepSCNoSIC', log_file_path=None, stdout=True, stdout_tqdm_write=True)
    trainer = get_trainer_from_checkpoint(
        logger=logger,
        checkpoint_path=Path('checkpoint/20251105/rayleigh_12_udeepsc_msa_symbols_24_cmu-mosei_3/checkpoint'),
        args_path=Path('checkpoint/20251105/rayleigh_12_udeepsc_msa_symbols_24_cmu-mosei_3/args.pkl'),
        gpus=[0]
    )

    n_user = 3
    trainer.channel = RayleighFadingMultiUplinkChannel(
            n_user, 
            snr_db=[12.0], 
            channel_gain_var=[[1.0], [1.0], [1.0]],
            divide_gain=True, 
            noise_power_density_dBm=-90,    # ref. ISSNOMATrainer's note
            reference_distance=1,
            reference_path_loss=pow(10, -30/10),
            path_loss_exponent=4,
            distance=torch.Tensor([33, 83, 133]).reshape(3, 1),
            interfere_mode='all', 
            fading_mode='slow',
        )
    # trainer.channel = AWGNMultiUplinkChannel(
    #         n_user=n_user,
    #         snr_db = [-2.0], # test for different snr here
    #         interfere_mode='all'
    #     )

    args = pickle.load(open(Path('checkpoint/20251105/rayleigh_12_udeepsc_msa_symbols_24_cmu-mosei_3/args.pkl'), 'rb'))

    # data_dir = './data/MF'
    test_dataloader = make_udeepsc_msa_testdataloader(batch_size=10, root='data/msadata')
    
    for batch in test_dataloader:
        print(type(batch), len(batch), type(batch[0]))
        break

    test_dataset(
        trainer, test_dataloader, n_batch = len(test_dataloader), result_folder = Path('./tmp/20251105/model_test/rayleigh_12_udeepsc_msa_symbols_24_cmu-mosei_3'), save_predict = True, save_per_nbatch = 15
    )

if __name__ == '__main__':
    # main_test_20250902()
    # main_test_only_2_modal_20250910()
    # test_get_features()
    # main_test_SHAP_KernelExplainer_20250908()
    main_test_20251120()