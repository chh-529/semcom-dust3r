import torch
import torch.nn as nn
from ...utils import *
import matplotlib.pyplot as plt
import shap
from tqdm import tqdm


class DeepExplainerWrapper_MSA(nn.Module):
    def __init__(self, trainer, modalites: list, bg_targets: torch.Tensor, targets: torch.Tensor, *args, **kwargs):
        """
            Args:
                targets: Ground truth with the batch passed to forward
                shape (batch_size, *). Note that batch_size need to be the same as the batch passed to forward
                
        """
        super().__init__(*args, **kwargs)
        
        self.trainer = trainer
        self.device = trainer.device
        self.modalities = modalites
        self.n_modal = len(modalites)
        self.bg_targets = bg_targets.to(self.trainer.device) # labels aligned with the batch passed to forward
        self.targets = targets.to(self.trainer.device)
        
    def forward(self, texts, images, speechs):
        """
            Returns: 
                a scalar score per sample in Tensor of shape (batch, )
                
            Can't work for SHAP
        """
        batch_size = texts.shape[0]
        preds = self.trainer.get_decode_result((texts, images, speechs))
        preds = preds.view(-1) # (batch,)
        
        if batch_size == self.bg_targets.shape[0]:
            targets = self.bg_targets.view(-1)
            margin = (2 * targets - 1.0) * preds
            q = torch.sigmoid(1 * margin) 
        elif batch_size == self.targets.shape[0]:
            targets = self.targets.view(-1)
            margin = (2 * targets - 1.0) * preds
            q = torch.sigmoid(1 * margin) 
        # accs = torch.zeros(batch_size)        
        # for i in range(batch_size):
        #     pred = preds[i].unsqueeze(0)
        #     tar = self.targets[i].unsqueeze(0)
        #     acc = calc_metrics(pred, tar)
        #     accs[i] = acc
        
        return q.unsqueeze(1)
    
def make_reference_vectors(bg_mask_ls: list[Tuple[torch.Tensor]],
                           how: str = "mean") -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create reference vectors used to replace positions when masked.
    We use a single vector per modality (global mean over valid positions).
    If you prefer per-position refs, compute mean along batch for each time index separately.

    Inputs:
      bg_*: [B, T_mod, D_mod] encoded features
      pad_mask_* (optional): [B, T_mod] booleans, True for valid positions, False for pad
      how: 'mean' | 'zero' | 'median' (extend as needed)

    Returns:
      ref_text:  [D_text]
      ref_image:[D_vis]
      ref_speech:[D_speech]
    """
    def _make_ref(x, mask):
        # x: [B, T, D], mask: [B, D] (True=keep, False=ignore)
        B, D = x.shape
        if mask is not None:
            valid = x[mask].view(-1, D) if m.any() else x.view(-1, D)  # [N_valid, D]
        else:
            valid = x.view(-1, D)
        if how == "mean":
            return valid.mean(dim=0)
        elif how == "median":
            return valid.median(dim=0).values
        elif how == "zero":
            return torch.zeros(D, device=x.device, dtype=x.dtype)
        else:
            raise ValueError(f"Unknown how={how}")

    return tuple(_make_ref(bg, pad_mask) for (bg, pad_mask) in bg_mask_ls)

class KernelAccuracyWrapper(nn.Module):
    """
    KernelExplainer will pass in batches (B, M). We interpret each row as a mask z'.
    For each distinct z', compute F(z') = accuracy over the fixed eval set with that mask.
    Cache results so repeated coalitions (due to multiple background rows) are free
    """
    def __init__(self,
                 trainer,
                 targets: torch.tensor,  
                 bg_size: int,
                 batch_size: int = 2048,
                 is_JSCC: bool = False,
                 ):
        """
        Targets: [N, 1]
        """
        self.trainer = trainer
        self.device = trainer.device
        self.targets = targets  
        self.batch_size = batch_size   
        self.bg_size = bg_size
        self.is_JSCC = is_JSCC
        self._cache: Dict[Tuple[int, ...], float] = {}  # mask tuple -> accuracy

    def _apply_mask(self, mask: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Z: [1, D_pos] where D_pos = Tt + Tv + Ts, values ~ [0,1]
        Returns masked (text, image, speech) each with shape [N, D]
        """
        raise NotImplementedError

    def _eval_accuracy_under_mask(self, mask01: np.ndarray) -> float:
        """
        Predict and calculate the accuracy by masked features
        """
        raise NotImplementedError

    def ones_instance(self) -> np.ndarray:
        """The mask instance to explain: vector of all ones (keep every position)."""
        return np.ones((1, self.total_positions), dtype=np.float32)

    def background_masks(self, m: int = 50) -> np.ndarray:
        """Background in mask-space: typically zeros (all replaced by ref) or random."""
        # You can also sample random 0/1 masks to provide a richer background.
        return np.zeros((m, self.total_positions), dtype=np.float32)
    
    def predict(self, mask: np.ndarray):
        """
        mask: (B, M). Under our dummy setup, each row equals the coalition mask (or close).
        Return:  (B,) the same scalar accuracy for each row (broadcast).
        """
        # Extract the mask from the first row; rows in a batch share the same coalition in KernelExplainer. Size of a batch is same as the size of the background
        batch = mask.shape[0]
        res = np.zeros((batch,), dtype=np.float32)
        
        for i in tqdm(range(batch), leave=False, dynamic_ncols=True):
            z = mask[i]  # shape (M,)
            # print(str_type(mask))
            
            # Robust binarization (in case of float drift):
            mask01 = (z >= 0.5).astype(np.float32)
            acc = self._eval_accuracy_under_mask(mask01)
            res[i] = acc
            
            self.trainer.logger.info(f'Test [Mask={i}/{batch}] accuracy: {acc}')
        
        return res

class KernelAccuracyWrapper_MSA(KernelAccuracyWrapper):
    """
    Use for MSA task with 3 modalities: Text, Image, Speech
    """
    def __init__(self,
                 instance_feats: Dict[str, torch.Tensor],
                 ref_vectors: Dict[str, torch.Tensor],     # per-modality reference vectors
                 *args, 
                 **kwargs
                 ):
        super().__init__(*args, **kwargs)
        """
        instance_feats: {
           'text':   [N, D_text],
           'image': [N, D_vis],
           'speech': [N, D_speech]
        }
        ref_vectors: {
           'text':   [1, D_text],
           'image': [1, D_vis],
           'speech': [1, D_speech]
        }

        Targets: [N, 1]
        """
        # Freeze the instance features:
        self.x_text   = instance_feats['text'].to(self.device)   # [N, Dt]
        self.x_image = instance_feats['image'].to(self.device) # [N, Dv]
        self.x_speech = instance_feats['speech'].to(self.device) # [N, Ds]
        
        self.ref_text  = ref_vectors['text'].to(self.device).expand_as(self.x_text)     # [N, Dt]
        self.ref_image = ref_vectors['image'].to(self.device).expand_as(self.x_image)   # [N, Dv]
        self.ref_speech = ref_vectors['speech'].to(self.device).expand_as(self.x_speech)   # [N, Ds]

        # Build feature mapping: flat → (modality, time_index)
        self.Tt = self.x_text.shape[1]
        self.Tv = self.x_image.shape[1]
        self.Ts = self.x_speech.shape[1]

        # Flat index layout: [0..Tt-1]=text, [Tt..Tt+Tv-1]=image, [..]=speech
        self.slices = {
            'text':   (0,            self.Tt),
            'image': (self.Tt,      self.Tt + self.Tv),
            'speech': (self.Tt+self.Tv, self.Tt+self.Tv+self.Ts)
        }
        self.total_positions = self.Tt + self.Tv + self.Ts
        
        self._cache: Dict[Tuple[int, ...], float] = {}  # mask tuple -> accuracy
        
        assert self.targets.shape[0] == self.x_text.shape[0], "Data and targets must have the same sample s"

    def _apply_mask(self, mask: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Z: [1, D_pos] where D_pos = Tt + Tv + Ts, values ~ [0,1]
        Returns masked (text, image, speech) each with shape [N, D]
        """
        batch = self.targets.shape[0]
        device = self.device

        # Expand mask to batch
        mask = torch.tensor(mask, device=device, dtype=self.x_text.dtype).unsqueeze(0).expand(batch, -1)

        # Split masks by modality and broadcast over channel dim
        M_t = mask[:, self.slices['text'][0]: self.slices['text'][1]]# [N,Dt]
        M_v = mask[:, self.slices['image'][0]: self.slices['image'][1]] # [N,Dv]
        M_s = mask[:, self.slices['speech'][0]: self.slices['speech'][1]] # [N,Ds]

        # Linear imputation between instance vector and reference vector
        # x' = z'*x + (1-z') * mu 
        
        masked_text   = M_t * self.x_text   + (1.0 - M_t) * self.ref_text
        masked_image = M_v * self.x_image + (1.0 - M_v) * self.ref_image
        masked_speech = M_s * self.x_speech + (1.0 - M_s) * self.ref_speech
        
        return masked_text, masked_image, masked_speech

    def _eval_accuracy_under_mask(self, mask01: np.ndarray) -> float:
        key = tuple(mask01.astype(int).tolist())
        if key in self._cache:
            return self._cache[key]

        # Impute/mask the entire eval set
        masked_t, masked_i, masked_s = self._apply_mask(mask01)

        # Batched forward passes
        acc_sum = 0.0
        N = self.targets.shape[0]
        cnt = 0

        with torch.no_grad():
            for s in range(0, N, self.batch_size):
                e = min(s + self.batch_size, N)
                
                m_t = masked_t[s:e].unsqueeze(1)
                m_i = masked_i[s:e].unsqueeze(1)
                m_s = masked_s[s:e].unsqueeze(1)
                signal = (m_t, m_i, m_s)
                if not self.is_JSCC:
                    signal = self.trainer._channel_encode(signal)
                preds = self.trainer.get_decode_result(signal)
                # print(str_type(preds))
                acc = calc_metrics(preds, self.targets[s:e])
                acc_sum += acc
                cnt += 1
        
        avg_acc = (acc_sum / cnt)
        self._cache[key] = avg_acc
        return avg_acc
    
class KernelAccuracyWrapper_AVE(KernelAccuracyWrapper):
    """
    Use for AVE task with 2 modalities: Image, Speech
    """
    def __init__(self,
                 instance_feats: Dict[str, torch.Tensor],
                 ref_vectors: Dict[str, torch.Tensor],     # per-modality reference vectors
                 *args, 
                 **kwargs
                 ):
        super().__init__(*args, **kwargs)
        """
        instance_feats: {
           'image': [N, D_vis],
           'speech': [N, D_speech]
        }
        ref_vectors: {
           'image': [1, D_vis],
           'speech': [1, D_speech]
        }

        Targets: [N, 1]
        """
        # Freeze the instance features:
        self.x_image = instance_feats['image'].to(self.device) # [N, Dv]
        self.x_speech = instance_feats['speech'].to(self.device) # [N, Ds]
        
        self.ref_image = ref_vectors['image'].to(self.device).expand_as(self.x_image)   # [N, Dv]
        self.ref_speech = ref_vectors['speech'].to(self.device).expand_as(self.x_speech)   # [N, Ds]

        # Build feature mapping: flat → (modality, time_index)
        self.Tv = self.x_image.shape[1]
        self.Ts = self.x_speech.shape[1]

        self.slices = {
            'image': (0,      self.Tv),
            'speech': (self.Tv, self.Tv+self.Ts)
        }
        self.total_positions = self.Tv + self.Ts
        
        self._cache: Dict[Tuple[int, ...], float] = {}  # mask tuple -> accuracy
        
        assert self.targets.shape[0] == self.x_image.shape[0], "Data and targets must have the same sample s"

    def _apply_mask(self, mask: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Z: [1, D_pos] where D_pos = Tv + Ts, values ~ [0,1]
        Returns masked (text, image, speech) each with shape [N, D]
        """
        batch = self.targets.shape[0]
        device = self.device

        # Expand mask to batch
        mask = torch.tensor(mask, device=device, dtype=self.x_image.dtype).unsqueeze(0).expand(batch, -1)

        # Split masks by modality and broadcast over channel dim
        M_v = mask[:, self.slices['image'][0]: self.slices['image'][1]] # [N,Dv]
        M_s = mask[:, self.slices['speech'][0]: self.slices['speech'][1]] # [N,Ds]

        # Linear imputation between instance vector and reference vector
        # x' = z'*x + (1-z') * mu 
        
        masked_image = M_v * self.x_image + (1.0 - M_v) * self.ref_image
        masked_speech = M_s * self.x_speech + (1.0 - M_s) * self.ref_speech
        
        return masked_image, masked_speech

    def _eval_accuracy_under_mask(self, mask01: np.ndarray) -> float:
        key = tuple(mask01.astype(int).tolist())
        if key in self._cache:
            return self._cache[key]

        # Impute/mask the entire eval set
        masked_i, masked_s = self._apply_mask(mask01)

        # Batched forward passes
        acc_sum = 0.0
        N = self.targets.shape[0]
        cnt = 0

        with torch.no_grad():
            for s in range(0, N, self.batch_size):
                e = min(s + self.batch_size, N)
                
                m_i = masked_i[s:e].unsqueeze(1)
                m_s = masked_s[s:e].unsqueeze(1)
                signal = (m_i, m_s)
                if not self.is_JSCC:
                    signal = self.trainer._channel_encode(signal)
                preds = self.trainer.get_decode_result(signal)
                acc = compute_acc_AVE(preds, self.targets[s:e])
                acc_sum += acc
                cnt += 1
        
        avg_acc = (acc_sum / cnt)
        self._cache[key] = avg_acc
        return avg_acc
    
def split_by_modality(shap_flat: np.ndarray, slices: Dict[str, Tuple[int, int]]) -> Dict[str, np.ndarray]:
    out = {}
    for m, (a, b) in slices.items():
        out[m] = shap_flat[a:b]
    return out

def group_mod_shap(shap_mod: Dict[str, np.ndarray], X, slices: Dict[str, Tuple[int, int]], agg: Literal['mean_abs', 'sum']="mean_abs"):
    """
    shap_mod: Dict{Mod: np.ndarray}: splited modality result ouput by split_by_modality()
    X:        (N, D) features (just to pass into SHAP plotting, can be dummy)
    agg: "mean_abs" (default) or "sum"

    Returns:
        group_names: list of str
    """
    N, D = X.shape
    shap_group, X_group, group_names = [], [], []
    for mod, vals in shap_mod.items():
        group_names.append(mod)
        (s, e) = slices[mod]
        vals = np.expand_dims(vals, 0)
        if agg == "mean_abs":
            # Average of absolute shap values with sign preserved
            shap_group.append(vals.mean(axis=1))
        elif agg == "sum":
            shap_group.append(vals.sum(axis=1))
        X_group.append(X[:, s:e].mean(axis=1))  # dummy aggregation of feature values
    
    shap_group = np.stack(shap_group, axis=1)
    X_group = np.stack(X_group, axis=1)
    return shap_group, X_group, group_names


def aggregate_pos_neg_by_mod(shap_vals:dict[np.ndarray],  return_std:bool = False): 
    shap_mods = {}
    
    for mod, val_arr in shap_vals.items():
        shap_mods[mod] = {
            'pos': val_arr[val_arr > 0].sum(),
            'neg': val_arr[val_arr < 0].sum()
        }
        
    return shap_mods

def plot_mod_pos_neg(shap_vals:dict[np.ndarray], title: str, result_path: Path, dpi: int = 300,):
    
    shap_pos_neg = aggregate_pos_neg_by_mod(shap_vals)
    gnames = [k for k, _ in shap_pos_neg.items()]
    pos_sums = [v['pos'] for _, v in shap_pos_neg.items()]
    neg_sums = [abs(v['neg']) for _, v in shap_pos_neg.items()]
    
    # Plot
    fig, ax = plt.subplots(figsize=(7, max(3, 0.5 * len(gnames))))

    y = np.arange(len(gnames))  # one row per array
    bar_width = 0.2

    # Bars
    ax.barh(y - bar_width/2, neg_sums, height=bar_width, color="#197fc7", label="Negative sum")   # left
    ax.barh(y + bar_width/2, pos_sums,  height=bar_width, color="#d62929", label="Positive sum") # right

    # Labels
    ax.set_yticks(y, labels=gnames)
    # ax.axvline(0, color="black")  # vertical center line
    ax.set_xlabel("SHAP value sum")
    ax.legend()
    plt.title(title)
    plt.savefig(result_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    
def plot_mod_pos_neg_num(shap_vals:dict[np.ndarray], title: str, result_path: Path, dpi: int = 300,):
    
    gnames = [k for k, _ in shap_vals.items()]
    pos_nums = [(v > 0).sum() for _, v in shap_vals.items()]
    neg_nums = [(v < 0).sum() for _, v in shap_vals.items()]
    
    # Plot
    fig, ax = plt.subplots(figsize=(4, 6))

    y = np.arange(len(gnames))  # one row per array
    bar_width = 0.2

    # Bars
    ax.bar(y - bar_width/2, pos_nums, width=bar_width, color="#197fc7", label="Number of negative value")   # left
    ax.bar(y + bar_width/2, neg_nums,  width=bar_width, color="#d62929", label="Number of positive value") # right
    
    # 顯示數值標籤
    for i, (n_pos, n_neg) in enumerate(zip(pos_nums, neg_nums)):
        plt.text(i - bar_width/2, n_pos + 0.5, str(n_pos), ha='center', va='bottom')
        plt.text(i + bar_width/2, n_neg + 0.5, str(n_neg), ha='center', va='bottom')

    # Labels
    ax.set_xticks(y, labels=gnames)
    # ax.axvline(0, color="black")  # vertical center line
    ax.set_xlabel("Number of positive/negtive Shapley value")
    ax.legend()
    plt.title(title)
    plt.savefig(result_path, dpi=dpi, bbox_inches="tight")
    plt.close()

def plot_group_signed_mean_shap(
    shap_mod,
    x_ref,
    slices: Dict[str, Tuple[int, int]],
    agg: Literal['mean_abs', 'sum']="mean_abs",
    normalize_by_len: bool = False,
    with_error: bool = True,
    sort_by_abs: bool = True,
    top_k: int | None = None,
    title: str = "SHAP value (impact on model output)",
    path: Path=Path('shap_modality_bar_plot'),
    dpi: int = 300,
):
    gvals, X_group, gnames = group_mod_shap(shap_mod, x_ref, slices, agg)

    # sort by absolute magnitude (or keep input order)
    order = np.argsort(np.abs(gvals[0]))[::-1] if sort_by_abs else np.arange(len(gvals))
    if top_k is not None:
        order = order[:top_k]
    gnames = [gnames[i] for i in order]
    gvals  = gvals[0][order]

    # colors by sign
    colors = ["#d62929" if v > 0 else "#197fc7" for v in gvals]  # red=positive, blue=negative

    plt.figure(figsize=(7, max(3, 0.5 * len(gnames))))
    y = np.arange(len(gnames))
    plt.barh(y, gvals, align="center", color=colors, alpha=0.9)

    plt.yticks(y, gnames)
    plt.gca().invert_yaxis()
    xlabel = "Signed mean SHAP" if agg == "mean" else "Signed sum SHAP"
    if normalize_by_len:
        xlabel += " (normalized by group length)"
    plt.xlabel(xlabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()
    

def plot_group_bar(explainer, shap_group, X_group, group_names, path:Path=Path('shap_modality_bar_mean_plot')):
    """
    Draw mean absolute Shap value bar plot by shap.plots.bar
    Group features as modality
    
    Args:
        shap_group, X_group, group_names are the same as group_mod_shap() returns
    """
    N, G = shap_group.shape

    # Make safe base_values
    base = explainer.expected_value  # or your own baseline
    base_arr = None if base is None else np.full((N,), float(np.asarray(base).flat[0]))
    
    exp = shap.Explanation(
        values=shap_group,      # (N, G)
        base_values=base_arr,
        data=X_group,
        feature_names=group_names
    )
    
    fig = shap.plots.bar(exp, show=False)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    


def save_shap_summary(shap_values: np.ndarray, X: np.ndarray, feature_names, path:Path=Path('shap_summary_plot'), plot_type:Literal["bar", "dot"]="dot", dpi=300):
    """
    shap_values: (N, D) or list[(N, D)] for multiclass
    X:            (N, D) features matrix
    """
    plt.figure()
    shap.summary_plot(shap_values, X, feature_names=feature_names,
                      plot_type=plot_type, show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()
    
def save_shap_force_html(explainer, shap_values, X, i, html_path):
    """
    Save an interactive force plot to HTML
    shap_values: (N, D) for the selected target
    """
    base = explainer.expected_value
    if isinstance(base, list):
        base = base[0]
    fp = shap.force_plot(base, shap_values[i], X[i], matplotlib=False)
    shap.save_html(html_path, fp)
    
    
def read_shap_from_file(shap_path: Path) -> np.ndarray:
    """
    Read shap values from a .npy file
    """
    return np.load(shap_path, allow_pickle=True)

if __name__ == "__main__":
    path = Path('./tmp/20250915/kernel_shap_3000_features/awgn_12_udeepsc_msa_symbols_24_cmu-mosei')
    
    data = read_shap_from_file(path / 'feat_contribs.npz')
    shap_mod = data['shap_mod'].item()
    
    print(str_type(data))
    # print(data['shap_mod'])
    print(str_type(shap_mod))
    data.close()
    
    # res = aggregate_pos_neg_by_mod(shap_mod)
    # print(res)
    
    result_main_folder = Path('./tmp')
    result_path = result_main_folder / 'shap_test'
    result_path.mkdir(parents=True, exist_ok=True)
    # plot_mod_pos_neg(shap_mod, "Negative vs Positive SHAP value", result_path  / 'shap_sum_plot.png')
    plot_mod_pos_neg_num(shap_mod, "Number of negative and positive value", result_path  / 'shap_num_plot.png')
    