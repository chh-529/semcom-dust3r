n_symbols=48
oma_symbols=16

# python -m src.train.train_UDeepSC \
#     --model_name "awgn_12_udeepsc_msa_symbols_$((n_symbols/2))_cmu-mosei" \
#     --gpu 0 \
#     --interfere_mode "all" \
#     --fading_mode "slow" \
#     --num_symbols $n_symbols \
#     --seed 1000

# UDeepSC OMA
# python -m src.train.train_UDeepSC \
#     --model_name "awgn_12_udeepscOMA_ave_symbols_$((oma_symbols/2))_ave_2" \
#     --gpu 0\
#     --interfere_mode "all" \
#     --fading_mode "slow" \
#     --num_symbols $oma_symbols \
#     --seed 1000 \

# UDeepSC SIC
# python -m src.train.train_UDeepSC \
#     --model_name "awgn_12_udeepscSIC_msa_symbols_$((n_symbols/2))_cmu-mosei" \
#     --gpu 1 \
#     --interfere_mode "all" \
#     --fading_mode "slow" \
#     --num_symbols $n_symbols \
    # --seed 1000 \


python -m src.train.train_UDeepSC \
    --config ./src/train_config/train_config_msa.json

# python -m src.train.train_UDeepSC \
#     --config ./src/train_config/train_config_ave.json