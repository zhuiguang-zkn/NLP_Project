## Roberta
python scripts/slu_baseline2.py --encoder_cell LSTM --mlp_num_layers 1 \
     --trainset_spoken_language_select asr_1best  --word_embedding Bert --device 2

# python scripts/slu_baseline2.py --encoder_cell LSTM --mlp_num_layers 1 \
#      --trainset_spoken_language_select manual_transcript  --word_embedding Bert --device 2

# python scripts/slu_baseline2.py --encoder_cell LSTM --mlp_num_layers 1 \
#      --trainset_spoken_language_select both  --word_embedding Bert --device 2

# python scripts/slu_baseline2.py --encoder_cell LSTM --mlp_num_layers 2 \
#      --trainset_spoken_language_select both  --word_embedding Bert --device 2
    
python scripts/slu_baseline2.py --encoder_cell LSTM --mlp_num_layers 2 --trainset_augmentation \
     --trainset_spoken_language_select both  --word_embedding Bert --device 2

##
python scripts/slu_baseline2.py --encoder_cell GRU --mlp_num_layers 1 \
     --trainset_spoken_language_select asr_1best  --word_embedding Bert --device 2

python scripts/slu_baseline2.py --encoder_cell GRU --mlp_num_layers 1 \
     --trainset_spoken_language_select both  --word_embedding Bert --device 2

python scripts/slu_baseline2.py --encoder_cell GRU --mlp_num_layers 1 \
     --trainset_spoken_language_select manual_transcript  --word_embedding Bert --device 2

python scripts/slu_baseline2.py --encoder_cell GRU --mlp_num_layers 2 \
     --trainset_spoken_language_select both  --word_embedding Bert --device 2
    
python scripts/slu_baseline2.py --encoder_cell GRU --mlp_num_layers 2 --trainset_augmentation \
     --trainset_spoken_language_select both  --word_embedding Bert --device 2
