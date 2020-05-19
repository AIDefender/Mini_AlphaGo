cd ../MCTS 
python3 alg_vs_random.py --output_channels 2,4,8,16,32 --hidden_layers_sizes 64  
python3 alg_vs_random.py --output_channels 2,4,8,16,32 --hidden_layers_sizes 128  
python3 alg_vs_random.py --output_channels 2,2,4,4,8,16 --hidden_layers_sizes 64  
python3 alg_vs_random.py --output_channels 2,2,4,4,8,16 --hidden_layers_sizes 128  