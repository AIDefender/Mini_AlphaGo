cd ../MCTS 
python3 alg_vs_random.py --output_channels 2,4,8,16,32 --hidden_layers_sizes 32,32  
python3 alg_vs_random.py --output_channels 2,4,8,16,32 --hidden_layers_sizes 32,64,14  
python3 alg_vs_random.py --output_channels 2,2,4,4,8,16 --hidden_layers_sizes 32,32  
python3 alg_vs_random.py --output_channels 2,2,4,4,8,16 --hidden_layers_sizes 32,64,14  