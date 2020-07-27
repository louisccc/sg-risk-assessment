CUDA_VISIBLE_DEVICES=1 python ./3_train_graph_learning.py  --device cuda --cache_path /home/louisccc/NAS/louisccc/av/synthesis_data/carla_pkls/1043_dataset_image.pkl --seed 1 --epochs 200 --model mrgcn --layer_spec 64,128 --num_layers 2 --conv_type FastRGCNConv --batch_size 16 --learning_rate 0.0005 --activation relu --readout_type mean --temporal_type lstm_attn --dropout 0.1 --pooling_type sagpool --pooling_ratio 0.5 --lstm_input_dim 50 --lstm_output_dim 20 --test_step 1 --stats_path 1043_2_layer_image.csv
CUDA_VISIBLE_DEVICES=1 python ./3_train_graph_learning.py  --device cuda --cache_path /home/louisccc/NAS/louisccc/av/synthesis_data/carla_pkls/1043_dataset_image.pkl --seed 1 --epochs 200 --model mrgcn --layer_spec 64,128 --num_layers 2 --conv_type FastRGCNConv --batch_size 16 --learning_rate 0.0005 --activation relu --readout_type mean --temporal_type lstm_attn --dropout 0.1 --pooling_type topk --pooling_ratio 0.5 --lstm_input_dim 50 --lstm_output_dim 20 --test_step 1 --stats_path 1043_2_layer_image.csv
CUDA_VISIBLE_DEVICES=1 python ./3_train_graph_learning.py  --device cuda --cache_path /home/louisccc/NAS/louisccc/av/synthesis_data/carla_pkls/1043_dataset_image.pkl --seed 1 --epochs 200 --model mrgcn --layer_spec 64,128 --num_layers 2 --conv_type FastRGCNConv --batch_size 16 --learning_rate 0.0005 --activation relu --readout_type mean --temporal_type lstm_attn --dropout 0.1 --pooling_type None  --pooling_ratio 1.0  --lstm_input_dim 50 --lstm_output_dim 20 --test_step 1 --stats_path 1043_2_layer_image.csv
CUDA_VISIBLE_DEVICES=1 python ./3_train_graph_learning.py  --device cuda --cache_path /home/louisccc/NAS/louisccc/av/synthesis_data/carla_pkls/1043_dataset_image.pkl --seed 1 --epochs 200 --model mrgcn --layer_spec 64,128 --num_layers 2 --conv_type FastRGCNConv --batch_size 16 --learning_rate 0.0005 --activation relu --readout_type add --temporal_type lstm_attn --dropout 0.1 --pooling_type sagpool --pooling_ratio 0.5 --lstm_input_dim 50 --lstm_output_dim 20 --test_step 1 --stats_path 1043_2_layer_image.csv
CUDA_VISIBLE_DEVICES=1 python ./3_train_graph_learning.py  --device cuda --cache_path /home/louisccc/NAS/louisccc/av/synthesis_data/carla_pkls/1043_dataset_image.pkl --seed 1 --epochs 200 --model mrgcn --layer_spec 64,128 --num_layers 2 --conv_type FastRGCNConv --batch_size 16 --learning_rate 0.0005 --activation relu --readout_type add --temporal_type lstm_attn --dropout 0.1 --pooling_type topk --pooling_ratio 0.5 --lstm_input_dim 50 --lstm_output_dim 20 --test_step 1 --stats_path 1043_2_layer_image.csv
CUDA_VISIBLE_DEVICES=1 python ./3_train_graph_learning.py  --device cuda --cache_path /home/louisccc/NAS/louisccc/av/synthesis_data/carla_pkls/1043_dataset_image.pkl --seed 1 --epochs 200 --model mrgcn --layer_spec 64,128 --num_layers 2 --conv_type FastRGCNConv --batch_size 16 --learning_rate 0.0005 --activation relu --readout_type add --temporal_type lstm_attn --dropout 0.1 --pooling_type None  --pooling_ratio 1.0  --lstm_input_dim 50 --lstm_output_dim 20 --test_step 1 --stats_path 1043_2_layer_image.csv
CUDA_VISIBLE_DEVICES=1 python ./3_train_graph_learning.py  --device cuda --cache_path /home/louisccc/NAS/louisccc/av/synthesis_data/carla_pkls/1043_dataset_image.pkl --seed 1 --epochs 200 --model mrgcn --layer_spec 128,64 --num_layers 2 --conv_type FastRGCNConv --batch_size 16 --learning_rate 0.0005 --activation relu --readout_type mean --temporal_type lstm_attn --dropout 0.1 --pooling_type sagpool --pooling_ratio 0.5 --lstm_input_dim 50 --lstm_output_dim 20 --test_step 1 --stats_path 1043_2_layer_image.csv
CUDA_VISIBLE_DEVICES=1 python ./3_train_graph_learning.py  --device cuda --cache_path /home/louisccc/NAS/louisccc/av/synthesis_data/carla_pkls/1043_dataset_image.pkl --seed 1 --epochs 200 --model mrgcn --layer_spec 128,64 --num_layers 2 --conv_type FastRGCNConv --batch_size 16 --learning_rate 0.0005 --activation relu --readout_type mean --temporal_type lstm_attn --dropout 0.1 --pooling_type topk --pooling_ratio 0.5 --lstm_input_dim 50 --lstm_output_dim 20 --test_step 1 --stats_path 1043_2_layer_image.csv
CUDA_VISIBLE_DEVICES=1 python ./3_train_graph_learning.py  --device cuda --cache_path /home/louisccc/NAS/louisccc/av/synthesis_data/carla_pkls/1043_dataset_image.pkl --seed 1 --epochs 200 --model mrgcn --layer_spec 128,64 --num_layers 2 --conv_type FastRGCNConv --batch_size 16 --learning_rate 0.0005 --activation relu --readout_type mean --temporal_type lstm_attn --dropout 0.1 --pooling_type None  --pooling_ratio 1.0  --lstm_input_dim 50 --lstm_output_dim 20 --test_step 1 --stats_path 1043_2_layer_image.csv
CUDA_VISIBLE_DEVICES=1 python ./3_train_graph_learning.py  --device cuda --cache_path /home/louisccc/NAS/louisccc/av/synthesis_data/carla_pkls/1043_dataset_image.pkl --seed 1 --epochs 200 --model mrgcn --layer_spec 128,64 --num_layers 2 --conv_type FastRGCNConv --batch_size 16 --learning_rate 0.0005 --activation relu --readout_type add --temporal_type lstm_attn --dropout 0.1 --pooling_type sagpool --pooling_ratio 0.5 --lstm_input_dim 50 --lstm_output_dim 20 --test_step 1 --stats_path 1043_2_layer_image.csv
CUDA_VISIBLE_DEVICES=1 python ./3_train_graph_learning.py  --device cuda --cache_path /home/louisccc/NAS/louisccc/av/synthesis_data/carla_pkls/1043_dataset_image.pkl --seed 1 --epochs 200 --model mrgcn --layer_spec 128,64 --num_layers 2 --conv_type FastRGCNConv --batch_size 16 --learning_rate 0.0005 --activation relu --readout_type add --temporal_type lstm_attn --dropout 0.1 --pooling_type topk --pooling_ratio 0.5 --lstm_input_dim 50 --lstm_output_dim 20 --test_step 1 --stats_path 1043_2_layer_image.csv
CUDA_VISIBLE_DEVICES=1 python ./3_train_graph_learning.py  --device cuda --cache_path /home/louisccc/NAS/louisccc/av/synthesis_data/carla_pkls/1043_dataset_image.pkl --seed 1 --epochs 200 --model mrgcn --layer_spec 128,64 --num_layers 2 --conv_type FastRGCNConv --batch_size 16 --learning_rate 0.0005 --activation relu --readout_type add --temporal_type lstm_attn --dropout 0.1 --pooling_type None  --pooling_ratio 1.0  --lstm_input_dim 50 --lstm_output_dim 20 --test_step 1 --stats_path 1043_2_layer_image.csv
CUDA_VISIBLE_DEVICES=1 python ./3_train_graph_learning.py  --device cuda --cache_path /home/louisccc/NAS/louisccc/av/synthesis_data/carla_pkls/1043_dataset_image.pkl --seed 1 --epochs 200 --model mrgcn --layer_spec 100,100 --num_layers 2 --conv_type FastRGCNConv --batch_size 16 --learning_rate 0.0005 --activation relu --readout_type mean --temporal_type lstm_attn --dropout 0.1 --pooling_type sagpool --pooling_ratio 0.5 --lstm_input_dim 50 --lstm_output_dim 20 --test_step 1 --stats_path 1043_2_layer_image.csv
CUDA_VISIBLE_DEVICES=1 python ./3_train_graph_learning.py  --device cuda --cache_path /home/louisccc/NAS/louisccc/av/synthesis_data/carla_pkls/1043_dataset_image.pkl --seed 1 --epochs 200 --model mrgcn --layer_spec 100,100 --num_layers 2 --conv_type FastRGCNConv --batch_size 16 --learning_rate 0.0005 --activation relu --readout_type mean --temporal_type lstm_attn --dropout 0.1 --pooling_type topk --pooling_ratio 0.5 --lstm_input_dim 50 --lstm_output_dim 20 --test_step 1 --stats_path 1043_2_layer_image.csv
CUDA_VISIBLE_DEVICES=1 python ./3_train_graph_learning.py  --device cuda --cache_path /home/louisccc/NAS/louisccc/av/synthesis_data/carla_pkls/1043_dataset_image.pkl --seed 1 --epochs 200 --model mrgcn --layer_spec 100,100 --num_layers 2 --conv_type FastRGCNConv --batch_size 16 --learning_rate 0.0005 --activation relu --readout_type mean --temporal_type lstm_attn --dropout 0.1 --pooling_type None  --pooling_ratio 1.0  --lstm_input_dim 50 --lstm_output_dim 20 --test_step 1 --stats_path 1043_2_layer_image.csv
CUDA_VISIBLE_DEVICES=1 python ./3_train_graph_learning.py  --device cuda --cache_path /home/louisccc/NAS/louisccc/av/synthesis_data/carla_pkls/1043_dataset_image.pkl --seed 1 --epochs 200 --model mrgcn --layer_spec 100,100 --num_layers 2 --conv_type FastRGCNConv --batch_size 16 --learning_rate 0.0005 --activation relu --readout_type add --temporal_type lstm_attn --dropout 0.1 --pooling_type sagpool --pooling_ratio 0.5 --lstm_input_dim 50 --lstm_output_dim 20 --test_step 1 --stats_path 1043_2_layer_image.csv
CUDA_VISIBLE_DEVICES=1 python ./3_train_graph_learning.py  --device cuda --cache_path /home/louisccc/NAS/louisccc/av/synthesis_data/carla_pkls/1043_dataset_image.pkl --seed 1 --epochs 200 --model mrgcn --layer_spec 100,100 --num_layers 2 --conv_type FastRGCNConv --batch_size 16 --learning_rate 0.0005 --activation relu --readout_type add --temporal_type lstm_attn --dropout 0.1 --pooling_type topk --pooling_ratio 0.5 --lstm_input_dim 50 --lstm_output_dim 20 --test_step 1 --stats_path 1043_2_layer_image.csv
CUDA_VISIBLE_DEVICES=1 python ./3_train_graph_learning.py  --device cuda --cache_path /home/louisccc/NAS/louisccc/av/synthesis_data/carla_pkls/1043_dataset_image.pkl --seed 1 --epochs 200 --model mrgcn --layer_spec 100,100 --num_layers 2 --conv_type FastRGCNConv --batch_size 16 --learning_rate 0.0005 --activation relu --readout_type add --temporal_type lstm_attn --dropout 0.1 --pooling_type None  --pooling_ratio 1.0  --lstm_input_dim 50 --lstm_output_dim 20 --test_step 1 --stats_path 1043_2_layer_image.csv
