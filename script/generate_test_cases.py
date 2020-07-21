import os
import sys

batch_sizes = ["4","8","16"]
hidden_dims = [
    "128,128,128",
    "128,64,128",
    "128,64,32",
    "128,64,32,16",
    "32,64,128,256",
    "256,128,64,32"
    ]
activations = ['relu','leaky_relu']
learning_rates = ["0.0001", "0.00005"]
readouts = ["mean","add"]
pooling_types = ["None", "sagpool", "topk"]
pooling_ratios = ["0.25", "0.5", "0.75"]
temporal_types = ["lstm_attn"]
dropouts = ["0.1","0.25", "0.5"]
model = "mrgcn"
epochs = "500"
seed = "0"
cache_path = "before.pkl"
device = "cuda"
lstm_input_dim = "50"
lstm_output_dim = "20"
filename = "batch_tests.sh"
command = "python3 ./3_train_graph_learning.py "
lines = []
for dims in hidden_dims:
    layers = str(len(dims.split(",")))
    if "128" in dims or "256" in dims:
        conv = "RGCNConv"
    else:
        conv = "FastRGCNConv"
    for batch_size in batch_sizes:
        for lr in learning_rates:
            for activation in activations:
                for readout in readouts:
                    for temporal_type in temporal_types:
                        for dropout in dropouts:
                            for pool_type in pooling_types:
                                if not pool_type == "None":
                                    for pooling_ratio in pooling_ratios:
                                        lines.append(command + \
                                            " --device "+ device + \
                                            " --cache_path " + cache_path + \
                                            " --seed "+ seed + \
                                            " --epochs "+ epochs + \
                                            " --model " + model + \
                                            " --layer_spec " + dims + \
                                            " --num_layers " + layers + \
                                            " --conv_type " + conv + \
                                            " --batch_size " + batch_size + \
                                            " --learning_rate " + lr + \
                                            " --activation " + activation + \
                                            " --readout_type " + readout + \
                                            " --temporal_type " + temporal_type + \
                                            " --dropout " + dropout + \
                                            " --pooling_type " + pool_type + \
                                            " --pooling_ratio " + pooling_ratio + \
                                            " --lstm_input_dim " + lstm_input_dim + \
                                            " --lstm_output_dim " + lstm_output_dim + "\n")
                                else:
                                    lines.append(command + \
                                            " --device "+ device + \
                                            " --cache_path " + cache_path + \
                                            " --seed "+ seed + \
                                            " --epochs "+ epochs + \
                                            " --model " + model + \
                                            " --layer_spec " + dims + \
                                            " --num_layers " + layers + \
                                            " --conv_type " + conv + \
                                            " --batch_size " + batch_size + \
                                            " --learning_rate " + lr + \
                                            " --activation " + activation + \
                                            " --readout_type " + readout + \
                                            " --temporal_type " + temporal_type + \
                                            " --dropout " + dropout + \
                                            " --pooling_type None " + \
                                            " --pooling_ratio 1.0 " + \
                                            " --lstm_input_dim " + lstm_input_dim + \
                                            " --lstm_output_dim " + lstm_output_dim + "\n")


with open(filename,"w+") as f:
    f.writelines(lines)

print("batch file generated.")
