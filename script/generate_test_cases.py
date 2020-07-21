import os
import sys

batch_sizes = ["4","8","16"]
hidden_dims = [
    "64,64,64",
    "128,128,128",
    "64,32,64",
    "128,64,128",
    "128,64,32",
    "128,64,32,16",
    "32,64,128,256",
    "256,128,64,32"
    ]
learning_rates = ["0.0001", "0.00005","0.00002"]
readouts = ["mean","add"]
pooling_types = ["None", "sagpool", "topk"]
pooling_ratios = ["0.25", "0.5", "0.75"]
temporal_types = ["lstm_attn"]
dropouts = ["0.1","0.25", "0.5"]

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
            for readout in readouts:
                for temporal_type in temporal_types:
                    for dropout in dropouts:
                        for pool_type in pooling_types:
                            if not pool_type == "None":
                                for pooling_ratio in pooling_ratios:
                                    lines.append(command + \
                                        " --device cuda "+ \
                                        " --cache_path before.pkl " + \
                                        " --seed 0"+ \
                                        " --epochs 500 "+ \
                                        " --layer_spec " + dims + \
                                        " --layers " + layers + \
                                        " --conv_type " + conv + \
                                        " --batch_size " + batch_size + \
                                        " --learning_rate " + lr + \
                                        " --readout_type " + readout + \
                                        " --temporal_type " + temporal_type + \
                                        " --dropout " + dropout + \
                                        " --pooling_type " + pool_type + \
                                        " --pooling_ratio " + pooling_ratio + \
                                        " --lstm_input_dim 50 " + \
                                        " --lstm_output_dim 20 \n")
                            else:
                                lines.append(command + \
                                        " --device cuda "+ \
                                        " --cache_path before.pkl " + \
                                        " --seed 0 "+ \
                                        " --epochs 500 "+ \
                                        " --layer_spec " + dims + \
                                        " --layers " + layers + \
                                        " --conv_type " + conv + \
                                        " --batch_size " + batch_size + \
                                        " --learning_rate " + lr + \
                                        " --readout_type " + readout + \
                                        " --temporal_type " + temporal_type + \
                                        " --dropout" + dropout + \
                                        " --pooling_type None " + \
                                        " --pooling_ratio 1.0 " + \
                                        " --lstm_input_dim 50 " + \
                                        " --lstm_output_dim 20 \n")


with open(filename,"w+") as f:
    f.writelines(lines)

print("batch file generated.")
