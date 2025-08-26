import os
import time
import random

import numpy as np

import shutil
from enum import Enum

import torch
import torchvision.transforms as transforms

def ece_calculator(result_dict):
    
    list_max_confidence = result_dict['max_confidence']
    list_prediction = result_dict['prediction']
    list_label = result_dict['label']

    torch_list_prediction = torch.tensor(list_prediction).int()
    torch_list_label = torch.tensor(list_label).int()

    torch_correct = (torch_list_prediction == torch_list_label)
    list_correct = torch_correct.tolist()


    ece_data = ECE_Loss(20, list_prediction, list_max_confidence, list_correct)
    acc = sum(list_correct)/len(list_correct)

    # print('acc: ', acc*100)
    # print('ece: ', ece_data[0]*100)
          
    return acc*100, ece_data[0]*100

def ECE_Loss(num_bins, predictions, confidences, correct):
    #ipdb.set_trace()
    bin_boundaries = torch.linspace(0, 1, num_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    bin_accuracy = [0]*num_bins
    bin_confidence = [0]*num_bins
    bin_num_sample = [0]*num_bins

    for idx in range(len(predictions)):
        #prediction = predictions[idx]
        confidence = confidences[idx]
        bin_idx = -1
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            bin_idx += 1 
            bin_lower = bin_lower.item()
            bin_upper = bin_upper.item()
            #if bin_lower <= confidence and confidence < bin_upper:
            if bin_lower < confidence and confidence <= bin_upper:
                bin_num_sample[bin_idx] += 1
                bin_accuracy[bin_idx] += correct[idx]
                bin_confidence[bin_idx] += confidences[idx]
    
    for idx in range(num_bins):
        if bin_num_sample[idx] != 0:
            bin_accuracy[idx] = bin_accuracy[idx]/bin_num_sample[idx]
            bin_confidence[idx] = bin_confidence[idx]/bin_num_sample[idx]

    ece_loss = 0.0
    for idx in range(num_bins):
        temp_abs = abs(bin_accuracy[idx]-bin_confidence[idx])
        ece_loss += (temp_abs*bin_num_sample[idx])/len(predictions)

    return ece_loss, bin_accuracy, bin_confidence, bin_num_sample

def accuracy_writer(args, results, ece_res, log_path=None, file_path=None):

    print("======== Result Summary ========")
    print("params: nstep	lr	bs")
    print("params: {}	{}	{}".format(args.tta_steps, args.lr, args.batch_size))
    print("[set_id] \t Top-1 acc. \t Top-5 acc. \t ECE")
    for id in results.keys():
        print("{}".format(id), end=" \t\t")
        print("{:.2f}".format(results[id][0]), end=" \t\t")
        print("{:.2f}".format(results[id][1]), end=" \t\t")
        print("{:.2f}".format(ece_res[id]), end=" \t\t")
        print("\n")
    print("============== End =============")
    if args.log_dir is not None:
        assert log_path is not None and file_path is not None
        # Create a folder named with the current date
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        with open(file_path, 'a') as f:
            f.write("\n================== Result Summary ==================\n")
            f.write("params: nstep\tlr\tbs\n")
            f.write(f"params: {args.tta_steps}\t{args.lr}\t{args.batch_size}\n")
            f.write("[set_id] \t Top-1 acc. \t Top-5 acc. \t ECE \n")
            for id in results.keys():
                f.write(f"{id} \t\t\t")
                f.write(f"{results[id][0]:.2f} \t\t\t")
                f.write(f"{results[id][1]:.2f} \t\t\t")
                f.write(f"{ece_res[id]:.2f} \t\t\t")
                f.write("\n")
            f.write("======================== End =======================\n\n")