import torch

from pgae import PGAE, PGAEOPP
from config import PGAEConfig, TrainConfig
from data_util import save_latent
import numpy as np
from dataset import PairedNico2BlocksDataset
from torch.utils.data import DataLoader
from data_util import normalise, pad_with_zeros

# Extract shared representations, viz. binding layer features
def main():
    # get the network configuration (parameters such as number of layers and units)
    paramaters = PGAEConfig()
    paramaters.set_conf("../train/pgae_conf.txt")

    # get the training configuration (batch size, initialisation, number of iterations, saving and loading directory)
    train_conf = TrainConfig()
    train_conf.set_conf("../train/train_conf.txt")
    save_dir = train_conf.save_dir

    # Load the dataset
    training_data = PairedNico2BlocksDataset(train_conf)
    test_data = PairedNico2BlocksDataset(train_conf, True)

    # Get the max and min values for normalisation
    max_joint = np.concatenate((training_data.B_fw, test_data.B_fw), 1).max()
    min_joint = np.concatenate((training_data.B_fw, test_data.B_fw), 1).min()
    max_vis = np.concatenate((training_data.V_fw, test_data.V_fw), 1).max()
    min_vis = np.concatenate((training_data.V_fw, test_data.V_fw), 1).min()

    # normalise the joint angles, visual features between -1 and 1 and pad the fast actions with zeros
    training_data.B_bw = pad_with_zeros(normalise(training_data.B_bw, max_joint, min_joint))
    training_data.B_fw = pad_with_zeros(normalise(training_data.B_fw, max_joint, min_joint))
    test_data.B_bw = pad_with_zeros(normalise(test_data.B_bw, max_joint, min_joint), True)
    test_data.B_fw = pad_with_zeros(normalise(test_data.B_fw, max_joint, min_joint), True)
    training_data.V_bw = pad_with_zeros(normalise(training_data.V_bw, max_vis, min_vis))
    training_data.V_fw = pad_with_zeros(normalise(training_data.V_fw, max_vis, min_vis))
    test_data.V_bw = pad_with_zeros(normalise(test_data.V_bw, max_vis, min_vis), True)
    test_data.V_fw = pad_with_zeros(normalise(test_data.V_fw, max_vis, min_vis), True)

    train_dataloader = DataLoader(training_data)
    test_dataloader = DataLoader(test_data)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using {} device'.format(device))
    model = PGAEOPP(paramaters).to(device)

    # Load the trained model
    checkpoint = torch.load(save_dir + '/pgae_inference_opp_agent_incl.tar')  # get the checkpoint
    model.load_state_dict(checkpoint['model_state_dict'])  # load the model state

    model.eval()
    signal = 'execute'
    # Feed the dataset as input
    for input in train_dataloader:
        L_fw_before = input["L_fw"].transpose(0, 1)
        sentence_idx = 0#np.random.randint(8)
        input["B_fw"] = input["B_fw"].transpose(0, 1).to(device)
        input["V_fw"] = input["V_fw"].transpose(0, 1).to(device)
        input["B_bw"] = input["B_bw"].transpose(0, 1).to(device)
        input["V_bw"] = input["V_bw"].transpose(0, 1).to(device)
        input["V_opp_fw"] = input["V_opp_fw"].transpose(0, 1).to(device)
        input["V_opp_bw"] = input["V_opp_bw"].transpose(0, 1).to(device)
        input["B_bin"] = input["B_bin"].transpose(0, 1).to(device)
        input["VB_fw"] = [input["V_fw"][:, :, :], input["B_fw"][0, :, :]]
        # Choose one of eight description alternatives
        if sentence_idx == 0:
            L_fw_feed = L_fw_before[0:5, :, :]
        elif sentence_idx == 1:
            L_fw_feed = L_fw_before[5:10, :, :]
        elif sentence_idx == 2:
            L_fw_feed = L_fw_before[10:15, :, :]
        elif sentence_idx == 3:
            L_fw_feed = L_fw_before[15:20, :, :]
        elif sentence_idx == 4:
            L_fw_feed = L_fw_before[20:25, :, :]
        elif sentence_idx == 5:
            L_fw_feed = L_fw_before[25:30, :, :]
        elif sentence_idx == 6:
            L_fw_feed = L_fw_before[30:35, :, :]
        else:
            L_fw_feed = L_fw_before[35:40, :, :]

        input["L_fw"] = L_fw_feed.to(device)

        with torch.no_grad():
            h = model.extract_representations(input, signal)
        save_latent(h.cpu(), input["L_filenames"][0], dirname='common_latent')

    # Do the same for the test set
    if train_conf.test:
        for input in test_dataloader:
            L_fw_before = input["L_fw"].transpose(0, 1)
            sentence_idx = 0#np.random.randint(8)
            # Choose one of eight description alternatives
            input["B_fw"] = input["B_fw"].transpose(0, 1).to(device)
            input["V_fw"] = input["V_fw"].transpose(0, 1).to(device)
            input["B_bw"] = input["B_bw"].transpose(0, 1).to(device)
            input["V_bw"] = input["V_bw"].transpose(0, 1).to(device)
            input["V_opp_fw"] = input["V_opp_fw"].transpose(0, 1).to(device)
            input["V_opp_bw"] = input["V_opp_bw"].transpose(0, 1).to(device)
            input["B_bin"] = input["B_bin"].transpose(0, 1).to(device)
            input["VB_fw"] = [input["V_fw"][:, :, :], input["B_fw"][0, :, :]]
            if sentence_idx == 0:
                L_fw_feed = L_fw_before[0:5, :, :]
            elif sentence_idx == 1:
                L_fw_feed = L_fw_before[5:10, :, :]
            elif sentence_idx == 2:
                L_fw_feed = L_fw_before[10:15, :, :]
            elif sentence_idx == 3:
                L_fw_feed = L_fw_before[15:20, :, :]
            elif sentence_idx == 4:
                L_fw_feed = L_fw_before[20:25, :, :]
            elif sentence_idx == 5:
                L_fw_feed = L_fw_before[25:30, :, :]
            elif sentence_idx == 6:
                L_fw_feed = L_fw_before[30:35, :, :]
            else:
                L_fw_feed = L_fw_before[35:40, :, :]

            input["L_fw"] = L_fw_feed.to(device)

            with torch.no_grad():
                h = model.extract_representations(input, signal)
            save_latent(h.cpu(), input["L_filenames"][0], dirname='common_latent')

if __name__ == "__main__":
    main()
