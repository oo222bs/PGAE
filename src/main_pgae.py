import torch
from config import PGAEConfig, TrainConfig
import os
import numpy as np
from pgae import PGAE, PGAEBERT, PGAEOPP, PGAEBERTOPP, train_gmu_opp, validate_gmu_opp
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from dataset import PairedNico2BlocksDataset
from data_util import normalise
import datetime

def main():
    # get the network configuration (parameters such as number of layers and units)
    paramaters = PGAEConfig()
    paramaters.set_conf("../train/pgae_conf.txt")

    # get the training configuration
    # (batch size, initialisation, num_of_epochs number, saving and loading directory)
    train_conf = TrainConfig()
    train_conf.set_conf("../train/train_conf.txt")
    seed = train_conf.seed
    batch_size = train_conf.batch_size
    num_of_epochs = train_conf.num_of_epochs
    learning_rate = train_conf.learning_rate
    save_dir = train_conf.save_dir
    if not os.path.exists(os.path.dirname(save_dir)):
        os.mkdir(os.path.dirname(save_dir))

    # Random Initialisation
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Use GPU if possible
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using {} device'.format(device))
    print("The currently selected GPU is number:", torch.cuda.current_device(),
          ", it's a ", torch.cuda.get_device_name(device=None))
    # Create a model instance
    model = PGAEOPP(paramaters).to(device)
    # Initialise the optimiser
    #optimiser = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)      # Adam optimiser
    scheduler = MultiStepLR(optimiser, milestones=[10000], gamma=0.5)
    #scheduler = ReduceLROnPlateau(optimiser, mode='min', factor=0.5, patience=8000)
    #  Inspect the model with tensorboard
    model_name = 'pgae_inference_opp_agent_incl'
    date = str(datetime.datetime.now()).split('.')[0]
    writer = SummaryWriter(log_dir='.././logs/'+model_name+date)  # initialize the writer with folder "./logs"

    # Load the trained model to continue training
    #checkpoint = torch.load(save_dir + '/pgae_inference_33pc_unimodal_limited_data.tar')       # get the checkpoint
    #model.load_state_dict(checkpoint['model_state_dict'])       # load the model state
    #optimiser.load_state_dict(checkpoint['optimiser_state_dict'])   # load the optimiser state

    model.train()  # tell the model that it's training time

    # Load the dataset
    training_data = PairedNico2BlocksDataset(train_conf)
    test_data = PairedNico2BlocksDataset(train_conf, True)

    # Get the max and min values for normalisation
    max_joint = np.concatenate((training_data.B_fw, test_data.B_fw), 1).max()
    min_joint = np.concatenate((training_data.B_fw, test_data.B_fw), 1).min()
    max_vis = np.concatenate((training_data.V_fw, test_data.V_fw), 1).max()
    min_vis = np.concatenate((training_data.V_fw, test_data.V_fw), 1).min()
    max_vis_opp = np.concatenate((training_data.V_opp_fw, test_data.V_opp_fw), 1).max()
    min_vis_opp = np.concatenate((training_data.V_opp_fw, test_data.V_opp_fw), 1).min()

    # normalise the joint angles, visual features between -1 and 1 and pad the fast actions with zeros
    training_data.B_bw = normalise(training_data.B_bw, max_joint, min_joint) * training_data.B_bin
    training_data.B_fw = normalise(training_data.B_fw, max_joint, min_joint) * training_data.B_bin
    test_data.B_bw = normalise(test_data.B_bw, max_joint, min_joint) * test_data.B_bin
    test_data.B_fw = normalise(test_data.B_fw, max_joint, min_joint) * test_data.B_bin
    training_data.V_bw = normalise(training_data.V_bw, max_vis, min_vis) * training_data.V_bin
    training_data.V_fw = normalise(training_data.V_fw, max_vis, min_vis) * training_data.V_bin
    test_data.V_bw = normalise(test_data.V_bw, max_vis, min_vis) * test_data.V_bin
    test_data.V_fw = normalise(test_data.V_fw, max_vis, min_vis) * test_data.V_bin
    training_data.V_opp_bw = normalise(training_data.V_opp_bw, max_vis_opp, min_vis_opp) * training_data.V_opp_bin
    training_data.V_opp_fw = normalise(training_data.V_opp_fw, max_vis_opp, min_vis_opp) * training_data.V_opp_bin
    test_data.V_opp_bw = normalise(test_data.V_opp_bw, max_vis, min_vis_opp) * test_data.V_opp_bin
    test_data.V_opp_fw = normalise(test_data.V_opp_fw, max_vis, min_vis_opp) * test_data.V_opp_bin

    # Load the training and testing sets with DataLoader
    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    step = 0

    # Training
    for epoch in range(num_of_epochs):
        epoch_loss = []
        for input in train_dataloader:
            input["L_fw"] = input["L_fw"].transpose(0,1)
            input["B_fw"] = input["B_fw"].transpose(0,1).to(device)
            input["V_fw"] = input["V_fw"].transpose(0,1).to(device)
            input["B_bw"] = input["B_bw"].transpose(0,1).to(device)
            input["V_bw"] = input["V_bw"].transpose(0,1).to(device)
            input["V_opp_fw"] = input["V_opp_fw"].transpose(0, 1).to(device)
            input["V_opp_bw"] = input["V_opp_bw"].transpose(0, 1).to(device)
            input["B_bin"] = input["B_bin"].transpose(0,1).to(device)
            input["VB_fw"] = [input["V_fw"][:, :, :], input["B_fw"][0, :, :]]

            sentence_idx = np.random.randint(8)  # Generate random index for description alternatives

            #Choose one of the eight description alternatives according to the generated random index
            L_fw_feed = input["L_fw"][5*sentence_idx:5+5*sentence_idx, :, :]
            input["L_fw"] = L_fw_feed.to(device)

            # Train and print the losses
            l, b, t, signal = train_gmu_opp(model, input, optimiser, epoch_loss, paramaters)
            print("step:{} total:{}, language:{}, behavior:{}, signal:{}".format(step, t, l, b, signal))
            step = step +1

        writer.add_scalar('Training Loss', np.mean(epoch_loss), epoch)     # add the overall loss to the Tensorboard
        scheduler.step()

        # Testing
        if train_conf.test and (epoch+1) % train_conf.test_interval == 0:
            epoch_loss_t = []
            for input in test_dataloader:
                input["L_fw"] = input["L_fw"].transpose(0, 1)
                input["B_fw"] = input["B_fw"].transpose(0, 1).to(device)
                input["V_fw"] = input["V_fw"].transpose(0, 1).to(device)
                input["B_bin"] = input["B_bin"].transpose(0, 1).to(device)
                input["V_opp_fw"] = input["V_opp_fw"].transpose(0, 1).to(device)
                input["V_opp_bw"] = input["V_opp_bw"].transpose(0, 1).to(device)
                input["VB_fw"] = [input["V_fw"][:, :, :], input["B_fw"][0, :, :]]
                input["B_bw"] = input["B_bw"].transpose(0, 1).to(device)
                input["V_bw"] = input["V_bw"].transpose(0, 1).to(device)

                sentence_idx = np.random.randint(8)
                # Choose one of the eight description alternatives according to the generated random index
                L_fw_feed = input["L_fw"][5 * sentence_idx:5 + 5 * sentence_idx, :, :]
                input["L_fw"] = L_fw_feed.to(device)

                # Calculate and print the losses
                l, b, t, signal = validate_gmu_opp(model, input, epoch_loss_t, paramaters)
                print("test")
                print("step:{} total:{}, language:{}, behavior:{}, signal:{}".format(step, t, l, b, signal))
            writer.add_scalar('Test Loss', np.mean(epoch_loss_t), epoch)  # add the overall loss to the Tensorboard

        # Save the model parameters at every log interval
        if (epoch+1) % train_conf.log_interval == 0:
            torch.save({'model_state_dict': model.state_dict(),
                        'optimiser_state_dict': optimiser.state_dict()},
                       save_dir + '/'+model_name+'.tar')
    # Flush and close the summary writer of Tensorboard
    writer.flush()
    writer.close()

if __name__ == '__main__':
    main()