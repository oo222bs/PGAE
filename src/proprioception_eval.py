import matplotlib
matplotlib.use('TkAgg')
import numpy as np
from data_util import read_sequential_target
from config import TrainConfig

def evaluate(signal='execute'):
    train_conf = TrainConfig()
    train_conf.set_conf("../train/train_conf.txt")
    B_fw, B_bw, B_bin, B_len, filenames = read_sequential_target(train_conf.B_dir, True)
    B_fw_u, B_bw_u, B_bin_u, B_len_u, filenames_u = read_sequential_target(train_conf.B_dir_test, True)
    max_joint = np.concatenate((B_fw, B_fw_u), 1).max()
    min_joint = np.concatenate((B_fw, B_fw_u), 1).min()
    B_fw = B_fw.transpose((1,0,2))
    B_fw_u = B_fw_u.transpose((1,0,2))
    B_bw = B_bw.transpose((1,0,2))
    B_bw_u = B_bw_u.transpose((1,0,2))
    B_bin = B_bin.transpose((1,0,2))
    B_bin_u = B_bin_u.transpose((1,0,2))
    predict_train, _, _, predtrain_len, _ = read_sequential_target('../train/inference/prediction/behavior_train/', True)
    predict_test, _, _, predtest_len, _ = read_sequential_target('../train/inference/prediction/behavior_test/', True)
    predict_train = predict_train.transpose((1,0,2))
    predict_test = predict_test.transpose((1,0,2))
    predict_test = predict_test * B_bin_u[:, 1:, :]
    predict_train = predict_train * B_bin[:, 1:, :]
    if signal == 'describe':
        gt_train = np.repeat(np.expand_dims(B_bw[:, 0, :], 1), B_bw.shape[1] - 1, axis=1) * B_bin[:, 1:, :]
        gt_test = np.repeat(np.expand_dims(B_bw_u[:,0,:], 1), B_bw_u.shape[1] - 1, axis=1)*B_bin_u[:,1:,:]
    elif signal == 'execute' or signal == 'repeat action':
        gt_train = B_fw[:, 1:, :]
        gt_test = B_fw_u[:,1:,:]
    else:
        gt_train = np.repeat(np.expand_dims(B_fw[:, 0, :], 1), B_fw.shape[1] - 1, axis=1) * B_bin[:, 1:, :]
        gt_test = np.repeat(np.expand_dims(B_fw_u[:, 0, :], 1), B_fw_u.shape[1] - 1, axis=1) * B_bin_u[:, 1:, :]

    mse_train = np.mean(np.square(predict_train - gt_train))  # action loss (MSE)
    rmse_train = np.sqrt(mse_train)
    nrmse_train = rmse_train / (max_joint-min_joint)

    mse_test = np.mean(np.square(predict_test - gt_test))  # action loss (MSE)
    rmse_test = np.sqrt(mse_test)
    nrmse_test = rmse_test / (max_joint-min_joint)

    print('Normalised Root-Mean squared error (NRMSE) for predicted joint values on the training set: ', nrmse_train*100)
    print('Normalised Root-Mean squared error (NRMSE) for predicted joint values on the test set: ', nrmse_test*100)

if __name__ == '__main__':
    evaluate('repeat action')

