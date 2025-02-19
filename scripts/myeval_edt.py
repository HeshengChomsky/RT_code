import os
import random
from datetime import datetime

# import d4rl
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from decision_transformer.model import ReveseDecisionTransformer
from decision_transformer.utils import (
    RTTrajectoryDataset,
    ModelSaver,
    encode_return,
    parse,
    EnvalutionD4RLTrajectoryDataset
)
from torch.utils.data import DataLoader


# from omegaconf import OmegaConf


def train(args):
    scaler = torch.cuda.amp.GradScaler()
    model_saver = ModelSaver(args)

    # seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    dataset = args.dataset  # medium / medium-replay
    rtg_scale = args.rtg_scale  # normalize returns to go
    num_bin = args.num_bin
    top_percentile = args.top_percentile
    dt_mask = args.dt_mask
    expert_weight = args.expert_weight
    exp_loss_weight = args.exp_loss_weight

    if args.env == "walker2d":
        rtg_target = 5000
        env_d4rl_name = f"walker2d-{dataset}-v2"
        env_name = env_d4rl_name
    elif args.env == "halfcheetah":
        rtg_target = 6000
        env_d4rl_name = f"halfcheetah-{dataset}-v2"
        env_name = env_d4rl_name
    elif args.env == "hopper":
        rtg_target = 3600
        env_d4rl_name = f"hopper-{dataset}-v2"
        env_name = env_d4rl_name
    elif args.env == "ant":
        rtg_target = 3600
        env_d4rl_name = f"ant-{dataset}-v2"
        env_name = env_d4rl_name
    else:
        raise NotImplementedError

    max_eval_ep_len = args.max_eval_ep_len  # max len of one episode
    num_eval_ep = args.num_eval_ep  # num of evaluation episodes

    batch_size = args.batch_size  # training batch size
    lr = args.lr  # learning rate
    wt_decay = args.wt_decay  # weight decay
    warmup_steps = args.warmup_steps  # warmup steps for lr scheduler

    # total updates = max_train_iters x num_updates_per_iter
    max_train_iters = args.max_train_iters
    max_train_iters = 5
    num_updates_per_iter = args.num_updates_per_iter
    mgdt_sampling = args.mgdt_sampling

    context_len = args.context_len  # K in decision transformer
    n_blocks = args.n_blocks  # num of transformer blocks
    embed_dim = args.embed_dim  # embedding (hidden) dim of transformer
    n_heads = args.n_heads  # num of transformer heads
    dropout_p = args.dropout_p  # dropout probability
    expectile = args.expectile
    rs_steps = args.rs_steps
    state_loss_weight = args.state_loss_weight
    rs_ratio = args.rs_ratio
    real_rtg = args.real_rtg
    data_ratio = args.data_ratio
    eval_chk_pt_dir = args.chk_pt_dir

    eval_chk_pt_name = args.chk_pt_name
    eval_chk_pt_list = [eval_chk_pt_name]

    eval_d4rl_score_mson = None

    # load data from this file
    dataset_path_u = os.path.join(args.dataset_dir, f"{env_d4rl_name}.pkl")
    # dataset_path_u = "/mnt/f/Ubuntu/workspace/pepars/Elastic-DT-master/data/hopper-medium-v2.pkl"
    # saves model and csv in this directory
    log_dir = args.log_dir
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # training and evaluation device
    device = torch.device(args.device)

    start_time = datetime.now().replace(microsecond=0)
    start_time_str = start_time.strftime("%y-%m-%d-%H-%M-%S")

    prefix = "edt_" + env_d4rl_name + f"_{args.seed}"

    save_model_name = prefix + "_model_" + start_time_str + ".pt"
    save_model_path = os.path.join(log_dir, save_model_name)

    log_csv_name = prefix + "_log_" + start_time_str + ".csv"
    log_csv_path = os.path.join(log_dir, log_csv_name)



    traj_dataset_u = EnvalutionD4RLTrajectoryDataset(
        dataset_path_u, context_len, rtg_scale, data_ratio=data_ratio
    )

    ## get state stats from dataset
    state_mean, state_std = traj_dataset_u.get_state_stats()

    env = gym.make('Hopper-v3')
    env.seed(args.seed)

    state_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    new_data=[]

    for eval_chk_pt_name in eval_chk_pt_list:
        model = ReveseDecisionTransformer(
            state_dim=state_dim,
            act_dim=act_dim,
            n_blocks=n_blocks,
            h_dim=embed_dim,
            context_len=context_len,
            n_heads=n_heads,
            drop_p=dropout_p,
            env_name=env_name,
            num_bin=num_bin,
            dt_mask=dt_mask,
            rtg_scale=rtg_scale,
            real_rtg=real_rtg,
        ).to(device)

        eval_chk_pt_path = os.path.join(eval_chk_pt_dir, eval_chk_pt_name)

        # load checkpoint
        model.load_state_dict(torch.load(eval_chk_pt_path, map_location=device))

        print("model loaded from: " + eval_chk_pt_path)

        timesteps_u,states_u,next_states_u,actions_u,returns_to_go_u,rewards_u,traj_mask_u= get_toaken(traj_dataset_u)


        timesteps_u = timesteps_u.to(device)  # B x T
        states_u = states_u.to(device)  # B x T x state_dim
        next_states_u = next_states_u.to(device)
        actions_u = actions_u.to(device)  # B x T x act_dim
        returns_to_go_u = returns_to_go_u.to(device).unsqueeze(
            dim=-1
        )  # B x T x 1
        rewards_u = rewards_u.to(device).unsqueeze(dim=-1)  # B x T x 1
        traj_mask_u = traj_mask_u.to(device)  # B x T

        for i in range(args.max_genrate):
            (
                state_preds_u,
                action_preds_u,
                return_preds_u,
                imp_return_preds_u,
                reward_preds_u,
            ) = model.forward(
                timesteps=timesteps_u,
                states=states_u,
                actions=actions_u,
                returns_to_go=returns_to_go_u,
                rewards=rewards_u,
            )
            timesteps_u=torch.cat([timesteps_u,torch.ones(1)],dim=-1)
            new_state=state_preds_u[-1].reshape([-1,state_dim])
            states_u=torch.cat([states_u,new_state],dim=-1)
            new_action=action_preds_u[-1].reshape([-1,act_dim])
            states_u=torch.cat([states_u,new_action])
            new_rtgo=return_preds_u[-1]
            returns_to_go_u=torch.cat([returns_to_go_u,new_rtgo],dim=-1)
            new_reward=_return_search_heuristic(model,traj_dataset_u)
            rewards_u=torch.cat([rewards_u,new_reward],dim=-1)

        traj={'observations':states_u.numpy(),'actions':actions_u.numpy(),'timestep':timesteps_u.numpy(),'rewards':rewards_u.numpy()}
        new_data.append(traj)
        np.savez(args.data_path+"/"+args.env_name+"agument_data.npz",trajectories=new_data)








if __name__ == "__main__":
    args = parse()
    args.data_path=str("data")
    # wandb.init(project=args.project_name, config=OmegaConf.to_container(args, resolve=True))

    train(args)
