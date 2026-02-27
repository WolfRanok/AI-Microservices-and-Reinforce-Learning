import argparse


def get_args():
    # 创建解析器
    parser = argparse.ArgumentParser("Hyperparameter Setting for GNN-PPO")
    parser.add_argument("--ms_num", type=int, default=12,help="Number of common microservices")
    parser.add_argument("--aims_num", type=int, default=3, help="Number of ai microservices")
    parser.add_argument("--user_num", type=int, default=30, help="Number of users")
    parser.add_argument("--node_num", type=int, default=10, help="Number of servers")
    parser.add_argument("--resource", type=int, default=3, help="Number of resource categories")
    parser.add_argument("--max_length_of_service", type=int, default=7, help="Max length of user requests")
    parser.add_argument("--gpu_of_node_num", type=int, default=5, help="Number of servers")
    parser.add_argument("--min_arrival_rate", type=int, default=2, help="min_arrival_rate")
    parser.add_argument("--max_arrival_rate", type=int, default=4, help="max_arrival_rate")
    parser.add_argument("--AI_service_num", type=int, default=10, help="AI_service_num")
    parser.add_argument("--min_length_of_f_service", type=int, default=3, help="min_length_of_f_service")
    parser.add_argument("--max_length_of_f_service", type=int, default=6, help="max_length_of_f_service")
    parser.add_argument("--min_length_of_f_service_in_ai", type=int, default=3, help="min_length_of_f_service_in_ai")
    parser.add_argument("--max_length_of_f_service_in_ai", type=int, default=4, help="max_length_of_f_service_in_ai")
    parser.add_argument("--min_length_of_ai_service_in_ai", type=int, default=1, help="min_length_of_ai_service_in_ai")
    parser.add_argument("--max_length_of_ai_service_in_ai", type=int, default=3, help="max_length_of_ai_service_in_ai")


    parser.add_argument("--max_train_episodes", type=int, default=20000, help="Maximum number of training episodes")
    parser.add_argument("--max_episode_steps", type=int, default=None,
                        help="Maximum number of steps per training episode")
    parser.add_argument("--mini_batch_size", type=int, default=None, help="Minibatch size")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--high_return_buffer_size", type=int, default=0, help="high_return_buffer_size")
    parser.add_argument("--SIL_sample_batch_size", type=int, default=0, help="self imitation batch_size")
    parser.add_argument("--Server_nums", type=int, default=None, help="Number of server nodes")
    parser.add_argument("--solved_delay", type=float, default=None,
                        help="Delay used to determine whether the algorithm is converging")
    parser.add_argument("--state_dim", type=int, default=None, help="State space dimension")
    parser.add_argument("--action_dim", type=int, default=None, help="Action space dimension")
    parser.add_argument("--hidden_width", type=int, default=256, help="The number of neurons in hidden layers of the neural network")
    parser.add_argument("--evaluate_freq", type=int, default=10, help="Evaluate the policy every 'evaluate_freq' episodes")
    parser.add_argument("--log_interval", type=int, default=100, help="Log printing interval")
    parser.add_argument("--epsilon", type=float, default=0.5, help="Epsilon-Greedy")
    parser.add_argument("--lr_a", type=float, default=5e-5, help="Learning rate of actor")
    parser.add_argument("--lr_c", type=float, default=1e-5, help="Learning rate of critic")
    parser.add_argument("--gamma", type=float, default=0.9, help="Discount factor")
    parser.add_argument("--lamda", type=float, default=0.95, help="GAE parameter")
    parser.add_argument("--clip_range", type=float, default=0.2, help="PPO clip parameter")
    parser.add_argument("--SIL_clip_range", type=float, default=0.2, help="self imitation clip parameter")
    parser.add_argument("--ppo_epochs", type=int, default=5, help="PPO parameter")
    parser.add_argument("--use_epsilon_greedy", type=bool, default=False, help="epsilon greedy")
    parser.add_argument("--use_self_imitation", type=bool, default=False, help="self imitation")
    parser.add_argument("--use_adv_norm", type=bool, default=False, help="Trick 1:advantage normalization")
    parser.add_argument("--use_state_norm", type=bool, default=False, help="Trick 2:state normalization")
    parser.add_argument("--use_reward_norm", type=bool, default=False, help="Trick 3:reward normalization")
    parser.add_argument("--use_reward_scaling", type=bool, default=True, help="Trick 4:reward scaling")
    parser.add_argument("--entropy_coef", type=float, default=0.05, help="Trick 5: policy entropy")
    parser.add_argument("--use_lr_decay", type=bool, default=True, help="Trick 6:learning rate Decay")
    parser.add_argument("--use_grad_clip", type=bool, default=True, help="Trick 7: Gradient clip")
    parser.add_argument("--use_orthogonal_init", type=bool, default=True, help="Trick 8: orthogonal initialization")
    parser.add_argument("--set_adam_eps", type=float, default=True, help="Trick 9: set Adam epsilon=1e-5")
    parser.add_argument("--use_tanh", type=float, default=True, help="Trick 10: tanh activation function")

    # parser.add_argument("--buffer_size", type=int, default=512, help="Size of buffer")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate of DQN algorithm")

    parser.add_argument("--epsilon_start", type=float, default=1.0, help="Initial exploration rate for epsilon-greedy strategy")
    parser.add_argument("--epsilon_end", type=float, default=0.01, help="Final exploration rate for epsilon-greedy strategy")
    parser.add_argument("--epsilon_decay", type=float, default=0.999, help="Decay factor for exploration rate")

    parser.add_argument("--update_freq", type=int, default=10, help="Target update frequency")

    parser.add_argument("--target_update_type", type=str, default='soft', help="Target Update Type")
    parser.add_argument("--tau", type=float, default=1.0, help="Soft update parameter (tau=1.0 for hard update)")

    # 新增路径参数
    # parser.add_argument("--training_data_path", type=str,default="Environmental_parameters/length_of_service/Training_data/dqn_training.csv",help="Path to save training data")
    # parser.add_argument("--evaluate_data_path", type=str,default="Environmental_parameters/length_of_service/Evaluation_data/dqn_evaluate.csv",help="Path to save evaluation data")
    parser.add_argument("--model_path", type=str, default="Environmental_parameters/length_of_service/dqn_Model",help="Path to save/load models")

    args = parser.parse_args()
    return args