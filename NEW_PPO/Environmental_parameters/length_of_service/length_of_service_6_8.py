import argparse


def get_args():
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
    parser.add_argument("--min_length_of_f_service", type=int, default=6, help="min_length_of_f_service")
    parser.add_argument("--max_length_of_f_service", type=int, default=8, help="max_length_of_f_service")
    parser.add_argument("--min_length_of_f_service_in_ai", type=int, default=5, help="min_length_of_f_service_in_ai")
    parser.add_argument("--max_length_of_f_service_in_ai", type=int, default=5, help="max_length_of_f_service_in_ai")
    parser.add_argument("--min_length_of_ai_service_in_ai", type=int, default=1, help="min_length_of_ai_service_in_ai")
    parser.add_argument("--max_length_of_ai_service_in_ai", type=int, default=3, help="max_length_of_ai_service_in_ai")

    args = parser.parse_args()
    return args