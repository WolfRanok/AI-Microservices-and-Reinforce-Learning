# 这是一个示例 Python 脚本。
import csv
import os

import wandb

import Arguments
from Environment import ENV
from Agent import *
from Normalization import *


# 按 Shift+F10 执行或将其替换为您的代码。
# 按 双击 Shift 在所有地方搜索类、文件、工具窗口、操作和设置。


def print_hi(name):
    # 在下面的代码行中使用断点来调试脚本。
    print(f'Hi, {name}')  # 按 Ctrl+F8 切换断点。

def save_mode(Agent, actor_path, critic_path):
    """
        用于保存模型，由于不同模型接受的参数规模不一样（这是由于服务器数量，微服务类型数量等因素导致的），所以这里按照输入的类型进行命名
        :return:None
        """
    try:
        # 创建目录（如果不存在）
        os.makedirs("Model", exist_ok=True)
        # 保存模型
        torch.save(Agent.actor.state_dict(), actor_path)
        torch.save(Agent.critic.state_dict(), critic_path)
        print("模型已保存！")
    except FileNotFoundError as e:
        print(f"保存路径错误: {e}. 请检查目录是否正确。")
    except PermissionError as e:
        print(f"权限错误: {e}. 请检查文件写入权限。")
    except TypeError as e:
        print(f"数据序列化错误: {e}. 请确保 self.statistics 中的数据可被 JSON 序列化。")
    except Exception as e:
        print(f"保存模型和数据时发生未知错误: {e}")

def load_model(agent):
    """
    当模型存在时，用于加载模型
    :return: None
    """

    actor_url = f"actor_Model/SIL_GNN_PPO_actor_target_model_5e-5_1e-5.pth"
    critic_url = f"Model/SIL_GNN_PPO_critic_target_model_5e-5_1e-5.pth"

    try:
        if os.path.exists(actor_url) and os.path.exists(critic_url):
            agent.actor.load_state_dict(torch.load(actor_url))
            agent.critic.load_state_dict(torch.load(critic_url))

            # # 网络复制
            # agent.actor.load_state_dict(agent.actor_target.state_dict())
            # agent.critic.load_state_dict(agent.critic_target.state_dict())
            print("模型成功加载")
        else:
            print("模型文件不存在，请检查路径")
    except Exception as e:
        print(f"加载模型时出错: {e}")

def write_training_data(path, total_episodes, actor_loss, critic_loss, entropy, advantage):
    # os.makedirs("Data", exist_ok=True)
    with open(path, 'a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        # 第一次写入表头
        if file.tell() == 0:  # 检查文件是否为空
            data = ["episode", "actor_loss", "critic_loss", "entropy", "advantage"]
            writer.writerow(data)
        # 写入新行
        new_row = [total_episodes, actor_loss, critic_loss, entropy, advantage]
        writer.writerow(new_row)
def write_evaluate_data(path, evaluate_num, evaluate_reward, evaluate_delay):
    # os.makedirs("Data", exist_ok=True)
    with open(path, 'a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        # 第一次写入表头
        if file.tell() == 0:  # 检查文件是否为空
            data = ["evaluate_num", "evaluate_reward", "evaluate_delay"]
            writer.writerow(data)
        # 写入新行
        new_row = [evaluate_num, evaluate_reward, evaluate_delay]
        writer.writerow(new_row)

def evaluate_policy(env, agent):
    s = env.reset()
    total_steps=0
    done = False
    episode_reward = 0
    rs = []
    while not done:
        total_steps += 1
        mark = s[0][10:]
        action = agent.evaluate([s], mark, env)
        s_, r, done = env.agent_step(action)
        rs.append(r)
        episode_reward += r
        s = s_
    evaluate_reward = episode_reward
    evaluate_sum_delay = env.last_episode_sum_delay
    return evaluate_reward, evaluate_sum_delay, rs

def text_evaluate(env, args):
    args.action_dim = env.NODE_NUM
    agent = PPOAgent(env, args)
    load_model(agent)
    s = env.reset()
    total_steps=0
    done = False
    episode_reward = 0
    rs = []
    Ts = []
    reward_scaling = RewardScaling(shape=1, gamma=args.gamma)


    while not done:
        total_steps += 1
        mark = s[0][10:]
        action = agent.evaluate([s], mark, env)
        # action = env.greedy_deployment()
        s_, r, done = env.agent_step(action)

        episode_reward += r
        if args.use_reward_scaling:
            r = reward_scaling(r)
        rs.append(r)
        Ts.append(env.last_step_sum_delay)
        s = s_
    evaluate_reward = episode_reward
    evaluate_sum_delay = env.last_episode_sum_delay
    load = env.cal_load_balance()
    delay = env.get_total_delay()
    return evaluate_reward, evaluate_sum_delay


def train(env,args):
    run = wandb.init(project="PPO_DATA",name="GNN_PPO_SIL_length_of_service_6_8",dir="MSD_PPO", job_type="training", reinit=True) # wandb日志记录

    args.action_dim = env.NODE_NUM
    args.max_episode_steps = env.Ms_instance_sum
    args.mini_batch_size = env.Ms_instance_sum
    args.batch_size = env.Ms_instance_sum*5
    args.high_return_buffer_size = env.Ms_instance_sum*5
    args.SIL_sample_batch_size = int(env.Ms_instance_sum/2)
    if args.use_reward_scaling:
        reward_scaling = RewardScaling(shape=1, gamma=args.gamma)
    epsilon_scheduler = EpsilonScheduler()
    agent = PPOAgent(env, args)
    # 上次训练中断，这次继续训练
    # load_model(agent, args)

    evaluate_num = 0
    total_steps = 0
    total_episodes=0
    buffer_size= 0
    min_delay = float('inf')
    self_learning_threshold = 0
    e_rs = []
    while total_episodes < args.max_train_episodes:
        total_episodes += 1
        done = False
        state = env.reset()
        reward_scaling.reset()
        T = []
        this_e_r = 0
        this_r_r_after_scaling = 0
        while not done:
            buffer_size += 1
            total_steps += 1
            mark = state[0][10:]
            if args.use_epsilon_greedy:
                action, action_logprob = agent.get_action_epsilon([state], mark, env, args.epsilon)
            else:
                # print(mark)
                if total_episodes%int(args.batch_size/args.mini_batch_size)!=0:
                    action, action_logprob = agent.get_action([state], mark, env)
                else:
                    action, action_logprob = agent.get_max_prob_action([state], mark, env)

            s_, r, done = env.agent_step(action)
            e_rs.append(r)
            this_e_r += r

            if args.use_reward_scaling:
                r = reward_scaling(r)
                this_r_r_after_scaling +=r

            agent.buffer.store(state, action, action_logprob, r, s_, False, done, mark)
            agent.SIL_buffer.store(state, action, action_logprob, r, s_, False, done, mark)
            T.append(env.last_step_sum_delay)
            state = s_
            # 打印本轮采样日志
            if done:
                print(f"---------------------------------采样日志---------------------------------")
                print(f"episode:{total_episodes},本轮样本的奖励值:{this_e_r},奖励伸缩后的奖励值:{this_r_r_after_scaling},本轮样本时延:{env.last_episode_sum_delay}")
            if buffer_size == args.batch_size:
                actor_loss, critic_loss, entropy, advantage = agent.update_model(env, args, total_steps)
                print(f"---------------------------------训练日志---------------------------------")
                print(f"use self imitation:{args.use_self_imitation}")
                agent.buffer.clean()
                buffer_size = 0
                run.log({"actor_loss": actor_loss, "critic_loss": critic_loss, "entropy": entropy, "advantage": advantage})
                write_training_data('Environmental_parameters/length_of_service/Training_data/length_of_service_6_8', total_episodes, actor_loss, critic_loss, entropy, advantage)
                print(f"episode:{total_episodes}, actor_loss: {actor_loss}, critic_loss: {critic_loss}, entropy: {entropy}, advantage: {advantage}")
        if args.use_epsilon_greedy:
            args.epsilon = epsilon_scheduler.step()
        # 打印程序日志
        if total_episodes % args.evaluate_freq == 0:
            evaluate_num += 1
            evaluate_reward, evaluate_sum_delay, rs = evaluate_policy(env, agent)
            if evaluate_sum_delay<=min_delay: ## 保证self imitation可以帮助最低延迟更新
                min_delay = evaluate_sum_delay
            if self_learning_threshold == 1:
                if evaluate_sum_delay <= min_delay:
                    self_learning_threshold = 0
            elif evaluate_sum_delay<min_delay+5:
                self_learning_threshold = 0
            else:
                self_learning_threshold += 1
            # 判断是否需要进行self_imitation
            if self_learning_threshold >= 1:
                args.use_self_imitation = True
            else:
                args.use_self_imitation = False
            e_rs.append(rs)
            write_evaluate_data('Environmental_parameters/length_of_service/Evaluation_data/length_of_service_6_8',evaluate_num, evaluate_reward, evaluate_sum_delay)
            print(f"---------------------------------评测日志---------------------------------")
            print(f"evaluate_num:{evaluate_num},evaluate_reward:{evaluate_reward},evaluate_delay:{evaluate_sum_delay}, min_delay:{min_delay}")
            run.log({"evaluate_reward": evaluate_reward, "evaluate_delay": evaluate_sum_delay})
            # 评估结束后保存模型
            save_mode(agent, 'Environmental_parameters/length_of_service/actor_Model/length_of_service_6_8',
                      'Environmental_parameters/length_of_service/critic_Model/length_of_service_6_8')



# 按间距中的绿色按钮以运行脚本。
if __name__ == '__main__':
    import Environmental_parameters.length_of_service.length_of_service_6_8
    torch.cuda.set_device(0)
    args = Arguments.get_args() # 获取训练参数
    env_args = Environmental_parameters.length_of_service.length_of_service_6_8.get_args()  # 获取环境参数
    env = ENV(env_args)
    # train(env, args)    # 带着环境参数和训练参数进行训练
    text_evaluate(env,args)


# 访问 https://www.jetbrains.com/help/pycharm/ 获取 PyCharm 帮助
