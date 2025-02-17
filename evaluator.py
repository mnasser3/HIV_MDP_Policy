import numpy as np
import re

def simulate_policy(policy, num_states, num_actions, T, R, num_episodes=100, max_steps=300):
    total_rewards = []
    episode_start_end_states = []  

    for _ in range(num_episodes):
        state = np.random.choice(num_states)  
        first_state = state  
        cumulative_reward = 0
        last_state = state  
        for _ in range(max_steps):
            action = policy[state]#
            if T[state * num_actions + action].sum() == 0:
                break

            next_state = np.random.choice(num_states, p=T[state * num_actions + action])
            cumulative_reward += R[state]
            last_state = next_state  
            state = next_state

        total_rewards.append(cumulative_reward)
        episode_start_end_states.append((first_state, last_state))  

    return np.mean(total_rewards), episode_start_end_states

def get_cd4_and_vl_from_rollouts(resulting_states,int_to_state):
    ini_and_final_states=[]
    for episode in resulting_states:
        IS, FS = episode[0], episode[1]
        ini_state = int_to_state[str(IS+1)]
        final_state = int_to_state[str(FS+1)]
        ini_and_final_states.append([ini_state,final_state])
    return ini_and_final_states

def evaluate_cd4_and_vl_change(start, end):
    def calculate_averages(range_string):
        matches = re.findall(r'(\d+|inf)-(\d+|inf)', range_string)
        averages = []
        for low, high in matches:
            if high == 'inf':
                averages.append(int(low))
            elif low == 'inf':
                averages.append(int(high))
            else:
                avg = (int(low) + int(high)) / 2
                averages.append(avg)
        return averages

    averages_start = calculate_averages(start)
    averages_end = calculate_averages(end)
    
    return np.array(averages_end) - np.array(averages_start)

def evaluate_change_policies(rollout_results):
    changes=[]
    for start,end in rollout_results:
        change = evaluate_cd4_and_vl_change(start,end)
        changes.append(change)
    return list(changes)

def check_improvment_ratio(changes):
    best_count,good_count,bad_count,worst_count=0,0,0,0
    for change in changes:
        if change[0]>=0 and change[1]<=0:
            best_count +=1
        if change[0]<0 and change[1]<=0:
            good_count+=1
        if change[0]>=0 and change[1]>0:
            bad_count+=1
        if change[0]<0 and change[1]>0:
            worst_count+=1
    return best_count,good_count,bad_count,worst_count

def simulate_policy_on_all_s(policy, num_states, num_actions, T, R, max_steps=300):
    total_rewards = []
    episode_start_end_states = []

    for start_state in range(num_states):  
        state = start_state  # Set the initial state to the current starting state
        first_state = state  
        cumulative_reward = 0
        last_state = state  

        for _ in range(max_steps):
            action = policy[state]
            if T[state * num_actions + action].sum() == 0:
                break

            next_state = np.random.choice(num_states, p=T[state * num_actions + action])
            cumulative_reward += R[state]
            last_state = next_state  
            state = next_state

        total_rewards.append(cumulative_reward)
        episode_start_end_states.append((first_state, last_state))  

    return np.mean(total_rewards), episode_start_end_states
