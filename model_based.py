import numpy as np
import pandas as pd

def get_transition_and_reward_matrices(df,state_indeces,action_indeces):
    num_states=len(state_indeces)
    num_actions=len(action_indeces)
    T = np.zeros((num_states*num_actions, num_states))
    R = np.zeros(num_states)

    for state, reward in df.groupby('s')['r'].first().items():
        R[state_indeces[state]] = reward

    for _, row in df.iterrows():
        s = state_indeces[row['s']]
        a = action_indeces[row['a']]
        sp = state_indeces[row['sp']]
        T[s*num_actions+a, sp] += 1
        
    T_sums = T.sum(axis=1, keepdims=True)
    T_sums[np.isclose(T_sums, 0, atol=1e-9)] = 1  # Avoid division by zero
    T = np.divide(T, T_sums, where=(T_sums != 0))
    return T,R

def check_transition_matrix_validity(T):
    T_row_sums = T.sum(axis=1)
    is_valid = np.all((T_row_sums == 0) | np.isclose(T_row_sums, 1, atol=1e-6))

    print("All rows of T sum to 0 or 1:", is_valid)

    invalid_rows = T_row_sums[(T_row_sums != 0) & ~np.isclose(T_row_sums, 1, atol=1e-6)]
    print("Invalid row sums (if any):", invalid_rows)
    
def Q(s, a, num_actions, R, T, V, gamma=0.9):
    row_index = s * num_actions + a
    return R[s] + gamma * np.dot(T[row_index], V)

def Value_Iteration(df,state_indeces,action_indeces,convergence_threshold=1e-9,gamma=0.9):
    num_states=len(state_indeces)
    num_actions=len(action_indeces)
    T,R=get_transition_and_reward_matrices(df,state_indeces,action_indeces)
    check_transition_matrix_validity(T)
    V = np.zeros(num_states)
    delta = 100000 
    
    while delta > convergence_threshold:
        delta = 0  
        for s in range(num_states):
            v = max(Q(s, a,num_actions,R,T,V,gamma) for a in range(num_actions))
            delta = max(delta, abs(V[s] - v))
            V[s] = v
    return (R,T,V)

def policy_extraction(num_states,num_actions,R,T,V):
    policy = np.zeros(num_states, dtype=int)
    for s in range(num_states):
        policy[s] = np.argmax([Q(s, a,num_actions,R,T,V) for a in range(num_actions)])
    return policy