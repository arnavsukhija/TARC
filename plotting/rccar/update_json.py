import json
import wandb
import numpy as np
from collections import defaultdict

def update_json():
    api = wandb.Api()
    runs = api.runs("arnavsukhija-eth-zurich/PPO_RCCar_lowFrequency")
    
    # Read existing JSON
    file_path = '/Users/arnavsukhija/Library/CloudStorage/OneDrive-ETHZurich/Bachelor/Thesis/code/TARC/plotting/rccar/RCCar_simulation_results.json'
    with open(file_path, 'r') as f:
        data = json.load(f)
        
    groups = defaultdict(list)
    
    # Process WandB runs
    for run in runs:
        c = run.config
        s = run.summary
        
        ep_len = s.get("eval/avg_episode_length", 200)
        dt_div = c.get("dt_divisor", c.get("base_dt_divisor", "N/A"))
        repeat = c.get("max_time_repeat", "N/A")
        ep_time = c.get("episode_time", "N/A")
        
        freq = float(ep_len) / float(ep_time) if ep_time != "N/A" and ep_time else "N/A"
        if freq == "N/A":
             freq = float(ep_len) / (200.0/30.0)
        freq = round(freq, 2)
        
        # Calculate rewards and actions for this run
        unpen_rewards = []
        raw_actions = []
        
        for i in range(10):
            r_key = f"results/total_reward_{i}"
            a_key = f"results/num_actions_{i}"
            if r_key in s:
                unpen_rewards.append(s[r_key])
            if a_key in s:
                raw_actions.append(s[a_key])
                
        if not unpen_rewards:
             if "total reward" in s:
                 unpen_rewards.append(s.get("total reward", 0))
             elif "total_reward" in s:
                 unpen_rewards.append(s.get("total_reward", 0))
        if not raw_actions:
             if "number of actions" in s:
                 raw_actions.append(s.get("number of actions", 0))
        
        if unpen_rewards:
            mean_unpen = np.mean(unpen_rewards)
            mean_actions = np.mean(raw_actions) if raw_actions else 0
            mean_pen = mean_unpen - (0.1 * mean_actions) # switch_cost=0.1
            groups[freq].append((mean_unpen, mean_pen, freq))

    # Add to JSON
    for freq, metrics in groups.items():
        if freq == 15.0:
            name = "PPO-15"
        elif freq == 9.9 or freq == 10.0 or freq == 10.05:
            name = "PPO-10"
        elif freq == 7.5:
            name = "PPO-7.5"
        else:
            name = f"PPO-{freq}"
            
        unpen = [x[0] for x in metrics]
        pen = [x[1] for x in metrics]
        freqs = [x[2] for x in metrics]
        
        data["unpenalized_rewards"][name] = {
            "mean": np.mean(unpen),
            "sem": np.std(unpen) / np.sqrt(len(unpen)) if len(unpen) > 1 else 0
        }
        data["penalized_rewards"][name] = {
            "mean": np.mean(pen),
            "sem": np.std(pen) / np.sqrt(len(pen)) if len(pen) > 1 else 0
        }
        data["frequencies"][name] = {
            "mean": np.mean(freqs),
            "sem": np.std(freqs) / np.sqrt(len(freqs)) if len(freqs) > 1 else 0
        }
    
    # Save JSON
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)
    print("Successfully updated JSON with new PPO baselines.")

if __name__ == "__main__":
    update_json()
