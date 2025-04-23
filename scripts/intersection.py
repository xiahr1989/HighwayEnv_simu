import gymnasium as gym
import highway_env
from typing import List, Dict
import numpy as np
import time
import math
import itertools
from itertools import product
import random
from gymnasium.wrappers import RecordVideo

def main():
    env = gym.make('intersection-v0', render_mode='rgb_array')
    obs, info = env.reset()
    env = RecordVideo(env, video_folder="intersection/videos", episode_trigger=lambda episode: True)
    env.unwrapped.config["simulation_frequency"] = 30
    env.unwrapped.set_record_video_wrapper(env)

    decision_times = []
    dt = 0.3

    for episode in range(1):
        done = truncated = False
        obs, info = env.reset()
        stop_count = 0

        while not (done or truncated):
            step_start = time.time()
            ego_state, non_ego_list = get_and_print_vehicle_info(env)

            safe_candidates = search_composite_motion_primitives(ego_state, non_ego_list, dt, num_primitives=3)
            final_candidates = safe_candidates

            solid_candidates = filter_solid(final_candidates)
            if solid_candidates:
                final_candidates = solid_candidates

            dashed_candidates = filter_solid(final_candidates)
            if dashed_candidates:
                final_candidates = dashed_candidates

            stop_candidates, stop_count = filter_stop_zone(final_candidates, ego_state, stop_count)
            if stop_candidates:
                final_candidates = stop_candidates
            print(f"stop_count={stop_count}")

            orient_candidates = filter_solid(final_candidates)
            if orient_candidates:
                final_candidates = orient_candidates

            speed_candidates1 = filter_speed1(final_candidates, min_avg_speed=3.0)
            if speed_candidates1:
                final_candidates = speed_candidates1

            speed_candidates2 = filter_speed2(final_candidates, max_avg_speed=6.0)
            if speed_candidates2:
                final_candidates = speed_candidates2

            print(f"candidate num={len(final_candidates)}")

            if final_candidates:
                chosen_seq, chosen_traj = random.choice(final_candidates)
                action = chosen_seq[0]
                print(f"Chosen Seq: {chosen_seq} => first action={action}")
            else:
                chosen_seq = [2] * 15
                action = chosen_seq[0]
                print("No feasible trajectory found, fallback to IDLE action.")

            step_end = time.time()
            elapsed = step_end - step_start
            decision_times.append(elapsed)
            print(f"Decision step time: {elapsed:.4f} seconds")

            print("-----------")

            obs, reward, done, truncated, info = env.step(action)
            env.render()
    decision_times = np.array(decision_times)

    filtered_times = np.array([t for t in decision_times if t != 0])

    if filtered_times.size == 0:
        print("No valid (non‑zero) decision times found!")
    else:
        mean_time = filtered_times.mean()
        std_time  = filtered_times.std()

        if std_time == 0:
            final_times = filtered_times
        else:
            zscores = (filtered_times - mean_time) / std_time

            upper_thr =  3.0 
            lower_thr = -1.0 

            keep_mask = (zscores <  upper_thr) & (zscores > lower_thr)
            final_times = filtered_times[keep_mask]

        if final_times.size == 0:
            print("All data were outliers after Z‑score filtering.")
        else:
            print("Decision time statistics:")
            print(f"  Mean:   {final_times.mean():.4f} s")
            print(f"  Std:    {final_times.std():.4f} s")
            print(f"  Median: {np.median(final_times):.4f} s")
            print(f"  Min:    {final_times.min():.4f} s")
            print(f"  Max:    {final_times.max():.4f} s")

    env.close()

def get_and_print_vehicle_info(env):

    ego_vehicle = env.unwrapped.vehicle
    ego_x = ego_vehicle.position[0]
    ego_y = ego_vehicle.position[1]
    ego_speed = ego_vehicle.speed
    ego_heading = ego_vehicle.heading

    ego_destination = "Unknown"
    if ego_vehicle.route and len(ego_vehicle.route) > 0:
        ego_destination = ego_vehicle.route[-1][1]

    print(f"[Ego] position=({ego_x:.2f}, {ego_y:.2f}), speed={ego_speed:.2f}, dest={ego_destination}")

    ego_state = {
        "ego_x": ego_x,
        "ego_y": ego_y,
        "ego_speed": ego_speed,
        "ego_heading": ego_heading
    }

    non_ego_list = []
    for v in env.unwrapped.road.vehicles:
        if v is not ego_vehicle:
            x = v.position[0]
            y = v.position[1]
            spd = v.speed

            non_ego_destination = "Unknown"
            if v.route and len(v.route) > 0:
                non_ego_destination = v.route[-1][1]

            print(f"[Non-Ego] position=({x:.2f}, {y:.2f}), speed={spd:.2f}, dest={non_ego_destination}")

            non_ego_list.append({
                "non_ego_x": x,
                "non_ego_y": y,
                "non_ego_speed": spd
            })

    return ego_state, non_ego_list

def predict_trajectory_intersection(current_state, action_sequence, dt=1):
    x = current_state["ego_x"]
    y = current_state["ego_y"]
    speed = current_state["ego_speed"]
    heading = current_state["ego_heading"]

    trajectory = []
    
    for action in action_sequence:
        if action == 0:   # SLOWER
            speed = max(0.0, speed - 1.5)
        elif action == 2: # FASTER
            speed = speed + 1.5
        else:
            pass
        
        y = y - speed * dt

        state_dict = {
            "x": x,
            "y": y,
            "speed": speed,
            "heading": heading
        }
        trajectory.append(state_dict)

    return trajectory

def predict_trajectory_intersection_non_ego(non_ego_state, horizon=15, dt=1):

    x = non_ego_state["non_ego_x"]
    y = non_ego_state["non_ego_y"]
    speed = non_ego_state["non_ego_speed"]

    trajectory = []
    for _ in range(horizon):
        x = x + speed * dt

        state_dict = {
            "x": x,
            "y": y,
            "speed": speed
        }
        trajectory.append(state_dict)

    return trajectory

def detect_collision(pos1, pos2, length=6.0, width=2.0):

    if abs(pos1[0] - pos2[0]) < length and abs(pos1[1] - pos2[1]) < width:
        return True
    return False

def check_ego_trajectory_collision(ego_traj, non_ego_list, dt=1):

    horizon = len(ego_traj)
    non_ego_trajs = []
    for ne_state in non_ego_list:
        ne_traj = predict_trajectory_intersection_non_ego(ne_state, horizon, dt)
        non_ego_trajs.append(ne_traj)

    for t in range(horizon):
        ego_pos = (ego_traj[t]["x"], ego_traj[t]["y"])
        for i, ne_traj in enumerate(non_ego_trajs):
            ne_pos = (ne_traj[t]["x"], ne_traj[t]["y"])
            if detect_collision(ego_pos, ne_pos, length=15.0, width=5.5):
                return True  
    return False 

def search_composite_motion_primitives(ego_state, non_ego_list, dt=1, num_primitives=3):
    MOTION_PRIMITIVES = [
        [2, 2, 2, 2, 2],
        [2, 2, 2, 1, 1],
        [2, 1, 1, 1, 1], 
        [1, 1, 1, 1, 1], 
        [1, 1, 1, 2, 2],
        [1, 1, 1, 0, 0], 
        [0, 1, 1, 1, 1], 
        [0, 0, 0, 1, 1], 
        [0, 0, 0, 0, 0],
    ]

    candidate_space = []
    for primitives in itertools.product(MOTION_PRIMITIVES, repeat=num_primitives):
        composite_seq = []
        for prim in primitives:
            composite_seq.extend(prim)
        ego_traj = predict_trajectory_intersection(ego_state, composite_seq, dt)
        collision = check_ego_trajectory_collision(ego_traj, non_ego_list, dt)
        if not collision:
            candidate_space.append( (composite_seq, ego_traj) )
    return candidate_space

def build_safe_trajectories(ego_state, non_ego_list, horizon=15, dt=1):
    all_candidates = []
    for seq_tuple in product([0,1,2], repeat=horizon):
        seq = list(seq_tuple)
        ego_traj = predict_trajectory_intersection(ego_state, seq, dt)
        collision = check_ego_trajectory_collision(ego_traj, non_ego_list, dt)
        if not collision:
            all_candidates.append((seq, ego_traj))
    return all_candidates

def filter_stop_zone(final_candidates, ego_state, stop_count):
    y = ego_state["ego_y"]
    speed = ego_state["ego_speed"]
    if not (5 <= y <= 23):
        return final_candidates, stop_count
    if stop_count > 2:
        return final_candidates, stop_count
    filtered = []
    if speed > 0.1:
        for (seq, traj) in final_candidates:
            if seq[0] == 0:
                filtered.append((seq, traj))
    else:
        # speed=0 => stop_count+1
        stop_count += 1
        temp = []
        for (seq, traj) in final_candidates:
            if seq[0] == 1:
                temp.append((seq, traj))
        filtered = temp

    return filtered, stop_count

def filter_solid(final_candidates):
    filtered = []
    for seq, traj in final_candidates:
        keep_this_traj = True
        for step in traj:
            x_val = step["x"]
            if x_val > 4 or x_val < -4:
                keep_this_traj = False
                break
        if keep_this_traj:
            filtered.append((seq, traj))

    return filtered

def filter_dashed(final_candidates):
    filtered = []
    for seq, traj in final_candidates:
        keep_this_traj = True
        for step in traj:
            x_val = step["x"]
            if x_val < 0 and x_val > -4:
                keep_this_traj = False
                break
        if keep_this_traj:
            filtered.append((seq, traj))

    return filtered

def filter_orient(final_candidates):
    tolerance = 0.1 
    desired_headings = [0.0, math.pi/2, math.pi, 3*math.pi/2]

    def is_aligned_with_axes(heading):
        h_mod = heading % (2*math.pi)
        for desired in desired_headings:
            if abs(h_mod - desired) < tolerance:
                return True
        return False

    filtered = []
    for (seq, traj) in final_candidates:
        last_heading = traj[-1].get("heading", None)
        if last_heading is None:
            continue
        if is_aligned_with_axes(last_heading):
            filtered.append((seq, traj))
    return filtered

def filter_speed1(final_candidates, min_avg_speed=2.0):
    filtered1 = []
    for seq, traj in final_candidates:
        speeds = [step["speed"] for step in traj]
        if speeds[0] >= min_avg_speed:
            filtered1.append((seq, traj))
    if filtered1:
        return filtered1
    
    filtered2 = []
    for seq, traj in final_candidates:
        speeds = [step["speed"] for step in traj]
        first_5 = speeds[:5]
        avg_5 = sum(first_5) / 5
        if avg_5 > min_avg_speed:
            filtered2.append((seq, traj))
    if filtered2:
        return filtered2
    
    filtered3 = []
    for seq, traj in final_candidates:
        speeds = [step["speed"] for step in traj]
        avg_total = sum(speeds) / len(speeds)
        if avg_total > min_avg_speed:
            filtered3.append((seq, traj))

    return filtered3

def filter_speed2(final_candidates, max_avg_speed=5.0):
    filtered = []
    for seq, traj in final_candidates:
        speeds = [step["speed"] for step in traj]
        avg_speed = sum(speeds)/len(speeds)
        if avg_speed <= max_avg_speed:
            filtered.append((seq, traj))

    filtered1 = []
    for seq, traj in final_candidates:
        speeds = [step["speed"] for step in traj]
        if speeds[0] <= max_avg_speed:
            filtered1.append((seq, traj))
    if filtered1:
        return filtered1
    
    filtered2 = []
    for seq, traj in final_candidates:
        speeds = [step["speed"] for step in traj]
        first_5 = speeds[:5]
        avg_5 = sum(first_5) / 5
        if avg_5 < max_avg_speed:
            filtered2.append((seq, traj))
    if filtered2:
        return filtered2
    
    filtered3 = []
    for seq, traj in final_candidates:
        speeds = [step["speed"] for step in traj]
        avg_total = sum(speeds) / len(speeds)
        if avg_total < max_avg_speed:
            filtered3.append((seq, traj))

    return filtered3


if __name__ == "__main__":
    main()
