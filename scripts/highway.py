import numpy as np
import gymnasium
import highway_env
from gymnasium.wrappers import RecordVideo
from model import predict_state, predict_trajectory, detect_collision
from model import MPCConfig
from highway_env.vehicle.kinematics import VehicleGhost
from highway_env.vehicle.kinematics import TrajectoryCurve
import itertools
import random
import os
import imageio
import time

horizon = 15
frame_idx = 0
all_ghost_vehicles = []
MOTION_PRIMITIVES = [
    [2, 2, 2, 2, 2], 
    [3, 3, 3, 2, 2],   
    [4, 4, 4, 2, 2],   
    [0, 2, 2, 2, 2],   
    [0, 3, 3, 2, 2],   
    [0, 4, 4, 2, 2],   
    [1, 2, 2, 2, 2],   
    [1, 3, 3, 2, 2],   
    [1, 4, 4, 2, 2]
]

def main():
    dt = MPCConfig.dt
    env = gymnasium.make('highway-v0', render_mode='rgb_array')
    obs, info = env.reset()
    env = RecordVideo(env, video_folder="highway/videos", episode_trigger=lambda episode: True)
    env.unwrapped.config["simulation_frequency"] = 30
    env.unwrapped.set_record_video_wrapper(env)

    decision_times = []

    for episode in range(1):
        done = truncated = False
        obs, info = env.reset()
        while not (done or truncated):
            step_start = time.time()
            current_state = extract_current_state(env)
            ego_y = current_state['ego_position'][1]

            if any(abs(ego_y - target) < 0.1 for target in [0.0, 4.0, 8.0, 12.0]):
                candidate_primitives = search_composite_motion_primitives(current_state, dt, num_primitives=3)
                final_candidates = candidate_primitives

                solidlane_candidates = search_solidlane_candidates_from_candidates(final_candidates, current_state, dt)
                if solidlane_candidates:
                    final_candidates = solidlane_candidates

                ego_lane = current_state['ego_lane']
                if ego_lane == 3:
                    zero_first_candidates = []
                    for seq in final_candidates:
                        if seq[0] == 0:
                            zero_first_candidates.append(seq)
                    if zero_first_candidates:
                        final_candidates = zero_first_candidates

                dashedlane_candidates = search_dashedlane_candidates_from_candidates(final_candidates, current_state)
                if dashedlane_candidates:
                    final_candidates = dashedlane_candidates

                orient_candidates = Orient_along_the_lane(final_candidates)
                if orient_candidates:
                    final_candidates = orient_candidates

                speed_candidates = search_speed_candidates_from_candidates(final_candidates, current_state, dt, 19)
                if speed_candidates:
                    final_candidates = speed_candidates

                speed_candidates2 = search_speed_candidates2_from_candidates(final_candidates, current_state, dt, 24)
                if speed_candidates2:
                    final_candidates = speed_candidates2

                # if final_candidates:
                #     best_candidate_seq = None
                #     max_avg_speed = -float('inf')
                #     for seq in final_candidates:
                #         trajectory = predict_trajectory(current_state, seq, dt)
                #         avg_speed = compute_avg_speed(trajectory)
                #         if avg_speed > max_avg_speed:
                #             max_avg_speed = avg_speed
                #             best_candidate_seq = seq
                # else:
                #     best_candidate_seq = [4] * 15
                if final_candidates:
                    best_candidate_seq = random.choice(final_candidates)
                else:
                    best_candidate_seq = [4] * 15
                
                step_end = time.time()
                elapsed = step_end - step_start
                decision_times.append(elapsed)
            else:
                best_candidate_seq = [2] * 15

            predicted_trajectory = predict_trajectory(current_state, best_candidate_seq, dt)
            visualize_predicted_trajectory(env, predicted_trajectory, color=(144, 238, 144, 0.6))

            final_action = best_candidate_seq[0]
            obs, reward, done, truncated, info = env.step(final_action)
            for v in all_ghost_vehicles:
                if v in env.unwrapped.road.vehicles:
                    env.unwrapped.road.vehicles.remove(v)
            all_ghost_vehicles.clear()
            env.render()

    decision_times = np.array(decision_times)

    filtered_times = [t for t in decision_times if t != 0]

    if len(filtered_times) == 0:
        m = 0
    else:
        filtered_times = np.array(filtered_times)
        
        mean_time   = np.mean(filtered_times)
        std_time    = np.std(filtered_times)
        median_time = np.median(filtered_times)
        min_time    = np.min(filtered_times)
        max_time    = np.max(filtered_times)

        print("Decision time statistics (excluding zero):")
        print(f"  Mean:   {mean_time:.4f} s")
        print(f"  Std:    {std_time:.4f} s")
        print(f"  Median: {median_time:.4f} s")
        print(f"  Min:    {min_time:.4f} s")
        print(f"  Max:    {max_time:.4f} s")

    env.close()
    print("Simulation terminated.")

def extract_current_state(env):
    ego_vehicle = env.unwrapped.vehicle
    
    ego_lane_index = ego_vehicle.lane_index  # e.g., (from, to, lane_id)
    ego_lane = env.unwrapped.road.network.get_lane(ego_lane_index)
    ego_lane_num = ego_lane_index[2]
    
    ego_long, ego_lat = ego_lane.local_coordinates(ego_vehicle.position)
    
    front_vehicle = None
    min_delta = np.inf
    for veh in env.unwrapped.road.vehicles:
        if veh is ego_vehicle:
            continue
        if veh.lane_index == ego_lane_index:
            veh_long, veh_lat = ego_lane.local_coordinates(veh.position)
            if veh_long > ego_long:
                delta = veh_long - ego_long
                if delta < min_delta:
                    min_delta = delta
                    front_vehicle = veh
    if front_vehicle is not None:
        front_position = [front_vehicle.position[0], front_vehicle.position[1]]
        front_speed = front_vehicle.speed
    else:
        front_position = [ego_vehicle.position[0], ego_vehicle.position[1]]
        front_speed = ego_vehicle.speed
        print("No vehicle ahead in the same lane")
    
    other_lane_candidates = {}  # { lane_id: [ (x_diff, veh), (x_diff, veh), ... ] }
    for veh in env.unwrapped.road.vehicles:
        if veh is ego_vehicle:
            continue
        if veh.lane_index != ego_lane_index:
            x_diff = abs(veh.position[0] - ego_vehicle.position[0])
            lane_id = veh.lane_index 

            if lane_id not in other_lane_candidates:
                other_lane_candidates[lane_id] = []

            other_lane_candidates[lane_id].append((x_diff, veh))
            other_lane_candidates[lane_id].sort(key=lambda x: x[0])  # 按距离从小到大排序
    
    other_positions = []
    other_speeds = []
    if other_lane_candidates:
        for lane, veh_list in other_lane_candidates.items():
            for (x_diff, veh) in veh_list:
                other_positions.append([veh.position[0], veh.position[1]])
                other_speeds.append(veh.speed)
    else:
        print("No vehicles in other lanes")

    state = {
        'ego_lane': ego_lane_num,
        'ego_position': [ego_vehicle.position[0], ego_vehicle.position[1]],
        'ego_speed': ego_vehicle.speed,
        'front_position': front_position,
        'front_speed': front_speed,
        'other_positions': other_positions, 
        'other_speeds': other_speeds
    }
    return state

def check_collision(trajectory):

    for state in trajectory:
        ego_pred_pos = state['ego_position']
        if detect_collision(ego_pred_pos, state['front_position']):
            return 1
        for other_pos in state['other_positions']:
            if detect_collision(ego_pred_pos, other_pos):
                return 1
    return 0

def has_dynamic_lane_violation(trajectory, action_seq):

    for i in range(len(action_seq)):
        lane_i = trajectory[i]['ego_lane']
        a_i = action_seq[i]
        if lane_i == 0 and a_i == 0:
            return True
        if lane_i == 3 and a_i == 1:
            return True
    return False

def search_composite_motion_primitives(current_state, dt, num_primitives=3):

    candidate_space = []
    for primitives in itertools.product(MOTION_PRIMITIVES, repeat=num_primitives):
        composite_seq = []
        for prim in primitives:
            composite_seq.extend(prim)
        trajectory = predict_trajectory(current_state, composite_seq, dt)
        if has_dynamic_lane_violation(trajectory, composite_seq):
            continue
        if check_collision(trajectory) == 0:
            candidate_space.append(composite_seq)
    return candidate_space

def search_solidlane_candidates_from_candidates(candidate_list, current_state, dt):
    filtered_candidates = []
    for action_seq in candidate_list:
        trajectory = predict_trajectory(current_state, action_seq, dt)
        if has_dynamic_lane_violation(trajectory, action_seq):
            continue
        else:
            filtered_candidates.append(action_seq)
    return filtered_candidates

def search_speed_candidates_from_candidates(candidate_list, current_state, dt, speed_threshold): 
    filtered_candidates = []
    for action_seq in candidate_list:
        trajectory = predict_trajectory(current_state, action_seq, dt)
        avg_speed = compute_avg_speed(trajectory)
        if avg_speed > speed_threshold:
            filtered_candidates.append(action_seq)
    return filtered_candidates

def search_speed_candidates2_from_candidates(candidate_list, current_state, dt, speed_threshold):
    filtered_candidates = []
    for action_seq in candidate_list:
        trajectory = predict_trajectory(current_state, action_seq, dt)
        avg_speed = compute_avg_speed(trajectory)
        if avg_speed < speed_threshold:
            filtered_candidates.append(action_seq)
    return filtered_candidates

def search_dashedlane_candidates_from_candidates(candidate_list, current_state, distance_threshold=-100):

    ego_x = current_state['ego_position'][0]
    distances = []
 
    front_position = current_state.get('front_position', None)
    if front_position is not None:
        distances.append(abs(front_position[0] - ego_x))

    min_distance = min(distances) if distances else float('inf')
    
    if min_distance > distance_threshold:
        candidate_space = []
        for action_seq in candidate_list:
            if all(a in [2, 3, 4] for a in action_seq):
                candidate_space.append(action_seq)
        return candidate_space
    else:

        return candidate_list

def Orient_along_the_lane(candidate_list):
    filtered_candidates = []
    for seq in candidate_list:
            last_three = seq[-3:]
            if not any(a in [0, 1] for a in last_three):
                filtered_candidates.append(seq)
    return filtered_candidates

def compute_avg_speed(trajectory):
    speeds = [state['ego_speed'] for state in trajectory]
    return sum(speeds) / len(speeds)

def compute_sequence_cost(action_seq):
    return sum(1 for _ in action_seq)

def visualize_predicted_trajectory(env, predicted_trajectory, color=(144, 238, 144, 0.3), save_folder="predicted_frames"):
    global frame_idx
    ghost_vehicles = []
    for state in predicted_trajectory[1:]:
        x, y = state['ego_position']
        heading_rad = state.get('heading', 0.0)  
        ghost = VehicleGhost(
            road=env.unwrapped.road,
            position=(x, y),
            heading=heading_rad,
            speed=0
        )
        ghost.color = color
        ghost_vehicles.append(ghost)
        env.unwrapped.road.vehicles.append(ghost)
        all_ghost_vehicles.append(ghost)

    env.render()

    if hasattr(env.unwrapped, "viewer") and env.unwrapped.viewer is not None:
        img = env.unwrapped.viewer.get_image()
        os.makedirs(save_folder, exist_ok=True)
        filename = f"predicted_{frame_idx:06d}.png"
        imageio.imwrite(os.path.join(save_folder, filename), img)
        frame_idx += 1
    # for v in ghost_vehicles:
    #     env.unwrapped.road.vehicles.remove(v)

if __name__ == "__main__":
    main()
