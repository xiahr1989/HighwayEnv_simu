import numpy as np

class MPCConfig:
    dt = 0.3           # Time interval per step (seconds)
    horizon = 15      # Predict the next 2 dt steps
    action = 2

def predict_state(current_state, ego_action, dt):
    """
    Single-step state prediction model (including ego vehicle speed update).

    Parameters:
      current_state (dict): The current state, which includes:
          - 'ego_lane': The lane index of the ego vehicle (e.g., an integer)
          - 'ego_position': [x, y] position of the ego vehicle
          - 'ego_speed': Speed of the ego vehicle
          - 'front_position': [x, y] position of the leading vehicle
          - 'front_speed': Speed of the leading vehicle
          - 'other_positions': List of [x, y] positions of the nearest vehicles in other lanes
          - 'other_speeds': List of speeds of vehicles in other lanes
      ego_action (int): Predicted action for the ego vehicle (3: accelerate, 4: decelerate, others: maintain)
      dt (float): Time step (seconds)

    Returns:
      new_state (dict): The updated state.
    """
    new_state = {}

    # Update lane index and lateral position
    current_lane = current_state['ego_lane']
    ego_x, ego_y = current_state['ego_position']

    lane_change_phase = current_state.get('lane_change_phase', 0)
    target_y = current_state.get('target_y', ego_y)
    lane_change_direction = current_state.get('lane_change_direction', 0)  # -1(左), 0(无变道), 1(右)
    current_heading = current_state.get('heading', 0)              # 上一步的车头朝向

    # Default remains unchanged
    new_lane = current_lane
    new_y = ego_y

    if ego_action == 0:  # 向左变道
        # 如果当前不在变道过程，才开始新的变道过程
        if lane_change_phase == 0:
            new_lane = max(current_lane - 1, 0)  # 车道编号立刻变为左侧车道
            target_y = ego_y - 4               # 假设相邻车道y差=4
            lane_change_phase = 3              # 总共需要3步过渡
            lane_change_direction = -1  # -1代表向左
    elif ego_action == 1:  # 向右变道
        if lane_change_phase == 0:
            new_lane = min(current_lane + 1, 3)  # 车道编号立刻变为右侧车道
            target_y = ego_y + 4
            lane_change_phase = 3
            lane_change_direction = 1  # +1代表向右

    # ========== 4) 如果还在变道过渡阶段，则每步平滑移动一部分 ==========
    if lane_change_phase > 0:
        # 每步移动 (target_y - ego_y) / lane_change_phase
        step_delta = (target_y - ego_y) / lane_change_phase
        new_y = ego_y + step_delta
        lane_change_phase -= 1  # 走完一步后剩余步数再减1
    else:
        # 如果不在变道，则重置方向为 0
        lane_change_direction = 0
    
    new_state['ego_lane'] = new_lane

    def get_heading_degrees(phase, direction):
        if phase == 3:
            return 0 * direction
        elif phase == 2:
            return 0.35 * direction
        elif phase == 1:
            return 0.17 * direction
        else:
            return 0

    new_heading = get_heading_degrees(lane_change_phase, lane_change_direction)
    new_state['heading'] = new_heading

    # Update ego vehicle speed based on the predicted action
    current_ego_speed = current_state['ego_speed']
    if ego_action == 3:  # Accelerate
        next_ego_speed = current_ego_speed + 0.9
    elif ego_action == 4:  # Decelerate
        next_ego_speed = max(current_ego_speed - 0.9, 0)
    else:
        next_ego_speed = current_ego_speed
    new_state['ego_speed'] = next_ego_speed

    # Update ego vehicle position (assume the vehicle travels at the updated speed during dt)
    new_x = ego_x + next_ego_speed * dt
    new_state['ego_position'] = [new_x, new_y]

    # Update the leading vehicle's position (assume its speed remains constant)
    front_x, front_y = current_state['front_position']
    new_state['front_position'] = [front_x + current_state['front_speed'] * dt, front_y]
    new_state['front_speed'] = current_state['front_speed']

    # Update positions of vehicles in other lanes (assume their speeds remain constant and y-position stays unchanged)
    new_other_positions = []
    for pos, speed in zip(current_state['other_positions'], current_state['other_speeds']):
        other_x, other_y = pos
        new_other_positions.append([other_x + speed * dt, other_y])
    new_state['other_positions'] = new_other_positions
    new_state['other_speeds'] = current_state['other_speeds']

    # ★ 一定要把这两个关键信息记录回去 ★
    new_state['lane_change_phase'] = lane_change_phase
    new_state['target_y'] = target_y
    new_state['lane_change_direction'] = lane_change_direction

    return new_state

def predict_trajectory(initial_state, ego_action_seq, dt):
    """
    Predict the future state trajectory based on a sequence of ego vehicle actions.

    Parameters:
      initial_state (dict): The initial state (see predict_state for details)
      ego_action_seq (list): A list of predicted ego actions (e.g., [3, 2] means accelerate at step 1 and maintain at step 2)
      dt (float): Time step (seconds)

    Returns:
      trajectory (list): A list of predicted states (each state is a dictionary)
    """
    trajectory = [initial_state]
    current_state = initial_state
    for action in ego_action_seq:
        next_state = predict_state(current_state, action, dt)
        trajectory.append(next_state)
        current_state = next_state
    return trajectory

def detect_collision(pos1, pos2, length=8, width=2.2):
    """
    Detect whether two vehicles have collided.

    Parameters:
      pos1, pos2 (list): Center positions [x, y] of the two vehicles
      length (float): Vehicle length in meters
      width (float): Vehicle width in meters

    Returns:
      bool: True if the absolute difference in x is less than length and the absolute difference in y is less than width; otherwise, False.
    """
    if abs(pos1[0] - pos2[0]) < length and abs(pos1[1] - pos2[1]) < width:
        return True
    else:
        return False
