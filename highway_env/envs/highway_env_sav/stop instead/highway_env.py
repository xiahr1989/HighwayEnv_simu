from __future__ import annotations
import numpy as np
import torch
from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.common.action import Action
from highway_env.road.lane import LineType, StraightLane
from highway_env.road.road import Road, RoadNetwork
from highway_env.utils import near_split
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.vehicle.kinematics import Vehicle

# import all the stlcg rules
from highway_env.rule_hierarchy.rule_hierarchies.rule_hierarchy import RuleHierarchy
from highway_env.rule_hierarchy.rule_hierarchies.rules.demo import AvoidCollision    # No collision
from highway_env.rule_hierarchy.rule_hierarchies.rules.demo import SolidLaneLine     # Do not cross solid lane line
from highway_env.rule_hierarchy.rule_hierarchies.rules.demo import DashedLaneLine    # Do not cross dashed lane line
from highway_env.rule_hierarchy.rule_hierarchies.rules.demo import AlwaysGreater     # speed > v_min
from highway_env.rule_hierarchy.rule_hierarchies.rules.demo import AlwaysLesser      # speed < v_max

Observation = np.ndarray

class HighwayEnv(AbstractEnv):

    # initialize rule_hierarchy and the rules
    def __init__(self, config: dict = None, render_mode: str = None) -> None:
        super().__init__(config=config, render_mode=render_mode)
        self.rule_hierarchy = self._init_rule_hierarchy()
        self.wr_values = (2, 0.2, 0.02, 0.002, 0.0002) # set weight for each rule here

    def _init_rule_hierarchy(self) -> RuleHierarchy:

        # rules
        rule1 = AvoidCollision()                      # No collision
        rule2 = SolidLaneLine()                       # Do not cross solid lane line
        rule3 = DashedLaneLine()                      # Do not cross dashed lane line
        rule4 = AlwaysGreater(rule_threshold=16.0)    # speed > v_min
        rule5 = AlwaysLesser(rule_threshold=24.0)     # speed < v_max
        rules = [rule1, rule2, rule3, rule4, rule5]

        init = [1.0, 1.0, 1.0, 1.0, 1.0]
        return RuleHierarchy(rules, init)

    # reward calculation and print
    def _reward(self, action: Action) -> float:

        wr1, wr2, wr3, wr4, wr5 = self.wr_values # receive the weight of each rule
        
        # No Collision
        collision_trajectory = torch.tensor([[float(self.vehicle.crashed)]], device="cuda:0" if torch.cuda.is_available() else "cpu",)
        if hasattr(self.rule_hierarchy, "traj_cost"):
            collision_cost, _ = self.rule_hierarchy.traj_cost(collision_trajectory, selected_rule_name="AvoidCollision", get_robustness_vector=True)
            collision_reward = -wr1*collision_cost.item() # calculate reward1

        # Do not cross solid lane line
        # vehicle_x, vehicle_y = self.vehicle.position
        solidlane_trajectory = torch.tensor([self.vehicle.position[1]], device="cuda:0" if torch.cuda.is_available() else "cpu",).unsqueeze(0)
        if hasattr(self.rule_hierarchy, "traj_cost"):
            solidlane_cost, _ = self.rule_hierarchy.traj_cost(solidlane_trajectory, selected_rule_name="SolidLaneLine", get_robustness_vector=True)
            solidlane_reward = -wr2*solidlane_cost.item() # calculate reward2

        # Do not cross dashed lane line
        dashedlane_trajectory = torch.tensor([action], device="cuda:0" if torch.cuda.is_available() else "cpu",).unsqueeze(0)
        if hasattr(self.rule_hierarchy, "traj_cost"):
            dashedlane_cost, _ = self.rule_hierarchy.traj_cost(dashedlane_trajectory, selected_rule_name="DashedLaneLine", get_robustness_vector=True)
            dashedlane_reward = -wr3*dashedlane_cost.item() # calculate reward3

        # Speed > v_min
        speed1_trajectory = torch.tensor([self.vehicle.speed], device="cuda:0" if torch.cuda.is_available() else "cpu",).unsqueeze(0)
        if hasattr(self.rule_hierarchy, "traj_cost"):
            speed1_cost, _ = self.rule_hierarchy.traj_cost(speed1_trajectory, selected_rule_name="AlwaysGreater", get_robustness_vector=True)
            speed1_reward = -wr4*speed1_cost.item() # calculate reward4
        
        # Speed < v_max
        speed2_trajectory = torch.tensor([self.vehicle.speed], device="cuda:0" if torch.cuda.is_available() else "cpu",).unsqueeze(0)
        if hasattr(self.rule_hierarchy, "traj_cost"):
            speed2_cost, _ = self.rule_hierarchy.traj_cost(speed2_trajectory, selected_rule_name="AlwaysLesser", get_robustness_vector=True)
            speed2_reward = -wr5*speed2_cost.item() # calculate reward5

        reward = collision_reward + solidlane_reward + dashedlane_reward + speed1_reward + speed2_reward # reward sum

        # print
        action_name = self.action_type.actions.get(action, "UNKNOWN") # aquire names of actions
        # print(f"Action: {action}({action_name}), Speed: {self.vehicle.speed:.2f}, Collision: {self.vehicle.crashed}, Total Reward: {reward:.2f}")
        # print(f"Reward(No Collision): {collision_reward:.2f}")
        # print(f"Reward(No Cross Solid Line): {solidlane_reward:.2f}")
        # print(f"Reward(No Cross Dashed Line): {dashedlane_reward:.2f}")
        # print(f"Reward(Speed>16): {speed1_reward:.2f}, Reward(Speed<24): {speed2_reward:.2f}")
        # print("-" * 50)
        return reward
    
# -------------------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------------------

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update(
            {
                "observation": {"type": "Kinematics"},
                "action": {
                    "type": "DiscreteMetaAction",
                },
                "lanes_count": 3,
                #"vehicles_count": 50,
                "vehicles_count": 40,
                "controlled_vehicles": 1,
                "initial_lane_id": None,
                "duration": 13,  # [s]
                "ego_spacing": 2,
                "vehicles_density": 1,
                "collision_reward": -1,
                "right_lane_reward": 0.1,
                "high_speed_reward": 0.4,
                "lane_change_reward": 0,
                "reward_speed_range": [10, 30],
                "normalize_reward": True,
                "offroad_terminal": False,
            }
        )
        return config

    def _reset(self) -> None:
        self._create_road()
        self._create_vehicles()

    def _create_road(self) -> None:

        road_network = RoadNetwork()

        lane_width = StraightLane.DEFAULT_WIDTH

        lane0 = StraightLane(
            start=[0, 0],
            end=[2000, 0],  
            width=lane_width,

            line_types=(LineType.CONTINUOUS, LineType.STRIPED),
            speed_limit=30.0,
            forbidden=False
        )
        road_network.add_lane("0", "1", lane0)

        lane1 = StraightLane(
            start=[0, lane_width],
            end=[2000, lane_width],
            width=lane_width,
            line_types=(LineType.STRIPED, LineType.STRIPED),
            speed_limit=30.0,
            forbidden=False
        )
        road_network.add_lane("0", "1", lane1)

        lane2 = StraightLane(
            start=[0, lane_width*2],
            end=[2000, lane_width*2],
            width=lane_width,
            line_types=(LineType.STRIPED, LineType.CONTINUOUS),
            speed_limit=30.0,
            forbidden=False
        )
        road_network.add_lane("0", "1", lane2)

        lane3 = StraightLane(
            start=[0, lane_width*3],
            end=[2000, lane_width*3],
            width=lane_width,
            line_types=(LineType.STRIPED, LineType.CONTINUOUS),
            speed_limit=30.0,
            forbidden=False
        )
        road_network.add_lane("0", "1", lane3)

        self.road = Road(
            network=road_network,
            np_random=self.np_random,
            record_history=self.config["show_trajectories"]
        )

    def _create_vehicles(self) -> None:

        self.controlled_vehicles = []
        lane_id = 2
        longitudinal = 20.0
        lane = self.road.network.get_lane(("0", "1", lane_id))  
        ego_position = lane.position(longitudinal, 0)
        ego_heading = lane.heading_at(longitudinal)
        ego_speed = 21.0

        ego = Vehicle(self.road, ego_position, ego_heading, ego_speed)
        ego = self.action_type.vehicle_class(self.road, ego.position, ego.heading, ego.speed)
        self.controlled_vehicles.append(ego)
        self.road.vehicles.append(ego)

        other_configs = [
            (0, 570.0, 22.0),  
            (1, 560.0, 20.5),  
            (1, 25.0, 27),
            (2, 110.0, 0), 
            (2, 180.0, 17.0),  
            (3, 900.0, 0.0)
            
        ]

        colors = [
            (255, 0,   0),   
            (0,   0,   255), 
            (222, 222, 0),   
            (255, 0,   0), 
            (0,   255, 255), 
            (128, 128, 128)  
        ]

        for i, (o_lane_id, o_longitudinal, o_speed) in enumerate(other_configs):
            lane = self.road.network.get_lane(("0", "1", o_lane_id))
            position = lane.position(o_longitudinal, 0)
            heading = lane.heading_at(o_longitudinal)

            other_vehicle = Vehicle(
                road=self.road,
                position=position,
                heading=heading,
                speed=o_speed
            )
            other_vehicle.enable_lane_change = False
            
            other_vehicle.color = colors[i]

            self.road.vehicles.append(other_vehicle)


    def _is_terminated(self) -> bool:
        """The episode is over if the ego vehicle crashed."""
        return (
            self.vehicle.crashed
            or self.config["offroad_terminal"]
            and not self.vehicle.on_road
        )

    def _is_truncated(self) -> bool:
        """The episode is truncated if the time limit is reached."""
        return self.time >= self.config["duration"]


class HighwayEnvFast(HighwayEnv):
    """
    A variant of highway-v0 with faster execution:
        - lower simulation frequency
        - fewer vehicles in the scene (and fewer lanes, shorter episode duration)
        - only check collision of controlled vehicles with others
    """

    @classmethod
    def default_config(cls) -> dict:
        cfg = super().default_config()
        cfg.update(
            {
                "simulation_frequency": 5,
                "lanes_count": 3,
                "vehicles_count": 20,
                "duration": 30,  # [s]
                "ego_spacing": 1.5,
            }
        )
        return cfg

    def _create_vehicles(self) -> None:
        super()._create_vehicles()
        # Disable collision check for uncontrolled vehicles
        for vehicle in self.road.vehicles:
            if vehicle not in self.controlled_vehicles:
                vehicle.check_collisions = False
