from __future__ import annotations
import numpy as np
import torch
from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.common.action import Action
from highway_env.road.road import Road, RoadNetwork
from highway_env.utils import near_split
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.vehicle.kinematics import Vehicle

# import all the stlcg rules
from highway_env.rule_hierarchy.rule_hierarchies.rule_hierarchy import RuleHierarchy
from highway_env.rule_hierarchy.rule_hierarchies.rules.demo import AvoidCollision    # No collision
from highway_env.rule_hierarchy.rule_hierarchies.rules.demo import SolidLaneLine     # Do not cross solid lane line
from highway_env.rule_hierarchy.rule_hierarchies.rules.demo import DashedLaneLine   # Do not cross dashed lane line
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
        rule4 = AlwaysGreater(rule_threshold=20.5)    # speed > v_min
        rule5 = AlwaysLesser(rule_threshold=29.0)     # speed < v_max
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
        print(f"Action: {action}({action_name}), Speed: {self.vehicle.speed:.2f}, Collision: {self.vehicle.crashed}, Total Reward: {reward:.2f}")
        print(f"Reward(No Collision): {collision_reward:.2f}")
        print(f"Reward(No Cross Solid Line): {solidlane_reward:.2f}")
        print(f"Reward(No Cross Dashed Line): {dashedlane_reward:.2f}")
        print(f"Reward(Speed>20.5): {speed1_reward:.2f}, Reward(Speed<29.0): {speed2_reward:.2f}")
        print("-" * 50)
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
                "lanes_count": 4,
                "vehicles_count": 50,
                "controlled_vehicles": 1,
                "initial_lane_id": None,
                "duration": 40,  # [s]
                "ego_spacing": 2,
                "vehicles_density": 1,
                "collision_reward": -1,
                "right_lane_reward": 0.1,
                "high_speed_reward": 0.4,
                "lane_change_reward": 0,
                "reward_speed_range": [20, 30],
                "normalize_reward": True,
                "offroad_terminal": False,
            }
        )
        return config

    def _reset(self) -> None:
        self._create_road()
        self._create_vehicles()

    def _create_road(self) -> None:
        self.road = Road(
            network=RoadNetwork.straight_road_network(
                self.config["lanes_count"], speed_limit=30
            ),
            np_random=self.np_random,
            record_history=self.config["show_trajectories"],
        )

    def _create_vehicles(self) -> None:
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        other_per_controlled = near_split(
            self.config["vehicles_count"], num_bins=self.config["controlled_vehicles"]
        )

        self.controlled_vehicles = []
        for others in other_per_controlled:
            vehicle = Vehicle.create_random(
                self.road,
                speed=25,
                lane_id=self.config["initial_lane_id"],
                spacing=self.config["ego_spacing"],
            )
            vehicle = self.action_type.vehicle_class(
                self.road, vehicle.position, vehicle.heading, vehicle.speed
            )
            self.controlled_vehicles.append(vehicle)
            self.road.vehicles.append(vehicle)

            for _ in range(others):
                vehicle = other_vehicles_type.create_random(
                    self.road, spacing=1 / self.config["vehicles_density"]
                )
                vehicle.randomize_behavior()
                self.road.vehicles.append(vehicle)

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
