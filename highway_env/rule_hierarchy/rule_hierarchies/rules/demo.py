from typing import Tuple, Union
import stlcg
import torch
from rule_hierarchy.rules.rule import Rule

### No Collision
class AvoidCollision(Rule[torch.Tensor]):
    def __init__(self, rule_threshold: float = 0.5) -> None:
        self.rule_threshold = rule_threshold

    def as_stl_formula(self) -> stlcg.Expression:
        stl_expression = stlcg.Expression("input", [], reversed=True)
        return stlcg.Always(stl_expression < self.rule_threshold)

    def prepare_signals(self, traj: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor]]:
        input = traj
        assert input.dim() == 2
        return self.shape_stl_signal_batch(input)

### Do not cross solid lane line
class SolidLaneLine(Rule[torch.Tensor]):
    def __init__(self, left_boundary: float = 9.0, right_boundary: float = -1.0) -> None:
        self.left_boundary = left_boundary
        self.right_boundary = right_boundary

    def as_stl_formula(self) -> stlcg.Expression:
        stl_expression = stlcg.Expression("input", [], reversed=True)
        return stlcg.Always((stl_expression > self.right_boundary) & (stl_expression < self.left_boundary))

    def prepare_signals(self, traj: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor]]:
        assert (traj.dim() == 2)
        if traj.dim() == 2:
            signal = self.shape_stl_signal_batch(traj)
        else:
            signal = self.shape_stl_signal(traj)
        return (signal, signal)

### Do not cross dashed lane line
class DashedLaneLine(Rule[torch.Tensor]):
    def as_stl_formula(self) -> stlcg.Expression:
        stl_expression = stlcg.Expression("input", [], reversed=True)
        return stlcg.Always(stl_expression>1.5)

    def prepare_signals(self, traj: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor]]:
        input = traj
        assert (input.dim() == 2)
        return self.shape_stl_signal_batch(input)

### Speed > v_min
class AlwaysGreater(Rule[torch.Tensor]):
    def __init__(self, rule_threshold: float) -> None:
        self.rule_threshold = rule_threshold

    def as_stl_formula(self) -> stlcg.Expression:
        stl_expression = stlcg.Expression("input", [], reversed=True)
        return stlcg.Always(stl_expression > self.rule_threshold)

    def prepare_signals(self, traj: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor]]:
        input = traj
        assert (input.dim() == 2)
        return self.shape_stl_signal_batch(input)

### Speed < v_max
class AlwaysLesser(Rule[torch.Tensor]):
    def __init__(self, rule_threshold: float) -> None:
        self.rule_threshold = rule_threshold

    def as_stl_formula(self) -> stlcg.Expression:
        stl_expression = stlcg.Expression("input", [], reversed=True)
        return stlcg.Always(stl_expression < self.rule_threshold)

    def prepare_signals(self, traj: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor]]:
        input = traj
        assert (input.dim() == 2)
        return self.shape_stl_signal_batch(input)

class EventuallyAlwaysLess(Rule[torch.Tensor]):
    def __init__(self, rule_threshold: float) -> None:
        self.rule_threshold = rule_threshold

    def as_stl_formula(self) -> stlcg.Expression:
        stl_expression = stlcg.Expression("input", [], reversed=True)
        return stlcg.Eventually(subformula=stlcg.Always(stl_expression < self.rule_threshold),)

    def prepare_signals(self, traj: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor]]:
        input = traj
        assert (input.dim() == 2)
        return self.shape_stl_signal_batch(input)