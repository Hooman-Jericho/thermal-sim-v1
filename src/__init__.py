"""thermal-sim-v1: an OOP class hierarchy simulating basic heat exchange.

Public API:
    ThermalSystem, SystemState   -- from .core
    NewtonianCoolingSystem       -- from .systems
    HeatExchangerSystem          -- from .systems
"""

from src.core import SystemState, ThermalSystem
from src.systems import HeatExchangerSystem, NewtonianCoolingSystem

__all__ = [
    "ThermalSystem",
    "SystemState",
    "NewtonianCoolingSystem",
    "HeatExchangerSystem",
]
