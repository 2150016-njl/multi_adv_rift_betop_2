#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File    : __init__.py
@Date    : 2024/8/15
"""
# collect policy models from scenarios
from rift.cbv.planning.dummy_policy import DummyPolicy
from rift.cbv.planning.fine_tuner.rlft.grpo_pluto.grpo_pluto import GRPOPluto
from rift.cbv.planning.fine_tuner.rlft.ppo_pluto.ppo_pluto import PPOPluto
from rift.cbv.planning.fine_tuner.rlft.reinforce_pluto.reinforce_pluto import ReinforcePluto
from rift.cbv.planning.fine_tuner.rlft.rift_pluto.rift_pluto import RIFTPluto
from rift.cbv.planning.fine_tuner.rlft.joint_rift.joint_rift_pluto import JointRIFTPluto
from rift.cbv.planning.fine_tuner.sft.rs_pluto.rs_pluto import RewardShapingPluto
from rift.cbv.planning.fine_tuner.sft.rtr_pluto.rtr_pluto import RTRPluto
from rift.cbv.planning.fine_tuner.sft.sft_pluto import SFTPluto
from rift.cbv.planning.rl.frea import FREA, FPPORs
from rift.cbv.planning.rl.ppo import PPO
from rift.cbv.planning.pluto.pluto import PLUTO


CBV_POLICY_LIST = {
    'standard': DummyPolicy,
    'ppo': PPO,
    'frea': FREA,
    'fppo_rs': FPPORs,
    'pluto': PLUTO,
    'sft_pluto': SFTPluto,
    'rtr_pluto': RTRPluto,
    'rs_pluto': RewardShapingPluto,
    'reinforce_pluto': ReinforcePluto,
    'ppo_pluto': PPOPluto,
    'grpo_pluto': GRPOPluto,
    'rift_pluto': RIFTPluto,
    'joint_rift_pluto': JointRIFTPluto,
}
