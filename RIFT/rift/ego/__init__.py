#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File    : __init__.py
@Date    : 2023/10/4
"""

# for planning scenario

# BeTop/SparseDrive is an optional E2E stack with additional third-party
# dependencies (including a top-level ``utils`` package).  Importing it here
# used to make the lightweight PDM-Lite / RIFT path fail before policy
# selection.  Keep its registry entries conditional so non-E2E experiments do
# not require those dependencies.
try:
    from rift.ego.b2d.e2e_agent import VAD, SparseDrive, UniAD
except ImportError:
    VAD = SparseDrive = UniAD = None
from rift.ego.rl.ppo import PPO
from rift.ego.behavior import Behavior
from rift.ego.expert_disturb import ExpertDisturb
from rift.ego.expert.expert import Expert
from rift.ego.plant.plant import PlanT
from rift.ego.pdm_lite.pdm_lite import PDM_LITE


EGO_POLICY_LIST = {
    'behavior': Behavior,
    'ppo': PPO,
    'expert': Expert,
    'plant': PlanT,
    'expert_disturb': ExpertDisturb,
    'pdm_lite': PDM_LITE,
}

if VAD is not None:
    EGO_POLICY_LIST.update({
        'vad': VAD,
        'uniad': UniAD,
        'sparsedrive': SparseDrive,
    })
