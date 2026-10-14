#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File    : rule_cbv.py
@Date    : 2024/7/7
"""
import math

from rift.cbv.recognition.base_cbv import BaseCBVRecog
from rift.scenario.tools.carla_data_provider import CarlaDataProvider
from rift.cbv.recognition.rule_based.interaction_pair_selection import select_conflict_pair


class RuleCBVRecog(BaseCBVRecog):
    name = 'rule'

    """ This is the template for implementing the CBV candidate selection for a scenario. """

    def __init__(self, config, logger):
        super().__init__(config, logger)

    @staticmethod
    def _velocity_speed_mps(velocity):
        """CarlaDataProvider returns a Vector3D in m/s, or None before caching."""
        if velocity is None:
            return 0.
        return math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)

    def get_CBVs(self, ego_vehicle, CBVs_id, local_route_waypoints, rest_route_waypoints, red_light_state=None):
        CBV_candidates = self.get_CBV_candidates(ego_vehicle, CBVs_id, rest_route_waypoints)
        if self.config[self.mode].get('pair_selection', 'nearest') == 'shared_conflict' and len(CBVs_id) < 2:
            ego_location = CarlaDataProvider.get_location(ego_vehicle)
            ego_speed = max(self._velocity_speed_mps(CarlaDataProvider.get_velocity(ego_vehicle)), 2.)
            actors = {actor.id: actor for actor in CBV_candidates[:12]}
            for actor_id in CBVs_id:
                actors[actor_id] = CarlaDataProvider.get_actor_by_id(actor_id)
            records = []
            for actor_id, actor in actors.items():
                info = self.route_planner.inter_info.get(actor_id, {})
                waypoint = info.get('inter_waypoint')
                distance = info.get('inter_CBV_dis')
                if actor is None or waypoint is None or distance is None:
                    continue
                location = waypoint.transform.location
                records.append(dict(id=actor_id, conflict_x=location.x, conflict_y=location.y,
                    ego_eta=ego_location.distance(location) / ego_speed,
                    cbv_eta=float(distance) / max(self._velocity_speed_mps(CarlaDataProvider.get_velocity(actor)), 2.)))
            chosen = select_conflict_pair(records, CBVs_id)
            if chosen:
                return [actors[i] for i in chosen]
        CBVs = self.find_closest_vehicle(CBV_candidates, CBVs_id)
        return CBVs

    def find_closest_vehicle(self, CBV_candidates, CBVs_id):
        '''
            rule-based method to find the CBV:
            find the closest N vehicle among all the CBV candidates
        '''
        sample_length = self.max_agent_num - len(CBVs_id)

        return CBV_candidates[:sample_length]


