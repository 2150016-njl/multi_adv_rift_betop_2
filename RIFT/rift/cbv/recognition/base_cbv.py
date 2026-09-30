#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File    : base_cbv.py
@Date    : 2024/7/7
"""


from collections import deque
import math
import carla
from rift.cbv.planning.route_planner.base_planner import CBVBasePlanner
from rift.scenario.tools.global_route_planner import GlobalRoutePlanner
from rift.scenario.tools.carla_data_provider import CarlaDataProvider
from rift.gym_carla.utils.misc import is_within_distance_ahead


class BaseCBVRecog:
    name = 'base'

    """ This is the template for implementing the CBV candidate selection for a scenario. """

    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.mode = None
        self.desired_speed = config['desired_speed']

    def get_CBVs(self, ego_vehicle, CBVs_id, local_route_waypoints, rest_route_waypoints, red_light_state=None):
        raise NotImplementedError()

    def set_mode(self, mode):
        # use different config for different mode
        self.mode = mode

        # CBV spawn and activation parameters
        mode_config = self.config[self.mode]
        self.spawn_radius = mode_config['spawn_radius']
        self.activate_radius = mode_config['activate_radius']
        self.coord_interval = mode_config['coord_interval']
        self.traffic_intensity = mode_config['traffic_intensity']

        # CBV selection parameters
        self.max_route_len = mode_config['max_route_len']
        self.max_ego_route_len = mode_config['max_ego_route_len']
        self.search_radius = mode_config['search_radius']
        self.max_interaction_fov = mode_config['max_interaction_fov']
        self.max_interaction_dis = mode_config['max_interaction_dis']
        self.min_interaction_dis_diff = mode_config['min_interaction_dis_diff']
        self.down_sample_step = mode_config['down_sample_step']
        self.max_agent_num = mode_config['max_agent_num']

    def set_route_planner(self, route_planner):
        self.route_planner:CBVBasePlanner = route_planner
    
    def set_world(self):
        self._map = CarlaDataProvider.get_map()
        self.global_route_planner: GlobalRoutePlanner = CarlaDataProvider.get_global_route_planner()

    def load_model(self):
        return None

    def save_model(self, map_name, episode):
        pass

    def get_CBV_candidates(self, ego_vehicle, CBVs_id, ego_rest_route_waypoints):
        """
        Select nearby vehicles as candidate CBVs based on specific traffic rules.
        Args:
            ego_vehicle: The ego vehicle.
            CBVs_id: The IDs of currently selected CBVs.
            ego_rest_route_waypoints: The remaining route waypoints for the ego vehicle.
        Returns:
            A sorted list of candidate vehicles, ordered by distance from the ego vehicle (closest first).
        """
        # Get IDs of currently selected CBVs and those that have already reached the goal
        CBVs_reach_goal_id = set(CarlaDataProvider.get_CBVs_reach_goal_by_ego(ego_vehicle.id).keys())

        # If the maximum number of CBVs is already selected, return an empty list
        if len(CBVs_id) >= self.max_agent_num:
            return []

        # Retrieve ego vehicle location and cumulative distance along the planned route
        ego_location = CarlaDataProvider.get_location(ego_vehicle)
        ego_cumulative_distance = self._get_ego_cumulative_distance(ego_rest_route_waypoints)

        # Get all vehicles in the environment, excluding ego vehicle and already selected/reached CBVs
        all_actors = CarlaDataProvider.get_actors()
        candidates = {
            actor_id: actor
            for actor_id, actor in all_actors.items()
            if actor_id != ego_vehicle.id and actor_id not in CBVs_id and actor_id not in CBVs_reach_goal_id
        }

        valid_candidates = []
        
        # Iterate through candidate vehicles and filter them based on specific criteria
        for vehicle_id, vehicle in candidates.items():
            vehicle_transform = CarlaDataProvider.get_transform(vehicle)
            vehicle_location = vehicle_transform.location
            distance = ego_location.distance(vehicle_location)

            # 1. Exclude vehicles that are too far or too close
            if distance > self.search_radius or distance < 10:
                continue

            # 2. Get the vehicle's preview waypoint
            preview_waypoint = self.get_preview_waypoint(vehicle_transform)
            
            # 3. Exclude vehicles that are not on a valid driving road
            if preview_waypoint is None:
                continue

            # 4. If the vehicle is at a junction, check its forward direction angle
            if preview_waypoint.is_junction:
                vehicle_forward = vehicle_transform.get_forward_vector()
                preview_forward = preview_waypoint.transform.get_forward_vector()
                if math.degrees(vehicle_forward.get_vector_angle(preview_forward)) > 20:
                    continue

            # 5. Check if an interaction waypoint exists between ego and the candidate vehicle
            if not self.find_interaction_waypoint(ego_rest_route_waypoints, preview_waypoint, vehicle_id, ego_cumulative_distance):
                continue

            # If the vehicle passes all filters, add it to the valid candidates list
            valid_candidates.append((distance, vehicle))

        # Sort the candidate vehicles by distance to the ego vehicle (ascending order)
        valid_candidates.sort(key=lambda x: x[0])

        # Return a sorted list of candidate vehicles
        return [vehicle for _, vehicle in valid_candidates]
    
    def _get_ego_cumulative_distance(self, ego_rest_route_waypoints):
        """
            Calculate cumulative distances and times for the ego vehicle from its current position to each waypoint in its global route
        """
        ego_cumulative_distances = {}
        ego_total_distance = 0.0
        start_index = 0
        ego_rest_route_waypoints = ego_rest_route_waypoints[:self.max_ego_route_len]
        # Downsample the ego rest route waypoints by selecting every 'step' waypoint
        downsampled_indices = list(range(start_index, len(ego_rest_route_waypoints), self.down_sample_step))
        if downsampled_indices[-1] != len(ego_rest_route_waypoints) - 1:
            # Ensure the last waypoint is included
            downsampled_indices.append(len(ego_rest_route_waypoints) - 1)
        
        # Calculate distances from the start index to each waypoint in the ego route
        previous_loc = ego_rest_route_waypoints[0].transform.location
        for idx in downsampled_indices[1:]:
            current_loc = ego_rest_route_waypoints[idx].transform.location
            distance = previous_loc.distance(current_loc)
            ego_total_distance += distance
            ego_cumulative_distances[idx] = ego_total_distance
            previous_loc = current_loc

        return ego_cumulative_distances

    def find_interaction_waypoint(self, ego_rest_route_waypoints, vehicle_waypoint, vehicle_id, ego_cumulative_distance):
        # find the interaction point
        # Initialize the minimum time difference and the inter waypoint
        temp_min_dis_diff = self.min_interaction_dis_diff
        inter_waypoint = None
        inter_CBV_route_elements = None
        inter_CBV_route_ids = None
        inter_CBV_dis = None
        
        for idx, ego_distance in ego_cumulative_distance.items():
            target_waypoint = ego_rest_route_waypoints[idx]
            
            # the target waypoint should be in front of the current vehicle
            if not is_within_distance_ahead(target_waypoint.transform, vehicle_waypoint.transform, self.max_interaction_dis, self.max_interaction_fov // 2):
                continue

            CBV_route, CBV_route_ids, CBV_distance = self.global_route_planner.trace_route(vehicle_waypoint.transform.location, target_waypoint.transform.location)

            CBV_route_elements = deque(CBV_route, maxlen=self.max_route_len)
            
            # Calculate the distance difference
            dis_diff = abs(CBV_distance - ego_distance)

            # Update the minimum distance difference and inter waypoint
            if dis_diff < temp_min_dis_diff:
                temp_min_dis_diff = dis_diff
                inter_waypoint = target_waypoint
                inter_CBV_route_elements = CBV_route_elements
                inter_CBV_route_ids = CBV_route_ids
                inter_CBV_dis = CBV_distance

        self.route_planner.update_inter_info(vehicle_id, inter_waypoint, inter_CBV_route_elements, inter_CBV_route_ids, inter_CBV_dis)
        
        return inter_waypoint

    def get_preview_waypoint(self, vehicle_trans, dis=2):
        """
            get the preview waypoint for the vehicle
        """
        vehicle_loc = vehicle_trans.location
        forward_vector = vehicle_trans.get_forward_vector().make_unit_vector()
        preview_loc = carla.Location(
            x=vehicle_loc.x + forward_vector.x * dis,
            y=vehicle_loc.y + forward_vector.y * dis,
            z=vehicle_loc.z + forward_vector.z * dis
        )
        return self._map.get_waypoint(preview_loc, project_to_road=None, lane_type=carla.LaneType.Driving)
