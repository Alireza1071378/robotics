import numpy as np
import inspect
import rectangle_overlap as ro
import cv2
import matplotlib.pyplot as plt
import time
from dash import Dash, html, dash_table, dcc, callback, Output, Input, State, callback_context
import plotly.graph_objs as go
import plotly.express as px
from dash.exceptions import PreventUpdate


def get_y_rotation(angle):
    result = np.array([
        [np.cos(angle), 0, np.sin(angle)],
        [0, 1, 0],
        [-np.sin(angle), 0, np.cos(angle)]
    ])
    return result

def get_z_rotation(angle):
    result = np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle), np.cos(angle), 0],
        [0, 0, 1]
    ])
    return result


def get_euler_transform(angles):
    
    T_1 = get_z_rotation(angles[0])
    T_2 = get_y_rotation(angles[1])
    T_3 = get_z_rotation(angles[2])
    
    return np.matmul(T_1, np.matmul(T_2, T_3))


class gadget:
    def __init__(self, id):
        self.id = id


class motor(gadget):
    def __init__(self, id, range):
        super().__init__(id)
        self.motor_angle = 0
        self.range = tuple(np.deg2rad(range))
    
    def rotate_by(self, angle):
        final_angle = self.motor_angle + angle
        if (final_angle < self.range[0]) or (final_angle > self.range[1]):
            raise Exception('rotation out of range!')
        else:
            self.motor_angle = final_angle



class camera(gadget):
    def __init__(self, id, resolution, angular_width, source = None):
        super().__init__(id)
        self.resolution = resolution
        self.angular_width = angular_width
        if source == 'simulator':
            self.source = cameraSimulator()
            self.source.simulated_camera = self
        else:
            self.source = source

    def get_direction_from_point_on_screen(self, location_on_screen):
        direction = np.insert(np.flip((np.array(self.resolution) - 2 * np.array(location_on_screen)) * np.tan(np.deg2rad(self.angular_width) / 2) / self.resolution[1]), 0, 1)
        direction /= np.linalg.norm(direction)
        return direction.reshape((3, 1))  
    
    def capture(self):
        image_array = np.empty((*self.resolution, 4))
        self.source.capture(image_array, format = 'rgba')
        return image_array


class cameraSimulator:
    def __init__(self):
        self.simulated_camera = None
        self.simulated_node = None
        self.configuration_simulator = None

    def capture(self, target_image, format):
        if format == 'rgba':
            i, j = np.indices(target_image.shape[:2])
            pixels = np.stack((i, j), axis = -1)
            directions_relative_to_camera = np.apply_along_axis(self.simulated_camera.get_direction_from_point_on_screen, axis = -1, arr = pixels).squeeze()
            directions_relative_to_root = np.matmul(directions_relative_to_camera, self.simulated_node.node_basis_relative_to_origin.T)
            self.configuration_simulator.color_the_query_rays(target_image, self.simulated_node.position, directions_relative_to_root)
            
        else:
            raise Exception('format not supported!')


class configurationSimulator:
    def __init__(self):
        self.location = None
        self.field = None
        self.dimensions_of_the_field = None
    
    def set_location(self, location):
        self.location = np.array(location)

    def color_the_query_rays(self, target_image, beginning_of_the_rays, direction_of_the_rays):
        t = -beginning_of_the_rays[2, 0] / direction_of_the_rays[:, :, 2]
        mask = (t < 0) & (t == np.inf)
        hitting_location_of_the_rays_that_hit_the_floor = np.ma.masked_array(direction_of_the_rays[..., :2], np.stack([mask] * 2, axis = 2)) * t[..., np.newaxis] + beginning_of_the_rays[:2, 0] + self.location
        hitting_pixel_of_the_rays_that_hit_the_floor = (hitting_location_of_the_rays_that_hit_the_floor / np.array(self.dimensions_of_the_field) * np.array(self.field.shape[:2])).astype(int)

        in_vertical_range_mask = ((hitting_pixel_of_the_rays_that_hit_the_floor[:, :, 0] < 0) | (hitting_pixel_of_the_rays_that_hit_the_floor[:, :, 0] >= self.field.shape[0]))
        
        in_horizontal_range_mask = ((hitting_pixel_of_the_rays_that_hit_the_floor[:, :, 1] < 0) | (hitting_pixel_of_the_rays_that_hit_the_floor[:, :, 1] >= self.field.shape[1]))
        in_range_mask = in_vertical_range_mask | in_horizontal_range_mask

        hitting_pixel_of_the_rays_that_hit_the_floor.mask = np.logical_or(hitting_pixel_of_the_rays_that_hit_the_floor.mask, np.stack([in_range_mask] * 2, axis = 2))
        
        target_image[:, :, :] = 0
        valid_mask = ~hitting_pixel_of_the_rays_that_hit_the_floor.mask
        target_image[np.where(valid_mask[:, :, 0])] = self.field[hitting_pixel_of_the_rays_that_hit_the_floor[:, :, 0][valid_mask[:, :, 0]], hitting_pixel_of_the_rays_that_hit_the_floor[:, :, 1][valid_mask[:, :, 1]]]


class configuration_node:
    def __init__(self, data):
        self.data = data
        self.parent = self
        self.children = []
        self.position_relative_to_parent = (0, 0, 0)
        self.initial_Euler_angles_relative_to_parent = (0, 0, 0)
        self.node_basis_relative_to_parent = np.identity(3)
        self.parent_basis_relative_to_node = np.identity(3)
        self.node_basis_relative_to_origin = np.identity(3)
        self.origin_basis_relative_to_node = np.identity(3)
        self.position = np.zeros((3, 1))
        
    def add_child(self, new_node, position_relative_to_parent, initial_Euler_angles_relative_to_parent):
        new_node.position_relative_to_parent = position_relative_to_parent
        new_node.initial_Euler_angles_relative_to_parent = initial_Euler_angles_relative_to_parent
        new_node.node_basis_relative_to_parent = np.matmul(get_euler_transform(initial_Euler_angles_relative_to_parent), new_node.node_basis_relative_to_parent)
        new_node.parent_basis_relative_to_node = np.matmul(new_node.parent_basis_relative_to_node, get_euler_transform(-np.flip(initial_Euler_angles_relative_to_parent)))
        self.children.append(new_node)
        new_node.parent = self
        new_node.rotate_as_child_by(0)
        
    def cut_from_parent(self):
        position_relative_to_parent = self.position_relative_to_parent
        initial_Euler_angles_relative_to_parent = self.initial_Euler_angles_relative_to_parent
        self.position_relative_to_parent = (0, 0, 0)
        self.initial_Euler_angles_relative_to_parent = (0, 0, 0)
        self.node_basis_relative_to_parent = np.matmul(get_euler_transform(-np.flip(initial_Euler_angles_relative_to_parent)), self.node_basis_relative_to_parent)
        self.parent_basis_relative_to_node = np.matmul(self.parent_basis_relative_to_node, get_euler_transform(initial_Euler_angles_relative_to_parent))
        self.node_basis_relative_to_origin = np.matmul(get_euler_transform(-np.flip(initial_Euler_angles_relative_to_parent)), self.node_basis_relative_to_parent)
        self.origin_basis_relative_to_node = np.matmul(self.parent_basis_relative_to_node, get_euler_transform(initial_Euler_angles_relative_to_parent))
        self.position = np.zeros((3, 1))
        self.parent.children.remove(self)
        self.parent = self
        self.rotate_as_child_by(0)
        
        return self, position_relative_to_parent, initial_Euler_angles_relative_to_parent
    
    def get_ids(self):
        ids = ({self.data.id: self} if self.data != None else {})
        for child in self.children:
            ids = {**ids, **child.get_ids()}
        return ids

    def rotate_motor_by(self, angle):
        if self.data.__class__.__name__ == 'motor':
            self.data.rotate_by(angle)
            self.node_basis_relative_to_parent = np.matmul(self.node_basis_relative_to_parent, get_z_rotation(angle))
            self.parent_basis_relative_to_node = np.matmul(get_z_rotation(-angle), self.parent_basis_relative_to_node)
            self.node_basis_relative_to_origin = np.matmul(self.node_basis_relative_to_origin, get_z_rotation(angle))
            self.origin_basis_relative_to_node = np.matmul(get_z_rotation(-angle), self.origin_basis_relative_to_node)
            for child in self.children:
                child.rotate_as_child_by(angle)
        else:
            raise Exception('not a motor!')
            
    def rotate_as_child_by(self, angle):
        self.node_basis_relative_to_origin = np.matmul(self.parent.node_basis_relative_to_origin, self.node_basis_relative_to_parent)
        self.origin_basis_relative_to_node = np.matmul(self.parent_basis_relative_to_node, self.parent.origin_basis_relative_to_node)
        self.position = self.parent.position + np.matmul(self.parent.node_basis_relative_to_origin, np.array(self.position_relative_to_parent).reshape((3, 1)))
        #print(self.children)
        for child in self.children:
            child.rotate_as_child_by(angle)

    def get_subtree_data(self):
        data = {self.data.id: {'parent id': self.parent.data.id,
                               'children id': [child.data.id for child in self.children],
                                'position_relative_to_parent': self.position_relative_to_parent, 
                                'initial_Euler_angles_relative_to_parent': tuple(np.rad2deg(self.initial_Euler_angles_relative_to_parent)),  
                                'node_basis_relative_to_parent': self.node_basis_relative_to_parent, 
                                'parent_basis_relative_to_node': self.parent_basis_relative_to_node, 
                                'node_basis_relative_to_origin': self.node_basis_relative_to_origin, 
                                'origin_basis_relative_to_node': self.origin_basis_relative_to_node, 
                                'position': self.position, 
                                'gadget_data': vars(self.data)}}
        for child in self.children:
            data = {**data, **child.get_subtree_data()}
        return data

    def is_ancestor_of(self, query_node):
        x = query_node
        y = x
        while True:
            if x == self:
                return True
            else:
                x = x.parent
                if x == y:
                    return False
                else:
                    y = y.parent

    def get_max_distance_in_subtree(self):
        if len(self.children) == 0:
            return 0
        else:
            max_distance = 0
            for child in self.children:
                distance = child.get_max_distance_in_subtree() + np.linalg.norm(child.position_relative_to_parent)
                if distance > max_distance:
                    max_distance = distance
            return max_distance
        

    
class configuration:
    def __init__(self, *nodes):
        self.root = configuration_node(gadget('root'))
        self.nodes = {'root': self.root}
        self.simulator = configurationSimulator()
        for given_gadget, position_relative_to_parent, initial_Euler_angles_relative_to_parent in nodes:
            node = configuration_node(given_gadget)
            self.root.add_child(node, position_relative_to_parent, initial_Euler_angles_relative_to_parent)
            self.nodes[node.data.id] = node
            if given_gadget.__class__.__name__ == 'camera' and given_gadget.source.__class__.__name__ == 'cameraSimulator':
                given_gadget.source.simulated_node = node
                given_gadget.source.configuration_simulator = self.simulator
    
    def get_simulator(self):
        return self.simulator
    
    def merge(self, destination_id, new_config):
        if destination_id in self.nodes:
            duplication = (set(self.nodes) & set(new_config.nodes)) - {'root'}
            if not duplication:
                destination = self.nodes.get(destination_id)
                for descendant_id in new_config.nodes:
                    descendant = new_config.nodes.get(descendant_id)
                    if descendant.data.__class__.__name__ == 'camera' and descendant.data.source.__class__.__name__ == 'cameraSimulator':
                        descendant.data.source.configuration_simulator = self.simulator
                
                for child in new_config.root.children:
                    destination.add_child(*child.cut_from_parent())
                new_config.nodes.pop('root')
                self.nodes = {**self.nodes, **new_config.nodes}
                del new_config
            else:
                raise Exception('duplication in nodes: ' + str(duplication) + '!')
        else:
            raise Exception('no such id!')

    def add_motor(self, destination_id, new_motor, position_relative_to_parent, initial_Euler_angles_relative_to_parent):#########
        self.merge(destination_id, configuration((new_motor, position_relative_to_parent, tuple(np.deg2rad(initial_Euler_angles_relative_to_parent)))))

    def add_camera(self, destination_id, new_camera, position_relative_to_parent, initial_Euler_angles_relative_to_parent):#########
        self.merge(destination_id, configuration((new_camera, position_relative_to_parent, tuple(np.deg2rad(initial_Euler_angles_relative_to_parent)))))
    
    def cut_compartment(self, target_id):
        if target_id in self.nodes and target_id != 'root':
            target, position_relative_to_parent, initial_Euler_angles_relative_to_parent = self.nodes.get(target_id).cut_from_parent()
            dissected_compartment = configuration(target)
            for id in dissected_compartment.nodes:
                self.nodes.pop(id)
            return dissected_compartment, position_relative_to_parent, initial_Euler_angles_relative_to_parent
        else:
            raise Exception('no such id!')
    
    def rotate_motor_by(self, motor_id, angle):
        if motor_id in self.nodes:
            angle = np.deg2rad(angle)
            self.nodes.get(motor_id).rotate_motor_by(angle)
        else:
            raise Exception('no such id!')
    
    def set_motors_angle(self, **angles):
        for id in angles:
            try:
                angle = angles[id]
                if angle != np.rad2deg(self.nodes[id].data.motor_angle):
                    angle = np.deg2rad(angle)
                    node = self.nodes.get(id)
                    node.rotate_motor_by(angle - node.data.motor_angle)
            except:
                raise Exception('no motor called ' + id + '!')
    
    def get_nodes_locations(self):
        data = self.root.get_subtree_data()
        data.pop('root')
        return {id: data[id]['position'] for id in data}

    def get_parameters(self):
        parameters = dict()
        for id in self.nodes:
            node = self.nodes[id]
            if node.data.__class__.__name__ == 'motor':
                parameters[id] = node.data.motor_angle
        return parameters

    def get_node_data(self, target_id):
        target = self.nodes.get(target_id)
        data = {'parent id': target.parent.data.id,
                'children id': [child.data.id for child in target.children],
                'position_relative_to_parent': target.position_relative_to_parent, 
                'initial_Euler_angles_relative_to_parent': tuple(np.rad2deg(target.initial_Euler_angles_relative_to_parent)), 
                'node_basis_relative_to_parent': target.node_basis_relative_to_parent, 
                'parent_basis_relative_to_node': target.parent_basis_relative_to_node, 
                'node_basis_relative_to_origin': target.node_basis_relative_to_origin, 
                'origin_basis_relative_to_node': target.origin_basis_relative_to_node, 
                'position': target.position,
                'gadget_data': vars(target.data)}
        return data
    
    def get_screen_point_to_location_on_the_floor_transform(self, camera_id):
        if camera_id in self.nodes:
            def transform(location_on_screen):
                camera_node = self.nodes.get(camera_id)
                direction = camera_node.data.get_direction_from_point_on_screen(location_on_screen)
                direction = np.matmul(camera_node.node_basis_relative_to_origin, direction)
                camera_location = camera_node.position
                t = -camera_location[2, 0] / direction[2, 0]
                return (camera_location + t * direction)[: 2, 0]
                
            return transform
        else:
            raise Exception('no camera with this id!')
    
    def get_coordinate_transform(self, output_id, input_id):
        if output_id in self.nodes:
            if input_id in self.nodes:
                def transform(input_coordinates):
                    output_node = self.nodes.get(output_id)
                    input_node = self.nodes.get(input_id)
                    coordinates_relative_to_origin = input_node.position + np.matmul(input_node.motor_basis_relative_to_origin, input_coordinates)
                    coordinates_relative_to_output_node = np.matmul(output_node.origin_basis_relative_to_motor, -output_node.position + coordinates_relative_to_origin)
                    return coordinates_relative_to_output_node
                return transform
            else:
                raise Exception('input id not found!')  
        else:
            raise Exception('output id not found!')

    def get_radius(self):
        return self.root.get_max_distance_in_subtree()



class robot(configuration):
    def __init__(self, *nodes):
        super().__init__(*nodes)
        
        self.dimensions_of_the_field = None #(dimension[0], dimension[1])
        self.origine = (0, 0)# (0, 0) -> upper left corner, (1, 0) -> lower left corner, (1, 1) -> lower right corner, (0, 1) -> upper right corner
        self.line_identifier = None #(mean, std)
        self.line_width = None
        
        self.original_verticals = None
        self.original_horizontals = None
        
        self.raw_lines = np.zeros((1, 4))
        
    def set_the_line_identifier(self, line_identifier):# line_identifier = (mean, std)
        self.line_identifier = line_identifier
        
    def set_the_field(self, dimension_0, dimension_1, field):#field = (dimension[0], dimension[1], picture of the field)
        if self.line_identifier == None:
            raise Exception('you must set the line_identifier first!')
        else:
            self.simulator.field = field
            self.simulator.dimensions_of_the_field = (dimension_0, dimension_1)
            self.dimensions_of_the_field = (dimension_0, dimension_1)
            self.original_horizontals, self.original_verticals = robot.clean_the_lines(self.get_lines(field).squeeze(), 10, 9)
            self.original_verticals *= (np.array([self.dimensions_of_the_field[0], self.dimensions_of_the_field[1], self.dimensions_of_the_field[1]]) / np.array([field.shape[0], field.shape[1], field.shape[1]]))
            self.original_horizontals *= (np.array([self.dimensions_of_the_field[1], self.dimensions_of_the_field[0], self.dimensions_of_the_field[0]]) / np.array([field.shape[1], field.shape[0], field.shape[0]]))    
            self.line_width = 10 * self.dimensions_of_the_field[0] / field.shape[0]

    def get_lines(self, picture, threshold = 25, minLineLength = 5, maxLineGap = 50):
        mean_color = self.line_identifier[0]
        std_color = self.line_identifier[1]
        picture_filtered = (((np.abs(picture - mean_color) <= std_color).sum(axis = 2) == 4) * 255).astype(np.uint8)
        edges = cv2.Canny(picture_filtered, 100, 200)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold = threshold, minLineLength = minLineLength, maxLineGap = maxLineGap)
        return lines#.squeeze()

    @classmethod
    def clean_the_lines(cls, raw_lines, vertical_epsilon, horizontal_epsilon):
    
        angles = np.arctan(np.abs((raw_lines[:, 3] - raw_lines[:, 1]) / (np.where(raw_lines[:, 2] != raw_lines[:, 0] , raw_lines[:, 2] - raw_lines[:, 0], 1))))
    
        vertical_lines = raw_lines[angles > 19 * np.pi / 40]
        main_verticals = []
        if len(vertical_lines) > 0:
            vertical_lines = np.hstack((np.expand_dims((vertical_lines[:, 0] + vertical_lines[:, 2]) / 2, axis = 1), np.sort(vertical_lines[:, (1, 3)], axis = 1)))
            vertical_lines = vertical_lines[vertical_lines[:, 0].argsort()]
            vertical_groups = np.split(vertical_lines, np.where(np.diff(vertical_lines[:, 0]) >= vertical_epsilon)[0] + 1)

            for group in vertical_groups:
                group = group[np.argsort(group[:, 1])]
                gap = np.diff(group[:, 1]) - (group[: -1, 2] - group[: -1, 1])
                subgroups = np.split(group, np.where(gap > 10 * vertical_epsilon)[0] + 1)
                for subgroup in subgroups:
                    main_verticals.append([subgroup[:, 0].mean(), subgroup[0, 1], subgroup[-1, 2]])
        
        horizontal_lines = raw_lines[angles < np.pi / 40]
        main_horizontals = []
        if len(horizontal_lines) > 0:
            horizontal_lines = np.hstack((np.expand_dims((horizontal_lines[:, 1] + horizontal_lines[:, 3]) / 2, axis = 1), np.sort(horizontal_lines[:, (0, 2)], axis = 1)))
            horizontal_lines = horizontal_lines[horizontal_lines[:, 0].argsort()]
            horizontal_groups = np.split(horizontal_lines, np.where(np.diff(horizontal_lines[:, 0]) >= horizontal_epsilon)[0] + 1)
            for group in horizontal_groups:
                group = group[np.argsort(group[:, 1])]
                gap = np.diff(group[:, 1]) - (group[: -1, 2] - group[: -1, 1])
                subgroups = np.split(group, np.where(gap > 10 * horizontal_epsilon)[0] + 1)
                for subgroup in subgroups:
                    main_horizontals.append([subgroup[:, 0].mean(), subgroup[0, 1], np.max(subgroup[: , 2])])
            
        return np.array(main_verticals), np.array(main_horizontals)

    def get_the_observed_field(self):
        observed_field = np.zeros((2 * self.simulator.field.shape[0], 2 * self.simulator.field.shape[1]))
        lines = self.raw_lines.copy()
        lines = lines[1: , :]
        if len(lines) != 0:
            lines[:, (0, 2)] *= (self.simulator.field.shape[0] / self.dimensions_of_the_field[0])
            lines[:, (0, 2)] += self.simulator.field.shape[0]
            lines[:, (1, 3)] *= (self.simulator.field.shape[1] / self.dimensions_of_the_field[1])
            lines[:, (1, 3)] += self.simulator.field.shape[1]
            lines = lines.astype(int)
            for l in lines:
                cv2.line(observed_field, [l[1], l[0]], [l[3], l[2]], 255, 2)
        return observed_field
    
    def take_picture(self, camera_id, show_process = False):# = False):
        if camera_id in self.nodes and self.nodes.get(camera_id).data.__class__.__name__ == 'camera':
            camera = self.nodes.get(camera_id).data
            image = camera.capture()
            transform = self.get_screen_point_to_location_on_the_floor_transform(camera_id)
            mid_camera = (camera.resolution[0] / 2, camera.resolution[1] / 2)
            mid_camera_on_the_floor = transform(mid_camera)
            distance_from_mid_camera_on_the_floor = np.linalg.norm(mid_camera_on_the_floor)
            min_threshold = 5
            max_threshold = 20
            threshold = int(min_threshold + (max_threshold - min_threshold) * (distance_from_mid_camera_on_the_floor - self.dimensions_of_the_field[0] / (self.dimensions_of_the_field[1] - self.dimensions_of_the_field[0])))

            lines = self.get_lines(image, threshold = threshold, minLineLength = 5, maxLineGap = 10)
            if type(lines) != type(None):
                lines = lines.squeeze(axis = 1)[:, [1, 0, 3, 2]]
                beginning_of_the_lines = np.apply_along_axis(transform, axis = -1, arr = lines[:, :2])
                end_of_the_lines = np.apply_along_axis(transform, axis = -1, arr = lines[:, 2:])
                self.raw_lines = np.vstack((self.raw_lines, np.hstack((beginning_of_the_lines, end_of_the_lines))))
            if show_process:
                lines_image = np.zeros(image.shape[: 2])
                if type(lines) != type(None):    
                    for l in lines:
                        cv2.line(lines_image, [l[1], l[0]], [l[3], l[2]], 255, 2)
                fig, axs = plt.subplots(1, 2, figsize = (10, 3))
                axs[0].imshow(image)
                axs[1].imshow(lines_image)
                plt.show(block=False)
        else:
            raise Exception('no such camera!')

    def restart_the_coordination_memory(self):
        self.raw_lines = np.zeros((1, 4))
    
    def locate(self, show_process = False, calculate_time = False):
        
        t_1 = time.time()
        raw_lines = self.raw_lines.copy()
        raw_lines = self.raw_lines[1:]
                
        observed_verticals, observed_horizontals = robot.clean_the_lines(raw_lines, self.line_width, self.line_width)
        t_2 = time.time()

        if show_process:
            fig, axs = plt.subplots(1, 2, figsize = (10, 3))
            observed_field_raw = np.zeros((2 * self.simulator.field.shape[0], 2 * self.simulator.field.shape[1]))
            lines = self.raw_lines.copy()
            lines = lines[1: , :]
            lines[:, (0, 2)] *= (self.simulator.field.shape[0] / self.dimensions_of_the_field[0])
            lines[:, (0, 2)] += self.simulator.field.shape[0]
            lines[:, (1, 3)] *= (self.simulator.field.shape[1] / self.dimensions_of_the_field[1])
            lines[:, (1, 3)] += self.simulator.field.shape[1]
            lines = lines.astype(int)
            for l in lines:
                cv2.line(observed_field_raw, [l[1], l[0]], [l[3], l[2]], 255, 2)
                
            observed_field_cleaned = np.zeros((2 * self.simulator.field.shape[0], 2 * self.simulator.field.shape[1]))
                
            cleaned_verticals = observed_verticals.copy()
            cleaned_horizontals = observed_horizontals.copy()

            if len(cleaned_verticals) > 0:
                cleaned_verticals[:, 0] *= (self.simulator.field.shape[0] / self.dimensions_of_the_field[0])
                cleaned_verticals[:, 0] += self.simulator.field.shape[0]
                cleaned_verticals[:, 1:] *= (self.simulator.field.shape[1] / self.dimensions_of_the_field[1])
                cleaned_verticals[:, 1:] += self.simulator.field.shape[1]
                cleaned_verticals = cleaned_verticals.astype(int)
                for l in cleaned_verticals:
                    cv2.line(observed_field_cleaned, [l[1], l[0]], [l[2], l[0]], 255, 2)

            if len(cleaned_horizontals) > 0:
                cleaned_horizontals[:, 0] *= (self.simulator.field.shape[1] / self.dimensions_of_the_field[1])
                cleaned_horizontals[:, 0] += self.simulator.field.shape[1]
                cleaned_horizontals[:, 1:] *= (self.simulator.field.shape[0] / self.dimensions_of_the_field[0])
                cleaned_horizontals[:, 1:] += self.simulator.field.shape[0]
                cleaned_horizontals = cleaned_horizontals.astype(int)
                for l in cleaned_horizontals:
                    cv2.line(observed_field_cleaned, [l[0], l[1]], [l[0], l[2]], 255, 2)

            axs[0].imshow(observed_field_raw)
            axs[1].imshow(observed_field_cleaned)
            plt.show(block=False)
        
        t_3 = time.time()
        vertical_rectangles_count = self.original_verticals.shape[0] * observed_verticals.shape[0]
        horizontal_rectangles_count = self.original_horizontals.shape[0] * observed_horizontals.shape[0]
        rectangles = np.zeros((vertical_rectangles_count + horizontal_rectangles_count, 4))# (alignment, x/y, y/x, y/x)
        
        if len(observed_verticals) > 0:
            vertical_rectangles = np.fromfunction(lambda i, j: self.original_verticals[i] - observed_verticals[j], (self.original_verticals.shape[0], observed_verticals.shape[0]), dtype=int)
            vertical_rectangles = vertical_rectangles.reshape(self.original_verticals.shape[0] * observed_verticals.shape[0], 3)
            vertical_rectangles[:, 1:].sort(axis = 1)
            rectangles[: vertical_rectangles.shape[0], 1: ] = vertical_rectangles

        if len(observed_horizontals) > 0:
            horizontal_rectangles = np.fromfunction(lambda i, j: self.original_horizontals[i] - observed_horizontals[j], (self.original_horizontals.shape[0], observed_horizontals.shape[0]), dtype=int)
            horizontal_rectangles = horizontal_rectangles.reshape(self.original_horizontals.shape[0] * observed_horizontals.shape[0], 3)
            horizontal_rectangles[:, 1:].sort(axis = 1)
            rectangles[vertical_rectangles_count: , 1: ] = horizontal_rectangles
            rectangles[vertical_rectangles_count:, 0] = 1

        coordinate_stats = [] # list of dictionaries: [{x, x_error, y, y_error, x_lower, x_upper, y_lower, y_upper}]
        
        field_rectangle = ro.rectangle(-self.line_width, self.dimensions_of_the_field[0] + self.line_width, -self.line_width, self.dimensions_of_the_field[1] + self.line_width, match = 0, score = (observed_verticals.shape[0] + observed_horizontals.shape[0] + 1))
        if len(rectangles) > 0:
            rectanglize = np.vectorize(lambda i: ro.rectangle(rectangles[i, 2], rectangles[i, 3], rectangles[i, 1] - self.line_width, rectangles[i, 1] + self.line_width, i + 1) if rectangles[i, 0] else ro.rectangle(rectangles[i, 1] - self.line_width, rectangles[i, 1] + self.line_width, rectangles[i, 2], rectangles[i, 3], i + 1))
            rectangles = np.hstack((rectangles, rectanglize(np.arange(rectangles.shape[0]))[:, np.newaxis]))
            rectangles = np.vstack((np.array([-1, np.nan, np.nan, np.nan, field_rectangle]), rectangles))
        else:
            rectangles = np.array([[-1, np.nan, np.nan, np.nan, field_rectangle]])

        best_matches = ro.get_maximum_overlapping_rectangles(rectangles[:, 4])
        for match in best_matches:
            d = {}
            match_ids = tuple(np.vectorize(lambda rect: rect.match)(match))
            matched_rectangles = np.take(rectangles, match_ids, axis = 0)
            
            vertical_data = matched_rectangles[np.where(matched_rectangles[:, 0] == 0)][:, 1: 4]
            vertical_cores = vertical_data[:, 0]
            horizontal_intervals = vertical_data[:, 1:]
            horizontal_intervals = np.vstack((np.array([0, self.dimensions_of_the_field[1]]), horizontal_intervals))

            horizontal_data = matched_rectangles[np.where(matched_rectangles[:, 0] == 1)][:, 1: 4]
            horizontal_cores = horizontal_data[:, 0]
            vertical_intervals = horizontal_data[:, 1:]
            vertical_intervals = np.vstack((np.array([0, self.dimensions_of_the_field[0]]), vertical_intervals))
                
            d["x"], d["x_error"] = ro.get_coordinate_from_cores_and_intervals(vertical_cores, vertical_intervals)
            d["y"], d["y_error"] = ro.get_coordinate_from_cores_and_intervals(horizontal_cores, horizontal_intervals)

            d["x_range"] = best_matches[match].x_range
            d["y_range"] = best_matches[match].y_range

            coordinate_stats.append(d)
        
        t_4 = time.time()
        if calculate_time:
            return coordinate_stats, (t_2 - t_1 + t_4 - t_3)    
        else:
            return coordinate_stats

    def visualize(self, height = 500):
        app = Dash()

        radious = self.get_radius() + 1
        field_x = radious
        field_y = radious
        field_z = radious

        panels = {} #camera_id: camera_fig
        gadget_controllers = []

        cmap = plt.get_cmap("tab10")
        colorscale = [[0, 'rgb' + str(cmap(1)[0:3])], 
        [1, 'rgb' + str(cmap(2)[0:3])]]

        sketch_components = {'floor': 0, 'nodes': 1, 'cameras': dict()}

        sketch = go.Figure()
        sketch.add_trace(go.Surface(x = np.linspace(-field_x, field_x, 10), y = np.linspace(-field_y, field_y, 10), z = 0 * np.ones((10, 10)), cmin=0, 
        cmax=1, showscale=False, opacity = 0.3, surfacecolor= np.ones((10, 10)), colorscale=colorscale))


        sketch.add_trace(go.Scatter3d(showlegend = False, 
        x = [], y = [], z = [], 
        marker=dict(size=4, colorscale='Viridis'), 
        line=dict(color='darkblue', width=6)))

        actual_nodes = {**self.nodes}
        actual_nodes.pop('root')

        panel_outputs = []
        click_inputs = []

        panel_inputs = []
        slider_inputs = []

        component_counter = 2

        for node_id in actual_nodes:
            node = self.nodes.get(node_id)
            gadget = node.data
            if gadget.__class__.__name__ == 'camera':
                camera_fig = px.imshow(
                    np.zeros(shape=(*gadget.resolution, 4)))
                camera_fig.add_scatter(
                    x=[], 
                    y=[], 
                    mode='markers',
                    marker_color='red',
                    marker_size=10)
                camera_fig.update_layout(margin=dict(l=30, r=30, t=0, b=0))

                controller = [dcc.Graph(id = node_id, style = {'height': '%dpx'%int(0.3 * height), 'width': '100%', 'padding': '0px', 'margine': '0px'}, figure = camera_fig)]
                panels[node_id] = camera_fig
        
                sketch_components['cameras'][node_id] = {'direction0': component_counter, 
                'direction1': component_counter + 1, 
                'direction2': component_counter + 2, 
                'direction_tips': component_counter + 3,
                'panel': component_counter + 4,
                'line_of_sight': component_counter + 5
                }
                component_counter += 6
        
                sketch.add_traces([go.Scatter3d(showlegend = False, 
                x = [], y = [], z = [], 
                marker=dict(size=4, colorscale='Viridis'), 
                mode = 'lines+markers+text', text = ['', "x'"], 
                line=dict(color='yellow', width=3)), 
                go.Scatter3d(showlegend = False, 
                x = [], y = [], z = [], 
                marker=dict(size=4, colorscale='Viridis'), 
                mode = 'lines+markers+text', text = ['', "y'"], 
                line=dict(color='yellow', width=3)), 
                go.Scatter3d(showlegend = False, 
                x = [], y = [], z = [], 
                marker=dict(size=4, colorscale='Viridis'), 
                mode = 'lines+markers+text', text = ['', "z'"], 
                line=dict(color='yellow', width=3)), 
                go.Cone(showlegend=False, 
                x=[], y=[], z=[], u=[], v=[], w=[], 
                colorscale = [[0, 'rgb(255,255,0)'], [1, 'rgb(255,255,0)']], showscale=False, cmin = 0, cmax = 1, sizemode="raw"), 
                go.Mesh3d(x=[], y=[], z=[], color='lightgray', opacity=0.5), 
                go.Scatter3d(showlegend = False, 
                x = [], y = [], z = [], 
                marker=dict(size=4, colorscale='Viridis'), 
                line=dict(color='red', width=2))
                ])

                panel_outputs.append(Output(node_id, 'figure'))
                click_inputs.append(Input(node_id, 'clickData'))
        
                panel_inputs.append(Input(node_id, 'figure'))
    
    
            elif gadget.__class__.__name__ == 'motor':
                controller = [
                dcc.Slider(id = node_id, min = np.rad2deg(gadget.range[0]), max = np.rad2deg(gadget.range[1]), value = self.nodes.get(node_id).data.motor_angle)]
        
                slider_inputs.append(Input(node_id, 'value'))
         
            gadget_controllers.append(html.Div(style = {'width': '100%', 'padding': '10px'}, children = [html.Label(node_id, style = {'margine': '10px', 'padding': '10px'})] + controller))

            sketch_components[node_id] = go.Scatter3d(showlegend = False, 
            marker=dict(size=4, colorscale='Viridis'), 
            line=dict(color='darkblue', width=6))
    
        
        


        app.layout = html.Div(style = {'width': '100%', 'height': '%dpx'%(height), 'display': 'flex'}, children = [
            html.Div(style = {'flex': '0 0 70%'}, children = [
                dcc.Graph(id = 'sketch', style = {'width': '100%', 'height': '%dpx'%(height), 'padding': '0px', 'margine': '0px'}, figure = sketch)
                ]), 
                html.Div(style = {'height': '%dpx'%int(height * 0.5), 'overflowY': 'scroll', 'flex': '0 0 30%', 'border': '1px solid black'}, children = gadget_controllers)
                ])



        @app.callback(
            panel_outputs, click_inputs, prevent_initial_call=True
            )
        def update_panels(*args):
            camera_id = callback_context.triggered[0].get('prop_id')[:-len('.clickData')]
            graph_figure = panels[camera_id]
            clickData = callback_context.triggered[0].get('value')
            if not clickData:
                raise PreventUpdate
            else:
                points = clickData.get('points')[0]
                x = points.get('x')
                y = points.get('y')
                graph_figure['data'][1].update(x = [x])
                graph_figure['data'][1].update(y = [y])

            return tuple(panels.values())


        @app.callback(
            Output('sketch', 'figure'), panel_inputs + slider_inputs
            )
        def update_sketch(*args):
            t = 0.1
            d = 0.9
            prop = callback_context.triggered[0].get('prop_id')
            if prop.endswith('figure'):
                node_id = prop[:-len('.figure')]
        
                camera_figure = callback_context.triggered[0].get('value')

                camera_node = self.nodes.get(node_id)
                directions = camera_node.node_basis_relative_to_origin
                point_on_screen = np.array(camera_figure['data'][1]['y'] + camera_figure['data'][1]['x'])
                camera_to_point = camera_node.data.get_direction_from_point_on_screen(point_on_screen)
                end_of_arrow = np.sqrt(field_x ** 2 + field_y ** 2 + field_z ** 2) * camera_to_point
                camera_to_point *= (d / camera_to_point[0, 0])
                point_in_front_of_camera = camera_node.position + np.matmul(directions, camera_to_point)
                end_of_arrow = camera_node.position + np.matmul(directions, end_of_arrow)
                the_line = np.hstack((camera_node.position, point_in_front_of_camera, end_of_arrow))

                sketch.data[sketch_components['cameras'][node_id]['line_of_sight']].x = the_line[0, :]
                sketch.data[sketch_components['cameras'][node_id]['line_of_sight']].y = the_line[1, :]
                sketch.data[sketch_components['cameras'][node_id]['line_of_sight']].z = the_line[2, :]
        
            else:
                if prop.endswith('value'):
                    node_id = prop[:-len('.value')]
                    angle = callback_context.triggered[0].get('value')
                    self.set_motors_angle(**{node_id: angle})
                elif prop == '.':
                    node_id = 'root'
            
                joints = np.array(list(self.get_nodes_locations().values()))
        
                sketch.data[sketch_components['nodes']].x = joints[:, 0, 0]
                sketch.data[sketch_components['nodes']].y = joints[:, 1, 0]
                sketch.data[sketch_components['nodes']].z = joints[:, 2, 0]

                for camera_id in list(panels.keys()):
                    camera_node = self.nodes.get(camera_id)
                    node = self.nodes.get(node_id)

                    if node.is_ancestor_of(camera_node):
                        camera_focal_point = camera_node.position[:, 0]
                        directions = camera_node.node_basis_relative_to_origin
                        x_prime, y_prime, z_prime = (directions).T
                
                        sketch.data[sketch_components['cameras'][camera_id]['direction0']].x = [camera_focal_point[0], camera_focal_point[0] + x_prime[0]]
                        sketch.data[sketch_components['cameras'][camera_id]['direction0']].y = [camera_focal_point[1], camera_focal_point[1] + x_prime[1]]
                        sketch.data[sketch_components['cameras'][camera_id]['direction0']].z = [camera_focal_point[2], camera_focal_point[2] + x_prime[2]]
                
                        sketch.data[sketch_components['cameras'][camera_id]['direction1']].x = [camera_focal_point[0], camera_focal_point[0] + y_prime[0]]
                        sketch.data[sketch_components['cameras'][camera_id]['direction1']].y = [camera_focal_point[1], camera_focal_point[1] + y_prime[1]]
                        sketch.data[sketch_components['cameras'][camera_id]['direction1']].z = [camera_focal_point[2], camera_focal_point[2] + y_prime[2]]

                        sketch.data[sketch_components['cameras'][camera_id]['direction2']].x = [camera_focal_point[0], camera_focal_point[0] + z_prime[0]]
                        sketch.data[sketch_components['cameras'][camera_id]['direction2']].y = [camera_focal_point[1], camera_focal_point[1] + z_prime[1]]
                        sketch.data[sketch_components['cameras'][camera_id]['direction2']].z = [camera_focal_point[2], camera_focal_point[2] + z_prime[2]]
                
                        sketch.data[sketch_components['cameras'][camera_id]['direction_tips']].x = [camera_focal_point[0] + x_prime[0], camera_focal_point[0] + y_prime[0], camera_focal_point[0] + z_prime[0]]
                        sketch.data[sketch_components['cameras'][camera_id]['direction_tips']].y = [camera_focal_point[1] + x_prime[1], camera_focal_point[1] + y_prime[1], camera_focal_point[1] + z_prime[1]]
                        sketch.data[sketch_components['cameras'][camera_id]['direction_tips']].z = [camera_focal_point[2] + x_prime[2], camera_focal_point[2] + y_prime[2], camera_focal_point[2] + z_prime[2]]
                        sketch.data[sketch_components['cameras'][camera_id]['direction_tips']].u = [t * x_prime[0], t * y_prime[0], t * z_prime[0]]
                        sketch.data[sketch_components['cameras'][camera_id]['direction_tips']].v = [t * x_prime[1], t * y_prime[1], t * z_prime[1]]
                        sketch.data[sketch_components['cameras'][camera_id]['direction_tips']].w = [t * x_prime[2], t * y_prime[2], t * z_prime[2]]

                        y_half_range_of_screen = np.tan(np.deg2rad(camera_node.data.angular_width / 2))
                        z_half_range_of_screen = y_half_range_of_screen * camera_node.data.resolution[0] / camera_node.data.resolution[1]
                        screen_points = d * np.array([
                            [1, y_half_range_of_screen, z_half_range_of_screen], 
                            [1, -y_half_range_of_screen, z_half_range_of_screen],
                            [1, -y_half_range_of_screen, -z_half_range_of_screen],
                            [1, y_half_range_of_screen, -z_half_range_of_screen]
                            ]).T
                        screen_points = camera_node.position + np.matmul(directions, screen_points)
                
                        sketch.data[sketch_components['cameras'][camera_id]['panel']].x = screen_points[0, :]
                        sketch.data[sketch_components['cameras'][camera_id]['panel']].y = screen_points[1, :]
                        sketch.data[sketch_components['cameras'][camera_id]['panel']].z = screen_points[2, :]

                        camera_figure = panels[camera_id]
                        point_on_screen = np.array(camera_figure['data'][1]['y'] + camera_figure['data'][1]['x'])
                        if len(point_on_screen) != 0:
                            camera_to_point = camera_node.data.get_direction_from_point_on_screen(point_on_screen)
                            end_of_arrow = np.sqrt(field_x ** 2 + field_y ** 2 + field_z ** 2) * camera_to_point
                            camera_to_point *= (d / camera_to_point[0, 0])
                            point_in_front_of_camera = camera_node.position + np.matmul(directions, camera_to_point)
                            end_of_arrow = camera_node.position + np.matmul(directions, end_of_arrow)
                            the_line = np.hstack((camera_node.position, point_in_front_of_camera, end_of_arrow))

                            sketch.data[sketch_components['cameras'][camera_id]['line_of_sight']].x = the_line[0, :]
                            sketch.data[sketch_components['cameras'][camera_id]['line_of_sight']].y = the_line[1, :]
                            sketch.data[sketch_components['cameras'][camera_id]['line_of_sight']].z = the_line[2, :]
                    
            return sketch
    

        sketch.update_layout(scene=dict(aspectmode='manual', 
        xaxis_title='X Axis', 
        yaxis_title='Y Axis', 
        zaxis_title='Z Axis', aspectratio=dict(x=2, y=2, z=1), 
        xaxis=dict(range=[-field_x, field_x]), 
        yaxis=dict(range=[-field_y, field_y]), 
        zaxis=dict(range=[0, field_z])), 
        title='The Sketch of the Robot', 
        margin=dict(l=30, r=30, t=0, b=0))

        return app