import numpy as np
import matplotlib.pyplot as plt
import inspect


class rectangle:
    def __init__(self, x_low, x_high, y_low, y_high, match = None, score = 1):
        self.match = match
        self.x_range = (x_low, x_high)
        self.y_range = (y_low, y_high)
        self.score = score

    @classmethod
    def intersection(cls, *rectangles):
        rectangles = np.array(rectangles)
        rectangles = np.vstack(np.vectorize(lambda rect: (rect.x_range[0], rect.x_range[1], rect.y_range[0], rect.y_range[1], rect.score))(rectangles)).T
        return rectangle(np.max(rectangles[:, 0]), np.min(rectangles[:, 1]), np.max(rectangles[:, 2]), np.min(rectangles[:, 3]), np.sum(rectangles[:, 4]))
    

class sweepLineNode:
    def __init__(self):
        self.point = []
        self.value = -np.inf
        self.subtree_sum = 0
        self.max_sum = -np.inf
        self.max_points = []
        self.color = 0
        self.parent = None
        self.right = None
        self.left = None

    def set_data_from_children(self):
        if len(self.point) != 0:
            self.subtree_sum = self.left.subtree_sum + self.value + self.right.subtree_sum
            self.max_sum = self.left.max_sum
            self.max_points = self.left.max_points
            if self.max_sum == self.left.subtree_sum + self.value:
                self.max_points = self.max_points + self.point
            elif self.max_sum < self.left.subtree_sum + self.value:
                self.max_sum = self.left.subtree_sum + self.value
                self.max_points = self.point
            if self.max_sum == self.left.subtree_sum + self.value + self.right.max_sum:
                self.max_points = self.max_points + self.right.max_points
            elif self.max_sum < self.left.subtree_sum + self.value + self.right.max_sum:
                self.max_sum = self.left.subtree_sum + self.value + self.right.max_sum
                self.max_points = self.right.max_points

    
    def print_subtree(self, stop):
        if self != stop:
            self.left.print_subtree(stop)
            print(self, '::::', vars(self))
            self.right.print_subtree(stop)
        
            
class sweepLine:
    def __init__(self):
        self.nil = sweepLineNode()
        self.root = self.nil
        self.nodes = dict()

    def tree_minimum(self, x):
        while x.left != self.nil:
            x = x.left
        return x
    
    def left_rotate(self, x):
        y = x.right
        x.right = y.left
        if y.left != self.nil:
            y.left.parent = x
        y.parent = x.parent
        if x.parent == self.nil:
            self.root = y
        elif x == x.parent.left:
            x.parent.left = y
        else:
            x.parent.right = y
        y.left = x
        x.parent = y
        x.set_data_from_children()
        y.set_data_from_children()

    def right_rotate(self, x):
        y = x.left
        x.left = y.right
        if y.right != self.nil:
            y.right.parent = x
        y.parent = x.parent
        if x.parent == self.nil:
            self.root = y
        elif x == x.parent.right:
            x.parent.right = y
        else:
            x.parent.left = y
        y.right = x
        x.parent = y
        x.set_data_from_children()
        y.set_data_from_children()
    
    def insert_fixup(self, z):
        while z.parent.color == 1:
            if z.parent == z.parent.parent.left:
                y = z.parent.parent.right
                if y.color == 1:
                    z.parent.color = 0
                    y.color = 0
                    z.parent.parent.color = 1
                    z = z.parent.parent
                else:
                    if z == z.parent.right:
                        z = z.parent
                        self.left_rotate(z)
                    z.parent.color = 0
                    z.parent.parent.color = 1
                    self.right_rotate(z.parent.parent)
            else:
                y = z.parent.parent.left
                if y.color == 1:
                    z.parent.color = 0
                    y.color = 0
                    z.parent.parent.color = 1
                    z = z.parent.parent
                else:
                    if z == z.parent.left:
                        z = z.parent
                        self.right_rotate(z)
                    z.parent.color = 0
                    z.parent.parent.color = 1
                    self.left_rotate(z.parent.parent)
        self.root.color = 0
            
    
    def phase_one_augmentation_fixup(self, x):
        while x != self.nil:
            x.set_data_from_children()
            x = x.parent
    
    def insert(self, point, value):
        z = sweepLineNode()
        z.point = [point]
        z.value = value
        self.nodes[(point, value)] = z
        x = self.root
        y = self.nil
        while x != self.nil:
            y = x
            if z.point[0] < x.point[0]:
                x = x.left
            else:
                x = x.right
        z.parent = y
        if y == self.nil:
            self.root  = z
        elif z.point[0] < y.point[0]:
            y.left = z
        else:
            y.right = z
        z.left = self.nil
        z.right = self.nil
        z.color = 1
        self.phase_one_augmentation_fixup(z)
        self.insert_fixup(z)

    def transplant(self, u, v):
        if u.parent == self.nil:
            self.root = v
        elif u == u.parent.left:
            u.parent.left = v
        else:
            u.parent.right = v
        v.parent = u.parent
    
    
    def delete_fixup(self, x):
        while x != self.root and x.color == 0:
            if x == x.parent.left:
                w = x.parent.right
                if w.color == 1:
                    w.color = 0
                    x.parent.color = 1
                    self.left_rotate(x.parent)
                    w = x.parent.right
                if w.left.color == 0 and w.right.color == 0:
                    w.color = 1
                    x = x.parent
                else:
                    if w.right.color == 0:
                        w.left.color = 0
                        w.color = 0
                        self.right_rotate(w)
                        w = x.parent.right
                    w.color = x.parent.color
                    x.parent.color = 0
                    w.right.color = 0
                    self.left_rotate(x.parent)
                    x = self.root
            else:
                w = x.parent.left
                if w.color == 1:
                    w.color = 0
                    x.parent.color = 1
                    self.right_rotate(x.parent)
                    w = x.parent.left
                if w.right.color == 0 and w.left.color == 0:
                    w.color = 1
                    x = x.parent
                else:
                    if w.left.color == 0:
                        w.right.color = 0
                        w.color = 1
                        self.left_rotate(w)
                        w = x.parent.left
                    w.color = x.parent.color
                    x.parent.color = 0
                    w.left.color = 0
                    self.right_rotate(x.parent)
                    x = self.root
        x.color = 0
    
    def delete(self, point, value):
        z = self.nodes.pop((point, value))
        y = z
        y_original_color = y.color
        if z.left == self.nil:
            x = z.right
            self.transplant(z, z.right)
        elif z.right == self.nil:
            x = z.left
            self.transplant(z, z.left)
        else:
            y = self.tree_minimum(z.right)
            y_original_color = y.color
            x = y.right
            if y != z.right:
                self.transplant(y, y.right)
                y.right = z.right
                y.right.parent = y
            else:
                x.parent = y
            self.transplant(z, y)
            y.left = z.left
            y.left.parent = y
            y.color = z.color

        self.phase_one_augmentation_fixup(x.parent)
        if y_original_color == 0:
            self.delete_fixup(x)
    
    def get_max_overlap_data(self):
        return self.root.max_sum, self.root.max_points
    
    def print_the_line(self):
        self.root.print_subtree(self.nil)
        

class sweepingLineIterator:
    def __init__(self, rectangles):
        self.rectangles = np.array(rectangles)
        self.rectangles = np.vstack((rectangles, np.ones(self.rectangles.shape[0]), -np.ones(self.rectangles.shape[0])))
        self.rectangles = np.hstack((self.rectangles[(0, 1), :], self.rectangles[(0, 2), :])).T.tolist()
        self.rectangles.sort(key = lambda epoch: (epoch[0].x_range[0] if epoch[1] == 1 else epoch[0].x_range[1]))
        self.active_rectangles = set()
        self.line = sweepLine()
        self.index = 0
        self.state = 1
        self.epoch = -1
        
    def __iter__(self):
        return self
        
    def __next__(self):
        if self.index == len(self.rectangles):
            raise StopIteration
        else:
            self.epoch += 1
            while True:
                if self.index < len(self.rectangles):
                    if self.state == self.rectangles[self.index][1]:
                        if self.state == 1:
                            self.active_rectangles.add(self.rectangles[self.index][0])
                            self.line.insert(self.rectangles[self.index][0].y_range[0], self.rectangles[self.index][0].score)
                            self.line.insert(self.rectangles[self.index][0].y_range[1], -self.rectangles[self.index][0].score)
                        else:
                            self.active_rectangles.remove(self.rectangles[self.index][0])
                            self.line.delete(self.rectangles[self.index][0].y_range[0], self.rectangles[self.index][0].score)
                            self.line.delete(self.rectangles[self.index][0].y_range[1], -self.rectangles[self.index][0].score)
                        self.index += 1
                    else:
                        self.state = self.rectangles[self.index][1]
                        return self.active_rectangles, *self.line.get_max_overlap_data()
                else:
                    return self.active_rectangles, *self.line.get_max_overlap_data()


class intervalTree:
    def __init__(self, interval_attribute):
        self.interval_attribute = interval_attribute
        self.discriminant = -np.inf
        self.crossing_intervals = []
        self.left = None
        self.right = None
    
    @classmethod
    def create_tree_from_intervals_sorted_by_endpoints(cls, intervals, intervals_sorted_by_endpoints, interval_attribute):
        root = None
        if len(intervals_sorted_by_endpoints) > 0:
            root = intervalTree(interval_attribute)
            mid = (len(intervals_sorted_by_endpoints) - 1) // 2
            root.discriminant = intervals_sorted_by_endpoints[mid, 0]
            left_intervals_sorted_by_endpoints = intervals_sorted_by_endpoints[np.where(intervals_sorted_by_endpoints[:, 3] < root.discriminant)]
            left_intervals = intervals[np.where(intervals[:, 2] < root.discriminant)]
            right_intervals_sorted_by_endpoints = intervals_sorted_by_endpoints[np.where(intervals_sorted_by_endpoints[:, 2] > root.discriminant)]
            right_intervals = intervals[np.where(intervals[:, 1] > root.discriminant)]
            root.crossing_intervals = intervals[np.where((intervals[:, 1] <= root.discriminant) & (intervals[:, 2] >= root.discriminant))]
            root.left = cls.create_tree_from_intervals_sorted_by_endpoints(left_intervals, left_intervals_sorted_by_endpoints, interval_attribute)
            root.right = cls.create_tree_from_intervals_sorted_by_endpoints(right_intervals, right_intervals_sorted_by_endpoints, interval_attribute)
            
        return root
    
    @classmethod
    def create_tree(cls, intervals, interval_attribute):
        lows = np.vectorize(lambda x: getattr(x, interval_attribute)[0])(intervals)
        highs = np.vectorize(lambda x: getattr(x, interval_attribute)[1])(intervals)
        intervals_sorted_by_endpoints = np.hstack((np.vstack((lows, intervals, lows, highs)), 
                                                   np.vstack((highs, intervals, lows, highs))))
        intervals_sorted_by_endpoints = intervals_sorted_by_endpoints.T
        intervals_sorted_by_endpoints  = intervals_sorted_by_endpoints[intervals_sorted_by_endpoints[:, 0].argsort()]
        return cls.create_tree_from_intervals_sorted_by_endpoints(np.vstack((intervals, lows, highs)).T, intervals_sorted_by_endpoints, interval_attribute)
        
    def query(self, point):
        node = self
        including_intervals = []
        while node != None:
            if point == node.discriminant:
                including_intervals = np.hstack((including_intervals, node.crossing_intervals[:, 0]))
                node = None
            else:
                
                if point < node.discriminant:
                    including_intervals = np.hstack((including_intervals, node.crossing_intervals[np.where(node.crossing_intervals[:, 1] <= point)][:, 0]))
                    node = node.left
            
                elif point > node.discriminant:
                    including_intervals = np.hstack((including_intervals, node.crossing_intervals[np.where(node.crossing_intervals[:, 2] >= point)][:, 0]))
                    node = node.right
            
        return including_intervals

    def print_subtree(self):
        if self.left != None:
            self.left.print_subtree()
        print(vars(self))
        if self.right != None:
            self.right.print_subtree()



def get_maximum_overlapping_rectangles(rectangles):
    epochs = sweepingLineIterator(rectangles)
    max_overlap = 0
    max_overlap_data = []
    for active_rectangles, max_overlap_at_epoch, points_of_max_overlap_at_epoch in epochs:
        if max_overlap_at_epoch == max_overlap:
            max_overlap_data.append((list(active_rectangles), points_of_max_overlap_at_epoch))
        
        elif max_overlap_at_epoch > max_overlap:
            max_overlap = max_overlap_at_epoch
            max_overlap_data = [(list(active_rectangles), points_of_max_overlap_at_epoch)]
        
    maximum_overlapping_rectangles = dict()
    
    for data in max_overlap_data:
        active_rectangles = data[0]
        query_points = data[1]
        intervals = intervalTree.create_tree(active_rectangles, 'y_range')
        overlapping_rectangles = []
        for point in query_points:
            query_results = intervals.query(point)
            range = rectangle.intersection(*query_results)
            query_results = np.vstack((query_results, np.vectorize(lambda rect: rect.match)(query_results))).T
            query_results = query_results[query_results[:, 1].argsort()][:, 0]
            maximum_overlapping_rectangles[tuple(query_results)] = range
            
        
    return maximum_overlapping_rectangles


def get_coordinate_from_cores_and_intervals(cores, intervals):
    intervals = np.vstack((intervals, np.array([-np.inf, np.inf])))
    virtual_core_epochs = np.hstack((intervals, np.arange(intervals.shape[0]).reshape(intervals.shape[0], 1)))
    virtual_core_epochs = np.vstack((virtual_core_epochs[:, (0, 2)], virtual_core_epochs[:, (1, 2)]))
    virtual_core_epochs = virtual_core_epochs[virtual_core_epochs[:, 0].argsort()]

    virtual_cores_state = -np.ones(intervals.shape[0])
    virtual_cores_state[-1] += 1
    error = np.inf
    for i in range(1, virtual_core_epochs.shape[0]):
        local_cores = np.hstack([cores, intervals[np.where(virtual_cores_state == -1)[0], 0], intervals[np.where(virtual_cores_state == 1)[0], 1]])
        if len(local_cores) > 0:
            local_coordinate = min(max([local_cores.mean(), virtual_core_epochs[i - 1, 0]]), virtual_core_epochs[i, 0])
            local_error = (local_cores - local_coordinate).std()
        else:
            local_coordinate = (virtual_core_epochs[i, 0] + virtual_core_epochs[i - 1, 0]) / 2
            local_error = (virtual_core_epochs[i, 0] - virtual_core_epochs[i - 1, 0]) / 2
            coordinate = local_coordinate
            error = local_error
            break
        if local_error < error:
            coordinate = local_coordinate
            error = local_error
            
        virtual_cores_state[int(virtual_core_epochs[i, 1])] += 1
    
    return coordinate, error