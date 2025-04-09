import datetime
import numpy as np
import os
import pytz
import random
import time

# This try-except enables local debugging of the PRM class
try:
    from amr_planning.maps import Map
except ImportError:
    from maps import Map

from matplotlib import pyplot as plt


class PRM:
    """Class to plan a path to a given destination using probabilistic roadmaps (PRM)."""

    def __init__(
        self,
        map_path: str,
        obstacle_safety_distance=0.08,
        use_grid: bool = False,
        node_count: int = 50,
        grid_size=0.2,
        connection_distance: float = 0.15,
        sensor_range_max: float = 8.0,
    ):
        """Probabilistic roadmap (PRM) class initializer.

        Args:
            map_path: Path to the map of the environment.
            obstacle_safety_distance: Distance to grow the obstacles by [m].
            use_grid: Sample from a uniform distribution when False.
                Use a fixed step grid layout otherwise.
            node_count: Number of random nodes to generate. Only considered if use_grid is False.
            grid_size: If use_grid is True, distance between consecutive nodes in x and y.
            connection_distance: Maximum distance to consider adding an edge between two nodes [m].
            sensor_range_max: Sensor measurement range [m].
        """
        self._map: Map = Map(
            map_path,
            sensor_range=sensor_range_max,
            safety_distance=obstacle_safety_distance,
            compiled_intersect=False,
            use_regions=False,
        )

        self._graph: dict[tuple[float, float], list[tuple[float, float]]] = self._create_graph(
            use_grid,
            node_count,
            grid_size,
            connection_distance,
        )

        self._figure, self._axes = plt.subplots(1, 1, figsize=(7, 7))
        self._timestamp = datetime.datetime.now(pytz.timezone("Europe/Madrid")).strftime(
            "%Y-%m-%d_%H-%M-%S"
        )

    def find_path(
        self, start: tuple[float, float], goal: tuple[float, float]
    ) -> list[tuple[float, float]]:
        """Computes the shortest path from a start to a goal location using the A* algorithm.

        Args:
            start: Initial location in (x, y) [m] format.
            goal: Destination in (x, y) [m] format.

        Returns:
            Path to the destination. The first value corresponds to the initial location.

        """
        # Check if the target points are valid
        if not self._map.contains(start):
            raise ValueError("Start location is outside the environment.")

        if not self._map.contains(goal):
            raise ValueError("Goal location is outside the environment.")

        ancestors: dict[tuple[float, float], tuple[float, float]] = {}  # {(x, y: (x_prev, y_prev)}

        # TODO: 4.3. Complete the function body (i.e., replace the code below).
        path: list[tuple[float, float]] = []
        # 1. Find the closest graph nodes to the start and goal points
        start_node = None
        goal_node = None
        min_start_dist = float('inf')
        min_goal_dist = float('inf')

        for node in self._graph.keys():
            # Calculate distance from node to start point
            start_dist = np.sqrt((node[0] - start[0])**2 + (node[1] - start[1])**2)
            if start_dist < min_start_dist and not self._map.crosses([node, start]):
                min_start_dist = start_dist
                start_node = node
                
            # Calculate distance from node to goal point
            goal_dist = np.sqrt((node[0] - goal[0])**2 + (node[1] - goal[1])**2)
            if goal_dist < min_goal_dist and not self._map.crosses([node, goal]):
                min_goal_dist = goal_dist
                goal_node = node


        # 2. Initialize open list (dictionary) and closed list (set)
        # Open list: {(x, y): (f, g)} where f = g + h (total cost), g = cost from start
        open_list = {}
        closed_list = set()
        
        # Helper function to calculate heuristic (Euclidean distance)
        def heuristic(node1, node2):
            return np.sqrt((node1[0] - node2[0])**2 + (node1[1] - node2[1])**2)
        
        # Add start node to open list with g=0 and f=heuristic
        g_start = 0
        f_start = g_start + heuristic(start_node, goal_node)
        open_list[start_node] = (f_start, g_start)

        # 3. Main A* loop
        while open_list and goal_node not in open_list:
            # a) Get node with lowest f value from open list
            current = min(open_list, key=lambda k: open_list.get(k)[0])
            
            # b) Get g value and remove from open list
            current_f, current_g = open_list[current]
            del open_list[current]
            
            # c) Expand the current node
            for neighbor in self._graph[current]:
                # Skip if neighbor is in closed list
                if neighbor in closed_list:
                    continue

                # Calculate tentative g value (cost from start to neighbor through current)
                edge_cost = heuristic(current, neighbor)
                tentative_g = current_g + edge_cost
                
                # If neighbor not in open list or has better g value
                if neighbor not in open_list or tentative_g < open_list[neighbor][1]:
                    # Update open list with new f and g values
                    f_neighbor = tentative_g + heuristic(neighbor, goal_node)
                    open_list[neighbor] = (f_neighbor, tentative_g)
                    
                    # Update ancestors dictionary
                    ancestors[neighbor] = current
            
            # d) Add current to closed list
            closed_list.add(current)

        # 4. Check if a path was found and reconstruct it
        if goal_node in open_list:
            # If start and goal are not graph nodes, add them to ancestors
            if start != start_node:
                ancestors[start_node] = start
            
            if goal != goal_node:
                ancestors[goal] = goal_node
            
            return self._reconstruct_path(start, goal, ancestors)
        else:
            raise ValueError("No path found between start and goal.")
            
    @staticmethod
    def smooth_path(
        path: list[tuple[float, float]],
        data_weight: float = 0.2,
        smooth_weight: float = 0.15,
        additional_smoothing_points: int = 1,
        tolerance: float = 1e-6,
    ) -> list[tuple[float, float]]:
        """Computes a smooth path from a piecewise linear path.

        Args:
            path: Non-smoothed path to the goal (start location first).
            data_weight: The larger, the more similar the output will be to the original path.
            smooth_weight: The larger, the smoother the output path will be.
            additional_smoothing_points: Number of equally spaced intermediate points to add
                between two nodes of the original path.
            tolerance: The algorithm will stop when after an iteration the smoothed path changes
                less than this value.

        Returns: Smoothed path (initial location first) in (x, y) format.

        """
        # TODO: 4.5. Complete the function body (i.e., load smoothed_path).

        if len(path) <= 2:  # If path has only start and goal, no smoothing needed
            return path.copy()
    
        # Create enhanced path with additional points if requested
        enhanced_path = []
        for i in range(len(path) - 1):
            # Add the original point
            enhanced_path.append(path[i])
            
            # Add additional points between current and next point if needed
            if additional_smoothing_points > 0:
                x1, y1 = path[i]
                x2, y2 = path[i + 1]
                
                for j in range(1, additional_smoothing_points + 1):
                    ratio = j / (additional_smoothing_points + 1)
                    intermediate_x = x1 + ratio * (x2 - x1)
                    intermediate_y = y1 + ratio * (y2 - y1)
                    enhanced_path.append((intermediate_x, intermediate_y))
        
        # Add the final goal point
        enhanced_path.append(path[-1])

        # Initialize smoothed path to be the same as the enhanced path
        smoothed_path = enhanced_path.copy()
    

        # Number of points in the new path
        n = len(smoothed_path)
        
        # Gradient descent optimization
        change = tolerance + 1  # Ensure we enter the loop
        while change > tolerance:
            change = 0.0
            
            # Skip the first and last points (they remain fixed)
            for i in range(1, n - 1):
                # Save original coordinates
                x_old = smoothed_path[i][0]
                y_old = smoothed_path[i][1]
                
                # Update x-coordinate using gradient descent
                # The formula combines both objectives:
                # 1. Stay close to original path (data_weight)
                # 2. Make the path smoother (smooth_weight)
                x_update = (data_weight * enhanced_path[i][0] + 
                            smooth_weight * (smoothed_path[i-1][0] + smoothed_path[i+1][0])) / \
                        (data_weight + 2 * smooth_weight)
                
                 # Update y-coordinate using the same approach
                y_update = (data_weight * enhanced_path[i][1] + 
                            smooth_weight * (smoothed_path[i-1][1] + smoothed_path[i+1][1])) / \
                        (data_weight + 2 * smooth_weight)
                
                # Update the smoothed path
                smoothed_path[i] = (x_update, y_update)
                
                # Accumulate the change
                change += abs(x_old - x_update) + abs(y_old - y_update)
                
        
       
        return smoothed_path

    def plot(
        self,
        axes,
        path: list[tuple[float, float]] = (),
        smoothed_path: list[tuple[float, float]] = (),
    ):
        """Draws particles.

        Args:
            axes: Figure axes.
            path: Path (start location first).
            smoothed_path: Smoothed path (start location first).

        Returns:
            axes: Modified axes.

        """
        # Plot the nodes
        x, y = zip(*self._graph.keys())
        axes.plot(list(x), list(y), "co", markersize=1)

        # Plot the edges
        for node, neighbors in self._graph.items():
            x_start, y_start = node

            if neighbors:
                for x_end, y_end in neighbors:
                    axes.plot([x_start, x_end], [y_start, y_end], "c-", linewidth=0.25)

        # Plot the path
        if path:
            x_val = [x[0] for x in path]
            y_val = [x[1] for x in path]

            axes.plot(x_val, y_val)  # Plot the path
            axes.plot(x_val[1:-1], y_val[1:-1], "bo", markersize=4)  # Draw nodes as blue circles

        # Plot the smoothed path
        if smoothed_path:
            x_val = [x[0] for x in smoothed_path]
            y_val = [x[1] for x in smoothed_path]

            axes.plot(x_val, y_val, "y")  # Plot the path
            axes.plot(x_val[1:-1], y_val[1:-1], "yo", markersize=2)  # Draw nodes as yellow circles

        if path or smoothed_path:
            axes.plot(
                x_val[0], y_val[0], "rs", markersize=7
            )  # Draw a red square at the start location
            axes.plot(
                x_val[-1], y_val[-1], "g*", markersize=12
            )  # Draw a green star at the goal location

        return axes

    def show(
        self,
        title: str = "",
        path=(),
        smoothed_path=(),
        display: bool = False,
        block: bool = False,
        save_figure: bool = False,
        save_dir: str = "images",
    ):
        """Displays the current particle set on the map.

        Args:
            title: Plot title.
            path: Path (start location first).
            smoothed_path: Smoothed path (start location first).
            display: True to open a window to visualize the particle filter evolution in real-time.
                Time consuming. Does not work inside a container unless the screen is forwarded.
            block: True to stop program execution until the figure window is closed.
            save_figure: True to save figure to a .png file.
            save_dir: Image save directory.

        """
        figure = self._figure
        axes = self._axes
        axes.clear()

        axes = self._map.plot(axes)
        axes = self.plot(axes, path, smoothed_path)

        axes.set_title(title)
        figure.tight_layout()  # Reduce white margins

        if display:
            plt.show(block=block)
            plt.pause(0.001)  # Wait for 1 ms or the figure won't be displayed

        if display:
            plt.show(block=block)

        if save_figure:
            save_path = os.path.join(os.path.dirname(__file__), "..", save_dir)

            if not os.path.isdir(save_path):
                os.makedirs(save_path)

            file_name = f"{self._timestamp} {title.lower()}.png"
            file_path = os.path.join(save_path, file_name)
            figure.savefig(file_path)

    def _connect_nodes(
        self,
        graph: dict[tuple[float, float], list[tuple[float, float]]],
        connection_distance: float = 0.15,
    ) -> dict[tuple[float, float], list[tuple[float, float]]]:
        """Connects every generated node with all the nodes that are closer than a given threshold.

        Args:
            graph: A dictionary with (x, y) [m] tuples as keys and empty lists as values.
            connection_distance: Maximum distance to consider adding an edge between two nodes [m].

        Returns: A modified graph with lists of connected nodes as values.

        """
        # TODO: 4.2. Complete the missing function body with your code.
        
        # Get all nodes as a list
        nodes = list(graph.keys())
        
        # For each node, check potential connections with all other nodes
        for i, node1 in enumerate(nodes):
            for node2 in nodes[i+1:]:  # Only check each pair once
                # Calculate Euclidean distance between nodes
                dx = node2[0] - node1[0]
                dy = node2[1] - node1[1]
                distance = np.sqrt(dx*dx + dy*dy)
                
                # If nodes are within the connection threshold
                if distance <= connection_distance:
                    # Check if connection doesn't cross any obstacles
                    if not self._map.crosses([node1, node2]):
                        # Create bidirectional connection
                        graph[node1].append(node2)
                        graph[node2].append(node1)
        
        return graph

    def _create_graph(
        self,
        use_grid: bool = False,
        node_count: int = 50,
        grid_size=0.1,
        connection_distance: float = 0.15,
    ) -> dict[tuple[float, float], list[tuple[float, float]]]:
        """Creates a roadmap as a graph with edges connecting the closest nodes.

        Args:
            use_grid: Sample from a uniform distribution when False.
                Use a fixed step grid layout otherwise.
            node_count: Number of random nodes to generate. Only considered if use_grid is False.
            grid_size: If use_grid is True, distance between consecutive nodes in x and y.
            connection_distance: Maximum distance to consider adding an edge between two nodes [m].

        Returns: A dictionary with (x, y) [m] tuples as keys and lists of connected nodes as values.
            Key elements are rounded to a fixed number of decimal places to allow comparisons.

        """
        graph = self._generate_nodes(use_grid, node_count, grid_size)
        graph = self._connect_nodes(graph, connection_distance)

        return graph

    def _generate_nodes(
        self, use_grid: bool = False, node_count: int = 50, grid_size=0.1
    ) -> dict[tuple[float, float], list[tuple[float, float]]]:
        """Creates a set of valid nodes to build a roadmap with.

        Args:
            use_grid: Sample from a uniform distribution when False.
                Use a fixed step grid layout otherwise.
            node_count: Number of random nodes to generate. Only considered if use_grid is False.
            grid_size: If use_grid is True, distance between consecutive nodes in x and y.

        Returns: A dictionary with (x, y) [m] tuples as keys and empty lists as values.
            Key elements are rounded to a fixed number of decimal places to allow comparisons.

        """
        graph: dict[tuple[float, float], list[tuple[float, float]]] = {}

        # TODO: 4.1. Complete the missing function body with your code.
        # Get the map bounds
        
        x_min, y_min, x_max, y_max = self._map.bounds()
        
        if use_grid:
            # Generate nodes in a grid pattern
            x_values = np.arange(x_min, x_max, grid_size)
            y_values = np.arange(y_min, y_max, grid_size)
            
            for x in x_values:
                for y in y_values:
                    point = (round(x, 6), round(y, 6))  # Round to allow comparison
                    # Check if the point is in free space
                    if self._map.contains(point):
                        graph[point] = []

        else:
            # Generate random nodes
            valid_nodes = 0
            attempts = 0
            max_attempts = node_count * 10  # Avoid infinite loops
            
            while valid_nodes < node_count and attempts < max_attempts:
                # Generate a random point within map bounds
                x = random.uniform(x_min, x_max)
                y = random.uniform(y_min, y_max)
                point = (round(x, 6), round(y, 6))  # Round to allow comparison
                
                # Check if the point is in free space and not already added
                if self._map.contains(point) and point not in graph:
                    graph[point] = []
                    valid_nodes += 1
                
                attempts += 1
        
        return graph

    def _reconstruct_path(
        self,
        start: tuple[float, float],
        goal: tuple[float, float],
        ancestors: dict[tuple[int, int], tuple[int, int]],
    ) -> list[tuple[float, float]]:
        """Computes the path from the start to the goal given the ancestors of a search algorithm.

        Args:
            start: Initial location in (x, y) [m] format.
            goal: Goal location in (x, y) [m] format.
            ancestors: Dictionary with (x, y) [m] tuples as keys and the node (x_prev, y_prev) [m]
                from which it was added to the open list as values.

        Returns: Path to the goal (start location first) in (x, y) [m] format.

        """
        path: list[tuple[float, float]] = []

        # TODO: 4.4. Complete the missing function body with your code.
        # Start from the goal and work backwards
        current = goal
        
        # Build the path in reverse order (from goal to start)
        while current != start:
            path.append(current)
            current = ancestors[current]
        
        # Add the start node
        path.append(start)
        
        # Reverse the path to get it from start to goal
        path.reverse()
        
        return path


if __name__ == "__main__":
    map_name = "project"
    map_path = os.path.realpath(
        os.path.join(os.path.dirname(__file__), "..", "maps", map_name + ".json")
    )

    # Create the roadmap
    start_time = time.perf_counter()
    prm = PRM(map_path, use_grid=True, node_count=250, grid_size=0.15, connection_distance=0.15)
    roadmap_creation_time = time.perf_counter() - start_time

    print(f"Roadmap creation time: {roadmap_creation_time:1.3f} s")

    # Find the path
    start_time = time.perf_counter()
    path = prm.find_path(start=(-1.0, -1.0), goal=(-0.6, 1.0))
    pathfinding_time = time.perf_counter() - start_time

    print(f"Pathfinding time: {pathfinding_time:1.3f} s")

    # Smooth the path
    start_time = time.perf_counter()
    smoothed_path = prm.smooth_path(
        path, data_weight=0.1, smooth_weight=0.3, additional_smoothing_points=3
    )
    smoothing_time = time.perf_counter() - start_time

    print(f"Smoothing time: {smoothing_time:1.3f} s")

    prm.show(path=path, smoothed_path=smoothed_path, save_figure=True)
