import mesa
from mesa import Agent

class CarAgent(Agent):
    """An agent representing a car moving from a parking lot to a destination parking lot."""

    def __init__(self, model):
        super().__init__(model)
        self.destination_pos = None
        self.active = True  # Indicates if the car is still moving
        self.path = []
        print(f"Car {self.unique_id} created at position {self.pos}")


    def calculate_path(self):
        # Implement a pathfinding algorithm that respects buildings and road directions
        self.path = self.find_path(self.pos, self.destination_pos)

    def find_path(self, start: tuple[int, int], goal: tuple[int, int]) -> list[tuple[int, int]]:
        #Find a valid path using A* algorithm
        open_set = {start}
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self.heuristic(start, goal)}

        while open_set:
            current = min(open_set, key=lambda pos: f_score.get(pos, float('inf')))
            print(f"Current position: {current}")
            if current == goal:
                print(f"Goal reached: {goal}")
                return self.reconstruct_path(came_from, current)

            open_set.remove(current)
            for neighbor in self.get_valid_neighbors(current):
                tentative_g_score = g_score[current] + 1  # Assumes uniform cost
                if tentative_g_score < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self.heuristic(neighbor, goal)
                    open_set.add(neighbor)
        return []

    def heuristic(self, a: tuple[int, int], b: tuple[int, int]) -> int:
        """Heuristic function for A* (Manhattan distance)."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def reconstruct_path(self, came_from: dict, current: tuple[int, int]) -> list[tuple[int, int]]:
        """Reconstruct the path from start to goal."""
        total_path = [current]
        while current in came_from:
            current = came_from[current]
            total_path.append(current)
        total_path.reverse()
        return total_path

    def get_valid_neighbors(self, pos: tuple[int, int]) -> list[tuple[int, int]]:
        """Get neighbors that are valid for movement."""
        neighbors = self.model.grid.get_neighborhood(pos, moore=False, include_center=False)
        valid_neighbors = []
        for neighbor in neighbors:
            if self.is_valid_move(pos, neighbor):
                valid_neighbors.append(neighbor)
        return valid_neighbors

    def is_valid_move(self, from_pos, to_pos):
        """Check whether moving from from_pos to to_pos is a valid move."""
        if self.model.grid.out_of_bounds(to_pos):
            print(f"Move from {from_pos} to {to_pos} is out of bounds.")
            return False

        # Get the list of agents at to_pos
        cell_contents = self.model.grid.get_cell_list_contents([to_pos])

        if any(isinstance(agent, CarAgent) for agent in cell_contents):
            print(f"Move from {from_pos} to {to_pos} is blocked by another car.")
            return False  # Cell is occupied by another car

        # Check for buildings
        if self.model.is_building(to_pos):
            print(f"Move from {from_pos} to {to_pos} is blocked by a building.")
            return False  # Can't move into a building

        # Check for traffic lights
        for agent in cell_contents:
            if isinstance(agent, TrafficLightAgent):
                if agent.state == 'Red':
                    print(f"Move from {from_pos} to {to_pos} is blocked by a red traffic light.")
                    return False  # Can't move on red light
                else:
                    print(f"Move from {from_pos} to {to_pos} is allowed by traffic light.")

        # Determine the actual movement direction
        dx = to_pos[0] - from_pos[0]
        dy = to_pos[1] - from_pos[1]
        movement_direction = None
        if dx == 1:
            movement_direction = 'E'
        elif dx == -1:
            movement_direction = 'W'
        elif dy == 1:
            movement_direction = 'N'
        elif dy == -1:
            movement_direction = 'S'
        else:
            print(f"Invalid movement from {from_pos} to {to_pos}.")
            return False  # Movement is not to an adjacent cell

        # Check if the movement direction is allowed
        if movement_direction in self.model.road_direction_layer[from_pos[0], from_pos[1]]:
            print(f"Move from {from_pos} to {to_pos} in direction '{movement_direction}' is valid.")
            return True
        else:
            print(f"Move from {from_pos} to {to_pos} in direction '{movement_direction}' is invalid.")
        return False

    def step(self):
        print(f"Car {self.unique_id} at position {self.pos} taking a step.")
        if not self.active:
            print(f"Car {self.unique_id} is inactive.")
            return

        if not self.path:
            # Try to recalculate the path or move randomly
            print(f"Car {self.unique_id} has no path. Recalculating...")
            self.calculate_path()
            if not self.path:
                print(f"Car {self.unique_id} still cannot find a path. Moving randomly.")
                # Move randomly to a valid neighboring cell
                valid_neighbors = self.get_valid_neighbors(self.pos)
                if valid_neighbors:
                    next_pos = self.random.choice(valid_neighbors)
                    self.model.grid.move_agent(self, next_pos)
                    print(f"Car {self.unique_id} moved randomly to {next_pos}")
                else:
                    print(f"Car {self.unique_id} cannot move from position {self.pos}.")
                return

        # Move to the next position in the path
        if self.path:
            next_pos = self.path[0]

            # Check if the next move is into the destination parking lot
            if next_pos == self.destination_pos:
                if self.model.is_parking_lot_available(next_pos):
                    # Move into the parking lot
                    self.model.grid.move_agent(self, next_pos)
                    self.path.pop(0)
                    self.active = False
                    print(f"Car {self.unique_id} has arrived at Parking Lot at position {self.destination_pos}")
                    self.remove()
                else:
                    # Parking lot is occupied; wait or choose a new destination
                    print(f"Parking Lot at {next_pos} is occupied. Assigning new destination.")
                    # self.model.assign_random_destination(self, exclude_pos=self.pos)
                    # self.calculate_path()
            else:
                # Move to the next position if it's a valid move
                if self.is_valid_move(self.pos, next_pos):
                    self.model.grid.move_agent(self, next_pos)
                    self.path.pop(0)
                    print(f"Car {self.unique_id} moved to {next_pos}")
                else:
                    print(f"Car {self.unique_id} cannot move to {next_pos}. Recalculating path.")
                    self.calculate_path()
        else:
            print(f"Car {self.unique_id} has no path to follow.")

    def remove(self):
        """Remove the car agent from the model and grid."""
        if self.pos is not None:
            self.model.grid.remove_agent(self)
        self.model.car_agents.remove(self)
        super().remove()

class TrafficLightAgent(Agent):
    """An agent representing a traffic light."""

    def __init__(self, model):
        super().__init__(model)
        self.state = 'Green'  # Initial state
        self.timer = 0
        self.durations = {'Green': 5, 'Yellow': 2, 'Red': 5}

    def step(self):
        """Advance the traffic light by one step."""
        self.timer += 1
        if self.timer >= self.durations[self.state]:
            self.change_state()
            self.timer = 0

    def change_state(self):
        """Cycle through traffic light states."""
        if self.state == 'Green':
            self.state = 'Yellow'
        elif self.state == 'Yellow':
            self.state = 'Red'
        elif self.state == 'Red':
            self.state = 'Green'