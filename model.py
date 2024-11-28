import mesa
from mesa import Model
from mesa.agent import AgentSet
from agents.agents import CarAgent, TrafficLightAgent
from mesa.space import MultiGrid
import numpy as np

class CityModel(Model):
    """A model representing a city grid with cars, buildings, parking lots, and traffic lights."""

    def __init__(self, num_cars, width, height, seed=None):
        super().__init__(seed=seed)

        self.num_cars = num_cars
        self.grid = MultiGrid(width, height, torus=False)
        self.steps = 0

        # Initialize property layers
        self.buildings_layer = np.full((width, height), False, dtype=bool)
        self.parking_lot_layer = np.full((width, height), False, dtype=bool)
        self.parking_lot_ids = {}
        self.road_direction_layer = np.full((width, height), None, dtype=object)


        # Set up buildings, parking lots, and roads
        self.setup_environment()

        # Agent sets
        self.car_agents = AgentSet([], random=self.random)
        self.traffic_light_agents = AgentSet([], random=self.random)

        # Create traffic lights (if any)
        self.create_traffic_lights()

        # Create cars
        self.create_cars()

    def setup_environment(self):
        """Set up buildings, parking lots, and road directions."""
        # Place buildings
        self.setup_buildings()
        # Place parking lots
        self.setup_parking_lots()
        # Set road directions
        self.setup_road_directions()

    def setup_buildings(self):
        """Place buildings on the grid."""
        # Example: Buildings occupy multiple cells
        # Define building positions
        buildings_positions = [
            # Building 1
            [
                (2, 2), (2, 3), (2, 4), (2, 5), (2, 6), (2, 7), (2, 8), (2, 10), (2, 11),
                (3, 3), (3, 4), (3, 5), (3, 6), (3, 7), (3, 8), (3, 9), (3, 10), (3, 11),
                (4, 2), (4, 3), (4, 4), (4, 5), (4, 6), (4, 7), (4, 8), (4, 9), (4, 10),
                (5, 2), (5, 3), (5, 4), (5, 5), (5, 7), (5, 8), (5, 9), (5, 10), (5, 11)
            ],
            # Building 2
            [
                (8, 2), (8, 3), (8, 4),
                (9, 2), (9, 3), (9, 4),
                (10, 2), (10, 3),
                (11, 2), (11, 3), (11, 4)
            ],
            # Building 3
            [
                (8, 7), (8, 9), (8, 10), (8, 11),
                (9, 7), (9, 8), (9, 9), (9, 10), (9, 11),
                (10, 7), (10, 8), (10, 9), (10, 10),
                (11, 7), (11, 8), (11, 9), (11, 10), (11, 11)
            ],
            # Building 4
            [
                (16, 2), (16, 3), (16, 4), (16, 5),
                (17, 3), (17, 4), (17, 5),
                (18, 2), (18, 3), (18, 4), (18, 5),
                (19, 2), (19, 3), (19, 4), (19, 5),
                (20, 2), (20, 3), (20, 4),
                (21, 2), (21, 3), (21, 4), (21, 5)
            ],
            # Building 5
            [
                (16, 8), (17, 8), (18, 8), (19, 8), (21, 8),
                (16, 9), (16, 10), (16, 11),
                (17, 9), (17, 10), (17, 11),
                (18, 9), (18, 10), (18, 11),
                (19, 9), (19, 10), (19, 11),
                (20, 9), (20, 10), (20, 11),
                (21, 9), (21, 10), (21, 11)
            ],
            # Building 6
            [
                (16, 16), (16, 17), (16, 18), (16, 19), (16, 20), (16, 21),
                (17, 16), (17, 18), (17, 20), (17, 21)
            ],
            # Building 7
            [
                (20, 16), (20, 17), (20, 18), (20, 20), (20, 21),
                (21, 16), (21, 17), (21, 18), (21, 19), (21, 20), (21, 21)
            ],
            # Building 8
            [
                (8, 16), (8, 17),
                (9, 16), (9, 17),
                (10, 17),
                (11, 16), (11, 17)
            ],
            # Building 9
            [
                (8, 20), (8, 21),
                (9, 20),
                (10, 20), (10, 21),
                (11, 20), (11, 21)
            ],
            # Building 10
            [
                (2, 16), (2, 17),
                (3, 16),
                (4, 16), (4, 17),
                (5, 16), (5, 17)
            ],
            # Building 11
            [
                (2, 20), (2, 21),
                (3, 20), (3, 21),
                (4, 21),
                (5, 20), (5, 21)
            ]
        ]
        for building in buildings_positions:
            for pos in building:
                x, y = pos
                self.buildings_layer[x, y] = True  # Mark cell as building

    def setup_parking_lots(self):
        """Place parking lots on the grid."""
        parking_lot_positions = [
            (2, 9), (3, 2), (3, 17), (4, 11), (4, 20), (5, 6), (8, 8),
            (9, 21), (10, 4), (10, 11), (10, 16), (17, 2), (17, 17), (17, 19),
            (20, 5), (20, 8), (20, 19)
        ]
        for idx, pos in enumerate(parking_lot_positions, start=1):
            x, y = pos
            self.parking_lot_layer[x, y] = True  # Mark cell as parking lot
            self.parking_lot_ids[pos] = idx

    def setup_road_directions(self):
        """Set up specific road directions on the grid."""
        # Define road sections with their cell positions and directions
        custom_road_directions = {
                # Only down road and east at the same time
                (0, 0): ['N', 'E'], (0, 1): ['N', 'E'], (0, 2): ['N', 'E'], (0, 3): ['N', 'E'], (0, 4): ['N', 'E'], 
                (0, 5): ['N', 'E'], (0, 6): ['N', 'E'], (0, 7): ['N', 'E'], (0, 8): ['N', 'E'], (0, 9): ['N', 'E'],
                (0, 10): ['N', 'E'], (0, 11): ['N', 'E'], (0, 12): ['N', 'E'], (0, 13): ['N', 'E'], (0, 14): ['N', 'E'],
                (0, 15): ['N', 'E'], (0, 16): ['N', 'E'], (0, 17): ['N', 'E'], (0, 18): ['N', 'E'], (0, 19): ['N', 'E'],
                (0, 20): ['N', 'E'], (0, 21): ['N', 'E'], (0, 22): ['N', 'E'], (0, 23): ['N', 'E'],
                (12, 2): ['N', 'E'], (12, 3): ['N', 'E'], (12, 4): ['N', 'E'], (12, 7): ['N', 'E'], (12, 8): ['N', 'E'],
                (12, 9): ['N', 'E'], (12, 10): ['N', 'E'], (12, 11): ['N', 'E'], (12, 14): ['N', 'E'], (12, 15): ['N', 'E'],
                (12, 16): ['N', 'E'], (12, 17): ['N', 'E'], (12, 18): ['N', 'E'], (12, 19): ['N', 'E'], (12, 20): ['N', 'E'],
                (12, 21): ['N', 'E'], (6, 16): ['N', 'E'], (6, 17): ['N', 'E'], (6, 20): ['N', 'E'], (6, 21): ['N', 'E'],
                (1, 14): ['N', 'E'], (1, 15): ['N', 'E'], (2, 14): ['N', 'E'], (3, 14): ['N', 'E'], (4, 14): ['N', 'E'],
                (5, 14): ['N', 'E'], (6, 14): ['N', 'E'], (7, 14): ['N', 'E'], (8, 14): ['N', 'E'], (9, 14): ['N', 'E'],
                (10, 14): ['N', 'E'], (11, 14): ['N', 'E'], (12, 14): ['N', 'E'], (6, 15): ['N', 'E'], (7, 15): ['N', 'E'],
                (1, 22): ['N', 'E'], (2, 22): ['N', 'E'], (3, 22): ['N', 'E'], (4, 22): ['N', 'E'], (5, 22): ['N', 'E'],
                (6, 22): ['N', 'E'], (7, 22): ['N', 'E'], (8, 22): ['N', 'E'], (9, 22): ['N', 'E'], (10, 22): ['N', 'E'],
                (11, 22): ['N', 'E'], (12, 22): ['N', 'E'], (13, 22): ['N', 'E'], (16, 22): ['N', 'E'], (17, 22): ['N', 'E'],
                (20, 22): ['N', 'E'], (21, 22): ['N', 'E'], (8, 18): ['N', 'E'], (9, 18): ['N', 'E'], (10, 18): ['N', 'E'],
                (11, 18): ['N', 'E'], (16, 14): ['N', 'E'], (17, 14): ['N', 'E'], (18, 14): ['N', 'E'], (19, 14): ['N', 'E'],
                (20, 14): ['N', 'E'], (21, 14): ['N', 'E'], (12, 14): ['N', 'E'], (13, 14): ['N', 'E'], (12, 15): ['N', 'E'],
                (13, 15): ['N', 'E'],
                # Only down road and west at the same time  
                (1, 0): ['N', 'W'], (1, 1): ['N', 'W'], (1, 2): ['N', 'W'], (1, 3): ['N', 'W'], (1, 4): ['N', 'W'],
                (1, 5): ['N', 'W'], (1, 6): ['N', 'W'], (1, 7): ['N', 'W'], (1, 8): ['N', 'W'], (1, 9): ['N', 'W'],
                (1, 10): ['N', 'W'], (1, 11): ['N', 'W'], (1, 12): ['N', 'W'], (1, 13): ['N', 'W'], (1, 16): ['N', 'W'],
                (7, 16): ['N', 'W'], (7, 17): ['N', 'W'], (7, 20): ['N', 'W'], (7, 21): ['N', 'W'], (13, 2): ['N', 'W'],
                (13, 3): ['N', 'W'], (13, 4): ['N', 'W'], (13, 5): ['N', 'W'], (13, 6): ['N', 'W'], (13, 7): ['N', 'W'],
                (13, 8): ['N', 'W'], (13, 9): ['N', 'W'], (13, 10): ['N', 'W'], (13, 11): ['N', 'W'], (13, 12): ['N', 'W'],
                (13, 13): ['N', 'W'], (13, 16): ['N', 'W'], (13, 17): ['N', 'W'],
                (13, 18): ['N', 'W'], (13, 19): ['N', 'W'], (13, 20): ['N', 'W'], (13, 21): ['N', 'W'],
                (2, 0): ['N', 'W'], (3, 0): ['N', 'W'], (4, 0): ['N', 'W'], (5, 0): ['N', 'W'], (6, 0): ['N', 'W'],
                (7, 0): ['N', 'W'], (8, 0): ['N', 'W'], (9, 0): ['N', 'W'], (10, 0): ['N', 'W'], (11, 0): ['N', 'W'],
                (12, 0): ['N', 'W'], (13, 0): ['N', 'W'], (14, 0): ['N', 'W'], (15, 0): ['N', 'W'], (16, 0): ['N', 'W'],
                (17, 0): ['N', 'W'], (18, 0): ['N', 'W'], (19, 0): ['N', 'W'], (20, 0): ['N', 'W'], (21, 0): ['N', 'W'],
                (12, 1): ['N', 'W'], (13, 1): ['N', 'W'], (12, 12): ['N', 'W'], (12, 13): ['N', 'W'], (1, 18): ['N', 'W'], (2, 18): ['N', 'W'],
                (3, 18): ['N', 'W'], (4, 18): ['N', 'W'], (5, 18): ['N', 'W'], (1, 19): ['N', 'W'], (1, 20): ['N', 'W'], (1, 21): ['N', 'W'],
                (8, 5): ['N', 'W'], (9, 5): ['N', 'W'], (10, 5): ['N', 'W'], (11, 5): ['N', 'W'], (16, 6): ['N', 'W'], (17, 6): ['N', 'W'],
                (18, 6): ['N', 'W'], (19, 6): ['N', 'W'], (20, 6): ['N', 'W'], (21, 6): ['N', 'W'], (16, 12): ['N', 'W'], (17, 12): ['N', 'W'],
                (18, 12): ['N', 'W'], (19, 12): ['N', 'W'], (20, 12): ['N', 'W'], (21, 12): ['N', 'W'], (2, 12): ['N', 'W'], (3, 12): ['N', 'W'],
                (4, 12): ['N', 'W'], (5, 12): ['N', 'W'], (8, 12): ['N', 'W'], (9, 12): ['N', 'W'], (10, 12): ['N', 'W'], (11, 12): ['N', 'W'],
                (22, 0): ['N', 'W'], (23, 0): ['N', 'W'],
                # Only up road and west at the same time
                (2, 1): ['S', 'W'], (3, 1): ['S', 'W'], (4, 1): ['S', 'W'], (5, 1): ['S', 'W'], (6, 1): ['S', 'W'],
                (7, 1): ['S', 'W'], (8, 1): ['S', 'W'], (9, 1): ['S', 'W'], (10, 1): ['S', 'W'], (11, 1): ['S', 'W'],
                (14, 1): ['S', 'W'], (15, 1): ['S', 'W'], (16, 1): ['S', 'W'], (17, 1): ['S', 'W'], (18, 1): ['S', 'W'],
                (19, 1): ['S', 'W'], (20, 1): ['S', 'W'], (21, 1): ['S', 'W'], (22, 1): ['S', 'W'],
                (23, 1): ['S', 'W'], (23, 2): ['S', 'W'], (23, 3): ['S', 'W'], (23, 4): ['S', 'W'], (23, 5): ['S', 'W'],
                (23, 6): ['S', 'W'], (23, 7): ['S', 'W'], (23, 8): ['S', 'W'], (23, 9): ['S', 'W'], (23, 10): ['S', 'W'],
                (23, 11): ['S', 'W'], (23, 12): ['S', 'W'], (23, 13): ['S', 'W'], (23, 14): ['S', 'W'], (23, 15): ['S', 'W'],
                (23, 16): ['S', 'W'], (23, 17): ['S', 'W'], (23, 18): ['S', 'W'], (23, 19): ['S', 'W'], (23, 20): ['S', 'W'],
                (23, 21): ['S', 'W'], (7, 2): ['S', 'W'], (7, 3): ['S', 'W'], (7, 4): ['S', 'W'], (7, 5): ['S', 'W'], (7, 6): ['S', 'W'],
                (7, 7): ['S', 'W'], (7, 8): ['S', 'W'], (7, 9): ['S', 'W'], (7, 10): ['S', 'W'], (7, 11): ['S', 'W'], (1, 13): ['S', 'W'],
                (2, 13): ['S', 'W'], (3, 13): ['S', 'W'], (4, 13): ['S', 'W'], (5, 13): ['S', 'W'], (6, 13): ['S', 'W'], (7, 13): ['S', 'W'],
                (8, 13): ['S', 'W'], (9, 13): ['S', 'W'], (10, 13): ['S', 'W'], (11, 13): ['S', 'W'], (3, 19): ['S', 'W'], (4, 19): ['S', 'W'],
                (5, 19): ['S', 'W'], (15, 2): ['S', 'W'], (15, 3): ['S', 'W'], (15, 4): ['S', 'W'], (15, 5): ['S', 'W'], (15, 6): ['S', 'W'],
                (15, 7): ['S', 'W'], (15, 8): ['S', 'W'], (15, 9): ['S', 'W'], (15, 10): ['S', 'W'], (15, 11): ['S', 'W'],
                (15, 12): ['S', 'W'], (15, 13): ['S', 'W'], (16, 13): ['S', 'W'], (17, 13): ['S', 'W'], (18, 13): ['S', 'W'],
                (19, 13): ['S', 'W'], (20, 13): ['S', 'W'], (21, 13): ['S', 'W'], (15, 16): ['S', 'W'], (15, 17): ['S', 'W'],
                (15, 18): ['S', 'W'], (15, 19): ['S', 'W'], (15, 20): ['S', 'W'], (15, 21): ['S', 'W'], (19, 16): ['S', 'W'],
                (19, 17): ['S', 'W'], (19, 18): ['S', 'W'], (19, 19): ['S', 'W'], (19, 20): ['S', 'W'], (19, 21): ['S', 'W'],
                (16, 7): ['S', 'W'], (17, 7): ['S', 'W'], (18, 7): ['S', 'W'], (19, 7): ['S', 'W'], (20, 7): ['S', 'W'], (21, 7): ['S', 'W'],
                (8, 6): ['S', 'W'], (9, 6): ['S', 'W'], (10, 6): ['S', 'W'], (11, 6): ['S', 'W'], (14, 12): ['S', 'W'], (14, 13): ['S', 'W'], (23, 23): ['S', 'W'],
                (23, 22): ['S', 'W'],
                # Only up road and east at the same time
                (1, 23): ['S', 'E'], (2, 23): ['S', 'E'], (3, 23): ['S', 'E'], (4, 23): ['S', 'E'], (5, 23): ['S', 'E'],
                (6, 23): ['S', 'E'], (7, 23): ['S', 'E'], (8, 23): ['S', 'E'], (9, 23): ['S', 'E'], (10, 23): ['S', 'E'],
                (11, 23): ['S', 'E'], (12, 23): ['S', 'E'], (13, 23): ['S', 'E'], (14, 23): ['S', 'E'], (15, 23): ['S', 'E'],
                (16, 23): ['S', 'E'], (17, 23): ['S', 'E'], (18, 23): ['S', 'E'], (19, 23): ['S', 'E'], (20, 23): ['S', 'E'],
                (21, 23): ['S', 'E'], (22, 23): ['S', 'E'], (22, 2): ['S', 'E'], (22, 3): ['S', 'E'], (22, 4): ['S', 'E'],
                (22, 5): ['S', 'E'], (22, 8): ['S', 'E'], (22, 9): ['S', 'E'], (22, 10): ['S', 'E'], (22, 11): ['S', 'E'],
                (22, 14): ['S', 'E'], (22, 15): ['S', 'E'], (22, 16): ['S', 'E'], (22, 17): ['S', 'E'], (22, 18): ['S', 'E'],
                (22, 19): ['S', 'E'], (22, 20): ['S', 'E'], (22, 21): ['S', 'E'], (22, 22): ['S', 'E'],
                (6, 2): ['S', 'E'], (6, 3): ['S', 'E'], (6, 4): ['S', 'E'], (6, 5): ['S', 'E'], (6, 6): ['S', 'E'], (6, 7): ['S', 'E'], (6, 8): ['S', 'E'],
                (6, 9): ['S', 'E'], (6, 10): ['S', 'E'], (6, 11): ['S', 'E'], (14, 2): ['S', 'E'], (14, 3): ['S', 'E'],
                (14, 4): ['S', 'E'], (14, 5): ['S', 'E'], (14, 6): ['S', 'E'], (14, 7): ['S', 'E'], (14, 8): ['S', 'E'], (14, 9): ['S', 'E'], (14, 10): ['S', 'E'],
                (14, 11): ['S', 'E'], (1, 15): ['S', 'E'], (2, 15): ['S', 'E'], (3, 15): ['S', 'E'], (4, 15): ['S', 'E'],
                (5, 15): ['S', 'E'], (8, 15): ['S', 'E'], (9, 15): ['S', 'E'], (10, 15): ['S', 'E'], (11, 15): ['S', 'E'], (8, 19): ['S', 'E'],
                (9, 19): ['S', 'E'], (10, 19): ['S', 'E'], (11, 19): ['S', 'E'], (14, 2): ['S', 'E'], (14, 3): ['S', 'E'], (14, 4): ['S', 'E'],
                (14, 5): ['S', 'E'], (14, 8): ['S', 'E'], (14, 9): ['S', 'E'], (14, 10): ['S', 'E'], (14, 11): ['S', 'E'], (14, 16): ['S', 'E'],
                (14, 17): ['S', 'E'], (14, 18): ['S', 'E'], (14, 19): ['S', 'E'], (14, 20): ['S', 'E'], (14, 21): ['S', 'E'], (16, 15): ['S', 'E'],
                (17, 15): ['S', 'E'], (18, 15): ['S', 'E'], (19, 15): ['S', 'E'], (20, 15): ['S', 'E'], (21, 15): ['S', 'E'],
                (18, 16): ['S', 'E'], (18, 17): ['S', 'E'], (18, 18): ['S', 'E'], (18, 19): ['S', 'E'], (18, 20): ['S', 'E'],
                (18, 21): ['S', 'E'], (14, 14): ['S', 'E'], (14, 15): ['S', 'E'], (15, 14): ['S', 'E'], (15, 15): ['S', 'E'],
                # Down, up and west at the same time
                (6, 12): ['N', 'S', 'W'], (7, 12): ['N', 'S', 'W'],
                # Down, east and west at the same time
                (6, 18): ['N', 'E', 'W'], (6, 19): ['N', 'E', 'W'], (7, 18): ['N', 'E', 'W'], (7, 19): ['N', 'E', 'W'], (12, 5): ['N', 'E', 'W'], (12, 6): ['N', 'E', 'W'],
                # Up, down and east at the same time
                (13, 22): ['N', 'S', 'E'], (14, 22): ['N', 'S', 'E'], (18, 22): ['N', 'S', 'E'], (19, 22): ['N', 'S', 'E'],
                # Up, east and west at the same time
                (22, 12): ['S', 'E', 'W'], (22, 13): ['S', 'E', 'W'], (22, 6): ['S', 'E', 'W'], (22, 7): ['S', 'E', 'W']
            }

        # Assign custom directions to the road_direction_layer
        for pos, directions in custom_road_directions.items():
            x, y = pos
            if not self.buildings_layer[x, y] and not self.parking_lot_layer[x, y]:
                # Initialize the cell directions if not already set
                if self.road_direction_layer[x, y] is None:
                    self.road_direction_layer[x, y] = []
                # Combine new directions with existing ones, avoiding duplicates
                self.road_direction_layer[x, y].extend(directions)
                self.road_direction_layer[x, y] = list(set(self.road_direction_layer[x, y]))

        # Optionally, assign default directions to any remaining road cells not specified
        width, height = self.grid.width, self.grid.height
        for x in range(width):
            for y in range(height):
                if not self.buildings_layer[x, y] and self.road_direction_layer[x, y] is None:
                    self.road_direction_layer[x, y] = ['N', 'S', 'E', 'W']

    def assign_random_destination(self, car, exclude_pos=None):
        """Assign a random unoccupied parking lot as the destination to the car."""
        possible_destinations = [
            pos for pos in self.parking_lot_ids.keys()
            if pos != exclude_pos and self.is_parking_lot_available(pos)
        ]

        if not possible_destinations:
            print("No unoccupied parking lots available for destination.")
            car.active = False  # Deactivate the car or implement alternative behavior
            return

        destination_pos = self.random.choice(possible_destinations)
        car.destination_pos = destination_pos

        dest_lot_id = self.parking_lot_ids[destination_pos]

        if exclude_pos in self.parking_lot_ids:
            start_lot_id = self.parking_lot_ids[exclude_pos]
            print(
                f"Car {car.unique_id} is leaving Parking Lot {start_lot_id} at position {exclude_pos}"
            )
        else:
            print(f"Car {car.unique_id} is starting from road position {car.pos}")

        print(
            f"Car {car.unique_id} has destination Parking Lot {dest_lot_id} at position {destination_pos}"
        )

    def create_cars(self):
        """Create car agents starting at parking lots or random road positions."""
        # Get positions of parking lots
        available_parking_lots = list(self.parking_lot_ids.keys())

        num_cars_to_create = self.num_cars

        for _ in range(num_cars_to_create):
            car = CarAgent(self)

            if available_parking_lots:
                # Assign a parking lot as the starting position
                start_pos = self.random.choice(available_parking_lots)
                # Remove the start position to prevent multiple cars at the same spot
                available_parking_lots.remove(start_pos)
            else:
                # No parking lots available, choose a random road position
                road_cells = []
                width, height = self.grid.width, self.grid.height
                for x in range(width):
                    for y in range(height):
                        if (
                            not self.buildings_layer[x, y]
                            and not self.parking_lot_layer[x, y]
                            and len(self.grid.get_cell_list_contents((x, y))) == 0
                        ):
                            road_cells.append((x, y))
                start_pos = self.random.choice(road_cells)

            # Place the car agent at the starting position
            self.grid.place_agent(car, start_pos)
            self.car_agents.add(car)
            print(f"Car {car.unique_id} placed at position {start_pos}")

            # Assign a random destination parking lot
            self.assign_random_destination(car, exclude_pos=start_pos)

            # Calculate the path
            car.calculate_path()

    def create_traffic_lights(self):
        """Create traffic light agents at specified positions."""
        traffic_light_positions = [
            (2, 18), (2, 19), (0, 17), (1, 17), (5, 22), (5, 23),
            (6, 2), (7, 2), (6, 7), (7, 7), (6, 21), (7, 21),
            (8, 0), (8, 1), (8, 5), (8, 6), (17, 14), (17, 15),
            (18, 16), (19, 16),
            # Add more traffic lights as needed
        ]
        for pos in traffic_light_positions:
            traffic_light = TrafficLightAgent(self, pos)
            self.traffic_light_agents.add(traffic_light)


    def is_parking_lot(self, pos: tuple[int, int]) -> bool:
        """Check if a position is occupied by a parking lot."""
        x, y = pos
        return self.parking_lot_layer[x, y]
    
    def is_parking_lot_available(self, pos: tuple[int, int]) -> bool:
        """Check if a parking lot at a position is available (not occupied by a car)."""
        if not self.is_parking_lot(pos):
            return False  # Not a parking lot
        # Check if any car agents are in the cell
        cell_contents = self.grid.get_cell_list_contents([pos])
        for agent in cell_contents:
            if isinstance(agent, CarAgent):
                return False  # Parking lot is occupied by a car
        return True  # Parking lot is unoccupied

    def is_building(self, pos: tuple[int, int]) -> bool:
        """Check if a position is occupied by a building."""
        x, y = pos
        return self.buildings_layer[x, y]

    def is_valid_road_direction(self, from_pos, to_pos):
        """Check if movement from from_pos to to_pos follows the road direction."""
        from_x, from_y = from_pos
        to_x, to_y = to_pos
        direction = self.road_direction_layer[from_x, from_y]
        if direction == 'N' and to_y == from_y + 1:
            return True
        if direction == 'S' and to_y == from_y - 1:
            return True
        if direction == 'E' and to_x == from_x + 1:
            return True
        if direction == 'W' and to_x == from_x - 1:
            return True
        return False

    def step(self):
        """Advance the model by one step."""
        print(f"Model stepping. Current step: {self.steps}")
        # Activate traffic lights
        self.traffic_light_agents.do("step")
        # Activate cars
        self.car_agents.do("step")

    def get_car_positions(self):
        """Return a list of positions of all active car agents."""
        car_positions = []
        for car in self.car_agents:
            if car.active:
                car_positions.append({'id': car.unique_id, 'position': car.pos})
        return car_positions