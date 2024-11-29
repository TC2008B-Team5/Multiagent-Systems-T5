from flask import Flask, jsonify
from model import CityModel
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Create an instance of the CityModel
city_model = CityModel(num_cars=5, width=24, height=24)

# Define a route to get car positions
@app.route('/car_positions', methods=['GET'])
def get_car_positions():
    # Initialize a new model instance instead of resetting
    global city_model
    city_model = CityModel(num_cars=5, width=24, height=24)
    positions_over_time = []
    
    # Get initial positions when cars are in parking lots
    positions = city_model.get_car_positions()
    positions_step = {'step': 0, 'cars': positions}
    positions_over_time.append(positions_step)
    
    # Run model for 30 steps and collect positions at each step
    for step in range(1, 31):
        city_model.step()
        positions = city_model.get_car_positions()
        positions_step = {'step': step, 'cars': positions}
        positions_over_time.append(positions_step)
        
    return jsonify(positions_over_time)
