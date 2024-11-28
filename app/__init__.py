from flask import Flask, jsonify
from model import CityModel

app = Flask(__name__)

# Create an instance of the CityModel
city_model = CityModel(num_cars=5, width=24, height=24)

# Define a route to get car positions
@app.route('/car_positions', methods=['GET'])
def get_car_positions():
    positions_over_time = []
    for step in range(30):
        city_model.step()
        positions = city_model.get_car_positions()
        positions_step = {'step': step, 'cars': positions}
        positions_over_time.append(positions_step)
    return jsonify(positions_over_time)
