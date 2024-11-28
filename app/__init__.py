from flask import Flask, jsonify
from model import CityModel

app = Flask(__name__)

# Create an instance of the CityModel
city_model = CityModel(num_cars=10, width=24, height=24)

# Define a route to get car positions
@app.route('/car_positions', methods=['GET'])
def get_car_positions():
    # Advance the model by one step
    city_model.step()
    # Get car positions from the model
    positions = city_model.get_car_positions()
    return jsonify(positions)
