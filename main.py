from flask import *
from flask_cors import CORS
import json
from Automated_Process import final_temporal_map_api, calculate_processing_time, api_to_latlong

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route("/",  methods=['GET'])

def home():
       
    data_set_map = {"map":"status ok"}
    json_dump_map = json.dumps(data_set_map)
    return json_dump_map


@app.route("/pollination/",  methods=['GET'])

def save_map():

    try:
        min_lat = float(str(request.args.get('minLat')))
        max_lat = float(str(request.args.get('maxLat')))
        min_lon = float(str(request.args.get('minLon')))
        max_lon = float(str(request.args.get('maxLon')))
        html_content = final_temporal_map_api(min_lat,max_lat,min_lon,max_lon)

        data_set_map = {"map":html_content}
        json_dump_map = json.dumps(data_set_map)
        return json_dump_map
    
    except:
        data_set_map = {"map":"0"}
        json_dump_map = json.dumps(data_set_map)
        return json_dump_map

        
@app.route("/timing/",  methods=['GET'])

def processing_time():    
    try:

        min_lat = float(str(request.args.get('minLat')))
        max_lat = float(str(request.args.get('maxLat')))
        min_lon = float(str(request.args.get('minLon')))
        max_lon = float(str(request.args.get('maxLon')))
        lat, long, cols, raws = api_to_latlong(min_lat,max_lat,min_lon,max_lon)

        process_time = calculate_processing_time(cols,raws)
        message = f"It will takes {process_time}  minutes to Complete!"
        data_set = {"time":process_time, "message":message}
        json_dump = json.dumps(data_set)
        return json_dump
     
    except:
        data_set = {"message":"0"}
        json_dump = json.dumps(data_set)
        return json_dump
    

if __name__ == "__main__":
    app.run(debug=True)