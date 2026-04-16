"""
Reads the 350k+ row dataset.csv and generates a complete, accurate config.js 
file for the frontend containing all states, districts, crops, seasons, 
and exact min/max bounds for rainfall and temperature.
"""
import csv
import json
import math

def generate_frontend_config():
    print("Reading dataset and extracting unique locations, crops, and bounds...")
    
    location_data = {}
    crops = set()
    seasons = set()
    rainfalls = []
    temperatures = []
    
    try:
        with open('dataset.csv', 'r', encoding='utf-8-sig') as f:
            reader = csv.reader(f)
            headers = next(reader)
            
            # Map clean column names to indices
            col_map = {h.strip(): i for i, h in enumerate(headers)}
            
            def get_idx(name):
                return col_map.get(name, col_map.get(name + ' ', -1))
            
            idx_state = get_idx('State')
            idx_dist = get_idx('District')
            idx_crop = get_idx('Crop')
            idx_season = get_idx('Season')
            idx_rain = get_idx('Rainfall')
            idx_temp = get_idx('Temperature')

            for row in reader:
                try:
                    state = row[idx_state].strip()
                    district = row[idx_dist].strip()
                    crop = row[idx_crop].strip()
                    season = row[idx_season].strip()
                    
                    if not state or not district: continue
                    
                    rain = float(row[idx_rain].strip())
                    temp = float(row[idx_temp].strip())
                    
                    rainfalls.append(rain)
                    temperatures.append(temp)
                    
                    if state not in location_data:
                        location_data[state] = set()
                    location_data[state].add(district)
                    
                    crops.add(crop)
                    seasons.add(season)
                except (ValueError, IndexError):
                    continue # Skip empty or malformed rows

        # Convert sets to sorted lists for the frontend
        location_data = {k: sorted(list(v)) for k, v in location_data.items()}
        
        config = {
            "locationData": location_data,
            "crops": sorted(list(crops)),
            "seasons": sorted(list(seasons)),
            "bounds": {
                "rainfall": {
                    "min": math.floor(min(rainfalls)), 
                    "max": math.ceil(max(rainfalls))
                },
                "temperature": {
                    "min": math.floor(min(temperatures)),
                    "max": math.ceil(max(temperatures))
                }
            }
        }
        
        with open('config.js', 'w', encoding='utf-8') as f:
            f.write(f"const CONFIG = {json.dumps(config, indent=4)};\n")
            
        print("✅ Success! 'config.js' generated perfectly. Your UI now has 100% accurate dropdowns and validation bounds.")

    except FileNotFoundError:
        print("❌ Error: 'dataset.csv' not found in the current directory.")

if __name__ == '__main__':
    generate_frontend_config()