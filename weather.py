import os
import time
import requests
import datetime
from typing import Dict, List, Optional

# Constants
CACHE_DURATION = 1800  # 30 minutes in seconds
WEATHER_CACHE = {}

# Map of 15 Indian states to their capital city coordinates (lat, lon)
STATE_COORDINATES = {
    "Maharashtra": (19.0760, 72.8777),       # Mumbai
    "Tamil Nadu": (13.0827, 80.2707),        # Chennai
    "Karnataka": (12.9716, 77.5946),         # Bengaluru
    "Gujarat": (23.2156, 72.6369),           # Gandhinagar
    "Rajasthan": (26.9124, 75.7873),         # Jaipur
    "Uttar Pradesh": (26.8467, 80.9462),     # Lucknow
    "Bihar": (25.5941, 85.1376),             # Patna
    "West Bengal": (22.5726, 88.3639),       # Kolkata
    "Madhya Pradesh": (23.2599, 77.4126),    # Bhopal
    "Telangana": (17.3850, 78.4867),         # Hyderabad
    "Kerala": (8.5241, 76.9366),             # Thiruvananthapuram
    "Punjab": (30.7333, 76.7794),            # Chandigarh
    "Odisha": (20.2961, 85.8245),            # Bhubaneswar
    "Jharkhand": (23.3441, 85.3096),         # Ranchi
    "Haryana": (30.7333, 76.7794),           # Chandigarh
}

def get_simulated_monsoon_intensity(month: int) -> float:
    """
    Fallback formula to calculate monsoon intensity based on the current month.
    Peak monsoon in India is July-August.
    """
    if month in [7, 8]:
        return 0.9
    elif month in [6, 9]:
        return 0.7
    elif month in [5, 10]:
        return 0.4
    else:
        return 0.1

def get_live_weather(state: str, api_key: Optional[str] = None) -> Dict:
    """
    Fetches real-time weather data from OpenWeatherMap for the given state's capital.
    Gracefully falls back to simulated data if API call fails or key is missing.
    Results are cached for 30 minutes.
    
    Args:
        state (str): Name of the Indian state.
        api_key (str, optional): OpenWeatherMap API key. Defaults to None.
        
    Returns:
        dict: Weather intelligence including temperature, humidity, rainfall, and risk metrics.
    """
    state_title = state.title()
    
    # Check cache
    current_time = time.time()
    if state_title in WEATHER_CACHE:
        cached_data, timestamp = WEATHER_CACHE[state_title]
        if current_time - timestamp < CACHE_DURATION:
            return cached_data
            
    api_key = api_key or os.getenv('OPENWEATHERMAP_API_KEY')
    month = datetime.datetime.now().month
    
    # Default/Fallback simulated data
    # `source` lets callers tell a real observation apart from the seasonal
    # fallback. Without it, a report can print "Severe monsoon conditions —
    # 0.0mm rain in last hr", which is both contradictory and untrustworthy.
    weather_data = {
        "temperature": 30.0,
        "humidity": 60,
        "rainfall_mm": 0.0,
        "wind_speed": 5.0,
        "weather_description": "Seasonal profile (no live feed configured)",
        "monsoon_intensity": get_simulated_monsoon_intensity(month),
        "source": "simulated",
    }

    if state_title not in STATE_COORDINATES:
        return weather_data
        
    lat, lon = STATE_COORDINATES[state_title]

    if api_key:
        try:
            url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}&units=metric"
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                weather_data["temperature"] = data.get("main", {}).get("temp", 30.0)
                weather_data["humidity"] = data.get("main", {}).get("humidity", 60)
                weather_data["wind_speed"] = data.get("wind", {}).get("speed", 5.0)
                weather_data["weather_description"] = data.get("weather", [{}])[0].get("description", "Clear").title()
                
                # Extract rainfall in the last 1 hour
                rainfall = data.get("rain", {}).get("1h", 0.0)
                weather_data["rainfall_mm"] = rainfall
                
                # Calculate monsoon intensity based on actual live data + month context
                # High humidity + rainfall + monsoon months = high intensity
                intensity = get_simulated_monsoon_intensity(month)
                if rainfall > 10:
                    intensity = min(1.0, intensity + 0.3)
                elif rainfall > 2:
                    intensity = min(1.0, intensity + 0.15)
                
                if weather_data["humidity"] > 85:
                    intensity = min(1.0, intensity + 0.1)
                    
                weather_data["monsoon_intensity"] = round(intensity, 2)
                weather_data["source"] = "live"
        except requests.RequestException:
            # Fallback to simulated on exception gracefully
            pass
            
    # Add to cache
    WEATHER_CACHE[state_title] = (weather_data, current_time)
    
    return weather_data

def get_weather_risk_summary(state: str, api_key: Optional[str] = None) -> Dict:
    """
    Generates a human-readable risk summary for a given state based on weather intelligence.
    
    Args:
        state (str): Name of the Indian state.
        api_key (str, optional): OpenWeatherMap API key. Defaults to None.
        
    Returns:
        dict: Summary containing risk_level, description, icon, and estimated delay impact.
    """
    weather = get_live_weather(state, api_key)
    intensity = weather.get("monsoon_intensity", 0.1)
    rainfall = weather.get("rainfall_mm", 0.0)
    
    if intensity > 0.8:
        risk_level = 'Critical'
        icon = '⛈️'
        impact = 40  # 40% increased delay probability
        desc_prefix = "Severe monsoon conditions"
    elif intensity > 0.6:
        risk_level = 'High'
        icon = '🌧️'
        impact = 25
        desc_prefix = "Heavy rainfall and monsoon weather"
    elif intensity > 0.3:
        risk_level = 'Medium'
        icon = '🌦️'
        impact = 10
        desc_prefix = "Moderate weather disruptions"
    else:
        risk_level = 'Low'
        icon = '☀️'
        impact = 2
        desc_prefix = "Clear weather conditions"
        
    # Only quote an observation when there actually is one. Appending
    # "0.0mm rain in last hr" to "Severe monsoon conditions" reads as a
    # contradiction and makes the whole panel look untrustworthy.
    if weather.get("source") == "live":
        description = (f"{desc_prefix} in {state.title()} — "
                       f"{weather['weather_description']}, {rainfall}mm rain in the last hour.")
    else:
        description = (f"{desc_prefix} expected in {state.title()} "
                       f"(seasonal profile — no live weather feed configured).")
    
    return {
        "risk_level": risk_level,
        "description": description,
        "icon": icon,
        "impact_on_delivery": impact,
        "source": weather.get("source", "simulated"),
        "raw_weather": weather
    }

def get_multi_state_weather(states: List[str], api_key: Optional[str] = None) -> Dict[str, Dict]:
    """
    Batch fetches weather intelligence for multiple states.
    
    Args:
        states (list): List of state names.
        api_key (str, optional): OpenWeatherMap API key. Defaults to None.
        
    Returns:
        dict: A mapping of state names to their weather risk summaries.
    """
    results = {}
    for state in states:
        results[state.title()] = get_weather_risk_summary(state, api_key)
    return results

if __name__ == '__main__':
    print("--- NirmanAI Weather Intelligence Module ---")
    print("Fetching weather for Maharashtra (Simulated/Fallback mode)...")
    summary = get_weather_risk_summary("Maharashtra")
    print(f"Risk Level: {summary['icon']} {summary['risk_level']}")
    print(f"Description: {summary['description']}")
    print(f"Delay Impact: +{summary['impact_on_delivery']}%")
    print(f"Raw Data: {summary['raw_weather']}")
    
    print("\nBatch fetching for multiple states...")
    states_to_check = ["Bihar", "Kerala", "Rajasthan"]
    batch_results = get_multi_state_weather(states_to_check)
    for s, data in batch_results.items():
        print(f"{s}: {data['icon']} {data['risk_level']} - Impact: +{data['impact_on_delivery']}%")
