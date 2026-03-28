import json
from app.models.deepseek_vl import DeepSeekVLDriver

driver = DeepSeekVLDriver(verbose=True)

tasks = {
    "scene_description": {
        "prompt": """Let's analyze the driving scene step by step:

1. Observe the road type (highway, city street, residential, rural).
2. Note the surroundings (buildings, trees, open field, etc.).
3. Determine the weather (clear, rain, snow, overcast).
4. Identify the time of day (day, night, dusk/dawn).
5. Classify the scene type (urban, suburban, rural).
6. Write a brief overall description.

After this reasoning, provide your answer as a JSON object with exactly the following keys:
- "road_type": (string)
- "surroundings": (string)
- "weather": (string)
- "time_of_day": (string)
- "scene_type": (string)
- "description": (string)

Output only the JSON object, no other text.""",
        "max_new_tokens": 600 
    },
    "object_listing": {
        "prompt": """Let's list the main objects in this driving scene step by step:

1. Identify all visible objects (cars, pedestrians, traffic lights, signs, etc.).
2. For each object, determine its approximate position (left, center, or right).

After this reasoning, provide your answer as a JSON array of objects, each with:
- "object": (string)
- "position": (string)

 Output only the JSON object. Do not output any other text, questions, or explanations.
""",
        "max_new_tokens": 600
    },
    "drivable_area": {
        "prompt": """Let's analyze the drivable area step by step:

1. Look at the road ahead. Is it clear?
2. Are there any obstacles blocking the path (e.g., stopped vehicles, pedestrians, debris)?
3. Consider traffic rules (traffic lights, stop signs) – do they allow proceeding?
4. Based on the above, decide if the vehicle can proceed forward.

After this reasoning, provide your answer as a JSON object with:
- "drivable": (boolean)
- "reason": (string, explanation)
- "obstacles": (list of strings)

Output only the JSON object, no other text.""",
        "max_new_tokens": 600
    },
    "lane_description": {
        "prompt": """Let's describe the lane markings and availability step by step:

1. Count the number of visible lanes.
2. For each lane, identify the type of lane markings (solid, dashed, none).
3. Assess lane availability (clear, blocked, merging).

After this reasoning, provide your answer as a JSON object with:
- "num_lanes": (integer)
- "lane_markings": (list of strings)
- "lane_availability": (string)

Output only the JSON object, no other text.""",
        "max_new_tokens": 600
    }
}

image_path = "./image.png"

for task_name, task_info in tasks.items():
    print(f"\n--- Task: {task_name} ---")
    response = driver.analyze(
        prompt=task_info["prompt"],
        image_paths=image_path,
        max_new_tokens=task_info["max_new_tokens"],
        do_sample=True,          # keep as originally
        repetition_penalty=1.05  # helps prevent repetition
    )
    print("Raw response:\n", response)

    # Attempt to parse JSON (as before)
    try:
        start_idx = response.find('{') if '{' in response else response.find('[')
        end_idx = response.rfind('}') if '}' in response else response.rfind(']')
        if start_idx != -1 and end_idx != -1:
            json_str = response[start_idx:end_idx+1]
            data = json.loads(json_str)
            print("Parsed JSON:")
            print(json.dumps(data, indent=2))
        else:
            print("Could not find JSON structure in response.")
    except json.JSONDecodeError as e:
        print(f"Failed to parse JSON: {e}")

driver.close()