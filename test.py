import json
from app.models.deepseek_vl import DeepSeekVLDriver

driver = DeepSeekVLDriver(verbose=True)

# Plain‑text prompts (no JSON instructions)
tasks = {
    "scene_description": {
        "prompt": """Describe the driving scene in a few sentences.
Include: road type, surroundings, weather, time of day, and overall scene.""",
        "max_new_tokens": 1000
    },
    "object_listing": {
        "prompt": """List the main objects visible in this driving scene (e.g., cars, pedestrians, traffic lights, signs)
and tell me their approximate positions (left, center, or right).""",
        "max_new_tokens": 1000
    },
    "drivable_area": {
        "prompt": """Look at the road ahead. Is it clear to drive forward?
If not, what obstacles or traffic rules are blocking the path? Explain briefly.""",
        "max_new_tokens": 1000
    },
    "lane_description": {
        "prompt": """Describe the lane markings and lane availability.
How many lanes? Are the markings solid or dashed? Is the lane clear or blocked?""",
        "max_new_tokens": 1000
    }
}

image_path = "./image.png"
all_results = {}   # store raw text answers

for task_name, task_info in tasks.items():
    print(f"\n--- Task: {task_name} ---")
    response = driver.analyze(
        prompt=task_info["prompt"],
        image_paths=image_path,
        max_new_tokens=task_info["max_new_tokens"],
        do_sample=False,                # deterministic, cleaner
        repetition_penalty=1.05
    )
    print("Answer:\n", response)
    all_results[task_name] = response

# Save all raw answers to a file (optional)
with open("driving_analysis_text.json", "w") as f:
    json.dump(all_results, f, indent=2)

driver.close()