import json
import re
from app.models.deepseek_vl import DeepSeekVLDriver

driver = DeepSeekVLDriver(verbose=True)

tasks = {
    "scene_description": {
        "prompt": """Think step by step about the driving scene. Consider road type, surroundings, weather, time of day, and overall scene. After reasoning, provide your final answer in at most 10 words. Start your final answer with 'Final answer:'""",
        "max_new_tokens": 1000
    },
    "object_listing": {
        "prompt": """Think step by step about the main objects in the scene and their approximate positions (left, center, right). After reasoning, give a final list in at most 10 words. Start your final answer with 'Final answer:'""",
        "max_new_tokens": 1000
    },
    "drivable_area": {
        "prompt": """Think step by step about whether the road ahead is clear, any obstacles, and traffic rules. After reasoning, give a final verdict in at most 10 words. Start your final answer with 'Final answer:'""",
        "max_new_tokens": 1000
    },
    "lane_description": {
        "prompt": """Think step by step about lane markings, number of lanes, and lane availability. After reasoning, give a final description in at most 10 words. Start your final answer with 'Final answer:'""",
        "max_new_tokens": 1000
    }
}

image_path = "./image.png"
all_results = {}

for task_name, task_info in tasks.items():
    print(f"\n--- Task: {task_name} ---")
    response = driver.analyze(
        prompt=task_info["prompt"],
        image_paths=image_path,
        max_new_tokens=task_info["max_new_tokens"],
        do_sample=False,
        repetition_penalty=1.05
    )
    print("Full response:\n", response)

    # Extract the part after "Final answer:"
    match = re.search(r'Final answer:\s*(.*?)(?:\n|$)', response, re.IGNORECASE)
    if match:
        short_answer = match.group(1).strip()
        print("Short answer (≤10 words):", short_answer)
        all_results[task_name] = short_answer
    else:
        # Fallback: take last sentence or first 10 words
        words = response.split()
        if len(words) > 10:
            short_answer = ' '.join(words[:10]) + "..."
        else:
            short_answer = response
        print("No 'Final answer:' found, using first 10 words.")
        all_results[task_name] = short_answer

# Save only short answers
with open("driving_analysis_short.json", "w") as f:
    json.dump(all_results, f, indent=2)

driver.close()