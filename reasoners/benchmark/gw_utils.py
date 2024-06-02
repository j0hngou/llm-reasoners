import torch
import webcolors
import re
import json

def closest_color(requested_color):
    min_colors = {}
    for key, name in webcolors.CSS3_HEX_TO_NAMES.items():
        r_c, g_c, b_c = webcolors.hex_to_rgb(key)
        rd = (r_c - requested_color[0]) ** 2
        gd = (g_c - requested_color[1]) ** 2
        bd = (b_c - requested_color[2]) ** 2
        min_colors[(rd + gd + bd)] = name
    return min_colors[min(min_colors.keys())]

def describe_latent(latent, keys):
    descriptions = []
    for value, key in zip(latent, keys):
        # Use regex to extract the entity and RGB values
        match = re.search(r'([a-zA-Z]+)_\((\d+,\s*\d+,\s*\d+)\)(.*)', key)
        if match:
            entity = match.group(1)
            color_str = match.group(2)
            attribute = match.group(3).replace('_', ' ').strip()
            color_tuple = tuple(map(int, color_str.split(',')))
            color_name = closest_color(color_tuple)
        else:
            entity = "unknown entity"
            color_name = "unknown color"
            attribute = "unknown attribute"

        descriptions.append(f"{color_name} {entity} {attribute} is {value:.2f}")

    return ", ".join(descriptions)


def load_gridworld(file_path, config, return_intermediate=False):
    # Load data using PyTorch's deserialization
    config = json.load(open(config, 'r'))
    keys = config['flattened_causals']
    trajectories = torch.load(file_path)

    data = []
    for images, actions, latents in trajectories:
        cur_data = {}
        # Generate descriptions for initial and final states
        cur_data['init'] = describe_latent(latents[0], keys)
        cur_data['goal'] = describe_latent(latents[-1], keys)
        cur_data['plan'] = "\n".join(actions) + "\n[PLAN END]\n"

        if return_intermediate:
            cur_data['states'] = [describe_latent(lat, keys) for lat in latents]
        
        cur_data['question'] = fill_template(cur_data['init'], cur_data['goal'], "") + '\n'
        cur_data['images'] = images
        data.append(cur_data)
    
    return data

def extract_init_state(example):
    """Extract the initial state from the example
    
    :param example: example
    """
    # print(example)
    init_statement = example["question"].split("[STATEMENT]\nAs initial conditions I have that, ")[1]\
        .split("My goal")[0].strip()
    return init_statement

def fill_template(INIT, GOAL, PLAN):
    text = ""
    if INIT != "":
        text += "\n[STATEMENT]\n"
        text += f"As initial conditions I have that, {INIT.strip()}."
    if GOAL != "":
        text += f"\nMy goal is to have that {GOAL}."
    text += f"\n\nMy plan is as follows:\n\n[PLAN]{PLAN}"

    # TODO: Add this replacement to the yml file -- Use "Translations" dict in yml
    text = text.replace("-", " ").replace("ontable", "on the table")
    return text

def generate_actions(cur_state, entities, action_types, can_move_car):
    import re

    state_dict = {}
    matches = re.finditer(r"(\w+ \w+) position [xy] is (\d+\.\d+)|(\w+ \w+) state is (\d+\.\d+)", cur_state)
    for match in matches:
        if match.group(1):
            state_dict[match.group(1)] = float(match.group(2))
        else:
            state_dict[match.group(3)] = float(match.group(4))

    actions = []
    
    vehicle_to_trafficlight = {
        'red vehicle': 'cyan trafficlight',
        'blue vehicle': 'silver trafficlight',
        'silver vehicle': 'olive trafficlight'
    }

    for entity in entities:
        if 'trafficlight' in entity:
            if 'change_state' in action_types:
                actions.append(f"You changed the state of the {entity}")
        elif 'obstacle' in entity:
            if 'move' in action_types:
                # # Assuming the obstacle is a boulder for natural language output
                # obstacle_type = "boulder" if "boulder" in entity else "obstacle"
                # actions.append(f"You moved the {entity.replace('obstacle', obstacle_type)}")
                pass
        elif 'vehicle' in entity:
            # if can_move_car and 'move' in action_types:
            #     # Check if the corresponding traffic light state allows movement
            #     corresponding_trafficlight = vehicle_to_trafficlight[entity]
            #     if state_dict.get(corresponding_trafficlight) == 1.00:
            #         actions.append(f"You moved the {entity}.")
                pass
    actions.append("You performed no action.")

    return actions
    # return [actions[0], actions[1]]

def extract_goals(example, return_raw=False):
    """Extract the goals from the example
    
    :param example: example
    """
    goal_statement = example["question"].split("[STATEMENT]")[-1]\
        .split("My goal is to ")[1].split("My plan is as follows")[0].strip()
    if return_raw:
        return goal_statement
    #TODO regex parse goal statement.
#     goals = re.findall("the [a-z]{0,10} block is on top of the [a-z]{0,10} block", goal_statement)
    return goals

import re

def extract_goals(example, return_raw=False):
    """Extract the goals from the example.
    
    :param example: dict with 'question' key containing goal and plan statements
    :param return_raw: if True, returns the raw goal statement
    :return: Either raw goal string or parsed goals as a dictionary
    """
    goal_statement = example["question"].split("[STATEMENT]")[-1]\
        .split("My goal is to ")[1].split("My plan is as follows")[0].strip()
    
    if return_raw:
        return goal_statement
    
    # Regex pattern to match the identifier, attribute, and value accurately
    pattern = r"(\w+)\s(\w+)\s(position\s\w+|state)\sis\s([\d\.]+)"
    goals = {}
    
    for match in re.finditer(pattern, goal_statement):
        key = f"{match.group(1)} {match.group(2)} {match.group(3)}"
        value = float(match.group(4).rstrip('.'))
        goals[key] = value
    
    return goals

def goal_check(goals, description, epsilon=0.07, ignore_obstacles=False):
    """Check if the description matches the goals with a tolerance and return the percentage of goals met.
    
    :param goals: dictionary of goal states
    :param description: description string containing current states
    :param epsilon: tolerance for numeric comparisons
    :param ignore_obstacles: if True, obstacle positions are ignored in the check
    :return: Tuple (boolean, float) where boolean is True if all goals are met within tolerance, and float is the percentage of goals met
    """
    # Parse the description into a dictionary
    # pattern = r"(\w+)\s(\w+\s\w+)\sis\s([\d\.]+)(?=\,|\s|$)"
    # current_states = {match.group(1) + ' ' + match.group(2): float(match.group(3).rstrip('.')) for match in re.finditer(pattern, description)}
    pattern = r"(\w+)\s(\w+)\s(position\s\w+|state)\sis\s([\d\.]+)"
    current_states = {}
    
    for match in re.finditer(pattern, description):
        key = f"{match.group(1)} {match.group(2)} {match.group(3)}"
        value = float(match.group(4).rstrip('.'))
        current_states[key] = value
    
    total_goals = 0
    met_goals = 0
    for key, goal_value in goals.items():
        if ignore_obstacles and 'obstacle' in key:
            continue
        total_goals += 1
        if key in current_states and (abs(current_states[key] - goal_value) <= epsilon):
            met_goals += 1

    all_goals_met = met_goals == total_goals
    percentage_met = (met_goals / total_goals) * 100 if total_goals > 0 else 0
    return all_goals_met, percentage_met

# def validate_plan(