import csv
import json
import os
import sys

def extract_conversation(json_data, json_turn_data):
    # Prepare a list to store conversations
    conversation = []
    temp_convo = {}

    # Navigate through nested structure safely
    debug_info = json_data.get('debug_info', {})
    all_tools_call = debug_info.get('all_tools_call', {})
    prompt_messages = all_tools_call.get('prompt_messages', [])
    user_metadata = json_data.get('user_metadata', {})  # Extract user_metadata

    # Ensure prompt_messages is a list
    if not isinstance(prompt_messages, list):
        print("Error: 'prompt_messages' is not a list.")
        return conversation

    # Extract data from json_turn
    timestamp = json_turn_data.get('now', 'Unknown Timestamp')
    query = json_turn_data.get('user_query', 'No Query Provided')

    # Add json_turn data and user metadata at the top of the conversation
    conversation.append({
        "Timestamp": timestamp,
        "Query": query
    })

    if user_metadata:
        conversation.append({"User_Metadata": user_metadata})

    # Iterate through the prompt messages
    for item in prompt_messages:
        if not isinstance(item, dict):
            print("Error: Item in 'prompt_messages' is not a dictionary.")
            continue
        
        # Get role and parts content
        role = item.get('message', {}).get('author', {}).get('role')
        parts = item.get('message', {}).get('content', {}).get('parts', [])

        if not parts:
            continue

        # Group user and assistant messages
        if role == 'user':
            if 'User' in temp_convo:  # If a previous user input exists without a response
                conversation.append(temp_convo)
                temp_convo = {}
            temp_convo['User'] = parts[0]
        elif role == 'assistant':
            if 'User' in temp_convo:  # Only pair with existing user input
                temp_convo['Assistant'] = parts[0]
                conversation.append(temp_convo)
                temp_convo = {}  # Reset for the next pair

    # Append the last conversation if it was not followed by an assistant response
    if temp_convo:
        conversation.append(temp_convo)

    return conversation

# File paths
csv_file_path = '/Users/mandip/Downloads/New_Query_2025-01-13_2_24pm_2025_02_05.csv'
output_json_path = '/Users/mandip/Downloads/extracted_conversations.json'

# Increase CSV field size limit to handle large fields
csv.field_size_limit(sys.maxsize)

# Check if the file exists
if not os.path.exists(csv_file_path):
    print(f"Error: File '{csv_file_path}' not found.")
else:
    all_conversations = []

    # Read the CSV and extract conversations
    with open(csv_file_path, mode='r', encoding='utf-8') as file:
        csv_reader = csv.reader(file)
        headers = next(csv_reader)  # Skip header row if present
        
        for row in csv_reader:
            if len(row) < 5:  # Ensure there are enough columns
                continue  
            
            try:
                # Assuming the 4th column is json_artifacts and 5th is json_turn
                json_artifacts = json.loads(row[3])  
                json_turn = json.loads(row[4])  
                
                # Extract conversation with additional json_turn data
                conversation = extract_conversation(json_artifacts, json_turn)
                
                if conversation:
                    all_conversations.append(conversation)
            except json.JSONDecodeError as e:
                print(f"JSONDecodeError in row {csv_reader.line_num}: {e}")

    # Write output to JSON file
    with open(output_json_path, 'w', encoding='utf-8') as json_file:
        json.dump(all_conversations, json_file, ensure_ascii=False, indent=4)

    print(f"Conversations extracted and saved to '{output_json_path}'")
