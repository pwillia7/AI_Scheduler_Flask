from flask import Flask, request, jsonify, session
from icalendar import Calendar, Event
import icalendar
import datetime
import re
from datetime import timedelta
import openai
import logging
import os

logging.basicConfig(filename='app.log', level=logging.INFO)


app = Flask(__name__)
app.secret_key = 'supersecretkey'  # Change this to a random secret key

openai.api_key = os.getenv('OPENAI_API_KEY')


def parse_ics(ics_file_content, target_day):
    ics_file = icalendar.Calendar.from_ics(ics_file_content)
    events = [event for event in ics_file.walk('VEVENT') if event.get('dtstart').dt.date() == target_day]
    return events

def get_preference_text(pref_key, pref_name, preferences):
    return f'{pref_name}: {preferences.get(pref_key)}'

def parse_tasks_text(tasks_text):
    return [task.strip() for task in tasks_text.strip().split('\n') if task.strip()]

def construct_gpt4_prompt_with_feedback(preferences, events, tasks, feedback):
    # Retrieve existing dialogue from the session, if available
    session_id = request.headers.get('X-Session-ID', 'default-session')
    existing_dialogue = session.get(session_id, {}).get('dialogue', [])
    
    # Increment feedback round counter
    feedback_round = session.get(session_id, {}).get('feedback_round', 0) + 1
    if feedback_round > 5:
        # Reset the dialogue after 5 rounds
        feedback_round = 0
        existing_dialogue = []
    
    # Construct feedback message
    feedback_message = {
        "role": "user",
        "content": feedback
    }
    
    # Append feedback to existing dialogue
    existing_dialogue.append(feedback_message)
    
    # Update the session with the new dialogue and feedback round counter
    session[session_id] = {'dialogue': existing_dialogue, 'feedback_round': feedback_round}
    
    return existing_dialogue

def construct_gpt4_prompt(preferences, events, tasks):
    events_text = '\n'.join([f'{event.get("dtstart").dt.strftime("%H:%M")}, {event.get("summary").value}' for event in events])
    tasks_text = '\n'.join(tasks)
    preference_texts = [get_preference_text(key, name, preferences) for key, name in [
        ('task_preference', 'Task Preference'),
        ('specific_times', 'Specific Times for Tasks'),
        ('schedule_breaks', 'Schedule Breaks'),
        ('break_length', 'Break Length'),
        ('break_frequency', 'Break Frequency'),
        ('start_time', 'Start Time'),
        ('end_time', 'End Time'),
        ('schedule_meals', 'Schedule Meals'),
        ('meal_prefs', 'Meal Preferences')
    ]]
    preferences_text = ', '.join(preference_texts)
    
    initial_prompt = f"""
    I am going to give you a list of calendar events and tasks for today that we will work together on to make a daily schedule.
    User Preferences: {preferences_text}

    Use your best guess to determine the length of tasks. Ask up to 6 clarifying questions you have. Feel free to make suggestions you think will make the user's day better and more productive.

    Your questions output must start with a new line that says "Questions:" followed by each question on a new line.

    Once the user has confirmed the schedule, output the schedule. The schedule output must have one event on each line and must be in the exact format like this: start time, duration in minutes, event description. An example line looks like this: 09:00, 30m, Book your NYC trip.

    Here are the list of events and tasks. Do not change the start time or duration of events:
    Events:
    {events_text}
    Tasks:
    {tasks_text}
    """
    
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant tasked with creating a daily schedule based on the user's calendar events, tasks, and personal preferences. Aim to design a balanced and productive day, incorporating any specific user requests and preferences."
        },
        {
            "role": "user",
            "content": initial_prompt
        }
    ]
    
    return messages

def interact_with_gpt4(messages):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        logging.error(f'Error interacting with GPT-4: {e}')
        return None


def parse_gpt4_response(response):
    lines = response.split('\n')
    questions_section = False
    schedule_section = False
    questions, schedule_lines, extra_text = [], [], []

    for line in lines:
        if line.startswith('Questions:'):
            questions_section = True
            continue
        elif line.startswith('Proposed Schedule:'):
            questions_section = False
            schedule_section = True
            continue
        
        if questions_section:
            questions.append(line)
        elif schedule_section:
            if re.match(r'\d{2}:\d{2}, \d{1,2}[hm], .+', line):
                schedule_lines.append(line)
            else:
                schedule_section = False
                extra_text.append(line)
        else:
            extra_text.append(line)

    return questions, schedule_lines, extra_text

def create_ics(schedule_lines):
    cal = icalendar.Calendar()
    for line in schedule_lines:
        time, duration, task = line.split(', ')
        start_time = datetime.datetime.strptime(time, '%H:%M')
        duration_hrs, duration_mins = map(int, re.findall(r'\d+', duration))
        duration = datetime.timedelta(hours=duration_hrs, minutes=duration_mins)
        end_time = start_time + duration

        event = icalendar.Event()
        event.add('summary', task)
        event.add('dtstart', start_time)
        event.add('dtend', end_time)
        cal.add_component(event)

    return cal.to_ics().decode()


def validate_ics(ics_content):
    try:
        cal = Calendar.from_ics(ics_content)
        for component in cal.walk():
            if component.name == "VEVENT":
                required_attrs = ["SUMMARY", "DTSTART", "DTEND"]
                for attr in required_attrs:
                    if not component.get(attr):
                        logging.error(f'Missing {attr} in event: {component}')
                        return False
    except ValueError as e:
        logging.error(f'Invalid ICS file: {e}')
        return False
    return True

def validate_schedule(schedule_lines):
    valid_schedule = []
    for line in schedule_lines:
        match = re.match(r'(\d{2}:\d{2}), (\d{1,2}h\d{1,2}m|\d{1,2}[hm]), (.+)', line)
        if match:
            valid_schedule.append(match.groups())
        else:
            return None  # Invalid format, return None
    return valid_schedule

def parse_schedule(valid_schedule):
    schedule_entries = []
    for entry in valid_schedule:
        start_time_str, duration_str, description = entry
        start_time = timedelta(hours=int(start_time_str.split(':')[0]), minutes=int(start_time_str.split(':')[1]))
        if 'h' in duration_str:
            hours, minutes = map(int, duration_str.split('h'))
            duration = timedelta(hours=hours, minutes=minutes)
        else:
            unit = duration_str[-1]
            value = int(duration_str[:-1])
            duration = timedelta(hours=value) if unit == 'h' else timedelta(minutes=value)
        schedule_entries.append((start_time, duration, description))
    return schedule_entries

def validate_and_parse_questions(text):
    lines = text.split('\n')
    questions_section = False
    questions = []

    for line in lines:
        if line.startswith('Questions:'):
            questions_section = True
            continue
        
        if questions_section:
            questions.append(line)
    return questions


def generate_ics(schedule):
    cal = Calendar()
    for item in schedule:
        event = Event()
        event.add('summary', item['description'])
        event.add('dtstart', item['start_time'])
        event.add('dtend', item['end_time'])
        cal.add_component(event)
    return cal.to_ics()


@app.route('/generate_schedule', methods=['POST'])
def generate_schedule():

    data = request.json
    ics_file = data.get('ics_file')
    tasks_text = data.get('tasks_text')
    preferences = data.get('preferences')
    session_id = request.headers.get('X-Session-ID')
    logging.info(f'Generate Schedule: Received request with session ID: {session_id}')


    if not session_id:
        return jsonify(error='Missing session ID'), 400

    if not validate_ics(ics_file):
        return jsonify(error='Invalid ICS file format'), 400

    events = parse_ics(ics_file, preferences.get('target_day'))
    tasks = parse_tasks_text(tasks_text)
    initial_prompt = construct_gpt4_prompt(preferences, events, tasks)

    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant tasked with creating a daily schedule based on the user's calendar events, tasks, and personal preferences. Aim to design a balanced and productive day, incorporating any specific user requests and preferences."
        },
        {
            "role": "user",
            "content": initial_prompt
        }
    ]

    response = interact_with_gpt4(messages)
    if not response:
        return jsonify(error='Error interacting with GPT-4'), 500


    questions = validate_and_parse_questions(response['choices'][0]['text'])

    if questions:
        return jsonify(status='questions', questions=questions), 200

    schedule = parse_schedule(response['choices'][0]['text'])
    if not schedule:
        logging.error(f'Error parsing schedule: {response["choices"][0]["text"]}')
        return jsonify(error='Error parsing schedule'), 500

    ics_output = generate_ics(schedule)

    return jsonify(status='success', schedule=schedule, ics_file=ics_output.decode()), 200



@app.route('/revise_schedule', methods=['POST'])
def revise_schedule():
    logging.info(f'Revised Schedule: Received request with session ID: {session_id}')
    data = request.json
    feedback = data.get('feedback')
    
    # Use the same session ID to retrieve the current state
    session_id = request.headers.get('X-Session-ID', 'default-session')
    current_state = session.get(session_id, {})

    if not current_state:
        return jsonify(error='No current schedule data found, please start with /generate_schedule endpoint'), 400
    
    # Update GPT-4 prompt with user feedback
    gpt4_prompt = construct_gpt4_prompt_with_feedback(current_state['preferences'], current_state['events'], current_state['tasks'], feedback)
    
    gpt4_response = interact_with_gpt4(gpt4_prompt)
    questions = validate_and_parse_questions(gpt4_response)
    valid_schedule = validate_schedule(gpt4_response.split('\n'))

    if valid_schedule:
        parsed_schedule = parse_schedule(valid_schedule)
        current_state['parsed_schedule'] = parsed_schedule
        session[session_id] = current_state
    
    return jsonify({"questions": questions, "schedule_lines": valid_schedule})


@app.route('/download_schedule', methods=['GET'])
def download_schedule():

    schedule_lines = session.get('accepted_schedule', [])
    ics_content = create_ics(schedule_lines)
    
    response = app.response_class(
        response=ics_content,
        mimetype='text/calendar',
        headers={'Content-Disposition': 'attachment;filename=schedule.ics'}
    )
    
    return response

if __name__ == '__main__':
    app.run(debug=True)
