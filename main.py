from flask import Flask, request, jsonify, session, send_file
from icalendar import Calendar, Event
import icalendar
import datetime
import re
from datetime import timedelta
import openai
import json
import logging
import os
import uuid
from flask_cors import CORS
from werkzeug.datastructures import FileStorage
import base64



logging.basicConfig(level=logging.DEBUG)


app = Flask(__name__)
CORS(app)

session = {}
app.secret_key = 'supersecretkey'  # Change this to a random secret key

openai.api_key = os.getenv('OPENAI_API_KEY')

@app.route('/start_session', methods=['GET'])
def start_session_endpoint():
    session_id = start_new_session()
    logging.debug(f'New session started: {session_id}')
    return jsonify(session_id=session_id), 200

def start_new_session():
    session_id = str(uuid.uuid4())  # Generate a unique session ID
    session[session_id] = {
        'preferences': None,
        'events': None,
        'tasks': None,
        'gpt4_dialogue': [],
        'questions': None,
        'answers': None,
        'schedule': None,
        'accepted_schedule': None,
        'feedback': None,
    }
    return session_id


def parse_ics(ics_file_content, target_day):
    ics_file = Calendar.from_ical(ics_file_content)
    events = [event for event in ics_file.walk('VEVENT') if event.get('dtstart').dt == target_day]
    return events

def get_preference_text(pref_key, pref_name, preferences):
    return f'{pref_name}: {preferences.get(pref_key)}'

def parse_tasks_text(tasks_text):
    return [task.strip() for task in tasks_text.strip().split('\n') if task.strip()]
def construct_gpt4_prompt_with_feedback(preferences, events, tasks, feedback):
    # Retrieve existing dialogue from the session, if available
    session_id = request.headers.get('X-Session-ID')


   
    existing_dialogue = session.get(session_id, {}).get('gpt4_dialogue', [])
    
    # Increment feedback round counter
    feedback_round = session.get(session_id, {}).get('feedback_round', 0) + 1
    if feedback_round > 5:
        # Reset the dialogue after 5 rounds
        feedback_round = 0
        existing_dialogue = []
    
    # Construct feedback message
    feedback_message = {
        "role": "user",
        "content": "Please revise the schedule based on the following feedback: " + feedback
    }
    
    # Append feedback to existing dialogue
    existing_dialogue.append(feedback_message)
    
    # Update the session with the new dialogue and feedback round counter
    session[session_id] = {'dialogue': existing_dialogue, 'feedback_round': feedback_round}
    
    return existing_dialogue


def construct_gpt4_prompt(preferences, events, tasks):
    events_text = '\n'.join([f'{event.get("dtstart").dt.strftime("%H:%M")}, {event.get("summary").value}' for event in events])
    tasks_text = '\n'.join(tasks)
    formatted_preferences = (
        f"Task Preference: {preferences.get('taskPreference', 'None')}, "
        f"Specific Times for Tasks: {preferences.get('specificTimes', 'None')}, "
        f"Break Length: {preferences.get('breakLength', 'None')}, "
        f"Break Frequency: {preferences.get('breakFrequency', 'None')}, "
        f"Start Time: {preferences.get('startTime', 'None')}, "
        f"End Time: {preferences.get('endTime', 'None')}, "
        f"Schedule Breaks: {preferences.get('scheduleBreaks', 'None')}, "
        f"Schedule Meals: {preferences.get('scheduleMeals', 'None')}, "
        f"Meal Preferences: {preferences.get('mealPrefs', 'None')}"
    )
    
    initial_prompt = f"""
    I am going to give you a list of calendar events and tasks for today that we will work together on to make a daily schedule.

    Use your best guess to determine the length of tasks. Ask up to 6 clarifying questions you have, then return a schedule.

    Your questions output must start with a new line that says "Questions:" followed by each question on a new line. After I answer the questions, return my schedule.

    The schedule output must be under the heading "Schedule" and must have one event on each line and must be in the exact format like this: start time, duration in minutes, event description. An example line looks like this: 09:00, 30m, Book your NYC trip.

    Here are the list of events and tasks. Do not change the start time or duration of events:
    Events:
    {events_text}
    Tasks:
    {tasks_text}
    User Preferences: 
    {formatted_preferences}
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
        # Ensure content is formatted as a string
        formatted_messages = [{"role": msg["role"], "content": str(msg["content"])} for msg in messages]

        response = openai.ChatCompletion.create(
            model="gpt-4",  # Update this to the correct GPT-4 model version
            messages=formatted_messages
        )
        logging.info("OPEN AI RESPONSE\n\n" + response['choices'][0]['message']['content'])
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
        elif line.startswith('Schedule:'):
            questions_section = False
            schedule_section = True
            continue
        
        if questions_section:
            questions.append(line)
        elif schedule_section:
            # Skip blank lines
            if not line.strip():
                continue
            
            # Check for matching schedule line format
            match = re.match(r'(\d{2}:\d{2}), (\d{1,2}[hm]|\d{1,2} hr), (.+)', line)  # Updated regex to include space before 'hr'
            if match:
                time, duration, description = match.groups()
                # Convert 'hr' to minutes if necessary
                if 'hr' in duration:
                    hours, _ = duration.split(' ')
                    hours = int(hours)
                    minutes = hours * 60
                    duration = f'{minutes}m'
                # Append the formatted schedule line
                schedule_lines.append(f'{time}, {duration}, {description}')
            else:
                # Skip lines that don't match the expected format
                extra_text.append(line)  # You may choose to log these lines for debugging
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
        Calendar.from_ical(ics_content)
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
        # Validate the format using a regular expression
        match = re.match(r'(\d{2}:\d{2}), (\d{1,2}[hm]), (.+)', entry)
        if not match:
            # Skip this line if it doesn't match the expected format
            continue
        
        start_time_str, duration_str, description = match.groups()
        start_time = timedelta(hours=int(start_time_str.split(':')[0]), minutes=int(start_time_str.split(':')[1]))

        # Check for the presence of 'hr' or 'h' to identify hour durations
        if 'hr' in duration_str:
            hours = int(duration_str.replace('hr', ''))
            duration = timedelta(hours=hours)
        elif 'h' in duration_str:
            hours = int(duration_str.replace('h', ''))
            duration = timedelta(hours=hours)
        else:  # Assume the remaining case is in minutes
            minutes = int(duration_str.replace('m', ''))
            duration = timedelta(minutes=minutes)

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
@app.route('/submit_answers', methods=['POST'])
def submit_answers():
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        if session_id not in session:
            return jsonify(error='Invalid session ID'), 400
        answers = data.get('answers')
        session[session_id]['answers'] = answers
        questions_list = [f'Question {key}: {value}' for key, value in answers.items()]
        questions_str = '\n'.join(questions_list)


        # Retrieve the existing session data
        current_state = session.get(session_id, {})

        # Retrieve existing dialogue from the session
        existing_dialogue = session[session_id].get('gpt4_dialogue', [])
        user_message = {
            "role": "user",
            "content": questions_str  # or format answers appropriately
        }
        existing_dialogue.append(user_message)

        # Add a system instruction to generate the schedule
        system_message = {
            "role": "system",
            "content": "Now generate the schedule based on the provided information and answers."
        }
        existing_dialogue.append(system_message)

        # Interact with GPT-4
        gpt4_response = interact_with_gpt4(existing_dialogue)

        # Update the dialogue history in the session
        session[session_id]['gpt4_dialogue'] = existing_dialogue

        # Parse GPT-4 response to extract the new schedule
        # Assuming parse_gpt4_response is modified to return a schedule
        _, new_schedule, _ = parse_gpt4_response(gpt4_response)

        if not new_schedule:
            return jsonify(error='Error generating new schedule'), 500

        # Store the new schedule in the session data
        session[session_id]['schedule'] = new_schedule

        return jsonify(status='success', schedule=new_schedule, session_id=session_id), 200
    except Exception as e:
        app.logger.error(f'Error in /submit_answers: {e}')
        return jsonify(error=str(e)), 500


@app.route('/generate_schedule', methods=['POST'])
def generate_schedule():
    logging.info('Entering generate_schedule endpoint')
    data = request.get_json()  # Get JSON data from request
    session_id = request.headers.get('X-Session-ID')
    # Decode Base64 file
    ics_file_content_base64 = data.get('ics_file', '')
    ics_file_content = base64.b64decode(ics_file_content_base64).decode('utf-8')
    if not ics_file_content:
        return jsonify(error='No ICS file uploaded'), 400
    tasks_text = data.get('tasks_text', '')
    preferences = data.get('preferences', {})
    if not validate_ics(ics_file_content):
        return jsonify(error='Invalid ICS file format'), 400
    events = parse_ics(ics_file_content, preferences.get('target_day'))
    tasks = parse_tasks_text(tasks_text)
    logging.info(f'Generate Schedule: Received request with session ID: {session_id}')
    session[session_id]['events'] = events
    session[session_id]['tasks'] = tasks
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
    
    session[session_id]['gpt4_dialogue'] = messages

    questions = validate_and_parse_questions(response)

    if questions:
        session[session_id]['questions'] = questions
        return jsonify(status='questions', questions=questions, session_id=session_id), 200

    schedule = parse_schedule(response)
    if not schedule:
        logging.error(f'Error parsing schedule: {response}')
        return jsonify(error='Error parsing schedule'), 500

    ics_output = generate_ics(schedule)
    return jsonify(status='success', schedule=schedule, ics_file=ics_output.decode()), 200
@app.route('/revise_schedule', methods=['POST'])
def revise_schedule():
    data = request.get_json()
    session_id = data.get('session_id')
    accepted = data.get('accepted', None)
    feedback = data.get('feedback', None)

    current_state = session.get(session_id, {})

    if not current_state:
        return jsonify(error='No current schedule data found, please start with /generate_schedule endpoint'), 400

    if accepted:
        return jsonify(status='success', session_id=session_id), 200

    if feedback:
        # Update GPT-4 prompt with user feedback
        gpt4_prompt = construct_gpt4_prompt_with_feedback(
            current_state['preferences'], 
            current_state['events'], 
            current_state['tasks'], 
            feedback
        )
        
        gpt4_response = interact_with_gpt4(gpt4_prompt)
        questions, valid_schedule, _ = parse_gpt4_response(gpt4_response)

        if valid_schedule:
            parsed_schedule = parse_schedule(valid_schedule)
            current_state['parsed_schedule'] = parsed_schedule
            session[session_id] = current_state
        
        return jsonify({"questions": questions, "schedule": valid_schedule, "session_id": session_id})

    return jsonify(error='Either acceptance or feedback is required'), 400


@app.route('/accept_schedule', methods=['POST'])
def accept_schedule():
    data = request.get_json()
    session_id = data.get('session_id')
    accepted = data.get('accepted')
    session[session_id]['accepted_schedule'] = accepted
    return jsonify(status='success', session_id=session_id), 200

@app.route('/download_schedule', methods=['GET'])
def download_schedule():
    session_id = request.args.get('session_id')

    if session_id not in session or 'accepted_schedule' not in session[session_id]:
        return jsonify(error='No accepted schedule found for this session ID'), 400

    schedule_lines = session[session_id].get('accepted_schedule', [])
    ics_content = create_ics(schedule_lines)
    
    response = app.response_class(
        response=ics_content,
        mimetype='text/calendar',
        headers={'Content-Disposition': 'attachment;filename=schedule.ics'}
    )
    
    return response

if __name__ == '__main__':
    app.run(debug=True)
