from flask import Flask, request, jsonify, session, send_file
from icalendar import Calendar, Event
import icalendar
from datetime import datetime, timedelta, date
import re
import openai
import json
import logging
import os
import uuid
import pytz
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
        'timezone': None,
    }
    return session_id

def parse_ics(ics_file_content, target_day):
    try:
        # Step 1: Extract home timezone
        ics_file = Calendar.from_ical(ics_file_content)
        home_timezone_id = ics_file.get('X-WR-TIMEZONE', default='UTC')
        home_timezone = pytz.timezone(home_timezone_id)
        
        # Step 2: Parse target_day to datetime object
        target_day_dt = datetime.strptime(target_day, '%Y-%m-%d')
        target_day_start = home_timezone.localize(target_day_dt)
        target_day_end = target_day_start + timedelta(days=1)
        
        events_on_target_day = []  # List to hold events on target_day
        
        # Step 3: Iterate through each event
        for event in ics_file.walk('VEVENT'):
            dtstart_ical = event.get('dtstart')
            dtend_ical = event.get('dtend')
            
            # Step 4: Convert DTSTART and DTEND to user's home timezone
            if dtstart_ical is not None and dtend_ical is not None:
                # Check if dtstart and dtend are datetime objects
                if isinstance(dtstart_ical.dt, datetime):
                    dtstart = dtstart_ical.dt
                else:  # it's a date object, convert to datetime at midnight UTC
                    dtstart = pytz.utc.localize(datetime.combine(dtstart_ical.dt, datetime.min.time()))
                
                if isinstance(dtend_ical.dt, datetime):
                    dtend = dtend_ical.dt
                else:  # it's a date object, convert to datetime at midnight UTC
                    dtend = pytz.utc.localize(datetime.combine(dtend_ical.dt, datetime.min.time()))
                

                
                if dtstart.tzinfo is None:
                    dtstart = pytz.utc.localize(dtstart)
                if dtend.tzinfo is None:
                    dtend = pytz.utc.localize(dtend)
                
                dtstart_home_tz = dtstart.astimezone(home_timezone)
                dtend_home_tz = dtend.astimezone(home_timezone)
                
                # Step 5: Check if event falls on target_day
                if (target_day_start <= dtstart_home_tz < target_day_end) or (target_day_start <= dtend_home_tz < target_day_end):
                    # Step 6: Append event to list if it falls on target_day
                    events_on_target_day.append(event)
        
        # Step 7: Return the list of events
        return events_on_target_day, home_timezone
    
    except Exception as e:
        print(f'Error parsing ICS file: {e}')
        return None




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


def construct_gpt4_prompt(preferences, events, tasks, timezone):
    events_text = '\n'.join([
        f'{event.get("dtstart").dt.astimezone(timezone).strftime("%H:%M")}-'
        f'{event.get("dtend").dt.astimezone(timezone).strftime("%H:%M")}, '
        f'{event.get("summary")}'
        for event in events
    ])   
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

def create_ics(schedule_entries):
    logging.info("Schedule: {}".format(schedule_entries))
    cal = icalendar.Calendar()
    for entry in schedule_entries:
        # Split the entry string into components
        start_time_str, duration_str, task = entry.split(', ')
        
        # Parse the start time string to a datetime object
        start_time = datetime.strptime(start_time_str, '%H:%M').time()
        
        # Extract the number of minutes from the duration string
        duration_mins = int(re.search(r'(\d+)m', duration_str).group(1))
        duration = timedelta(minutes=duration_mins)
        
        # Assuming the date is today, adjust as necessary
        today = date.today()
        start_datetime = datetime.combine(today, start_time)
        end_datetime = start_datetime + duration

        event = icalendar.Event()
        event.add('summary', task)
        event.add('dtstart', start_datetime)
        event.add('dtend', end_datetime)
        cal.add_component(event)

    return cal.to_ical().decode()


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
        if gpt4_response is None:
            app.logger.error('Error interacting with GPT-4')
            return jsonify(error='Error interacting with GPT-4'), 500

        # Update the dialogue history in the session
        session[session_id]['gpt4_dialogue'] = existing_dialogue

        # Parse GPT-4 response to extract the new schedule
        # Assuming parse_gpt4_response is modified to return a schedule
        _, new_schedule, _ = parse_gpt4_response(gpt4_response)

        if not new_schedule:
            return jsonify(error='Error generating new schedule'), 500

        # Store the new schedule in the session data
        session[session_id]['parsed_schedule'] = new_schedule

        return jsonify(status='success', schedule=new_schedule, session_id=session_id), 200
    except Exception as e:
        app.logger.error(f'Error in /submit_answers: {e}')
        return jsonify(error=str(e)), 500


@app.route('/generate_schedule', methods=['POST'])
def generate_schedule():
    logging.info('Entering generate_schedule endpoint')
    data = request.get_json()  # Get JSON data from request
    session_id = request.headers.get('X-Session-ID')
    if session_id not in session:
        return jsonify(error='Invalid session ID'), 400
    logging.info(f'session keys = {list(session.keys())}')
    # Decode Base64 file
    ics_file_content_base64 = data.get('ics_file', '')
    ics_file_content = base64.b64decode(ics_file_content_base64).decode('utf-8')
    if not ics_file_content:
        return jsonify(error='No ICS file uploaded'), 400
    tasks_text = data.get('tasks_text', '')
    preferences = data.get('preferences', {})
    target_day = data.get('target_day', '')
    logging.info(f'targetday = {target_day}')
    if not target_day:
        return jsonify(error='No target day specified'), 400
    events, timezone = parse_ics(ics_file_content, target_day)
    if events is None:
        return jsonify(error='Error parsing ICS file'), 400
    if not validate_ics(ics_file_content):
        return jsonify(error='Invalid ICS file format'), 400
    tasks = parse_tasks_text(tasks_text)
    logging.info(f'Generate Schedule: Received request with session ID: {session_id}')
    session[session_id]['events'] = events
    session[session_id]['tasks'] = tasks
    session[session_id]['target_day'] = target_day
    session[session_id]['timezone'] = timezone
    initial_prompt = construct_gpt4_prompt(preferences, events, tasks, timezone)
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

    if session_id not in session or 'parsed_schedule' not in session[session_id]:
        return jsonify(error='No schedule found for this session ID'), 400

    schedule_lines = session[session_id]['parsed_schedule']
    ics_content = create_ics(schedule_lines)
    
    response = app.response_class(
        response=ics_content,
        mimetype='text/calendar',
        headers={'Content-Disposition': 'attachment;filename=schedule.ics'}
    )
    
    return response

if __name__ == '__main__':
    app.run(debug=True)
