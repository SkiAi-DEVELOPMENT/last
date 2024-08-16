import os
import json
import webbrowser
import yfinance as yf
import sympy as sp13
from datetime import datetime
import logging
import pyjokes
import nltk
from nltk.tokenize import sent_tokenize
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import pyttsx3
import speech_recognition as sr
from deep_translator import GoogleTranslator
import requests

# Initialize logging
logging.basicConfig(filename='marcus.log', level=logging.DEBUG, format='%(asctime)s:%(levelname)s:%(message)s')

# File to log spoken responses and user inputs
log_file = "spoken_responses.txt"

# Load GPT-Neo or GPT-J model and tokenizer
def load_model(model_name="EleutherAI/gpt-neo-2.7B"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return tokenizer, model

# Initialize speech recognition and synthesis
recognizer = sr.Recognizer()
tts_engine = pyttsx3.init()

# Initialize the translator
translator = GoogleTranslator(source='auto', target='en')

# File for storing business details in ERP
business_details_file = "business_details.json"
erp_system_used = False

# Function to open the site
def open_site():
    url = "https://skyaidevelopment01.wixsite.com/skyai-1"
    webbrowser.open(url)
    return f"Opening the site: {url}"

def log_user_input(input_text):
    with open(log_file, "a", encoding="utf-8") as file:
        file.write(f"User: {input_text}\n")

def speak(text):
    prefixed_text = f"Marcus AI says: {text}"
    with open(log_file, "a", encoding="utf-8") as file:
        file.write(prefixed_text + "\n")
    tts_engine.say(prefixed_text)
    tts_engine.runAndWait()

def greet_user():
    hour = datetime.now().hour
    greetings = {
        range(0, 12): "Good morning!",
        range(12, 18): "Good afternoon!",
        range(18, 21): "Good evening!",
        range(21, 24): "Good night!"
    }
    greeting = next(g for h, g in greetings.items() if hour in h)
    speak(greeting)

def get_time_info(info_type):
    now = datetime.now()
    info_map = {
        'time': now.strftime("%H:%M:%S"),
        'day': now.strftime("%A"),
        'date': now.strftime("%d %B %Y"),
        'year': now.strftime("%Y")
    }
    return info_map.get(info_type, "Please try again.")

def translate_text(text, dest_language):
    try:
        return GoogleTranslator(source='auto', target=dest_language).translate(text)
    except Exception as e:
        return f"Error: {str(e)}"

def recognize_user_speech():
    with sr.Microphone() as source:
        print("Listening...")
        try:
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
            command = recognizer.recognize_google(audio).lower()
            print(f"You said: {command}")
            return command
        except sr.UnknownValueError:
            print("Could not understand audio")
        except sr.RequestError:
            print("Could not request results; check your network connection")
        except sr.WaitTimeoutError:
            print("Listening timed out while waiting for phrase to start")
        return None

def handle_translation(command):
    if 'translate' in command:
        parts = command.split('into')
        if len(parts) == 2:
            text = parts[0].replace("translate", "").strip()
            dest_language = parts[1].strip().lower()
            return f"The translation is: {translate_text(text, dest_language)}"
        return "Please specify the text and target language."
    return "No translation command found."

def get_gpt_neo_summary(query, model_name="EleutherAI/gpt-neo-2.7B"):
    tokenizer, model = load_model(model_name)
    input_ids = tokenizer(f"Summarize: {query}", return_tensors='pt').input_ids

    # Generate the summary using the model
    output = model.generate(
        input_ids,
        max_length=100,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        top_k=50,
    )

    # Decode the output tokens into a string
    summary = tokenizer.decode(output[0], skip_special_tokens=True)
    return summary.strip()

def handle_command(command):
    commands = {
        'open youtube': lambda: webbrowser.open("https://www.youtube.com"),
        'open chrome': lambda: os.system("start chrome"),
        'exit': lambda: (speak("Exiting..."), True),
        'open site': lambda: speak(open_site()),
        'stock market update': lambda: speak('handle_stock_market_updat'()),
        'translate': lambda: speak(handle_translation(command)),
        'tell me a joke': lambda: speak(handle_joke()),
        'solve': lambda: speak(handle_math(command)),
        'summarize': lambda: speak(handle_summarization(command)),
        'analyze my dream': lambda: speak(handle_dream_analysis(command)),
        'search wikipedia for': lambda: speak((command)),  # Deprecated
        'start erp': lambda: start_erp_system(),
        'start fintech': lambda: start_fintech_system(),
        'start personal ai': lambda: start_personal_ai(),
        'start healthcare robot': lambda: start_healthcare_robot(),
        'weather update': lambda: speak(handle_weather_update()),
        'news update': lambda: speak(handle_news_update())
    }

    for keyword, handler in commands.items():
        if keyword in command:
            return handler()
    
    # Default action for general queries
    response = handle_general_query(command)
    speak(response)
    return False

# ERP Functions
def save_business_details(details):
    with open(business_details_file, "w") as file:
        json.dump(details, file)

def load_business_details():
    if os.path.exists(business_details_file):
        with open(business_details_file, "r") as file:
            return json.load(file)
    return None

def get_business_details():
    details = load_business_details()
    if details:
        return details
    
    details = {}
    speak("Starting ERP system. This is your first time using the ERP system. Please provide the following details about your business.")
    details['name'] = input("What is the name of your business?\n")
    details['phone'] = input("What is the phone number of your business?\n")
    details['address'] = input("What is the address of your business?\n")
    details['type'] = input("What type of business is it (e.g., retail, manufacturing, etc.)?\n")
    details['employees'] = input("How many employees does your business have?\n")
    details['additional_info'] = input("Please describe any additional relevant information about your business.\n")
    
    save_business_details(details)
    return details

def handle_erp():
    business_details = get_business_details()
    speak("Thank you for providing the details. The ERP system is now started. What would you like to do next?")

def start_erp_system():
    global erp_system_used
    if not erp_system_used:
        handle_erp()
        erp_system_used = True
    else:
        speak("ERP system is already in use. What would you like to do next?")

    while True:
        erp_command = recognize_user_speech()
        if erp_command:
            if handle_erp_command(erp_command):
                break

def handle_erp_command(command):
    erp_commands = {
        'manage employee records': lambda: speak("Managing employee records."),
        'handle recruitment': lambda: speak("Handling recruitment."),
        'manage performance reviews': lambda: speak("Managing performance reviews."),
        'track attendance': lambda: speak("Tracking attendance."),
        'manage financial records': lambda: speak("Managing financial records."),
        'generate financial reports': lambda: speak("Generating financial reports."),
        'track expenses': lambda: speak("Tracking expenses."),
        'create budgets': lambda: speak("Creating budgets."),
        'manage customer data': lambda: speak("Managing customer data."),
        'track customer interactions': lambda: speak("Tracking customer interactions."),
        'handle customer support': lambda: speak("Handling customer support."),
        'manage projects': lambda: speak("Managing projects."),
        'assign tasks': lambda: speak("Assigning tasks."),
        'set deadlines': lambda: speak("Setting deadlines."),
        'track milestones': lambda: speak("Tracking milestones."),
        'manage supply chain': lambda: speak("Managing supply chain."),
        'track shipments': lambda: speak("Tracking shipments."),
        'manage vendors': lambda: speak("Managing vendors."),
        'perform business analytics': lambda: speak("Performing business analytics."),
        'generate dashboards': lambda: speak("Generating dashboards."),
        'predict trends': lambda: speak("Predicting trends."),
        'manage documents': lambda: speak("Managing documents."),
        'track approvals': lambda: speak("Tracking approvals."),
        'archive records': lambda: speak("Archiving records."),
        'handle compliance': lambda: speak("Handling compliance."),
        'open site': lambda: speak(open_site())
    }

    for keyword, handler in erp_commands.items():
        if keyword in command:
            return handler()

    speak("Command not recognized. Please try again.")
    return False

# Financial Tech Functions
def start_fintech_system():
    speak("Starting FinTech system. What would you like to do next?")

def handle_fintech_command(command):
    fintech_commands = {
        'manage investments': lambda: speak("Managing investments."),
        'track portfolio': lambda: speak("Tracking portfolio."),
        'analyze market trends': lambda: speak("Analyzing market trends."),
        'generate investment reports': lambda: speak("Generating investment reports."),
        'provide financial advice': lambda: speak("Providing financial advice."),
        'monitor stocks': lambda: speak("Monitoring stocks."),
        'open site': lambda: speak(open_site())
    }

    for keyword, handler in fintech_commands.items():
        if keyword in command:
            return handler()

    speak("Command not recognized. Please try again.")
    return False

# Personal AI Functions
def start_personal_ai():
    speak("Starting Personal AI system. How can I assist you today?")

def handle_personal_ai_command(command):
    personal_ai_commands = {
        'set reminders': lambda: speak("Setting reminders."),
        'manage calendar': lambda: speak("Managing calendar."),
        'provide daily tips': lambda: speak("Providing daily tips."),
        'organize tasks': lambda: speak("Organizing tasks."),
        'track goals': lambda: speak("Tracking goals."),
        'offer productivity suggestions': lambda: speak("Offering productivity suggestions."),
        'open site': lambda: speak(open_site())
    }

    for keyword, handler in personal_ai_commands.items():
        if keyword in command:
            return handler()

    speak("Command not recognized. Please try again.")
    return False

# Healthcare Robot Functions
def start_healthcare_robot():
    speak("Starting Healthcare Robot system. How can I assist you today?")

def handle_healthcare_robot_command(command):
    healthcare_robot_commands = {
        'track health metrics': lambda: speak("Tracking health metrics."),
        'manage medication': lambda: speak("Managing medication."),
        'provide health tips': lambda: speak("Providing health tips."),
        'schedule doctor appointments': lambda: speak("Scheduling doctor appointments."),
        'monitor fitness': lambda: speak("Monitoring fitness."),
        'offer wellness advice': lambda: speak("Offering wellness advice."),
        'open site': lambda: speak(open_site())
    }

    for keyword, handler in healthcare_robot_commands.items():
        if keyword in command:
            return handler()

    speak("Command not recognized. Please try again.")
    return False

# Joke Function
def handle_joke():
    return pyjokes.get_joke()

# Math Function
def handle_math(command):
    try:
        expression = command.replace("solve", "").strip()
        result = sp.sympify(expression)
        return f"The result is: {result}"
    except Exception as e:
        return f"Error solving math problem: {e}"

# Summarization Function
def handle_summarization(command):
    query = command.replace("summarize", "").strip()
    return get_gpt_neo_summary(query)

# Dream Analysis Function
def handle_dream_analysis(command):
    query = command.replace("analyze my dream", "").strip()
    return f"Analyzing your dream: {query}. Insights coming soon!"

# Weather Update Function
def handle_weather_update():
    city = "Delhi"
    api_key = "your_openweathermap_api_key"
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    response = requests.get(url)
    data = response.json()
    if data["cod"] == 200:
        weather = data["weather"][0]["description"]
        temp = data["main"]["temp"]
        return f"The current weather in {city} is {weather} with a temperature of {temp}°C."
    return "Could not retrieve weather data."

# News Update Function
def handle_news_update():
    api_key = "your_newsapi_api_key"
    url = f"https://newsapi.org/v2/top-headlines?country=in&apiKey={api_key}"
    response = requests.get(url)
    data = response.json()
    if data["status"] == "ok":
        articles = data["articles"]
        headlines = [article["title"] for article in articles[:5]]
        return "Here are the top news headlines: " + ", ".join(headlines)
    return "Could not retrieve news data."

def handle_general_query(query):
    # Placeholder for a general AI response or processing
    return "I'm not sure how to respond to that. Please try a different command."

def main():
    greet_user()
    speak("How can I assist you today?")

    while True:
        command = recognize_user_speech()
        if command:
            exit_program = handle_command(command)
            if exit_program:
                break

if __name__ == "__main__":
    main()
