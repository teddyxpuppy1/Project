# app.py
from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
import re
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import io
import base64
from datetime import datetime, timedelta
import matplotlib.dates as mdates
import seaborn as sns
import json
from werkzeug.security import generate_password_hash, check_password_hash
from train_diet_model import train_and_save_diet_model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from werkzeug.utils import secure_filename
import cv2
import numpy as np
from bson.objectid import ObjectId
from bson.errors import InvalidId
from flask_cors import CORS
# Handle dotenv import with try-except to make it optional
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # If dotenv is not installed, continue without it
    pass
app = Flask(__name__)
FRONTEND_URL = os.getenv("FRONTEND_URL", "*")  # set FRONTEND_URL in Render to https://your-frontend.vercel.app
CORS(app, resources={r"/*": {"origins": FRONTEND_URL}})


# Secret key for session
app.secret_key = os.getenv('SECRET_KEY', 'your-secret-key')

# Configure upload folder for meal images
UPLOAD_FOLDER = 'static/uploads/meals'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# ----------------- MongoDB Atlas Configuration -----------------
try:
    from pymongo import MongoClient
    from pymongo.errors import ConnectionFailure, PyMongoError

    # Get MongoDB URI from environment variable (set this in Render/your .env)
    MONGODB_URI = os.getenv("MONGODB_URI")

    if not MONGODB_URI:
        raise ValueError("MONGODB_URI not set in environment variables")

    # Initialize MongoDB client
    client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)

    # Test the connection (ping instead of deprecated ismaster)
    client.admin.command("ping")

    # Get database
    db = client["fitness_tracker"]

    # Collections
    users_collection = db["users"]
    user_info_collection = db["user_info"]
    workouts_collection = db["workouts"]
    meals_collection = db["meals"]
    hydration_settings_collection = db["hydration_settings"]
    water_logs_collection = db["water_logs"]

    db_available = True
    print("✅ Connected to MongoDB Atlas successfully!")

except ImportError:
    print("⚠️ pymongo not installed. Database functionality will not work.")
    client = None
    db = None
    db_available = False
except (ConnectionFailure, PyMongoError) as e:
    print(f"⚠️ Could not connect to MongoDB: {e}")
    client = None
    db = None
    db_available = False
except Exception as e:
    print(f"⚠️ Database configuration error: {e}")
    client = None
    db = None
    db_available = False


# Helper function to check if file extension is allowed
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Load food database for calorie estimation
food_db = pd.read_csv('calories.csv')

# Simplified food categories for image classification
FOOD_CATEGORIES = [
    'salad', 'pizza', 'pasta', 'burger', 'sandwich', 'soup', 
    'rice_dish', 'meat', 'fish', 'fruit', 'dessert', 'breakfast'
]

# Mock function to simulate food image classification model
def classify_food_image(image_path):
    try:
        # Pretend to analyze the image
        img = cv2.imread(image_path)
        if img is None:
            return None, "Image could not be read"
        
        # Simulate classification result
        import random
        food_category = random.choice(FOOD_CATEGORIES)
        confidence = random.uniform(0.7, 0.95)
        
        # Get a matching food item from our database
        matching_foods = food_db[food_db['Food_Item'].str.contains(food_category, case=False)]
        
        if not matching_foods.empty:
            food_item = matching_foods.iloc[0]
            result = {
                'food_name': food_item['Food_Item'],
                'calories': float(food_item['Calories_per_100g']),
                'protein': round(float(food_item['Calories_per_100g']) * 0.15 / 4, 1),
                'carbs': round(float(food_item['Calories_per_100g']) * 0.55 / 4, 1),
                'fat': round(float(food_item['Calories_per_100g']) * 0.3 / 9, 1),
                'confidence': round(confidence * 100, 1)
            }
            return result, None
        else:
            calories = random.randint(200, 600)
            result = {
                'food_name': food_category.replace('_', ' ').title(),
                'calories': calories,
                'protein': round(calories * 0.15 / 4, 1),
                'carbs': round(calories * 0.55 / 4, 1),
                'fat': round(calories * 0.3 / 9, 1),
                'confidence': round(confidence * 100, 1)
            }
            return result, None
            
    except Exception as e:
        return None, f"Error analyzing image: {str(e)}"

@app.route('/train_diet_model', methods=['GET'])
def train_diet_route():
    if 'loggedin' not in session:
        flash("Please log in to train model", 'danger')
        return redirect(url_for('login'))

    success, message = train_and_save_diet_model()
    flash(message, 'success' if success else 'danger')
    return render_template('train_diet.html', success=success, message=message)

# Route for home page/redirects
@app.route('/')
def index():
    if 'loggedin' in session:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

def train_and_save_model():
    df = pd.read_csv('fitness_data_1000_records.csv')
    df.columns = [col.strip().lower() for col in df.columns]

    print("✅ Cleaned Columns:", df.columns.tolist())

    df['days_to_goal'] = abs(df['currentweight'] - df['dreamweight']) * 10 + np.random.randint(10, 50, df.shape[0])

    features = ['currentweight', 'dreamweight']
    X = df[features]
    y = df['days_to_goal']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    joblib.dump(model, 'model.pkl')
    print("✅ Model trained and saved.")

# Load or train model
if not os.path.exists('model.pkl'):
    train_and_save_model()

model = joblib.load('model.pkl')

@app.route('/model1')
def model1():
    return render_template('model.html')

# Login route
@app.route('/login', methods=['GET', 'POST'])
def login():
    msg = ''
    if request.method == 'POST' and db_available:
        try:
            email = request.form['login-email']
            password = request.form['login-password']
            remember = 'remember' in request.form
            
            # Check user exists in MongoDB
            user = users_collection.find_one({'email': email})
            
            if user and check_password_hash(user['password'], password):
                session['loggedin'] = True
                session['id'] = str(user['_id'])
                session['name'] = user['name']
                session['email'] = user['email']
                
                if remember:
                    session.permanent = True
                    
                flash('Login successful!', 'success')
                return redirect(url_for('dashboard'))
            else:
                flash('Invalid email or password!', 'danger')
        except Exception as e:
            flash(f'An error occurred. Please try again later.', 'danger')
            print(f"Login error: {e}")
            
    return render_template('auth.html')

cal = pd.read_csv('calories.csv')

@app.route('/call', methods=['GET', 'POST'])
def call():
    if request.method == 'POST':
        food_items = request.form.getlist('food_items')
        total_calories = sum(cal.loc[cal['Food_Item'].isin(food_items), 'Calories_per_100g'])
        return render_template('index.html', total_calories=total_calories, food_items=food_items)
    return render_template('index.html', total_calories=0, food_items=[])

# Test route for MongoDB connection
@app.route('/test-db')
def test_db():
    try:
        if not db_available:
            return "Database not available - check your MongoDB connection"
        
        # Test connection
        client.admin.command('ismaster')
        
        # Test collection access
        count = users_collection.count_documents({})
        
        return f"MongoDB connection successful! Users in database: {count}"
    except Exception as e:
        return f"Database error: {str(e)}"

# Signup route
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        try:
            # Debug: Print form data
            print("Form data received:", dict(request.form))
            
            # Check if database is available
            if not db_available:
                flash('Database is not available. Please check your MongoDB connection.', 'danger')
                return redirect(url_for('login'))
            
            # Get form data with error checking
            name = request.form.get('signup-name', '').strip()
            email = request.form.get('signup-email', '').strip()
            password = request.form.get('signup-password', '')
            confirm_password = request.form.get('signup-confirm', '')
            
            print(f"Parsed data - Name: {name}, Email: {email}, Password length: {len(password)}")
            
            # Validation checks
            if not name:
                flash('Please enter your name!', 'danger')
            elif not email:
                flash('Please enter your email!', 'danger')
            elif not re.match(r'[^@]+@[^@]+\.[^@]+', email):
                flash('Invalid email address!', 'danger')
            elif not password:
                flash('Please enter a password!', 'danger')
            elif len(password) < 6:
                flash('Password must be at least 6 characters!', 'danger')
            elif password != confirm_password:
                flash('Passwords do not match!', 'danger')
            else:
                # Check if email already exists
                existing_user = users_collection.find_one({'email': email})
                print(f"Existing user check: {existing_user is not None}")
                
                if existing_user:
                    flash('Email already exists!', 'danger')
                else:
                    # Hash password and create user
                    hashed_password = generate_password_hash(password)
                    
                    user_doc = {
                        'name': name,
                        'email': email,
                        'password': hashed_password,
                        'created_at': datetime.now()
                    }
                    
                    # Insert new user into MongoDB
                    result = users_collection.insert_one(user_doc)
                    print(f"User insertion result: {result.inserted_id}")
                    
                    if result.inserted_id:
                        flash('Registration successful! Please login.', 'success')
                        return redirect(url_for('login'))
                    else:
                        flash('Failed to create account. Please try again.', 'danger')
                        
        except Exception as e:
            flash('An error occurred during registration. Please try again.', 'danger')
            print(f"Signup error: {e}")
            import traceback
            traceback.print_exc()
            
    elif request.method == 'POST' and not db_available:
        flash('Database connection is not available. Please check your MongoDB setup.', 'danger')
            
    return redirect(url_for('login'))

# Dashboard route - protected
@app.route('/dashboard')
def dashboard():
    if 'loggedin' not in session:
        flash('Please login first!', 'danger')
        return redirect(url_for('login'))

    user_id = session['id']
    info = {
        'gender': '',
        'age': '',
        'height': '',
        'weight': '',
        'dream_weight': '',
        'goal': ''
    }

    # Fetch user info from MongoDB
    if db_available:
        try:
            user_info = user_info_collection.find_one({'user_id': user_id})
            if user_info:
                info = {
                    'gender': user_info.get('gender', ''),
                    'age': user_info.get('age', ''),
                    'height': user_info.get('height', ''),
                    'weight': user_info.get('weight', ''),
                    'dream_weight': user_info.get('dream_weight', ''),
                    'goal': user_info.get('goal', '')
                }
        except Exception as e:
            print(f"Dashboard error: {e}")

    # Fetch stats
    stats = {}
    try:
        stats['total_workouts'] = workouts_collection.count_documents({'user_id': user_id})
        
        # Calculate total duration and calories
        pipeline = [
            {'$match': {'user_id': user_id}},
            {'$group': {
                '_id': None,
                'total_duration': {'$sum': '$duration'},
                'total_calories': {'$sum': '$calories_burned'}
            }}
        ]
        result = list(workouts_collection.aggregate(pipeline))
        if result:
            stats['total_duration'] = result[0].get('total_duration', 0)
            stats['total_calories'] = result[0].get('total_calories', 0)
        else:
            stats['total_duration'] = 0
            stats['total_calories'] = 0
        
        # Get favorite exercise
        pipeline = [
            {'$match': {'user_id': user_id}},
            {'$group': {'_id': '$exercise_type', 'count': {'$sum': 1}}},
            {'$sort': {'count': -1}},
            {'$limit': 1}
        ]
        result = list(workouts_collection.aggregate(pipeline))
        stats['favorite_exercise'] = result[0]['_id'] if result else 'None'
        
    except Exception as e:
        print(f"Error fetching stats: {e}")
        stats = {'total_workouts': 0, 'total_duration': 0, 'total_calories': 0, 'favorite_exercise': 'None'}

    # Get graphs
    graphs = generate_workout_graphs(user_id)

    return render_template('dashboard.html',
                           name=session.get('name', 'User'),
                           info=info,
                           stats=stats,
                           graphs=graphs)

@app.route('/settings')
def settings():
    if 'loggedin' not in session:
        flash('Please login first!', 'danger')
        return redirect(url_for('login'))
    
    user_id = session['id']
    user_info = None
    
    if db_available:
        try:
            # Get user from users collection
            user = users_collection.find_one({'_id': ObjectId(user_id)})
            
            if user:
                user_info = {
                    'id': str(user['_id']),
                    'name': user['name'],
                    'email': user['email']
                }
                
                # Get profile info if available
                user_profile = user_info_collection.find_one({'user_id': user_id})
                if user_profile:
                    user_info.update({
                        'gender': user_profile.get('gender', ''),
                        'age': user_profile.get('age', ''),
                        'height': user_profile.get('height', ''),
                        'weight': user_profile.get('weight', ''),
                        'dream_weight': user_profile.get('dream_weight', ''),
                        'goal': user_profile.get('goal', '')
                    })
                    
        except Exception as e:
            flash('An error occurred while loading settings.', 'danger')
            print(f"Settings error: {e}")
    
    return render_template('settings.html', user=user_info)

@app.route('/save_user_info', methods=['POST'])
def save_user_info():
    if 'loggedin' not in session:
        flash('Please login first!', 'danger')
        return redirect(url_for('login'))

    if not db_available:
        flash('Database functionality is not available.', 'danger')
        return redirect(url_for('dashboard'))

    try:
        user_id = session['id']
        gender = request.form['gender']
        age = int(request.form['age'])
        height = float(request.form['height'])
        weight = float(request.form['weight'])
        dream_weight = float(request.form['dream_weight'])
        goal = request.form['goal']

        print("✅ Saving user info:")
        print("User ID:", user_id)
        print("Gender:", gender)
        print("Age:", age)
        print("Height:", height)
        print("Weight:", weight)
        print("Dream Weight:", dream_weight)
        print("Goal:", goal)

        # Check if the user already has info saved
        existing = user_info_collection.find_one({'user_id': user_id})

        user_info_doc = {
            'user_id': user_id,
            'gender': gender,
            'age': age,
            'height': height,
            'weight': weight,
            'dream_weight': dream_weight,
            'goal': goal,
            'updated_at': datetime.now()
        }

        if existing:
            # Update existing record
            user_info_collection.update_one(
                {'user_id': user_id},
                {'$set': user_info_doc}
            )
            print("✅ Updated existing user info.")
        else:
            # Insert new record
            user_info_doc['created_at'] = datetime.now()
            user_info_collection.insert_one(user_info_doc)
            print("✅ Inserted new user info.")

        flash('Your fitness profile has been updated.', 'success')
    except Exception as e:
        flash('An error occurred while saving your information.', 'danger')
        print(f"❌ Save user info error: {e}")

    return redirect(url_for('dashboard'))

@app.route('/predict/<user_id>')
def predict(user_id):
    try:
        user_info = user_info_collection.find_one({'user_id': user_id})
        
        if not user_info:
            return "❌ User not found", 404

        currentweight = float(user_info['weight'])
        dreamweight = float(user_info['dream_weight'])
        height = float(user_info['height'])
        age = int(user_info['age'])
        
        prediction = model.predict([[currentweight, dreamweight]])[0]

        # Load dataset
        df = pd.read_csv('fitness_data_1000_records.csv')
        df.columns = [col.strip().lower() for col in df.columns]

        # Find closest match
        df['distance'] = (df['currentweight'] - currentweight).abs() + (df['dreamweight'] - dreamweight).abs()
        match = df.loc[df['distance'].idxmin()]

        # Extract weekly schedule if columns exist
        weekdays = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
        workout_schedule = {}

        for day in weekdays:
            if day in match:
                workout_schedule[day] = match[day]
            else:
                workout_schedule[day] = "Rest"

        return render_template('result.html',
                               actual=currentweight,
                               dream=dreamweight,
                               height=height,
                               age=age,
                               prediction=round(prediction, 2),
                               schedule=workout_schedule)
    except Exception as e:
        print(f"Prediction error: {e}")
        return "Error occurred", 500

# Initialize products functionality
dff = pd.read_csv('Fitness_trackers_updated.csv')

print("Columns in CSV:", dff.columns.tolist())
dff.columns = dff.columns.str.strip()

for col in dff.columns:
    if 'price' in col.lower():
        dff.rename(columns={col: 'Price'}, inplace=True)
        break

dff['Price'] = (
    dff['Price']
    .astype(str)
    .str.replace(r'[^\d.]', '', regex=True)
    .astype(float)
)

@app.route('/prod', methods=['GET', 'POST'])
def prod():
    products = []
    if request.method == 'POST':
        try:
            min_price = float(request.form['min_price'])
            max_price = float(request.form['max_price'])
            
            print("DataFrame columns:", dff.columns.tolist())
            
            if 'Price' in dff.columns:
                dff['Price'] = pd.to_numeric(dff['Price'], errors='coerce')
                print("Price column sample:", dff['Price'].head())
                
                filtered = dff[(dff['Price'] >= min_price) & (dff['Price'] <= max_price)]
                print(f"Found {len(filtered)} products in range ${min_price}-${max_price}")
                
                products = filtered.to_dict(orient='records')
                
                if len(products) > 0:
                    print("First product:", products[0])
            else:
                print("ERROR: 'Price' column not found!")
                
        except Exception as e:
            print("Error processing request:", e)
            import traceback
            traceback.print_exc()

    return render_template('productsss.html', products=products)

@app.route('/logout')
def logout():
    session.pop('loggedin', None)
    session.pop('id', None)
    session.pop('name', None)
    session.pop('email', None)
    
    flash('You have been logged out!', 'info')
    return redirect(url_for('login'))

@app.route('/forgot-password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST' and db_available:
        try:
            email = request.form['email']
            
            user = users_collection.find_one({'email': email})
            
            if user:
                flash('Password reset link has been sent to your email!', 'info')
            else:
                flash('Email not found!', 'danger')
        except Exception as e:
            flash('An error occurred. Please try again later.', 'danger')
            print(f"Forgot password error: {e}")
            
    return render_template('forgot_password.html')

# Diet functionality
def train_diet_model():
    df = pd.read_csv('fitness_diet_dataset.csv')

    df.rename(columns={
        'current_weight': 'weight',
        'desired_weight': 'dreamweight',
        'diet_type': 'diet'
    }, inplace=True)

    df['gender'] = 'male'
    df['age'] = 25
    df['height'] = 170
    df['goal'] = 'cut'

    X = df[['gender', 'age', 'height', 'weight', 'dreamweight', 'goal']]
    X = pd.get_dummies(X)

    y = df['diet']

    model = DecisionTreeClassifier()
    model.fit(X, y)

    joblib.dump(model, 'diet_model.pkl')
    print("✅ Model trained and saved as diet_model.pkl")

def predict_diet(user_data):
    model = joblib.load('diet_model.pkl')
    input_df = pd.DataFrame([user_data])
    input_df = pd.get_dummies(input_df)

    model_columns = model.feature_names_in_
    for col in model_columns:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[model_columns]

    prediction = model.predict(input_df)
    return prediction[0]

@app.route('/diet')
def diet():
    if 'loggedin' not in session:
        flash("Please login first", "warning")
        return redirect(url_for('login'))

    user_id = session['id']
    try:
        user_info = user_info_collection.find_one({'user_id': user_id})

        if not user_info:
            # If no user profile, show message instead of error
            flash("Please enter your details first", "info")
            return render_template('diet.html', diet=None, user=None)

        # If user exists, prepare data
        user_data = {
            'gender': user_info.get('gender', 'male'),
            'age': user_info.get('age', 25),
            'height': user_info.get('height', 170),
            'weight': user_info.get('weight', 70),
            'dreamweight': user_info.get('dream_weight', 65),
            'goal': user_info.get('goal', 'cut')
        }

        if not os.path.exists('diet_model.pkl'):
            train_diet_model()

        diet_result = predict_diet(user_data)
        return render_template('diet.html', diet=diet_result, user=user_data)

    except Exception as e:
        print("❌ Error:", e)
        return render_template('diet.html', diet=None, user=None, error="Something went wrong.")


def get_user_info(user_id):
    if not db_available:
        return None
        
    try:
        user_info = user_info_collection.find_one({'user_id': user_id})
        if user_info:
            return {
                'gender': user_info.get('gender', ''),
                'age': user_info.get('age', ''),
                'height': user_info.get('height', ''),
                'weight': user_info.get('weight', ''),
                'dream_weight': user_info.get('dream_weight', ''),
                'goal': user_info.get('goal', '')
            }
    except Exception as e:
        print(f"Error fetching user info: {e}")
    return None

@app.route('/add_workout', methods=['GET', 'POST'])
def add_workout():
    if 'loggedin' not in session:
        flash('Please login first!', 'danger')
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        try:
            user_id = session['id']
            workout_date = datetime.strptime(request.form['workout_date'], '%Y-%m-%d')
            exercise_type = request.form['exercise_type']
            duration = int(request.form['duration'])
            calories_burned = int(request.form.get('calories_burned', 0))
            weight = float(request.form.get('weight', 0))
            sets = int(request.form.get('sets', 0))
            reps = int(request.form.get('reps', 0))
            notes = request.form.get('notes', '')
            
            workout_doc = {
                'user_id': user_id,
                'workout_date': workout_date,
                'exercise_type': exercise_type,
                'duration': duration,
                'calories_burned': calories_burned,
                'weight': weight,
                'sets': sets,
                'reps': reps,
                'notes': notes,
                'created_at': datetime.now()
            }
            
            workouts_collection.insert_one(workout_doc)
            
            flash('Workout added successfully!', 'success')
            return redirect(url_for('workout_history'))
            
        except Exception as e:
            flash(f'An error occurred: {str(e)}', 'danger')
            print(f"Add workout error: {e}")
    
    return render_template('add_workout.html')

@app.route('/workout_history')
def workout_history():
    if 'loggedin' not in session:
        flash('Please login first!', 'danger')
        return redirect(url_for('login'))
    
    try:
        user_id = session['id']
        workouts = list(workouts_collection.find({'user_id': user_id}).sort('workout_date', -1))
        
        workout_list = []
        for workout in workouts:
            workout_list.append({
                'id': str(workout['_id']),
                'date': workout['workout_date'].strftime('%Y-%m-%d'),
                'exercise': workout['exercise_type'],
                'duration': workout['duration'],
                'calories': workout['calories_burned'],
                'weight': workout['weight'],
                'sets': workout['sets'],
                'reps': workout['reps'],
                'notes': workout['notes']
            })
        
        return render_template('workout_history.html', workouts=workout_list)
        
    except Exception as e:
        flash(f'An error occurred: {str(e)}', 'danger')
        print(f"Workout history error: {e}")
        return render_template('workout_history.html', workouts=[])

def generate_workout_graphs(user_id):
    graphs = {}
    
    try:
        thirty_days_ago = datetime.now() - timedelta(days=30)
        
        # Graph 1: Calories burned over time
        pipeline = [
            {'$match': {
                'user_id': user_id,
                'workout_date': {'$gte': thirty_days_ago}
            }},
            {'$group': {
                '_id': '$workout_date',
                'total_calories': {'$sum': '$calories_burned'}
            }},
            {'$sort': {'_id': 1}}
        ]
        
        calories_data = list(workouts_collection.aggregate(pipeline))
        
        if calories_data:
            dates = [item['_id'] for item in calories_data]
            calories = [item['total_calories'] for item in calories_data]
            
            plt.figure(figsize=(10, 6))
            plt.plot(dates, calories, marker='o', linestyle='-', color='#FF5722')
            plt.fill_between(dates, calories, alpha=0.2, color='#FF5722')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.title('Calories Burned Over Time', fontsize=16)
            plt.xlabel('Date', fontsize=12)
            plt.ylabel('Calories', fontsize=12)
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png')
            buffer.seek(0)
            image_png = buffer.getvalue()
            buffer.close()
            graphic = base64.b64encode(image_png).decode('utf-8')
            graphs['calories_chart'] = graphic
            plt.close()
        
        # Graph 2: Workout duration by exercise type
        pipeline = [
            {'$match': {
                'user_id': user_id,
                'workout_date': {'$gte': thirty_days_ago}
            }},
            {'$group': {
                '_id': '$exercise_type',
                'total_duration': {'$sum': '$duration'}
            }},
            {'$sort': {'total_duration': -1}},
            {'$limit': 5}
        ]
        
        duration_data = list(workouts_collection.aggregate(pipeline))
        
        if duration_data:
            exercise_types = [item['_id'] for item in duration_data]
            durations = [item['total_duration'] for item in duration_data]
            
            plt.figure(figsize=(10, 6))
            bars = plt.bar(exercise_types, durations, color='#2196F3')
            plt.title('Workout Duration by Exercise Type', fontsize=16)
            plt.xlabel('Exercise Type', fontsize=12)
            plt.ylabel('Total Duration (minutes)', fontsize=12)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                        f'{int(height)}',
                        ha='center', va='bottom', fontsize=10)
            
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png')
            buffer.seek(0)
            image_png = buffer.getvalue()
            buffer.close()
            graphic = base64.b64encode(image_png).decode('utf-8')
            graphs['duration_chart'] = graphic
            plt.close()
        
        # Graph 3: Workout frequency by day of week
        pipeline = [
            {'$match': {
                'user_id': user_id,
                'workout_date': {'$gte': thirty_days_ago}
            }},
            {'$addFields': {
                'day_of_week': {'$dayOfWeek': '$workout_date'}
            }},
            {'$group': {
                '_id': '$day_of_week',
                'count': {'$sum': 1}
            }},
            {'$sort': {'_id': 1}}
        ]
        
        frequency_data = list(workouts_collection.aggregate(pipeline))
        
        if frequency_data:
            day_names = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
            days = [day_names[item['_id'] - 1] for item in frequency_data]
            counts = [item['count'] for item in frequency_data]
            
            plt.figure(figsize=(10, 6))
            colors = sns.color_palette('viridis', len(days))
            plt.pie(counts, labels=days, autopct='%1.1f%%', startangle=90, colors=colors)
            plt.axis('equal')
            plt.title('Workout Frequency by Day of Week', fontsize=16)
            plt.tight_layout()
            
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png')
            buffer.seek(0)
            image_png = buffer.getvalue()
            buffer.close()
            graphic = base64.b64encode(image_png).decode('utf-8')
            graphs['frequency_chart'] = graphic
            plt.close()
        
    except Exception as e:
        print(f"Error generating workout graphs: {e}")
    
    return graphs

@app.route('/workout_dashboard')
def workout_dashboard():
    if 'loggedin' not in session:
        flash('Please login first!', 'danger')
        return redirect(url_for('login'))
    
    user_id = session['id']
    
    # Get workout statistics
    stats = {}
    try:
        stats['total_workouts'] = workouts_collection.count_documents({'user_id': user_id})
        
        # Aggregate total duration and calories
        pipeline = [
            {'$match': {'user_id': user_id}},
            {'$group': {
                '_id': None,
                'total_duration': {'$sum': '$duration'},
                'total_calories': {'$sum': '$calories_burned'}
            }}
        ]
        result = list(workouts_collection.aggregate(pipeline))
        if result:
            stats['total_duration'] = result[0].get('total_duration', 0)
            stats['total_calories'] = result[0].get('total_calories', 0)
        else:
            stats['total_duration'] = 0
            stats['total_calories'] = 0
        
        # Get favorite exercise type
        pipeline = [
            {'$match': {'user_id': user_id}},
            {'$group': {'_id': '$exercise_type', 'count': {'$sum': 1}}},
            {'$sort': {'count': -1}},
            {'$limit': 1}
        ]
        result = list(workouts_collection.aggregate(pipeline))
        stats['favorite_exercise'] = result[0]['_id'] if result else 'None'
        
    except Exception as e:
        print(f"Error fetching workout stats: {e}")
        stats = {'total_workouts': 0, 'total_duration': 0, 'total_calories': 0, 'favorite_exercise': 'None'}
    
    # Generate graphs
    graphs = generate_workout_graphs(user_id)
    
    return render_template('workout_dashboard.html', stats=stats, graphs=graphs)

@app.route('/delete_workout/<workout_id>', methods=['POST'])
def delete_workout(workout_id):
    if 'loggedin' not in session:
        flash('Please login first!', 'danger')
        return redirect(url_for('login'))
    
    try:
        user_id = session['id']
        
        # Verify the workout belongs to the user and delete
        result = workouts_collection.delete_one({
            '_id': ObjectId(workout_id),
            'user_id': user_id
        })
        
        if result.deleted_count > 0:
            flash('Workout deleted successfully!', 'success')
        else:
            flash('Workout not found or not authorized to delete!', 'danger')
            
    except InvalidId:
        flash('Invalid workout ID!', 'danger')
    except Exception as e:
        flash(f'An error occurred: {str(e)}', 'danger')
        print(f"Delete workout error: {e}")
    
    return redirect(url_for('workout_history'))

@app.route('/edit_workout/<workout_id>', methods=['GET', 'POST'])
def edit_workout(workout_id):
    if 'loggedin' not in session:
        flash('Please login first!', 'danger')
        return redirect(url_for('login'))
    
    try:
        user_id = session['id']
        
        if request.method == 'POST':
            # Update workout
            workout_date = datetime.strptime(request.form['workout_date'], '%Y-%m-%d')
            exercise_type = request.form['exercise_type']
            duration = int(request.form['duration'])
            calories_burned = int(request.form.get('calories_burned', 0))
            weight = float(request.form.get('weight', 0))
            sets = int(request.form.get('sets', 0))
            reps = int(request.form.get('reps', 0))
            notes = request.form.get('notes', '')
            
            # Update the workout
            result = workouts_collection.update_one(
                {
                    '_id': ObjectId(workout_id),
                    'user_id': user_id
                },
                {'$set': {
                    'workout_date': workout_date,
                    'exercise_type': exercise_type,
                    'duration': duration,
                    'calories_burned': calories_burned,
                    'weight': weight,
                    'sets': sets,
                    'reps': reps,
                    'notes': notes,
                    'updated_at': datetime.now()
                }}
            )
            
            if result.modified_count > 0:
                flash('Workout updated successfully!', 'success')
                return redirect(url_for('workout_history'))
            else:
                flash('Workout not found or not authorized to edit!', 'danger')
                return redirect(url_for('workout_history'))
        else:
            # GET request - show form with current values
            workout = workouts_collection.find_one({
                '_id': ObjectId(workout_id),
                'user_id': user_id
            })
            
            if workout:
                workout_data = {
                    'id': str(workout['_id']),
                    'date': workout['workout_date'].strftime('%Y-%m-%d'),
                    'exercise': workout['exercise_type'],
                    'duration': workout['duration'],
                    'calories': workout['calories_burned'],
                    'weight': workout['weight'],
                    'sets': workout['sets'],
                    'reps': workout['reps'],
                    'notes': workout['notes']
                }
                return render_template('edit_workout.html', workout=workout_data)
            else:
                flash('Workout not found or not authorized to edit!', 'danger')
                return redirect(url_for('workout_history'))
                
    except InvalidId:
        flash('Invalid workout ID!', 'danger')
        return redirect(url_for('workout_history'))
    except Exception as e:
        flash(f'An error occurred: {str(e)}', 'danger')
        print(f"Edit workout error: {e}")
        return redirect(url_for('workout_history'))

@app.route('/meal_tracker')
def meal_tracker():
    if 'loggedin' not in session:
        flash('Please login first!', 'danger')
        return redirect(url_for('login'))
    
    user_id = session['id']
    
    # Get all meal entries for the user
    meals = []
    try:
        meal_entries = list(meals_collection.find({'user_id': user_id}).sort('meal_date', -1))
        
        for meal in meal_entries:
            meals.append({
                'id': str(meal['_id']),
                'date': meal['meal_date'].strftime('%Y-%m-%d'),
                'type': meal['meal_type'],
                'food': meal['food_name'],
                'calories': meal['calories'],
                'protein': meal['protein'],
                'carbs': meal['carbs'],
                'fat': meal['fat'],
                'notes': meal['notes'],
                'image': meal.get('image_path', '')
            })
        
    except Exception as e:
        flash(f'An error occurred: {str(e)}', 'danger')
        print(f"Meal tracker error: {e}")
    
    return render_template('meal_tracker.html', meals=meals)

@app.route('/add_meal', methods=['GET', 'POST'])
def add_meal():
    if 'loggedin' not in session:
        flash('Please login first!', 'danger')
        return redirect(url_for('login'))

    predicted_data = {
        'food_name': '',
        'calories': '',
        'protein': '',
        'carbs': '',
        'fat': ''
    }

    if request.method == 'POST':
        try:
            user_id = session['id']
            meal_date = datetime.strptime(request.form['meal_date'], '%Y-%m-%d')
            meal_type = request.form['meal_type']
            food_name = request.form['food_name']
            calories = float(request.form['calories'])
            protein = float(request.form.get('protein', 0))
            carbs = float(request.form.get('carbs', 0))
            fat = float(request.form.get('fat', 0))
            notes = request.form.get('notes', '')

            image_path = None
            if 'meal_image' in request.files:
                file = request.files['meal_image']
                if file and file.filename != '' and allowed_file(file.filename):
                    filename = secure_filename(file.filename)
                    filename = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{filename}"
                    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    file.save(file_path)

                    image_path = f"uploads/meals/{filename}"

                    food_result, error = classify_food_image(file_path)
                    if food_result and not error:
                        predicted_data = {
                            'food_name': food_result['food_name'],
                            'calories': food_result['calories'],
                            'protein': food_result['protein'],
                            'carbs': food_result['carbs'],
                            'fat': food_result['fat']
                        }

                        flash('Food detected from image! Please confirm details below.', 'info')
                        return render_template('add_meal.html', predicted=predicted_data)

            # Save to MongoDB
            meal_doc = {
                'user_id': user_id,
                'meal_date': meal_date,
                'meal_type': meal_type,
                'food_name': food_name,
                'calories': calories,
                'protein': protein,
                'carbs': carbs,
                'fat': fat,
                'notes': notes,
                'image_path': image_path,
                'created_at': datetime.now()
            }
            
            meals_collection.insert_one(meal_doc)

            flash('Meal added successfully!', 'success')
            return redirect(url_for('meal_tracker'))

        except Exception as e:
            flash(f'An error occurred: {str(e)}', 'danger')
            print(f"Add meal error: {e}")

    return render_template('add_meal.html', predicted=predicted_data)

@app.route('/edit_meal/<meal_id>', methods=['GET', 'POST'])
def edit_meal(meal_id):
    if 'loggedin' not in session:
        flash('Please login first!', 'danger')
        return redirect(url_for('login'))
    
    try:
        user_id = session['id']
        
        if request.method == 'POST':
            # Update meal
            meal_date = datetime.strptime(request.form['meal_date'], '%Y-%m-%d')
            meal_type = request.form['meal_type']
            food_name = request.form['food_name']
            calories = float(request.form['calories'])
            protein = float(request.form.get('protein', 0))
            carbs = float(request.form.get('carbs', 0))
            fat = float(request.form.get('fat', 0))
            notes = request.form.get('notes', '')
            
            update_doc = {
                'meal_date': meal_date,
                'meal_type': meal_type,
                'food_name': food_name,
                'calories': calories,
                'protein': protein,
                'carbs': carbs,
                'fat': fat,
                'notes': notes,
                'updated_at': datetime.now()
            }
            
            # Handle image upload
            if 'meal_image' in request.files:
                file = request.files['meal_image']
                if file and file.filename != '' and allowed_file(file.filename):
                    filename = secure_filename(file.filename)
                    filename = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{filename}"
                    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    file.save(file_path)
                    
                    update_doc['image_path'] = f"uploads/meals/{filename}"
            
            # Update the meal
            result = meals_collection.update_one(
                {
                    '_id': ObjectId(meal_id),
                    'user_id': user_id
                },
                {'$set': update_doc}
            )
            
            if result.modified_count > 0:
                flash('Meal updated successfully!', 'success')
                return redirect(url_for('meal_tracker'))
            else:
                flash('Meal not found or not authorized to edit!', 'danger')
                return redirect(url_for('meal_tracker'))
        else:
            # GET request - show form with current values
            meal = meals_collection.find_one({
                '_id': ObjectId(meal_id),
                'user_id': user_id
            })
            
            if meal:
                meal_data = {
                    'id': str(meal['_id']),
                    'date': meal['meal_date'].strftime('%Y-%m-%d'),
                    'type': meal['meal_type'],
                    'food': meal['food_name'],
                    'calories': meal['calories'],
                    'protein': meal['protein'],
                    'carbs': meal['carbs'],
                    'fat': meal['fat'],
                    'notes': meal['notes'],
                    'image': meal.get('image_path', '')
                }
                return render_template('edit_meal.html', meal=meal_data)
            else:
                flash('Meal not found or not authorized to edit!', 'danger')
                return redirect(url_for('meal_tracker'))
                
    except InvalidId:
        flash('Invalid meal ID!', 'danger')
        return redirect(url_for('meal_tracker'))
    except Exception as e:
        flash(f'An error occurred: {str(e)}', 'danger')
        print(f"Edit meal error: {e}")
        return redirect(url_for('meal_tracker'))

@app.route('/delete_meal/<meal_id>', methods=['POST'])
def delete_meal(meal_id):
    if 'loggedin' not in session:
        flash('Please login first!', 'danger')
        return redirect(url_for('login'))
    
    try:
        user_id = session['id']
        
        # Get the meal first to check image path
        meal = meals_collection.find_one({
            '_id': ObjectId(meal_id),
            'user_id': user_id
        })
        
        if meal:
            # Delete the meal record
            result = meals_collection.delete_one({
                '_id': ObjectId(meal_id),
                'user_id': user_id
            })
            
            if result.deleted_count > 0:
                # Delete the image file if it exists
                image_path = meal.get('image_path')
                if image_path:
                    try:
                        full_path = os.path.join('static', image_path)
                        if os.path.exists(full_path):
                            os.remove(full_path)
                    except Exception as e:
                        print(f"Warning: Could not delete image file: {e}")
                
                flash('Meal deleted successfully!', 'success')
            else:
                flash('Meal not found or not authorized to delete!', 'danger')
        else:
            flash('Meal not found or not authorized to delete!', 'danger')
            
    except InvalidId:
        flash('Invalid meal ID!', 'danger')
    except Exception as e:
        flash(f'An error occurred: {str(e)}', 'danger')
        print(f"Delete meal error: {e}")
    
    return redirect(url_for('meal_tracker'))

@app.route('/meal_analysis')
def meal_analysis():
    if 'loggedin' not in session:
        flash('Please login first!', 'danger')
        return redirect(url_for('login'))
    
    user_id = session['id']
    nutrition_data = {}
    
    try:
        seven_days_ago = datetime.now() - timedelta(days=7)
        
        # Calories by day
        pipeline = [
            {'$match': {
                'user_id': user_id,
                'meal_date': {'$gte': seven_days_ago}
            }},
            {'$group': {
                '_id': '$meal_date',
                'total_calories': {'$sum': '$calories'}
            }},
            {'$sort': {'_id': 1}}
        ]
        
        calories_data = list(meals_collection.aggregate(pipeline))
        
        # Macronutrients breakdown
        pipeline = [
            {'$match': {
                'user_id': user_id,
                'meal_date': {'$gte': seven_days_ago}
            }},
            {'$group': {
                '_id': None,
                'total_protein': {'$sum': '$protein'},
                'total_carbs': {'$sum': '$carbs'},
                'total_fat': {'$sum': '$fat'}
            }}
        ]
        
        macros_result = list(meals_collection.aggregate(pipeline))
        macros_data = macros_result[0] if macros_result else None
        
        # Most frequent foods
        pipeline = [
            {'$match': {
                'user_id': user_id,
                'meal_date': {'$gte': seven_days_ago}
            }},
            {'$group': {
                '_id': '$food_name',
                'count': {'$sum': 1}
            }},
            {'$sort': {'count': -1}},
            {'$limit': 5}
        ]
        
        food_data = list(meals_collection.aggregate(pipeline))
        
        # Process data for charts
        if calories_data:
            dates = [item['_id'] for item in calories_data]
            calories = [item['total_calories'] for item in calories_data]
            
            plt.figure(figsize=(10, 6))
            plt.plot(dates, calories, marker='o', linestyle='-', color='#4CAF50')
            plt.fill_between(dates, calories, alpha=0.2, color='#4CAF50')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.title('Daily Calorie Intake', fontsize=16)
            plt.xlabel('Date', fontsize=12)
            plt.ylabel('Calories', fontsize=12)
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png')
            buffer.seek(0)
            image_png = buffer.getvalue()
            buffer.close()
            nutrition_data['calories_chart'] = base64.b64encode(image_png).decode('utf-8')
            plt.close()
        
        if macros_data and (macros_data['total_protein'] + macros_data['total_carbs'] + macros_data['total_fat']) > 0:
            labels = ['Protein', 'Carbs', 'Fat']
            sizes = [macros_data['total_protein'], macros_data['total_carbs'], macros_data['total_fat']]
            
            plt.figure(figsize=(8, 8))
            colors = ['#FF9800', '#2196F3', '#F44336']
            plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
            plt.axis('equal')
            plt.title('Macronutrient Distribution', fontsize=16)
            plt.tight_layout()
            
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png')
            buffer.seek(0)
            image_png = buffer.getvalue()
            buffer.close()
            nutrition_data['macros_chart'] = base64.b64encode(image_png).decode('utf-8')
            plt.close()
        
        if food_data:
            foods = [item['_id'] for item in food_data]
            counts = [item['count'] for item in food_data]
            
            plt.figure(figsize=(10, 6))
            plt.barh(foods, counts, color='#9C27B0')
            plt.title('Most Frequent Foods', fontsize=16)
            plt.xlabel('Count', fontsize=12)
            plt.tight_layout()
            
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png')
            buffer.seek(0)
            image_png = buffer.getvalue()
            buffer.close()
            nutrition_data['foods_chart'] = base64.b64encode(image_png).decode('utf-8')
            plt.close()
        
    except Exception as e:
        flash(f'An error occurred: {str(e)}', 'danger')
        print(f"Meal analysis error: {e}")
    
    return render_template('meal_analysis.html', nutrition_data=nutrition_data)

@app.route('/hydration_settings', methods=['GET', 'POST'])
def hydration_settings():
    if 'loggedin' not in session:
        flash('Please login first!', 'danger')
        return redirect(url_for('login'))

    user_id = session['id']
    settings = {'enable': True, 'interval': 120, 'goal': 2000}
    total_intake = 0

    if db_available:
        # Fetch settings
        hydration_settings_doc = hydration_settings_collection.find_one({'user_id': user_id})
        if hydration_settings_doc:
            settings = {
                'enable': hydration_settings_doc.get('enable_reminders', True),
                'interval': hydration_settings_doc.get('reminder_interval', 120),
                'goal': hydration_settings_doc.get('daily_goal', 2000)
            }

        # Calculate total intake for today
        today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        tomorrow = today + timedelta(days=1)
        
        pipeline = [
            {'$match': {
                'user_id': user_id,
                'log_date': {
                    '$gte': today,
                    '$lt': tomorrow
                }
            }},
            {'$group': {
                '_id': None,
                'total': {'$sum': '$quantity_ml'}
            }}
        ]
        
        result = list(water_logs_collection.aggregate(pipeline))
        if result:
            total_intake = int(result[0]['total'])

    if request.method == 'POST':
        enable = request.form.get('enable_reminders') == 'on'
        interval = int(request.form['reminder_interval'])
        daily_goal = int(request.form['daily_goal'])

        settings_doc = {
            'user_id': user_id,
            'enable_reminders': enable,
            'reminder_interval': interval,
            'daily_goal': daily_goal,
            'updated_at': datetime.now()
        }

        hydration_settings_collection.update_one(
            {'user_id': user_id},
            {'$set': settings_doc},
            upsert=True
        )

        flash('Hydration settings updated!', 'success')
        return redirect(url_for('hydration_settings'))

    return render_template('hydration_settings.html', settings=settings, total_intake=total_intake)

@app.route('/log_water', methods=['POST'])
def log_water():
    if 'loggedin' not in session:
        flash('Please login first!', 'danger')
        return redirect(url_for('login'))

    user_id = session['id']
    quantity = int(request.form['quantity_ml'])

    water_log_doc = {
        'user_id': user_id,
        'log_date': datetime.now().replace(hour=0, minute=0, second=0, microsecond=0),
        'quantity_ml': quantity,
        'created_at': datetime.now()
    }
    
    water_logs_collection.insert_one(water_log_doc)

    flash('Water intake logged!', 'success')
    return redirect(url_for('hydration_settings'))

# Main entry point
if __name__ == "__main__":
    
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
