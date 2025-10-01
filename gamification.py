# gamification.py
import datetime
from flask import session, flash

def update_streak(mysql, user_id, workout_date):
    """
    Update user streak when a workout is logged.
    Returns the current streak.
    """
    try:
        cursor = mysql.connection.cursor()
        
        # First check if user exists in streaks table
        cursor.execute("SELECT * FROM user_streaks WHERE user_id = %s", (user_id,))
        streak_data = cursor.fetchone()
        
        # Convert workout_date to datetime if it's a string
        if isinstance(workout_date, str):
            workout_date = datetime.datetime.strptime(workout_date, '%Y-%m-%d').date()
        else:
            # If it's already a datetime, just get the date part
            workout_date = workout_date.date()
            
        today = datetime.date.today()
        yesterday = today - datetime.timedelta(days=1)
        
        if not streak_data:
            # First time user, create streak record
            cursor.execute(
                "INSERT INTO user_streaks (user_id, current_streak, longest_streak, last_workout_date) VALUES (%s, 1, 1, %s)",
                (user_id, workout_date)
            )
            mysql.connection.commit()
            current_streak = 1
        else:
            # User exists in streak table
            current_streak = streak_data[1]
            longest_streak = streak_data[2]
            last_workout_date = streak_data[3]
            
            # If workout is for today or yesterday, update streak
            if workout_date == today:
                # Already logged for today, no change in streak
                if last_workout_date == today:
                    current_streak = current_streak
                # New log for today
                elif last_workout_date == yesterday:
                    # Continued streak
                    current_streak += 0  # Already counted yesterday
                else:
                    # New streak starts
                    current_streak = 1
            elif workout_date == yesterday:
                # Logging yesterday's workout
                if last_workout_date < yesterday:
                    current_streak = 1
                # Already logged for yesterday, no change
                else:
                    current_streak = current_streak
            # If workout is for an older date
            elif workout_date < yesterday:
                # Don't update streak for past workouts beyond yesterday
                cursor.close()
                return current_streak
            # If workout date is in the future, don't update streak
            elif workout_date > today:
                cursor.close()
                return current_streak
                
            # Update longest streak if current is longer
            if current_streak > longest_streak:
                longest_streak = current_streak
                
            # Update the database
            cursor.execute(
                "UPDATE user_streaks SET current_streak = %s, longest_streak = %s, last_workout_date = %s WHERE user_id = %s",
                (current_streak, longest_streak, workout_date, user_id)
            )
            mysql.connection.commit()
            
        cursor.close()
        
        # Check for streak-based badges
        check_streak_badges(mysql, user_id, current_streak)
        
        return current_streak
        
    except Exception as e:
        print(f"Error updating streak: {e}")
        return 0

def get_user_streak(mysql, user_id):
    """Get current streak for a user"""
    try:
        cursor = mysql.connection.cursor()
        cursor.execute("SELECT current_streak, longest_streak FROM user_streaks WHERE user_id = %s", (user_id,))
        result = cursor.fetchone()
        cursor.close()
        
        if result:
            return {
                'current': result[0],
                'longest': result[1]
            }
        return {
            'current': 0,
            'longest': 0
        }
    except Exception as e:
        print(f"Error getting streak: {e}")
        return {
            'current': 0,
            'longest': 0
        }

def add_points(mysql, user_id, points, reason=""):
    """
    Add points to user's account and return new total
    """
    try:
        cursor = mysql.connection.cursor()
        
        # Check if user has points record
        cursor.execute("SELECT points FROM user_points WHERE user_id = %s", (user_id,))
        result = cursor.fetchone()
        
        if result:
            # Update existing record
            new_total = result[0] + points
            cursor.execute("UPDATE user_points SET points = %s WHERE user_id = %s", 
                          (new_total, user_id))
        else:
            # Create new record
            new_total = points
            cursor.execute("INSERT INTO user_points (user_id, points) VALUES (%s, %s)", 
                          (user_id, points))
            
        mysql.connection.commit()
        cursor.close()
        
        # Check if any features can be unlocked with new point total
        check_unlockable_features(mysql, user_id, new_total)
        
        return new_total
    except Exception as e:
        print(f"Error adding points: {e}")
        return 0

def get_user_points(mysql, user_id):
    """Get current points for a user"""
    try:
        cursor = mysql.connection.cursor()
        cursor.execute("SELECT points FROM user_points WHERE user_id = %s", (user_id,))
        result = cursor.fetchone()
        cursor.close()
        
        if result:
            return result[0]
        return 0
    except Exception as e:
        print(f"Error getting points: {e}")
        return 0

def award_badge(mysql, user_id, badge_name):
    """
    Award a badge to a user if they don't already have it
    """
    try:
        cursor = mysql.connection.cursor()
        
        # Find badge id
        cursor.execute("SELECT id, points_reward FROM badges WHERE name = %s", (badge_name,))
        badge = cursor.fetchone()
        
        if not badge:
            cursor.close()
            return False, f"Badge {badge_name} does not exist"
            
        badge_id = badge[0]
        points_reward = badge[1]
        
        # Check if user already has this badge
        cursor.execute("SELECT * FROM user_badges WHERE user_id = %s AND badge_id = %s", 
                      (user_id, badge_id))
        existing = cursor.fetchone()
        
        if existing:
            cursor.close()
            return False, f"Already earned badge: {badge_name}"
            
        # Award the badge
        cursor.execute("INSERT INTO user_badges (user_id, badge_id) VALUES (%s, %s)", 
                      (user_id, badge_id))
        mysql.connection.commit()
        
        # Award points if applicable
        if points_reward > 0:
            add_points(mysql, user_id, points_reward, f"Earned badge: {badge_name}")
            
        cursor.close()
        return True, f"Earned new badge: {badge_name}"
    except Exception as e:
        print(f"Error awarding badge: {e}")
        return False, f"Error awarding badge: {str(e)}"

def check_streak_badges(mysql, user_id, streak):
    """
    Check and award badges based on streak achievements
    """
    streak_badges = {
        3: "3-Day Streak",
        7: "7-Day Streak",
        14: "2-Week Streak",
        30: "30-Day Streak",
        60: "60-Day Streak",
        90: "90-Day Streak",
        180: "6-Month Streak", 
        365: "1-Year Streak"
    }
    
    # Check if user qualifies for any streak badges
    for days, badge_name in streak_badges.items():
        if streak >= days:
            success, message = award_badge(mysql, user_id, badge_name)
            if success:
                # We'll handle notification to the user elsewhere
                pass

def check_workout_count_badges(mysql, user_id):
    """
    Check and award badges based on total workout count
    """
    try:
        cursor = mysql.connection.cursor()
        cursor.execute("SELECT COUNT(*) FROM workouts WHERE user_id = %s", (user_id,))
        count = cursor.fetchone()[0]
        cursor.close()
        
        # Define workout count badges
        count_badges = {
            1: "First Workout",
            10: "10 Workouts",
            25: "25 Workouts",
            50: "50 Workouts",
            100: "Century Club",
            250: "250 Workouts",
            500: "500 Workouts",
            1000: "1000 Workouts"
        }
        
        # Check if user qualifies for any count badges
        for threshold, badge_name in count_badges.items():
            if count >= threshold:
                award_badge(mysql, user_id, badge_name)
                
    except Exception as e:
        print(f"Error checking workout count badges: {e}")

def check_unlockable_features(mysql, user_id, points):
    """
    Check and unlock features based on user's points
    """
    try:
        cursor = mysql.connection.cursor()
        
        # Get all features user could unlock with their points
        cursor.execute("""
            SELECT id, name, points_required 
            FROM unlockable_features 
            WHERE points_required <= %s
            ORDER BY points_required
        """, (points,))
        
        available_features = cursor.fetchall()
        
        for feature in available_features:
            feature_id = feature[0]
            feature_name = feature[1]
            
            # Check if user already has this feature
            cursor.execute("""
                SELECT * FROM user_unlocked_features 
                WHERE user_id = %s AND feature_id = %s
            """, (user_id, feature_id))
            
            if not cursor.fetchone():
                # Unlock the feature
                cursor.execute("""
                    INSERT INTO user_unlocked_features (user_id, feature_id)
                    VALUES (%s, %s)
                """, (user_id, feature_id))
                mysql.connection.commit()
                
                # This would be a good place to notify the user
                # about their newly unlocked feature
                
        cursor.close()
    except Exception as e:
        print(f"Error checking unlockable features: {e}")

def get_user_badges(mysql, user_id):
    """Get all badges earned by a user"""
    try:
        cursor = mysql.connection.cursor()
        cursor.execute("""
            SELECT b.id, b.name, b.description, b.image_path, ub.earned_date
            FROM badges b
            JOIN user_badges ub ON b.id = ub.badge_id
            WHERE ub.user_id = %s
            ORDER BY ub.earned_date DESC
        """, (user_id,))
        
        badges = []
        for row in cursor.fetchall():
            badges.append({
                'id': row[0],
                'name': row[1],
                'description': row[2],
                'image': row[3],
                'earned_date': row[4]
            })
            
        cursor.close()
        return badges
    except Exception as e:
        print(f"Error getting user badges: {e}")
        return []

def get_available_badges(mysql):
    """Get all available badges in the system"""
    try:
        cursor = mysql.connection.cursor()
        cursor.execute("SELECT id, name, description, image_path, points_reward FROM badges")
        
        badges = []
        for row in cursor.fetchall():
            badges.append({
                'id': row[0],
                'name': row[1],
                'description': row[2],
                'image': row[3],
                'points': row[4]
            })
            
        cursor.close()
        return badges
    except Exception as e:
        print(f"Error getting available badges: {e}")
        return []

def get_user_unlocked_features(mysql, user_id):
    """Get all features unlocked by a user"""
    try:
        cursor = mysql.connection.cursor()
        cursor.execute("""
            SELECT f.id, f.name, f.description, f.feature_type, uf.unlocked_date
            FROM unlockable_features f
            JOIN user_unlocked_features uf ON f.id = uf.feature_id
            WHERE uf.user_id = %s
            ORDER BY uf.unlocked_date DESC
        """, (user_id,))
        
        features = []
        for row in cursor.fetchall():
            features.append({
                'id': row[0],
                'name': row[1],
                'description': row[2],
                'type': row[3],
                'unlocked_date': row[4]
            })
            
        cursor.close()
        return features
    except Exception as e:
        print(f"Error getting user features: {e}")
        return []

def initialize_badges(mysql):
    """Initialize default badges in the database"""
    default_badges = [
        # Streak badges
        ('First Workout', 'Completed your first workout', 'badge_first.png', 10, 'First workout logged'),
        ('3-Day Streak', 'Worked out for 3 days in a row', 'badge_streak_3.png', 15, '3-day streak'),
        ('7-Day Streak', 'Worked out for a full week', 'badge_streak_7.png', 25, '7-day streak'),
        ('2-Week Streak', 'Worked out for 14 days in a row', 'badge_streak_14.png', 50, '14-day streak'),
        ('30-Day Streak', 'Worked out for a month straight', 'badge_streak_30.png', 100, '30-day streak'),
        ('60-Day Streak', 'Worked out for 60 days in a row', 'badge_streak_60.png', 150, '60-day streak'),
        ('90-Day Streak', 'Worked out for 90 days in a row', 'badge_streak_90.png', 200, '90-day streak'),
        ('6-Month Streak', 'Worked out for 180 days in a row', 'badge_streak_180.png', 300, '180-day streak'),
        ('1-Year Streak', 'Worked out for a full year', 'badge_streak_365.png', 500, '365-day streak'),
        
        # Count badges
        ('10 Workouts', 'Completed 10 workouts', 'badge_count_10.png', 20, '10 workouts'),
        ('25 Workouts', 'Completed 25 workouts', 'badge_count_25.png', 30, '25 workouts'),
        ('50 Workouts', 'Completed 50 workouts', 'badge_count_50.png', 50, '50 workouts'),
        ('Century Club', 'Completed 100 workouts', 'badge_count_100.png', 100, '100 workouts'),
        ('250 Workouts', 'Completed 250 workouts', 'badge_count_250.png', 200, '250 workouts'),
        ('500 Workouts', 'Completed 500 workouts', 'badge_count_500.png', 300, '500 workouts'),
        ('1000 Workouts', 'Completed 1000 workouts', 'badge_count_1000.png', 500, '1000 workouts'),
        
        # Type-specific badges
        ('Cardio Enthusiast', 'Completed 10 cardio workouts', 'badge_cardio.png', 25, '10 cardio workouts'),
        ('Strength Champion', 'Completed 10 strength workouts', 'badge_strength.png', 25, '10 strength workouts'),
        ('Flexibility Master', 'Completed 10 flexibility workouts', 'badge_flexibility.png', 25, '10 flexibility workouts')
    ]
    
    try:
        cursor = mysql.connection.cursor()
        
        # Check if badges table exists
        cursor.execute("SHOW TABLES LIKE 'badges'")
        if not cursor.fetchone():
            # Create badges table if it doesn't exist
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS badges (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    name VARCHAR(100) NOT NULL,
                    description TEXT,
                    image_path VARCHAR(255),
                    points_reward INT DEFAULT 0,
                    requirements VARCHAR(255)
                )
            """)
        
        # Check if any badges exist
        cursor.execute("SELECT COUNT(*) FROM badges")
        count = cursor.fetchone()[0]
        
        # Only add default badges if none exist
        if count == 0:
            for badge in default_badges:
                cursor.execute("""
                    INSERT INTO badges (name, description, image_path, points_reward, requirements)
                    VALUES (%s, %s, %s, %s, %s)
                """, badge)
            
            mysql.connection.commit()
            print("Default badges initialized")
        
        cursor.close()
    except Exception as e:
        print(f"Error initializing badges: {e}")

def initialize_unlockable_features(mysql):
    """Initialize default unlockable features in the database"""
    default_features = [
        ('Custom Theme - Ocean', 'Unlock the Ocean color theme', 100, 'theme'),
        ('Custom Theme - Sunset', 'Unlock the Sunset color theme', 150, 'theme'),
        ('Custom Theme - Forest', 'Unlock the Forest color theme', 200, 'theme'),
        ('Custom Avatar - Silver', 'Unlock Silver tier avatars', 250, 'avatar'),
        ('Custom Avatar - Gold', 'Unlock Gold tier avatars', 500, 'avatar'),
        ('Detailed Analytics', 'Unlock detailed workout analytics', 300, 'feature'),
        ('Weekly Email Reports', 'Receive detailed weekly progress reports', 400, 'feature'),
        ('Custom Workout Planner', 'Create and save custom workout plans', 600, 'feature')
    ]
    
    try:
        cursor = mysql.connection.cursor()
        
        # Check if features table exists
        cursor.execute("SHOW TABLES LIKE 'unlockable_features'")
        if not cursor.fetchone():
            # Create features table if it doesn't exist
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS unlockable_features (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    name VARCHAR(100) NOT NULL,
                    description TEXT,
                    points_required INT NOT NULL,
                    feature_type VARCHAR(50) NOT NULL
                )
            """)
        
        # Check if any features exist
        cursor.execute("SELECT COUNT(*) FROM unlockable_features")
        count = cursor.fetchone()[0]
        
        # Only add default features if none exist
        if count == 0:
            for feature in default_features:
                cursor.execute("""
                    INSERT INTO unlockable_features (name, description, points_required, feature_type)
                    VALUES (%s, %s, %s, %s)
                """, feature)
            
            mysql.connection.commit()
            print("Default unlockable features initialized")
        
        cursor.close()
    except Exception as e:
        print(f"Error initializing unlockable features: {e}")