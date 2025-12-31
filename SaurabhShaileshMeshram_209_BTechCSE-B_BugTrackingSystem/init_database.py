from flask import Flask
from flask_pymongo import PyMongo
from bson.objectid import ObjectId
import bcrypt
from datetime import datetime, timedelta
import random

app = Flask(__name__)
app.config["MONGO_URI"] = "mongodb://localhost:27017/bugtracker"
mongo = PyMongo(app)

# Collections
users = mongo.db.users
bugs = mongo.db.bugs
comments = mongo.db.comments
projects = mongo.db.projects

def init_database():
    # Clear existing data
    users.delete_many({})
    bugs.delete_many({})
    comments.delete_many({})
    projects.delete_many({})

    # Create demo users
    demo_users = [
        {
            "username": "admin",
            "password": bcrypt.hashpw("admin123".encode("utf-8"), bcrypt.gensalt()),
            "role": "admin",
            "email": "admin@bugtracker.com",
            "full_name": "System Administrator",
            "created_at": datetime.now() - timedelta(days=30),
            "is_active": True
        },
        {
            "username": "john_dev",
            "password": bcrypt.hashpw("dev123".encode("utf-8"), bcrypt.gensalt()),
            "role": "developer",
            "email": "john@bugtracker.com",
            "full_name": "John Developer",
            "created_at": datetime.now() - timedelta(days=25),
            "is_active": True
        },
        {
            "username": "sarah_dev",
            "password": bcrypt.hashpw("dev123".encode("utf-8"), bcrypt.gensalt()),
            "role": "developer",
            "email": "sarah@bugtracker.com",
            "full_name": "Sarah Developer",
            "created_at": datetime.now() - timedelta(days=20),
            "is_active": True
        },
        {
            "username": "mike_tester",
            "password": bcrypt.hashpw("test123".encode("utf-8"), bcrypt.gensalt()),
            "role": "tester",
            "email": "mike@bugtracker.com",
            "full_name": "Mike Tester",
            "created_at": datetime.now() - timedelta(days=15),
            "is_active": True
        },
        {
            "username": "lisa_tester",
            "password": bcrypt.hashpw("test123".encode("utf-8"), bcrypt.gensalt()),
            "role": "tester",
            "email": "lisa@bugtracker.com",
            "full_name": "Lisa Tester",
            "created_at": datetime.now() - timedelta(days=10),
            "is_active": True
        }
    ]

    users.insert_many(demo_users)

    # Create demo projects
    demo_projects = [
        {
            "name": "E-Commerce Platform",
            "description": "Online shopping platform with payment integration",
            "created_by": "admin",
            "created_at": datetime.now() - timedelta(days=30),
            "is_active": True
        },
        {
            "name": "Mobile Banking App",
            "description": "Secure mobile banking application",
            "created_by": "admin",
            "created_at": datetime.now() - timedelta(days=20),
            "is_active": True
        },
        {
            "name": "HR Management System",
            "description": "Employee management and payroll system",
            "created_by": "admin",
            "created_at": datetime.now() - timedelta(days=15),
            "is_active": True
        }
    ]

    projects.insert_many(demo_projects)

    # Create demo bugs
    bug_titles = [
        "Login page not responsive on mobile devices",
        "Payment gateway timeout error during checkout",
        "Search results pagination broken on page 2+",
        "User profile picture upload fails with large files",
        "Dashboard charts not loading in Internet Explorer",
        "Email notifications not being sent for new registrations",
        "Password reset link expires too quickly",
        "File attachment feature crashes on certain file types",
        "Admin panel slow to load with many users",
        "Export functionality missing date filters",
        "Dark mode toggle not working on settings page",
        "API rate limiting too restrictive for legitimate users",
        "Database connection timeout during peak hours",
        "User permissions not updating in real-time",
        "Mobile app crashes when switching between tabs"
    ]

    bug_descriptions = [
        "The login page layout breaks on mobile devices, making it difficult to enter credentials and tap the login button. The form elements overlap and the submit button is not easily accessible.",
        "Customers are experiencing timeout errors when trying to complete payments through the checkout process. This happens intermittently and affects the conversion rate.",
        "When users navigate to page 2 or beyond in search results, the pagination controls disappear and users cannot navigate through the results.",
        "Users cannot upload profile pictures larger than 2MB. The upload fails silently without providing clear error messages to the user.",
        "Dashboard charts and graphs fail to render properly in Internet Explorer 11, showing blank spaces instead of the expected visualizations.",
        "New user registrations are not triggering email notifications to administrators, which delays the account approval process.",
        "Password reset links expire after only 15 minutes, which is too short for users who may not check their email immediately.",
        "The file attachment feature crashes the application when users try to upload certain file types like .tiff or .eps files.",
        "The admin panel takes more than 10 seconds to load when there are over 1000 users in the system, affecting productivity.",
        "The export functionality does not include date range filters, making it difficult for users to export data for specific time periods.",
        "The dark mode toggle button in settings does not actually change the application theme, though it appears to be functional.",
        "Legitimate API users are being rate limited unnecessarily, causing service disruptions for valid use cases.",
        "Database connections timeout during peak usage hours (9-11 AM), causing application errors and poor user experience.",
        "When user permissions are updated in the admin panel, the changes don't reflect immediately in the user interface.",
        "The mobile application crashes when users rapidly switch between different tabs, losing any unsaved data."
    ]

    priorities = ["High", "Medium", "Low"]
    severities = ["Critical", "Major", "Minor", "Trivial"]

    demo_bugs = []
    for i in range(15):
        created_date = datetime.now() - timedelta(days=random.randint(1, 30))
        status = random.choice(["Open", "In Progress", "Resolved", "Closed"])
        assigned_to = random.choice(["john_dev", "sarah_dev"]) if status in ["In Progress", "Resolved"] else None

        bug = {
            "title": bug_titles[i],
            "description": bug_descriptions[i],
            "status": status,
            "priority": random.choice(priorities),
            "severity": random.choice(severities),
            "created_by": random.choice(["mike_tester", "lisa_tester"]),
            "assigned_to": assigned_to,
            "project": random.choice(["E-Commerce Platform", "Mobile Banking App", "HR Management System"]),
            "created_at": created_date,
            "updated_at": created_date + timedelta(days=random.randint(1, 10)),
            "steps_to_reproduce": f"1. Navigate to {random.choice(['login page', 'dashboard', 'settings', 'profile'])}\n2. {random.choice(['Click on button', 'Enter invalid data', 'Switch to mobile view', 'Upload file'])}\n3. {random.choice(['Wait for response', 'Check console', 'Refresh page', 'Try different browser'])}",
            "expected_behavior": "System should work as expected without errors",
            "actual_behavior": "System shows error or behaves unexpectedly",
            "environment": "Chrome 91.0, Windows 10, Resolution 1920x1080",
            "tags": random.sample(["frontend", "backend", "ui", "api", "database", "mobile"], random.randint(1, 3))
        }
        demo_bugs.append(bug)

    bugs.insert_many(demo_bugs)

    # Create demo comments
    demo_comments = [
        {
            "bug_id": demo_bugs[0]["_id"] if "_id" in demo_bugs[0] else ObjectId(),
            "user": "john_dev",
            "comment": "I've identified the issue. The CSS media queries are not properly defined for mobile devices. I'll fix this today.",
            "created_at": datetime.now() - timedelta(days=2),
            "is_internal": False
        },
        {
            "bug_id": demo_bugs[1]["_id"] if "_id" in demo_bugs[1] else ObjectId(),
            "user": "sarah_dev",
            "comment": "This appears to be related to the payment gateway's timeout configuration. The server-side timeout is set to 30 seconds, but the client expects 60 seconds.",
            "created_at": datetime.now() - timedelta(days=1),
            "is_internal": False
        },
        {
            "bug_id": demo_bugs[2]["_id"] if "_id" in demo_bugs[2] else ObjectId(),
            "user": "admin",
            "comment": "This is a critical issue affecting user experience. Please prioritize this for the next release.",
            "created_at": datetime.now() - timedelta(hours=12),
            "is_internal": True
        }
    ]

    # Insert comments with proper ObjectIds
    for comment in demo_comments:
        bug = bugs.find_one({"title": bug_titles[0] if comment["bug_id"] == demo_bugs[0]["_id"] else bug_titles[1] if comment["bug_id"] == demo_bugs[1]["_id"] else bug_titles[2]})
        if bug:
            comment["bug_id"] = bug["_id"]
            comments.insert_one(comment)

    print("✅ Database initialized with demo data!")
    print(f"📊 Created {len(demo_users)} demo users")
    print(f"🐛 Created {len(demo_bugs)} demo bugs")
    print(f"💬 Created {len(demo_comments)} demo comments")
    print(f"📁 Created {len(demo_projects)} demo projects")

    # Print login credentials
    print("\n🔐 Demo Login Credentials:")
    print("Admin: admin / admin123")
    print("Developer: john_dev / dev123")
    print("Developer: sarah_dev / dev123")
    print("Tester: mike_tester / test123")
    print("Tester: lisa_tester / test123")

if __name__ == "__main__":
    with app.app_context():
        init_database()
