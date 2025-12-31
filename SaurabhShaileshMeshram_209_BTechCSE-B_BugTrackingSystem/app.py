from flask import Flask, render_template, request, redirect, url_for, session, flash
from flask_pymongo import PyMongo
from bson.objectid import ObjectId
import bcrypt
from datetime import datetime

app = Flask(__name__)
app.secret_key = "supersecretkey"

# MongoDB Config
app.config["MONGO_URI"] = "mongodb://localhost:27017/bugtracker"
mongo = PyMongo(app)

# Collections
users = mongo.db.users
bugs = mongo.db.bugs
comments = mongo.db.comments
projects = mongo.db.projects

# Authentication decorator
def require_login(func):
    def wrapper(*args, **kwargs):
        if 'username' not in session:
            flash("Please log in to access this page.")
            return redirect(url_for('login'))
        return func(*args, **kwargs)
    wrapper.__name__ = func.__name__
    return wrapper

# Role-based access control decorator
def require_role(role):
    def decorator(func):
        def wrapper(*args, **kwargs):
            if 'username' not in session:
                flash("Please log in to access this page.")
                return redirect(url_for('login'))
            if session.get('role') != role:
                flash(f"Access denied. This page requires {role} role.")
                if session.get('role') == 'admin':
                    return redirect(url_for('admin_dashboard'))
                elif session.get('role') == 'developer':
                    return redirect(url_for('developer_dashboard'))
                else:
                    return redirect(url_for('tester_dashboard'))
            return func(*args, **kwargs)
        wrapper.__name__ = func.__name__
        return wrapper
    return decorator

# ---------------- LANDING ----------------
@app.route("/")
@app.route("/landing.html")
@app.route("/favicon.ico")
def landing():
    return render_template("landing.html")

# ---------------- AUTH ----------------
@app.route("/register", methods=["GET", "POST"])
@app.route("/register.html", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        role = request.form["role"]
        email = request.form.get("email", "")  # Optional email field

        existing_user = users.find_one({"username": username})
        if existing_user:
            flash("User already exists!")
            return redirect(url_for("register"))

        hash_pw = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt())

        # Insert user (email field is optional in the form)
        user_data = {"username": username, "password": hash_pw, "role": role}
        if email:
            user_data["email"] = email

        users.insert_one(user_data)

        flash("Registration successful! Please login with your credentials.")
        return redirect(url_for("login"))

    return render_template("register.html")

@app.route("/login", methods=["GET", "POST"])
@app.route("/login.html", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        user = users.find_one({"username": username})
        if user and bcrypt.checkpw(password.encode("utf-8"), user["password"]):
            session["username"] = username
            session["role"] = user["role"]

            if user["role"] == "admin":
                return redirect(url_for("admin_dashboard"))
            elif user["role"] == "developer":
                return redirect(url_for("developer_dashboard"))
            else:
                return redirect(url_for("tester_dashboard"))

        flash("Invalid credentials!")
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    flash("Logged out successfully.")
    return redirect(url_for("landing"))

# ---------------- DASHBOARDS ----------------
@app.route("/admin")
@app.route("/admin-dashboard.html")
@require_role('admin')
def admin_dashboard():
    # Get dashboard statistics
    total_users = users.count_documents({})
    total_bugs = bugs.count_documents({})
    open_bugs = bugs.count_documents({"status": "Open"})
    resolved_bugs = bugs.count_documents({"status": "Resolved"})
    active_users = users.count_documents({"is_active": True})

    # Get recent bugs
    recent_bugs = list(bugs.find().sort("created_at", -1).limit(5))

    # Get user activity (recent registrations)
    recent_users = list(users.find().sort("created_at", -1).limit(5))

    # Get available developers for assignment
    available_developers = list(users.find({"role": "developer", "is_active": True}))

    # Get bug statistics by status
    bug_stats = {
        "total": total_bugs,
        "open": open_bugs,
        "in_progress": bugs.count_documents({"status": "In Progress"}),
        "resolved": resolved_bugs,
        "closed": bugs.count_documents({"status": "Closed"})
    }

    return render_template("admin-dashboard-enhanced.html",
                         stats=bug_stats,
                         recent_bugs=recent_bugs,
                         recent_users=recent_users,
                         total_users=total_users,
                         active_users=active_users,
                         available_developers=available_developers)

@app.route("/assign_bug/<id>", methods=["POST"])
@require_role('admin')
def assign_bug(id):
    developer = request.form.get("assigned_to")
    if developer:
        bugs.update_one(
            {"_id": ObjectId(id)},
            {"$set": {"assigned_to": developer, "updated_at": datetime.now()}}
        )
        flash("Bug assigned successfully!")
    else:
        flash("No developer selected.")
    return redirect(url_for('admin_dashboard'))

@app.route("/developer")
@app.route("/developer-dashboard.html")
@require_role('developer')
def developer_dashboard():
    username = session.get('username')

    # Get developer's assigned bugs
    assigned_bugs = list(bugs.find({"assigned_to": username}))
    open_bugs = [bug for bug in assigned_bugs if bug.get('status') == 'Open']
    in_progress_bugs = [bug for bug in assigned_bugs if bug.get('status') == 'In Progress']
    resolved_bugs = [bug for bug in assigned_bugs if bug.get('status') == 'Resolved']

    # Get recent activity
    recent_bugs = list(bugs.find({"assigned_to": username}).sort("updated_at", -1).limit(5))

    # Get bug statistics
    bug_stats = {
        "total_assigned": len(assigned_bugs),
        "open": len(open_bugs),
        "in_progress": len(in_progress_bugs),
        "resolved": len(resolved_bugs)
    }

    return render_template("developer-dashboard-enhanced.html",
                         assigned_bugs=assigned_bugs,
                         recent_bugs=recent_bugs,
                         stats=bug_stats)

@app.route("/tester")
@app.route("/tester-dashboard.html")
@require_role('tester')
def tester_dashboard():
    username = session.get('username')

    # Get tester's reported bugs
    reported_bugs = list(bugs.find({"created_by": username}))
    open_bugs = [bug for bug in reported_bugs if bug.get('status') == 'Open']
    resolved_bugs = [bug for bug in reported_bugs if bug.get('status') == 'Resolved']

    # Get all bugs for viewing (testers can see all bugs)
    all_bugs = list(bugs.find())

    # Get recent activity
    recent_bugs = list(bugs.find({"created_by": username}).sort("created_at", -1).limit(5))

    # Get bug statistics
    bug_stats = {
        "total_reported": len(reported_bugs),
        "open": len(open_bugs),
        "resolved": len(resolved_bugs),
        "total_system": bugs.count_documents({})
    }

    return render_template("tester-dashboard-enhanced.html",
                         reported_bugs=reported_bugs,
                         all_bugs=all_bugs,
                         recent_bugs=recent_bugs,
                         stats=bug_stats)

# ---------------- BUG MANAGEMENT ----------------
@app.route("/report-bug", methods=["GET", "POST"])
@app.route("/Bug-Report-Form.html", methods=["GET", "POST"])
@require_role('tester')
def report_bug():
    # Get available projects for dropdown
    available_projects = list(projects.find({"is_active": True}))

    if request.method == "POST":
        # Enhanced form validation
        title = request.form.get("title", "").strip()
        description = request.form.get("description", "").strip()
        priority = request.form.get("priority", "Medium")
        severity = request.form.get("severity", "Minor")
        project_id = request.form.get("project")
        steps_to_reproduce = request.form.get("steps_to_reproduce", "").strip()
        environment = request.form.get("environment", "").strip()
        expected_behavior = request.form.get("expected_behavior", "").strip()
        actual_behavior = request.form.get("actual_behavior", "").strip()
        operating_system = request.form.get("operating_system", "").strip()
        browser = request.form.get("browser", "").strip()

        # Validation errors
        errors = []

        # Required field validation
        if not title or len(title) < 5:
            errors.append("Title must be at least 5 characters long.")
        if not description or len(description) < 10:
            errors.append("Description must be at least 10 characters long.")
        if not steps_to_reproduce:
            errors.append("Steps to reproduce are required.")
        if not environment:
            errors.append("Environment information is required.")
        if not severity:
            errors.append("Please select a severity level.")
        if not priority:
            errors.append("Please select a priority level.")

        # Optional field validation
        if expected_behavior and len(expected_behavior) > 500:
            errors.append("Expected behavior must be less than 500 characters.")
        if actual_behavior and len(actual_behavior) > 500:
            errors.append("Actual behavior must be less than 500 characters.")

        # Check for duplicate titles (optional)
        existing_bug = bugs.find_one({
            "title": {"$regex": f"^{title}$", "$options": "i"},
            "created_by": session.get("username")
        })
        if existing_bug:
            errors.append("You have already reported a bug with this title.")

        if errors:
            flash("Validation failed: " + ", ".join(errors))
            return render_template("Bug-Report-Form.html", projects=available_projects)

        # Get project name if project selected
        project_name = "General"
        if project_id:
            project = projects.find_one({"_id": ObjectId(project_id)})
            if project:
                project_name = project["name"]

        # Create enhanced bug report
        created_by = session.get("username")
        bug_data = {
            "title": title,
            "description": description,
            "status": "Open",
            "priority": priority,
            "severity": severity,
            "project": project_name,
            "created_by": created_by,
            "assigned_to": None,
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
            "steps_to_reproduce": steps_to_reproduce,
            "environment": environment,
            "expected_behavior": expected_behavior,
            "actual_behavior": actual_behavior,
            "operating_system": operating_system,
            "browser": browser,
            "tags": request.form.get("tags", "").split(",") if request.form.get("tags") else [],
            "comments": []
        }

        result = bugs.insert_one(bug_data)

        # Add initial comment to comments collection
        if result.inserted_id:
            comment_data = {
                "bug_id": result.inserted_id,
                "user": created_by,
                "comment": f"Bug reported with {priority} priority and {severity} severity.",
                "created_at": datetime.now(),
                "is_internal": False
            }
            comments.insert_one(comment_data)

        flash("Bug reported successfully!")
        return redirect(url_for("admin_dashboard"))

    return render_template("Bug-Report-Form.html", projects=available_projects)

@app.route("/api/report-bug", methods=["POST"])
@require_role('tester')
def api_report_bug():
    """AJAX endpoint for bug reporting with enhanced validation"""
    try:
        # Enhanced form validation
        title = request.form.get("title", "").strip()
        description = request.form.get("description", "").strip()
        priority = request.form.get("priority", "Medium")
        severity = request.form.get("severity", "Minor")
        project_id = request.form.get("project")
        steps_to_reproduce = request.form.get("steps_to_reproduce", "").strip()
        environment = request.form.get("environment", "").strip()
        expected_behavior = request.form.get("expected_behavior", "").strip()
        actual_behavior = request.form.get("actual_behavior", "").strip()
        operating_system = request.form.get("operating_system", "").strip()
        browser = request.form.get("browser", "").strip()

        # Validation errors
        errors = []

        # Required field validation
        if not title or len(title) < 5:
            errors.append("Title must be at least 5 characters long.")
        if not description or len(description) < 10:
            errors.append("Description must be at least 10 characters long.")
        if not steps_to_reproduce:
            errors.append("Steps to reproduce are required.")
        if not environment:
            errors.append("Environment information is required.")
        if not severity:
            errors.append("Please select a severity level.")
        if not priority:
            errors.append("Please select a priority level.")

        # Optional field validation
        if expected_behavior and len(expected_behavior) > 500:
            errors.append("Expected behavior must be less than 500 characters.")
        if actual_behavior and len(actual_behavior) > 500:
            errors.append("Actual behavior must be less than 500 characters.")

        # Check for duplicate titles (optional)
        existing_bug = bugs.find_one({
            "title": {"$regex": f"^{title}$", "$options": "i"},
            "created_by": session.get("username")
        })
        if existing_bug:
            errors.append("You have already reported a bug with this title.")

        if errors:
            return {
                "success": False,
                "message": "Validation failed",
                "errors": errors
            }, 400

        # Get project name if project selected
        project_name = "General"
        if project_id:
            project = projects.find_one({"_id": ObjectId(project_id)})
            if project:
                project_name = project["name"]

        # Create enhanced bug report
        created_by = session.get("username")
        bug_data = {
            "title": title,
            "description": description,
            "status": "Open",
            "priority": priority,
            "severity": severity,
            "project": project_name,
            "created_by": created_by,
            "assigned_to": None,
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
            "steps_to_reproduce": steps_to_reproduce,
            "environment": environment,
            "expected_behavior": expected_behavior,
            "actual_behavior": actual_behavior,
            "operating_system": operating_system,
            "browser": browser,
            "tags": request.form.get("tags", "").split(",") if request.form.get("tags") else [],
            "comments": []
        }

        result = bugs.insert_one(bug_data)

        # Add initial comment to comments collection
        if result.inserted_id:
            comment_data = {
                "bug_id": result.inserted_id,
                "user": created_by,
                "comment": f"Bug reported with {priority} priority and {severity} severity.",
                "created_at": datetime.now(),
                "is_internal": False
            }
            comments.insert_one(comment_data)

        return {
            "success": True,
            "message": "Bug reported successfully!",
            "bug_id": str(result.inserted_id)
        }

    except Exception as e:
        return {
            "success": False,
            "message": "An error occurred while submitting the bug report.",
            "error": str(e)
        }, 500

@app.route("/bugs")

@app.route("/Bug-Detail-Page.html")
@require_login
def list_bugs():
    user_role = session.get('role')
    username = session.get('username')

    # Filter bugs based on user role
    if user_role == 'admin':
        # Admins can see all bugs
        all_bugs = list(bugs.find().sort("created_at", -1))
    elif user_role == 'developer':
        # Developers can see bugs assigned to them
        all_bugs = list(bugs.find({"assigned_to": username}).sort("created_at", -1))
    elif user_role == 'tester':
        # Testers can see all bugs
        all_bugs = list(bugs.find().sort("created_at", -1))
    else:
        all_bugs = []

    # Get bug statistics
    total_bugs = len(all_bugs)
    open_bugs = len([bug for bug in all_bugs if bug.get('status') == 'Open'])
    in_progress_bugs = len([bug for bug in all_bugs if bug.get('status') == 'In Progress'])
    resolved_bugs = len([bug for bug in all_bugs if bug.get('status') == 'Resolved'])

    return render_template("Bug-Detail-Page.html",
                         bugs=all_bugs,
                         total_bugs=total_bugs,
                         open_bugs=open_bugs,
                         in_progress_bugs=in_progress_bugs,
                         resolved_bugs=resolved_bugs)

@app.route("/bug/<id>", methods=["GET", "POST"])
@require_login
def bug_detail(id):
    try:
        bug = bugs.find_one({"_id": ObjectId(id)})
        if not bug:
            flash("Bug not found.")
            return redirect(url_for('landing'))

        user_role = session.get('role')
        username = session.get('username')

        # Check permissions based on role
        if user_role == 'admin':
            # Admins can access all bugs
            pass
        elif user_role == 'developer':
            # Developers can only access bugs assigned to them
            if bug.get('assigned_to') != username:
                flash("Access denied. You can only view bugs assigned to you.")
                return redirect(url_for('developer_dashboard'))
        elif user_role == 'tester':
            # Testers can access all bugs for viewing and commenting
            pass
        else:
            # This should not happen due to @require_login, but just in case
            flash("Access denied.")
            return redirect(url_for('login'))

        # Handle different POST actions based on user role
        if request.method == "POST":
            action = request.form.get("action", "")

            # Admin actions
            if user_role == 'admin':
                if action == "assign":
                    developer = request.form.get("assigned_to")
                    if developer:
                        try:
                            bugs.update_one(
                                {"_id": ObjectId(id)},
                                {"$set": {"assigned_to": developer, "updated_at": datetime.now()}}
                            )
                            flash("Bug assigned successfully!")
                            return redirect(url_for('bug_detail', id=id))
                        except Exception as e:
                            flash("Error assigning bug: " + str(e))
                            return redirect(url_for('bug_detail', id=id))

                elif action == "update_status":
                    new_status = request.form.get("status")
                    if new_status:
                        bugs.update_one(
                            {"_id": ObjectId(id)},
                            {"$set": {"status": new_status, "updated_at": datetime.now()}}
                        )
                        flash("Bug status updated successfully!")
                        return redirect(url_for('bug_detail', id=id))

                elif action == "delete":
                    bugs.delete_one({"_id": ObjectId(id)})
                    comments.delete_many({"bug_id": ObjectId(id)})
                    flash("Bug deleted successfully!")
                    return redirect(url_for('list_bugs'))

            # Developer actions
            elif user_role == 'developer' and bug.get('assigned_to') == username:
                if action == "update_status":
                    new_status = request.form.get("status")
                    valid_transitions = {
                        "Open": ["In Progress"],
                        "In Progress": ["Resolved", "Open"],
                        "Resolved": ["Closed", "In Progress"]
                    }
                    if new_status and new_status in valid_transitions.get(bug.get('status', ''), []):
                        bugs.update_one(
                            {"_id": ObjectId(id)},
                            {"$set": {"status": new_status, "updated_at": datetime.now()}}
                        )
                        flash("Bug status updated successfully!")
                        return redirect(url_for('bug_detail', id=id))

            # Comment submission (all roles can comment)
            if request.form.get("comment", "").strip():
                comment_text = request.form.get("comment", "").strip()
                comment_data = {
                    "bug_id": ObjectId(id),
                    "user": username,
                    "comment": comment_text,
                    "created_at": datetime.now(),
                    "is_internal": request.form.get("is_internal", "false").lower() == "true"
                }
                comments.insert_one(comment_data)

                # Update bug's updated_at timestamp
                bugs.update_one(
                    {"_id": ObjectId(id)},
                    {"$set": {"updated_at": datetime.now()}}
                )

                flash("Comment added successfully!")
                return redirect(url_for('bug_detail', id=id))

        # Get comments for this bug
        bug_comments = list(comments.find({"bug_id": ObjectId(id)}).sort("created_at", -1))

        # Get available developers for assignment (admin only)
        available_developers = []
        if user_role == 'admin':
            available_developers = list(users.find({"role": "developer", "is_active": True}))

        # Determine user permissions for template
        user_permissions = {
            "can_assign": user_role == 'admin',
            "can_update_status": (user_role == 'admin') or (user_role == 'developer' and bug.get('assigned_to') == username),
            "can_delete": user_role == 'admin',
            "can_comment": True,
            "can_upload": user_role in ['admin', 'developer', 'tester']
        }

        return render_template("Bug-Detail-Page.html",
                             bug=bug,
                             comments=bug_comments,
                             available_developers=available_developers,
                             user_permissions=user_permissions)

    except Exception as e:
        flash("Error accessing bug details.")
        return redirect(url_for('landing'))

# ---------------- ADMIN ONLY ----------------
@app.route("/user-management")
@app.route("/User-Management-(Admin Only).html")
@require_role('admin')
def user_management():
    all_users = list(users.find().sort("created_at", -1))

    # Get user statistics
    total_users = len(all_users)
    active_users = len([user for user in all_users if user.get('is_active', True)])
    admin_users = len([user for user in all_users if user.get('role') == 'admin'])
    developer_users = len([user for user in all_users if user.get('role') == 'developer'])
    tester_users = len([user for user in all_users if user.get('role') == 'tester'])

    # Get recent registrations
    recent_users = list(users.find().sort("created_at", -1).limit(5))

    return render_template("User-Management-(Admin Only).html",
                         users=all_users,
                         total_users=total_users,
                         active_users=active_users,
                         admin_users=admin_users,
                         developer_users=developer_users,
                         tester_users=tester_users,
                         recent_users=recent_users)

@app.route("/reports")
@app.route("/Reports&Analytics-(Admin Only).html")
@require_role('admin')
def reports():
    # Basic bug statistics
    bug_stats = {
        "total": bugs.count_documents({}),
        "open": bugs.count_documents({"status": "Open"}),
        "in_progress": bugs.count_documents({"status": "In Progress"}),
        "resolved": bugs.count_documents({"status": "Resolved"}),
        "closed": bugs.count_documents({"status": "Closed"})
    }

    # Priority and severity breakdown
    priority_stats = {
        "high": bugs.count_documents({"priority": "High"}),
        "medium": bugs.count_documents({"priority": "Medium"}),
        "low": bugs.count_documents({"priority": "Low"})
    }

    severity_stats = {
        "critical": bugs.count_documents({"severity": "Critical"}),
        "major": bugs.count_documents({"severity": "Major"}),
        "minor": bugs.count_documents({"severity": "Minor"}),
        "trivial": bugs.count_documents({"severity": "Trivial"})
    }

    # Project-wise bug distribution
    project_stats = list(bugs.aggregate([
        {"$group": {"_id": "$project", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}}
    ]))

    # Developer performance (bugs assigned and resolved)
    developer_stats = list(bugs.aggregate([
        {"$match": {"assigned_to": {"$ne": None}}},
        {"$group": {"_id": "$assigned_to", "assigned": {"$sum": 1}, "resolved": {"$sum": {"$cond": [{"$eq": ["$status", "Resolved"]}, 1, 0]}}}},
        {"$sort": {"assigned": -1}}
    ]))

    # Tester activity (bugs reported)
    tester_stats = list(bugs.aggregate([
        {"$group": {"_id": "$created_by", "reported": {"$sum": 1}}},
        {"$sort": {"reported": -1}}
    ]))

    # Recent activity (last 30 days)
    from datetime import timedelta
    thirty_days_ago = datetime.now() - timedelta(days=30)
    recent_bugs = list(bugs.find({"created_at": {"$gte": thirty_days_ago}}).sort("created_at", -1))

    return render_template("Reports&Analytics-(Admin Only).html",
                         stats=bug_stats,
                         priority_stats=priority_stats,
                         severity_stats=severity_stats,
                         project_stats=project_stats,
                         developer_stats=developer_stats,
                         tester_stats=tester_stats,
                         recent_bugs=recent_bugs)

if __name__ == "__main__":
    app.run(debug=True, port=5002)
