from flask import Flask, render_template, request, redirect, url_for, flash, session , send_file,jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import numpy as np
from datetime import timedelta
import json 
from authlib.integrations.flask_client import OAuth 
import matplotlib.pyplot as plt
import io
from scipy.optimize import linprog
from dotenv import load_dotenv
import os

load_dotenv()

GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
SECRET_KEY = os.getenv("SECRET_KEY")
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///local.db")

app = Flask(__name__)

app.config['SECRET_KEY'] = SECRET_KEY
app.config['SQLALCHEMY_DATABASE_URI'] = DATABASE_URL
app.config['SESSION_PERMANENT'] = True  # Enable permanent sessions
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=7)  # Session lasts 7 days

db = SQLAlchemy(app)

# Initialize LoginManager
login_manager = LoginManager(app)
login_manager.login_view = 'login'


oauth = OAuth(app)
google = oauth.register(
    name="google",
    client_id=GOOGLE_CLIENT_ID,
    client_secret=GOOGLE_CLIENT_SECRET,
    access_token_url="https://oauth2.googleapis.com/token",
    authorize_url="https://accounts.google.com/o/oauth2/auth",
    api_base_url="https://www.googleapis.com/oauth2/v1/",
    jwks_uri="https://www.googleapis.com/oauth2/v3/certs",  # Ensure this is set
    client_kwargs={"scope": "openid email profile"},
    redirect_uri="http://localhost:5000/signup/callback"
)

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), unique=True, nullable=False)  # Added username
    email = db.Column(db.String(64), unique=True, nullable=False)
    password = db.Column(db.String(128), nullable=True)

    def __init__(self, username, email, password=None):
        self.username = username  # Store username
        self.email = email
        self.password = generate_password_hash(password) if password else None

    def check_password(self, password):
        return check_password_hash(self.password, password)
    

    
class SavedProblem(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    optimization = db.Column(db.String(10), nullable=False)
    objective = db.Column(db.Text, nullable=False)
    constraints = db.Column(db.Text, nullable=False)
    rhs = db.Column(db.Text, nullable=False)
    constraint_types = db.Column(db.Text, nullable=False)
    optimal_value = db.Column(db.Float, nullable=False)
    solution_data = db.Column(db.Text, nullable=False)  # Store solution as JSON string
    num_constraints = db.Column(db.Integer, nullable=False)

    user = db.relationship('User', backref='saved_problems', lazy=True)

    def __init__(self, user_id, optimization, objective, constraints, rhs, constraint_types, optimal_value, solution_data , num_constraints):
        self.user_id = user_id
        self.optimization = optimization
        self.objective = ','.join(map(str, objective))  # Convert list to string
        self.constraints = '|'.join([','.join(map(str, row)) for row in constraints])
        self.rhs = ','.join(map(str, rhs))
        self.constraint_types = ','.join(constraint_types)
        self.optimal_value = optimal_value
        self.solution_data = json.dumps(solution_data)
        self.num_constraints = num_constraints

    

with app.app_context():
    db.create_all()  # This will create users.db with the User table
    print("Database created successfully!")

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]
        user = User.query.filter_by(email=email).first()

        if user and user.check_password(password):
            login_user(user)
            return redirect(url_for("dashboard"))
        else:
            flash("Invalid Credentials. Please enter your credentials" , 'error')
            return render_template("login.html", error="Invalid credentials")

    return render_template("login.html")

@app.route("/google-login")
def google_login():
    return google.authorize_redirect(url_for("google_callback", _external=True))

@app.route("/login/callback")
def google_callback():
    token = google.authorize_access_token()
    user_info = google.get("userinfo").json()

    email = user_info["email"]
    username = user_info.get("given_name", "GoogleUser")

    # Check if user exists
    user = User.query.filter_by(email=email).first()

    if not user:
        user = User(username=username, email=email, password=None)  # Store without password
        db.session.add(user)
        db.session.commit()

    login_user(user)
    return redirect(url_for("dashboard"))


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/dashboard')
@login_required
def dashboard():
    username = session.get("username", "User")  # Get username from session
    first_letter = username[0].upper() if username else "U"  # Extract first letter
    
    # ✅ Fetch the saved problems for the logged-in user
    problems = SavedProblem.query.filter_by(user_id=current_user.id).all()

    return render_template("dashboard.html", first_letter=first_letter, user_name=username, problems=problems)

@app.route("/instructions")
def instructions():
    return render_template("instructions.html")

@app.route('/new_problem')
def new_problem():
    return render_template('new_problem.html')


@app.route('/submit_problem', methods=['POST'])
def submit_problem():
    session['optimization'] = request.form.get('optimization')
    session['num_variables'] = int(request.form.get('num_variables'))
    session['objective'] = list(map(float, request.form.get('objective').split(',')))
    session['num_constraints'] = int(request.form.get('num_constraints'))

    constraints = []
    rhs = []
    constraint_types = []

    for i in range(session['num_constraints']):
        lhs = list(map(float, request.form.get(f'lhs_{i}').split(',')))
        constraint_types.append(request.form.get(f'constraint_type_{i}'))
        rhs.append(float(request.form.get(f'rhs_{i}')))
        constraints.append(lhs)

    session['constraints'] = constraints
    session['rhs'] = rhs
    session['constraint_types'] = constraint_types  # Store constraint conditions

    return redirect(url_for('solve_problem'))


def round_if_needed(value):
    """Round to 2 decimals if float, keep as integer if possible."""
    if isinstance(value, (float, np.floating)):
        return round(value, 2) if value % 1 != 0 else int(value)
    return value

import numpy as np

def round_if_needed(value):
    """Round to 2 decimals if float, keep as integer if possible."""
    if isinstance(value, (float, np.floating)):
        return round(value, 2) if value % 1 != 0 else int(value)
    return value

def simplex_method(n, c, n_cts, coeff_matrix, rhs):
    """Perform the Simplex algorithm."""
    c = np.array(c + [0] * n_cts).reshape(1, -1)  # Ensure correct shape
    rhs = np.array(rhs).reshape(-1, 1)

    history = {'coeff_mats': [], 'basic_ind': [], 'cz': [], 'rhs': [], 'z': []}

    for i in range(n_cts):
        slack_vars = [1 if i == j else 0 for j in range(n_cts)]
        coeff_matrix[i].extend(slack_vars)

    coeff_matrix = np.array(coeff_matrix, dtype=float)
    basic_vals = np.array([c[0, i] for i in range(n, n + n_cts)]).reshape(-1, 1)
    basic_ind = [i for i in range(n, n + n_cts)]

    while True:
        z = np.dot(basic_vals.T, coeff_matrix)  # Ensure correct shape for z
        cz = (c - z).reshape(1, -1)  # Ensure correct shape for cz 
        
        # Apply rounding
        tr = np.round(coeff_matrix, 2).tolist()
        tr1 = np.round(cz, 2).tolist()
        tr2 = basic_ind.copy()
        tr3 = np.round(rhs, 2).tolist()

        # Convert values in lists
        tr = [[round_if_needed(val) for val in row] for row in tr]
        tr1 = [[round_if_needed(val) for val in row] for row in tr1]
        tr3 = [[round_if_needed(val) for val in row] for row in tr3]

        history['coeff_mats'].append(tr)  
        history['cz'].append(tr1)
        history['basic_ind'].append(tr2)
        history['rhs'].append(tr3)

        # Finding entering variable
        entering_var, highest_positive = -1, -1
        for idx in range(n + n_cts):
            val = cz[0, idx]
            if val > 0 and val > highest_positive:
                entering_var = idx
                highest_positive = val

        if entering_var == -1:  # Optimal solution found
            break

        # Finding leaving variable
        ratios = []
        for i in range(n_cts):
            if coeff_matrix[i, entering_var] > 0:
                ratios.append(rhs[i, 0] / coeff_matrix[i, entering_var])
            else:
                ratios.append(float('inf'))

        leaving_var = np.argmin(ratios)

        if ratios[leaving_var] == float('inf'):
            return "Unbounded solution", [], [], {}

        # Pivoting
        pivot_element = coeff_matrix[leaving_var, entering_var]
        coeff_matrix[leaving_var] /= pivot_element
        rhs[leaving_var] /= pivot_element

        for i in range(n_cts):
            if i == leaving_var:
                continue
            factor = coeff_matrix[i, entering_var]
            coeff_matrix[i] -= factor * coeff_matrix[leaving_var]
            rhs[i, 0] -= factor * rhs[leaving_var, 0]

        basic_vals[leaving_var, 0] = c[0, entering_var]
        basic_ind[leaving_var] = entering_var
        optimal_value = np.dot(basic_vals.T, rhs)[0]

        # Append rounded optimal value
        history['z'].append(round_if_needed(optimal_value.tolist()))

    optimal_value = np.dot(basic_vals.T, rhs)[0]
    history['z'].append(round_if_needed(optimal_value.tolist()))

    print(history)
    return optimal_value, rhs.flatten(), basic_ind, history


def to_subscript(num):
    """Convert a number to its Unicode subscript equivalent."""
    subscript_digits = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
    return str(num).translate(subscript_digits)

def format_expression(coeffs):
    terms = []
    for i, coeff in enumerate(coeffs):
        sub_i = to_subscript(i + 1)  # Convert index to subscript
        if coeff == 1:
            terms.append(f"x{sub_i}")  # Remove "1x₁", just "x₁"
        elif coeff == -1:
            terms.append(f"-x{sub_i}")  # Handle negative 1
        elif coeff != 0:
            terms.append(f"{coeff}x{sub_i}")
    return " + ".join(terms).replace("+ -", "- ")  # Format signs properly


@app.route('/duality_analysis')
def duality():
    history = session.get('history', None)  # Retrieve stored history

    if history is None:
        return jsonify({"error": "No history found. Solve a problem first."})

    def to_subscript(num):
        """Convert a number to its Unicode subscript equivalent."""
        subscript_digits = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
        return str(num).translate(subscript_digits)

    iterations = []

    for i in range(len(history['coeff_mats']) - 1):
        coeff_matrix = history['coeff_mats'][i]
        rhs = history['rhs'][i]
        cz = history['cz'][i][0]
        opt = history['z'][i][0]
        basic_ind = history['basic_ind'][i]  # Extract basic indices

        # Format the objective function
        cz_terms = []
        for j in range(len(cz)):
            if cz[j] != 0:  # Skip terms where coefficient is 0
                sign = "-" if cz[j] < 0 else "+ "
                if abs(cz[j]) == 1 :
                    cz_terms.append(f"{sign}x{to_subscript(j+1)}")
                else :
                    cz_terms.append(f"{sign}{abs(cz[j])}x{to_subscript(j+1)}")

        objective_function = f"Z = {opt}"
        if cz_terms:
            objective_function += " " + " ".join(cz_terms)

        # Format constraints
        constraints = []
        for row, value, basic_var in zip(coeff_matrix, rhs, basic_ind):
            basic_coeff = row[basic_var]  # Get coefficient of the basic variable
            rhs_value = value  # Constant term on the right side

            # Move all non-basic terms to the right-hand side with flipped signs
            right_side_terms = []
            for j in range(len(row)):
                if j != basic_var and row[j] != 0:
                    sign = "-" if row[j] > 0 else "+"
                    if abs(row[j]) == 1 :
                        right_side_terms.append(f"{sign} x{to_subscript(j+1)}") 
                    else :
                        right_side_terms.append(f"{sign} {abs(row[j])}x{to_subscript(j+1)}")

            # Format equation: basic_variable * x_basic = rhs ± other terms
            constraint = f"x{to_subscript(basic_var+1)} = {rhs_value[0]}"
            if right_side_terms:
                constraint += " " + " ".join(right_side_terms)

            constraints.append(constraint)

        iterations.append({
            "iteration": i + 1,
            "objective": objective_function,
            "constraints": constraints
        })

        print(objective_function, constraints)

    return render_template('duality_analysis.html', iterations=iterations)


@app.route('/solve_problem')
def solve_problem():
    optimization = session.get('optimization', 'max')
    objective = session.get('objective', [])
    constraints = session.get('constraints', [])
    rhs = session.get('rhs', [])
    constraint_types = session.get('constraint_types', [])  # Retrieve constraint conditions
    n = session.get('num_variables', 0)
    n_cts = session.get('num_constraints', 0)

    if not constraints or not objective:
        return "No problem submitted yet."

    # Constructing constraints in "LHS condition RHS" format
    formatted_constraints = [
    f"{format_expression(constraints[i])} {constraint_types[i]} {rhs[i]}"
    for i in range(n_cts)
    ]

    optimal_value, solution, indices , history = simplex_method(n, objective, n_cts, constraints, rhs)

    solution_data = list(zip(indices, solution))
    objective_str = "Z = " + " + ".join([f"{objective[i]}x<sub>{i+1}</sub>" for i in range(n)])
    session['solution_data'] = solution_data
    session['optimal_value'] = optimal_value[0]
    session['history'] = history
    return render_template(
        'solve_problem.html',
        optimization=optimization,
        objective=objective,
        objective_str=objective_str,
        constraints=formatted_constraints,  # Send formatted constraints
        optimal_value=optimal_value,
        solution_data=solution_data , # 🔹 Pass the zipped data
        num_constraints=n,
        sess_history = history
    )

@app.route('/generate_graph')
def generate_graph():

    # Fetch first iteration data from session history
    first_iteration = session.get("history", []) # Get first iteration
    n = session.get('num_variables' , 0)

    if not first_iteration:
        return "No iteration data found!", 400
    
    A = np.array(first_iteration["coeff_mats"][0])[:,:n]  # Constraint Coefficients
    A = A.tolist()
    b = first_iteration["rhs"][0]  # Right-Hand Side values
    print(A)
    c = [-val for val in first_iteration["cz"][0][0]]
    print(c)
    c = [val for i,val in enumerate(c) if i < n]

    # Solve the LP
    res = linprog(c, A_ub=A, b_ub=b, bounds=(0, None), method='highs')

    # Plot Constraints
    x = np.linspace(0, 100, 1000)
    y_constraints = []
    
    for i in range(len(A)):  # Iterate through constraints
        if A[i][1] != 0:  
            y_constraints.append((b[i] - A[i][0] * x) / A[i][1])
        else:
            y_constraints.append(np.full_like(x, b[i] / A[i][0]))  # Vertical line case

    plt.figure(figsize=(6, 6))
    plt.xlim(0, 10)
    plt.ylim(0, 10)

    colors = ['blue', 'green', 'purple', 'orange']
    
    for i, y in enumerate(y_constraints):
        plt.plot(x, y, label=f'Constraint {i+1}', color=colors[i % len(colors)])

    # Feasible Region
    plt.fill_between(x, np.minimum.reduce(y_constraints), alpha=0.3, color='gray')

    # Optimal Solution
    if res.success:
        plt.scatter(res.x[0], res.x[1], color='red', marker='o', label='Optimal Solution')

    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid()
    
    # Save plot as image
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()

    return send_file(img, mimetype='image/png')


@app.route('/graphical_analysis')
def graphical_analysis():
    return render_template("graphical_analysis.html")


@app.route('/save_problem')
@login_required
def save_problem():
    if 'objective' not in session or 'constraints' not in session:
        flash("No problem to save.", "error")
        return redirect(url_for('dashboard'))

    new_problem = SavedProblem(
        user_id=current_user.id,
        optimization=session.get('optimization', 'max'),
        objective=session['objective'],
        constraints=session['constraints'],
        rhs=session['rhs'],
        constraint_types=session['constraint_types'],
        optimal_value=session.get('optimal_value', 0),
        solution_data=session.get('solution_data', []),
        num_constraints = session['num_constraints'],
    )

    db.session.add(new_problem)
    db.session.commit()
    flash("Problem saved successfully!", "success")
    return redirect(url_for('dashboard'))

@app.route('/solutions')
def view_solutions():
    saved_problems = SavedProblem.query.all()  # Fetch all saved problems
    return render_template('solutions.html', saved_problems=saved_problems)

@app.route('/view_problem/<int:problem_id>')
@login_required
def view_problem(problem_id):
    problem = SavedProblem.query.get_or_404(problem_id)

    objective = list(map(float, problem.objective.split(',')))
    constraints = [list(map(float, row.split(','))) for row in problem.constraints.split('|')]
    rhs = list(map(float, problem.rhs.split(',')))
    constraint_types = problem.constraint_types.split(',')
    num_constraints = problem.num_constraints
    print('num_constraints : ', num_constraints)

    # Load solution_data from JSON string
    solution_data = []
    if problem.solution_data:  # Ensure it's not empty
        try:
            solution_data = json.loads(problem.solution_data)  # Convert JSON string back to list of tuples ✅
        except Exception as e:
            print(f"Error parsing solution_data: {e}")  # Debugging
            solution_data = []  # Reset to avoid crashes

    return render_template(
        'solve_problem.html', 
        optimization=problem.optimization, 
        objective_str="Z = " + " + ".join([f"{objective[i]}x<sub>{i+1}</sub>" for i in range(len(objective))]), 
        constraints=[
            f"{format_expression(constraints[i])} {constraint_types[i]} {rhs[i]}"
            for i in range(len(rhs))
        ],
        optimal_value=problem.optimal_value,
        solution_data=solution_data , # Now correctly formatted
        num_constraints = num_constraints
    )

@app.route("/google-signup")
def google_signup():
    return google.authorize_redirect(url_for("google_signup_callback", _external=True))

@app.route("/signup/callback")
def google_signup_callback():
    token = google.authorize_access_token()
    user_info = google.get("userinfo").json()

    email = user_info["email"]
    username = user_info.get("given_name", "GoogleUser")  # Use given name if available

    # Check if user already exists
    user = User.query.filter_by(email=email).first()

    if not user:
        # Create a new user (no password since it's Google signup)
        user = User(username=username, email=email, password=None)
        db.session.add(user)
        db.session.commit()

    login_user(user)
    session["username"] = username  # Store username in session
    session.modified = True  

    return redirect(url_for("dashboard"))



@app.route('/delete_problem/<int:id>')
def delete_problem(id):
    problem = SavedProblem.query.get(id)
    if problem:
        db.session.delete(problem)
        db.session.commit()
    return redirect('/solutions')


@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('home'))

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm_password']

        if password != confirm_password:
            flash('Enter the same password in confirm password section.', 'error')
            return redirect(url_for('signup'))

        if User.query.filter_by(email=email).first():
            flash("Email already exists.", 'error')
            return redirect(url_for('signup'))

        if User.query.filter_by(username=username).first():
            flash("Username already taken. Choose another.", 'error')
            return redirect(url_for('signup'))

        new_user = User(username, email, password)
        db.session.add(new_user)
        db.session.commit()
        login_user(new_user)
        
        # Store username in session
        session["username"] = username  
        session.modified = True  

        return redirect(url_for('dashboard'))
    return render_template('signup.html')




if __name__ == '__main__':
    app.run(debug=True)