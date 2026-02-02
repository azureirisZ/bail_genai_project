import os
import json
from datetime import datetime
from flask import Flask, render_template_string, request, jsonify, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_bcrypt import Bcrypt
from cryptography.fernet import Fernet

# ==========================================
# ⚙️ CONFIGURATION & SECURITY
# ==========================================
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-123'  # Change this in production
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///bail_reckoner.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

# ENCRYPTION KEY (For Chat History)
# In production, save this key in a .env file!
ENCRYPTION_KEY = Fernet.generate_key()
cipher = Fernet(ENCRYPTION_KEY)

def encrypt_data(text):
    return cipher.encrypt(text.encode()).decode()

def decrypt_data(encrypted_text):
    try:
        return cipher.decrypt(encrypted_text.encode()).decode()
    except:
        return "[Error: Data Corrupted]"

# ==========================================
# 🗄️ DATABASE MODELS
# ==========================================
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password_hash = db.Column(db.String(150), nullable=False)
    chats = db.relationship('ChatHistory', backref='author', lazy=True)

class ChatHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    query = db.Column(db.Text, nullable=False)  # Encrypted
    response = db.Column(db.Text, nullable=False) # Encrypted
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# ==========================================
# 🧠 DUMMY AI ENGINE (Replace with your Real Logic)
# ==========================================
def get_ai_response(user_query):
    # TODO: Import your actual 'rag_engine' and 'model' here
    # results = rag.search(user_query)
    # answer = model.generate(prompt)
    
    # Fake response for demo
    answer = f"Based on the analysis of <b>Section 439</b>, and considering the precedents regarding '{user_query}', bail is likely to be <b>GRANTED</b>."
    
    # Fake Citations (RAG Results)
    citations = [
        {"title": "Satender Kumar Antil v. CBI", "text": "The Supreme Court held that bail is the rule and jail is the exception..."},
        {"title": "Section 439 CrPC", "text": "Special powers of High Court or Court of Session regarding bail..."},
        {"title": "Arnesh Kumar v. State of Bihar", "text": "Arrest should not be automatic. Police must follow Section 41A..."},
    ]
    return answer, citations

# ==========================================
# 🎨 FRONTEND TEMPLATES (HTML/CSS/JS)
# ==========================================
# We use one massive HTML string to handle all pages dynamically
BASE_LAYOUT = """
<!DOCTYPE html>
<html>
<head>
    <title>Bail Reckoner AI</title>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <style>
        :root { --bg: #121212; --panel: #1e1e1e; --primary: #bb86fc; --text: #e0e0e0; }
        body { background-color: var(--bg); color: var(--text); font-family: 'Segoe UI', sans-serif; margin: 0; display: flex; height: 100vh; }
        
        /* SIDEBAR */
        .sidebar { width: 250px; background-color: var(--panel); padding: 20px; display: flex; flex-direction: column; border-right: 1px solid #333; }
        .logo { font-size: 20px; font-weight: bold; color: var(--primary); margin-bottom: 30px; }
        .nav-link { display: block; padding: 10px; color: #aaa; text-decoration: none; border-radius: 5px; margin-bottom: 5px; }
        .nav-link:hover, .nav-link.active { background-color: #333; color: white; }
        .nav-link i { margin-right: 10px; }
        
        /* MAIN CONTENT */
        .main { flex: 1; padding: 0; display: flex; flex-direction: column; overflow: hidden; }
        
        /* AUTH PAGES */
        .auth-box { width: 350px; margin: 100px auto; background: var(--panel); padding: 30px; border-radius: 10px; box-shadow: 0 4px 15px rgba(0,0,0,0.5); }
        input { width: 90%; padding: 10px; margin: 10px 0; background: #333; border: 1px solid #444; color: white; border-radius: 5px; }
        button { width: 100%; padding: 10px; background: var(--primary); border: none; font-weight: bold; cursor: pointer; border-radius: 5px; margin-top: 10px; }
        
        /* CHAT UI */
        .chat-container { flex: 1; overflow-y: auto; padding: 20px; display: flex; flex-direction: column; gap: 15px; }
        .message { max-width: 70%; padding: 15px; border-radius: 10px; line-height: 1.5; position: relative; }
        .user-msg { align-self: flex-end; background-color: #3700b3; color: white; }
        .ai-msg { align-self: flex-start; background-color: var(--panel); border: 1px solid #333; }
        
        /* CITATIONS */
        .citation-box { font-size: 0.85em; margin-top: 10px; border-top: 1px solid #444; padding-top: 10px; }
        .citation-btn { background: #333; color: #aaa; border: 1px solid #555; padding: 5px 10px; font-size: 0.8em; margin-right: 5px; cursor: pointer; border-radius: 15px; }
        .citation-btn:hover { background: var(--primary); color: black; }
        
        /* INPUT AREA */
        .input-area { padding: 20px; background: var(--panel); display: flex; gap: 10px; }
        
        /* DASHBOARD TABLE */
        table { width: 100%; border-collapse: collapse; margin-top: 20px; }
        th, td { text-align: left; padding: 12px; border-bottom: 1px solid #333; }
        th { color: var(--primary); }
    </style>
</head>
<body>
    {% if current_user.is_authenticated %}
    <div class="sidebar">
        <div class="logo"><i class="fas fa-balance-scale"></i> Bail Reckoner</div>
        <a href="{{ url_for('chat_page') }}" class="nav-link"><i class="fas fa-comments"></i> New Case Analysis</a>
        <a href="{{ url_for('dashboard') }}" class="nav-link"><i class="fas fa-history"></i> Case History</a>
        <div style="flex:1;"></div>
        <div style="padding: 10px; font-size: 0.9em; color: #666;">Logged in as: <br><b style="color:white">{{ current_user.username }}</b></div>
        <a href="{{ url_for('logout') }}" class="nav-link" style="color: #ff5555;"><i class="fas fa-sign-out-alt"></i> Logout</a>
    </div>
    {% endif %}

    <div class="main">
        {% block content %}{% endblock %}
    </div>

    <script>
        function toggleCitation(id) {
            let el = document.getElementById(id);
            if (el.style.display === "none") el.style.display = "block";
            else el.style.display = "none";
        }
    </script>
</body>
</html>
"""

# ==========================================
# 🛤️ ROUTES
# ==========================================

@app.route('/')
def home():
    if current_user.is_authenticated:
        return redirect(url_for('chat_page'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()
        
        if user and bcrypt.check_password_hash(user.password_hash, password):
            login_user(user)
            return redirect(url_for('chat_page'))
        else:
            return "Invalid Credentials <a href='/login'>Try Again</a>"
            
    return render_template_string(BASE_LAYOUT + """
    {% block content %}
    <div class="auth-box">
        <h2 style="text-align:center; color:white;">Login</h2>
        <form method="POST">
            <input type="text" name="username" placeholder="Username" required>
            <input type="password" name="password" placeholder="Password" required>
            <button type="submit">Enter Chambers</button>
        </form>
        <p style="text-align:center; margin-top:15px; font-size:0.9em;">
            New User? <a href="{{ url_for('register') }}" style="color:#bb86fc;">Register Here</a>
        </p>
    </div>
    {% endblock %}
    """)

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        # Hash the password
        hashed_pw = bcrypt.generate_password_hash(password).decode('utf-8')
        
        # Create User
        new_user = User(username=username, password_hash=hashed_pw)
        try:
            db.session.add(new_user)
            db.session.commit()
            return redirect(url_for('login'))
        except:
            return "Username already exists."

    return render_template_string(BASE_LAYOUT + """
    {% block content %}
    <div class="auth-box">
        <h2 style="text-align:center; color:white;">Register</h2>
        <form method="POST">
            <input type="text" name="username" placeholder="Choose Username" required>
            <input type="password" name="password" placeholder="Choose Password" required>
            <button type="submit">Create Account</button>
        </form>
        <p style="text-align:center; margin-top:15px; font-size:0.9em;">
            <a href="{{ url_for('login') }}" style="color:#bb86fc;">Back to Login</a>
        </p>
    </div>
    {% endblock %}
    """)

@app.route('/chat')
@login_required
def chat_page():
    return render_template_string(BASE_LAYOUT + """
    {% block content %}
    <div class="chat-container" id="chat-box">
        <div class="message ai-msg">
            <b>Judge AI:</b> <br> Greetings, Counsel. Please provide the details of the bail application.
        </div>
    </div>
    
    <div class="input-area">
        <input type="text" id="user-input" placeholder="e.g. Murder accused under Section 302, in custody for 3 years...">
        <button onclick="sendMessage()" style="width:100px; margin:0;"><i class="fas fa-paper-plane"></i></button>
    </div>

    <script>
        async function sendMessage() {
            let input = document.getElementById("user-input");
            let chatBox = document.getElementById("chat-box");
            let text = input.value;
            if (!text) return;

            // User Message
            chatBox.innerHTML += `<div class="message user-msg">${text}</div>`;
            input.value = "";
            chatBox.scrollTop = chatBox.scrollHeight;

            // Call Backend
            let res = await fetch("/api/generate", {
                method: "POST",
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({query: text})
            });
            let data = await res.json();

            // AI Message with Citations
            let citationHtml = "";
            data.citations.forEach((cit, index) => {
                citationHtml += `
                    <button class="citation-btn" onclick="toggleCitation('cit-${index}')">📄 ${cit.title}</button>
                    <div id="cit-${index}" style="display:none; background:#222; padding:10px; margin-top:5px; border-radius:5px; border-left: 3px solid #bb86fc;">
                        <i>"${cit.text}"</i>
                    </div>
                `;
            });

            let aiHtml = `
                <div class="message ai-msg">
                    ${data.answer}
                    <div class="citation-box">
                        <b>📚 Legal Precedents Used:</b><br><br>
                        ${citationHtml}
                    </div>
                </div>
            `;
            
            chatBox.innerHTML += aiHtml;
            chatBox.scrollTop = chatBox.scrollHeight;
        }
    </script>
    {% endblock %}
    """)

@app.route('/dashboard')
@login_required
def dashboard():
    # Fetch user history and decrypt it
    history = ChatHistory.query.filter_by(user_id=current_user.id).order_by(ChatHistory.timestamp.desc()).all()
    
    decrypted_history = []
    for h in history:
        decrypted_history.append({
            "date": h.timestamp.strftime("%Y-%m-%d %H:%M"),
            "query": decrypt_data(h.query),
            "response": decrypt_data(h.response)[:100] + "..." # Show preview only
        })

    return render_template_string(BASE_LAYOUT + """
    {% block content %}
    <div style="padding:40px;">
        <h2>📊 Case History Dashboard</h2>
        <p style="color:#aaa;">All your client queries are encrypted for confidentiality.</p>
        
        <table>
            <thead>
                <tr>
                    <th>Date</th>
                    <th>Query</th>
                    <th>Verdict Preview</th>
                </tr>
            </thead>
            <tbody>
                {% for item in history %}
                <tr>
                    <td style="color:#666;">{{ item.date }}</td>
                    <td>{{ item.query }}</td>
                    <td style="color:#bbb;">{{ item.response }}</td>
                </tr>
                {% endfor %}
            </tbody>
        </table>
    </div>
    {% endblock %}
    """, history=decrypted_history)

@app.route('/api/generate', methods=['POST'])
@login_required
def generate_api():
    data = request.json
    user_query = data.get('query')
    
    # 1. Run AI
    answer, citations = get_ai_response(user_query)
    
    # 2. Encrypt & Save to DB
    new_chat = ChatHistory(
        user_id=current_user.id,
        query=encrypt_data(user_query),
        response=encrypt_data(answer)
    )
    db.session.add(new_chat)
    db.session.commit()
    
    return jsonify({"answer": answer, "citations": citations})

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

if __name__ == '__main__':
    with app.app_context():
        db.create_all()  # Create DB tables if they don't exist
    app.run(debug=True, port=5000)