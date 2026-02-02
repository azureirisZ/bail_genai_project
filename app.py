from flask import Flask, render_template_string, request, jsonify
import time

app = Flask(__name__)

# --- 1. THE BACKEND (API) ---
@app.route('/get_response', methods=['POST'])
def get_response():
    user_input = request.json.get("message")
    
    # [PLACEHOLDER] CONNECT YOUR AI HERE LATER
    # 1. results = rag.search(user_input)
    # 2. prompt = combine(results, user_input)
    # 3. response = model.generate(prompt)
    
    # Simulating a delay and response for now
    time.sleep(1) 
    fake_ai_response = f"<b>JUDGMENT:</b><br>Based on the provisions of <i>Section 439 CrPC</i>, and considering the precedents regarding '{user_input}', the bail is hereby <b>GRANTED</b>."
    
    return jsonify({"response": fake_ai_response})

# --- 2. THE FRONTEND (HTML/CSS/JS) ---
# We store the HTML inside Python for simplicity, but usually this is a separate file.
HTML_CODE = """
<!DOCTYPE html>
<html>
<head>
    <title>Bail Reckoner AI</title>
    <style>
        /* DARK MODE STYLING */
        body { background-color: #1e1e1e; color: #ffffff; font-family: 'Segoe UI', sans-serif; display: flex; justify-content: center; height: 100vh; margin: 0; }
        .chat-container { width: 800px; height: 90vh; background-color: #2d2d2d; border-radius: 15px; box-shadow: 0 10px 30px rgba(0,0,0,0.5); display: flex; flex-direction: column; overflow: hidden; margin-top: 20px; }
        
        /* HEADER */
        .header { background-color: #3d3d3d; padding: 20px; text-align: center; border-bottom: 1px solid #444; }
        .header h2 { margin: 0; color: #4CAF50; }
        .header p { margin: 5px 0 0; color: #aaa; font-size: 0.9em; }

        /* CHAT AREA */
        .chat-box { flex: 1; padding: 20px; overflow-y: auto; display: flex; flex-direction: column; gap: 15px; }
        .message { max-width: 80%; padding: 12px 18px; border-radius: 10px; line-height: 1.5; font-size: 15px; }
        
        .user-msg { align-self: flex-end; background-color: #0078D4; color: white; border-bottom-right-radius: 2px; }
        .ai-msg { align-self: flex-start; background-color: #3d3d3d; color: #e0e0e0; border-bottom-left-radius: 2px; border: 1px solid #555; }
        
        /* INPUT AREA */
        .input-area { padding: 20px; background-color: #3d3d3d; display: flex; gap: 10px; border-top: 1px solid #444; }
        input { flex: 1; padding: 15px; border-radius: 25px; border: none; background-color: #2d2d2d; color: white; outline: none; font-size: 16px; }
        button { padding: 10px 25px; border-radius: 25px; border: none; background-color: #4CAF50; color: white; font-weight: bold; cursor: pointer; transition: 0.2s; }
        button:hover { background-color: #45a049; }

        /* LOADING ANIMATION */
        .typing { font-style: italic; color: #888; font-size: 12px; margin-left: 20px; display: none; }
    </style>
</head>
<body>

<div class="chat-container">
    <div class="header">
        <h2>⚖️ Bail Reckoner AI</h2>
        <p>Powered by Custom LSTM + RAG Engine</p>
    </div>

    <div class="chat-box" id="chat-box">
        <div class="message ai-msg">Hello, Counsel. I am ready to review the case details.</div>
    </div>
    
    <div class="typing" id="typing-indicator">Judge AI is drafting the verdict...</div>

    <div class="input-area">
        <input type="text" id="user-input" placeholder="Enter case details here..." onkeypress="handleEnter(event)">
        <button onclick="sendMessage()">Analyze</button>
    </div>
</div>

<script>
    async function sendMessage() {
        let inputField = document.getElementById("user-input");
        let chatBox = document.getElementById("chat-box");
        let typingIndicator = document.getElementById("typing-indicator");
        let text = inputField.value;

        if (text.trim() === "") return;

        // 1. Add User Message
        chatBox.innerHTML += `<div class="message user-msg">${text}</div>`;
        inputField.value = "";
        chatBox.scrollTop = chatBox.scrollHeight;

        // 2. Show Typing Indicator
        typingIndicator.style.display = "block";

        // 3. Send to Python Backend
        let response = await fetch("/get_response", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({ "message": text })
        });
        
        let data = await response.json();

        // 4. Add AI Message
        typingIndicator.style.display = "none";
        chatBox.innerHTML += `<div class="message ai-msg">${data.response}</div>`;
        chatBox.scrollTop = chatBox.scrollHeight;
    }

    function handleEnter(e) {
        if (e.key === "Enter") sendMessage();
    }
</script>

</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HTML_CODE)

if __name__ == '__main__':
    print("🚀 Server starting at http://127.0.0.1:5000")
    app.run(debug=True)