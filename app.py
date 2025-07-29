# app.py

from flask import Flask, request, jsonify, render_template_string
import os
from datetime import datetime

# Import functions from your chatbot module
from chatbot import get_haven_response, reset_conversation, log_interaction

# Import the database manager
from db_manager import init_db

app = Flask(__name__)

# --- Dark Mode Colorful AI Chatbot HTML Template ---
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Haven AI - Your Dark Mode AI Companion</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        body {
            font-family: 'Inter', 'SF Pro Display', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #0a0a0a;
            background-image: 
                radial-gradient(circle at 20% 80%, rgba(120, 119, 198, 0.3) 0%, transparent 50%),
                radial-gradient(circle at 80% 20%, rgba(255, 119, 198, 0.3) 0%, transparent 50%),
                radial-gradient(circle at 40% 40%, rgba(120, 219, 226, 0.3) 0%, transparent 50%);
            min-height: 100vh;
            color: #ffffff;
            overflow-x: hidden;
        }
        .grid-overlay {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background-image: 
                linear-gradient(rgba(255, 255, 255, 0.03) 1px, transparent 1px),
                linear-gradient(90deg, rgba(255, 255, 255, 0.03) 1px, transparent 1px);
            background-size: 50px 50px;
            pointer-events: none;
            z-index: 0;
        }
        .floating-elements {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            pointer-events: none;
            z-index: 1;
        }
        .floating-orb {
            position: absolute;
            border-radius: 50%;
            filter: blur(1px);
            animation: float 8s ease-in-out infinite;
        }
        .orb-1 {
            width: 60px;
            height: 60px;
            background: linear-gradient(45deg, #ff6b9d, #c44569);
            top: 20%;
            left: 10%;
            animation-delay: 0s;
        }
        .orb-2 {
            width: 40px;
            height: 40px;
            background: linear-gradient(45deg, #4834d4, #686de0);
            top: 60%;
            right: 15%;
            animation-delay: 2s;
        }
        .orb-3 {
            width: 80px;
            height: 80px;
            background: linear-gradient(45deg, #00d2d3, #54a0ff);
            top: 30%;
            right: 30%;
            animation-delay: 4s;
        }
        @keyframes float {
            0%, 100% { transform: translateY(0px) translateX(0px) scale(1); opacity: 0.7; }
            33% { transform: translateY(-30px) translateX(20px) scale(1.1); opacity: 0.9; }
            66% { transform: translateY(15px) translateX(-15px) scale(0.9); opacity: 0.8; }
        }
        .container {
            position: relative;
            z-index: 2;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
        }
        .header {
            text-align: center;
            margin-bottom: 40px;
        }
        .logo {
            font-size: 3rem;
            font-weight: 800;
            background: linear-gradient(45deg, #ff6b9d, #4834d4, #00d2d3, #ffb347, #ff6b9d);
            background-size: 300% 300%;
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            animation: gradientFlow 3s ease-in-out infinite;
            margin-bottom: 15px;
            text-shadow: 0 0 30px rgba(255, 107, 157, 0.5);
        }
        @keyframes gradientFlow {
            0%, 100% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
        }
        .tagline {
            font-size: 1.3rem;
            color: #e2e8f0;
            font-weight: 400;
            margin-bottom: 8px;
        }
        .subtitle {
            font-size: 1rem;
            color: #94a3b8;
            font-weight: 300;
        }
        .chat-container {
            background: rgba(15, 15, 15, 0.9);
            backdrop-filter: blur(20px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 24px;
            width: 100%;
            max-width: 900px;
            height: 650px;
            display: flex;
            flex-direction: column;
            overflow: hidden;
            box-shadow: 
                0 25px 50px rgba(0, 0, 0, 0.5),
                0 0 0 1px rgba(255, 255, 255, 0.05),
                inset 0 1px 0 rgba(255, 255, 255, 0.1);
            position: relative;
        }
        .chat-header {
            background: linear-gradient(135deg, rgba(15, 15, 15, 0.95), rgba(30, 30, 30, 0.95));
            padding: 25px 35px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        }
        .chat-title {
            font-size: 1.4rem;
            font-weight: 700;
            background: linear-gradient(45deg, #ff6b9d, #4834d4);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        .header-controls {
            display: flex;
            align-items: center;
            gap: 20px;
        }
        .status-indicator {
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .status-dot {
            width: 10px;
            height: 10px;
            border-radius: 50%;
            background: linear-gradient(45deg, #00ff88, #00d2d3);
            animation: pulse 2s infinite;
            box-shadow: 0 0 10px rgba(0, 255, 136, 0.5);
        }
        @keyframes pulse {
            0%, 100% { opacity: 1; transform: scale(1); }
            50% { opacity: 0.7; transform: scale(1.2); }
        }
        .status-text {
            font-size: 0.95rem;
            color: #94a3b8;
            font-weight: 500;
        }
        .chat-window {
            flex: 1;
            padding: 30px 35px;
            overflow-y: auto;
            display: flex;
            flex-direction: column;
            gap: 25px;
            background: linear-gradient(180deg, rgba(10, 10, 10, 0.3), rgba(20, 20, 20, 0.3));
        }
        .chat-window::-webkit-scrollbar {
            width: 8px;
        }
        .chat-window::-webkit-scrollbar-track {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 4px;
        }
        .chat-window::-webkit-scrollbar-thumb {
            background: linear-gradient(45deg, #ff6b9d, #4834d4);
            border-radius: 4px;
        }
        .message {
            max-width: 75%;
            padding: 18px 24px;
            border-radius: 24px;
            line-height: 1.6;
            word-wrap: break-word;
            animation: messageSlide 0.4s cubic-bezier(0.4, 0, 0.2, 1);
            position: relative;
        }
        @keyframes messageSlide {
            from { opacity: 0; transform: translateY(30px) scale(0.95); }
            to { opacity: 1; transform: translateY(0) scale(1); }
        }
        .message.user {
            align-self: flex-end;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-bottom-right-radius: 8px;
            box-shadow: 
                0 8px 25px rgba(102, 126, 234, 0.3),
                0 0 0 1px rgba(255, 255, 255, 0.1);
        }
        .message.bot {
            align-self: flex-start;
            background: linear-gradient(135deg, rgba(30, 30, 30, 0.8), rgba(45, 45, 45, 0.8));
            color: #e2e8f0;
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-bottom-left-radius: 8px;
            backdrop-filter: blur(10px);
            box-shadow: 
                0 8px 25px rgba(0, 0, 0, 0.3),
                inset 0 1px 0 rgba(255, 255, 255, 0.1);
        }
        .chat-input-container {
            padding: 25px 35px;
            border-top: 1px solid rgba(255, 255, 255, 0.1);
            background: linear-gradient(135deg, rgba(15, 15, 15, 0.95), rgba(25, 25, 25, 0.95));
        }
        .chat-input {
            display: flex;
            gap: 15px;
            align-items: center;
        }
        .input-wrapper {
            flex: 1;
            position: relative;
        }
        .chat-input input[type="text"] {
            width: 100%;
            padding: 18px 24px;
            background: rgba(30, 30, 30, 0.8);
            border: 2px solid rgba(255, 255, 255, 0.1);
            border-radius: 30px;
            color: #ffffff;
            font-size: 1rem;
            outline: none;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            backdrop-filter: blur(10px);
        }
        .chat-input input[type="text"]:focus {
            border-color: #667eea;
            box-shadow: 
                0 0 0 3px rgba(102, 126, 234, 0.2),
                0 8px 25px rgba(102, 126, 234, 0.15);
            background: rgba(40, 40, 40, 0.9);
        }
        .chat-input input[type="text"]::placeholder {
            color: #64748b;
        }
        .send-btn {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border: none;
            border-radius: 50%;
            width: 56px;
            height: 56px;
            cursor: pointer;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-size: 1.3rem;
            box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
        }
        .send-btn:hover {
            transform: scale(1.1) translateY(-2px);
            box-shadow: 0 12px 35px rgba(102, 126, 234, 0.6);
        }
        .send-btn:active {
            transform: scale(0.95);
        }
        .reset-btn {
            background: linear-gradient(135deg, #ff6b9d, #ee5a6f);
            border: none;
            border-radius: 18px;
            padding: 10px 20px;
            cursor: pointer;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            color: white;
            font-size: 0.9rem;
            font-weight: 600;
            box-shadow: 0 4px 15px rgba(255, 107, 157, 0.3);
        }
        .reset-btn:hover {
            transform: translateY(-3px);
            box-shadow: 0 8px 25px rgba(255, 107, 157, 0.5);
        }
        .typing-indicator {
            display: none;
            align-items: center;
            gap: 12px;
            padding: 18px 24px;
            max-width: 100px;
            background: linear-gradient(135deg, rgba(30, 30, 30, 0.8), rgba(45, 45, 45, 0.8));
            border-radius: 24px;
            border-bottom-left-radius: 8px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            align-self: flex-start;
        }
        .typing-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: linear-gradient(45deg, #ff6b9d, #4834d4);
            animation: typing 1.4s infinite;
        }
        .typing-dot:nth-child(2) { animation-delay: 0.2s; }
        .typing-dot:nth-child(3) { animation-delay: 0.4s; }
        @keyframes typing {
            0%, 60%, 100% { transform: translateY(0); opacity: 0.4; }
            30% { transform: translateY(-12px); opacity: 1; }
        }
        @media (max-width: 768px) {
            .container {
                padding: 15px;
            }
            
            .chat-container {
                height: 75vh;
                max-width: 100%;
                border-radius: 20px;
            }
            
            .logo {
                font-size: 2.2rem;
            }
            
            .message {
                max-width: 85%;
                padding: 15px 20px;
            }
            
            .chat-window {
                padding: 20px 25px;
            }
            
            .chat-input-container {
                padding: 20px 25px;
            }
        }
    </style>
</head>
<body>
    <div class="grid-overlay"></div>
    <div class="floating-elements">
        <div class="floating-orb orb-1"></div>
        <div class="floating-orb orb-2"></div>
        <div class="floating-orb orb-3"></div>
    </div>
    
    <div class="container">
        <div class="header">
            <div class="logo">Haven AI</div>
            <div class="tagline">Understanding you better</div>
            <div class="subtitle">Conversations that matter</div>
        </div>
        
        <div class="chat-container">
            <div class="chat-header">
                <div class="chat-title">Haven Assistant</div>
                <div class="header-controls">
                    <div class="status-indicator">
                        <div class="status-dot"></div>
                        <span class="status-text">Online</span>
                    </div>
                    <button class="reset-btn" onclick="resetChat()">Reset Chat</button>
                </div>
            </div>
            
            <div class="chat-window" id="chatWindow">
                <div class="message bot">Hello! I'm Haven, your AI companion. How are you feeling today? 🌟</div>
            </div>
            
            <div class="typing-indicator" id="typingIndicator">
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
            </div>
            
            <div class="chat-input-container">
                <div class="chat-input">
                    <div class="input-wrapper">
                        <input type="text" id="userInput" placeholder="Share your thoughts..." onkeypress="handleKeyPress(event)">
                    </div>
                    <button class="send-btn" onclick="sendMessage()">
                        <svg width="22" height="22" viewBox="0 0 24 24" fill="none">
                            <path d="M22 2L11 13" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>
                            <path d="M22 2L15 22L11 13L2 9L22 2Z" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>
                        </svg>
                    </button>
                </div>
            </div>
        </div>
    </div>
    <script>
        const chatWindow = document.getElementById('chatWindow');
        const userInput = document.getElementById('userInput');
        const typingIndicator = document.getElementById('typingIndicator');
        function appendMessage(sender, message) {
            const messageDiv = document.createElement('div');
            messageDiv.classList.add('message', sender);
            messageDiv.innerHTML = message.replace(/\\n/g, '<br>');
            chatWindow.appendChild(messageDiv);
            chatWindow.scrollTop = chatWindow.scrollHeight;
        }
        function showTyping() {
            typingIndicator.style.display = 'flex';
            chatWindow.appendChild(typingIndicator);
            chatWindow.scrollTop = chatWindow.scrollHeight;
        }
        function hideTyping() {
            typingIndicator.style.display = 'none';
        }
        async function sendMessage() {
            const message = userInput.value.trim();
            if (message === '') return;
            appendMessage('user', message);
            userInput.value = '';
            showTyping();
            try {
                const response = await fetch('/chat', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ message: message })
                });
                const data = await response.json();
                
                // Realistic thinking time
                setTimeout(() => {
                    hideTyping();
                    appendMessage('bot', data.response);
                }, Math.random() * 1000 + 800);
                
            } catch (error) {
                console.error('Error:', error);
                hideTyping();
                appendMessage('bot', 'Oops! Something went wrong. Please try again. 😅');
            }
        }
        async function resetChat() {
            if (confirm("Are you sure you want to reset the conversation? This cannot be undone.")) {
                try {
                    const response = await fetch('/reset', {
                        method: 'POST'
                    });
                    const data = await response.json();
                    if (data.status === "Conversation reset") {
                        chatWindow.innerHTML = "<div class='message bot'>Hello! I'm Haven, your AI companion. How are you feeling today? 🌟</div>";
                        appendMessage('bot', 'Your conversation has been reset. Let\\'s start fresh! ✨');
                    } else {
                        appendMessage('bot', 'Failed to reset conversation. Please try again.');
                    }
                } catch (error) {
                    console.error('Error resetting chat:', error);
                    appendMessage('bot', 'Oops! Could not reset the chat. Please try refreshing the page.');
                }
            }
        }
        function handleKeyPress(event) {
            if (event.key === 'Enter') {
                sendMessage();
            }
        }
        // Auto-focus on input when page loads
        window.onload = function() {
            userInput.focus();
        };
        // Add some dynamic interactions
        document.addEventListener('mousemove', (e) => {
            const orbs = document.querySelectorAll('.floating-orb');
            orbs.forEach((orb, index) => {
                const speed = (index + 1) * 0.00005;
                const x = (e.clientX * speed);
                const y = (e.clientY * speed);
                orb.style.transform += ` translate(${x}px, ${y}px)`;
            });
        });
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    """Renders the main chat interface."""
    log_interaction("PAGE_LOAD", ip_address=request.remote_addr)
    print("--- Debug: Serving Dark Mode Colorful AI Chat Interface ---")
    return render_template_string(HTML_TEMPLATE)

@app.route('/chat', methods=['POST'])
def chat():
    """Handles user messages and returns bot responses."""
    user_message = request.json.get('message')
    session_id = request.remote_addr 

    if not user_message:
        log_interaction("CHAT_ERROR", user_input="None", bot_response="No message received", ip_address=session_id)
        return jsonify({"response": "No message received."}), 400

    bot_response = get_haven_response(user_message, session_id)
    return jsonify({"response": bot_response})

@app.route('/reset', methods=['POST'])
def reset():
    """Resets the conversation history for the current session."""
    session_id = request.remote_addr
    reset_conversation(session_id)
    log_interaction("RESET_REQUEST", ip_address=session_id)
    return jsonify({"status": "Conversation reset"}), 200

if __name__ == '__main__':
    init_db() 
    port = int(os.environ.get('PORT', 7860))
    debug_mode = os.environ.get('FLASK_DEBUG') == '1' or os.environ.get('FLASK_DEBUG', 'false').lower() == 'true'
    
    print(f"Starting Haven AI on http://0.0.0.0:{port} (Debug Mode: {debug_mode})")
    app.run(host='0.0.0.0', port=port, debug=debug_mode)