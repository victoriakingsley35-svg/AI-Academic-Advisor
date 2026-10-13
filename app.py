from flask import Flask, render_template_string, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained machine learning model safely
try:
    model = pickle.load(open('academic_model.pkl', 'rb'))
except Exception as e:
    model = None

# Session history store (in-memory for demonstration)
evaluation_history = []

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Academic Advisor System - Advanced Edition</title>
    <style>
        :root {
            --primary: #4f46e5;
            --primary-hover: #4338ca;
            --bg-color: #f1f5f9;
            --card-bg: #ffffff;
            --text-main: #1e293b;
            --text-muted: #64748b;
        }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: var(--bg-color);
            color: var(--text-main);
            margin: 0;
            padding: 30px;
            display: flex;
            justify-content: center;
        }
        .wrapper {
            width: 100%;
            max-width: 900px;
        }
        header {
            text-align: center;
            margin-bottom: 30px;
        }
        h1 {
            color: var(--primary);
            margin: 0 0 8px 0;
            font-size: 28px;
        }
        p.subtitle {
            color: var(--text-muted);
            margin: 0;
            font-size: 15px;
        }
        .grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 24px;
        }
        @media (max-width: 768px) {
            .grid { grid-template-columns: 1fr; }
        }
        .card {
            background: var(--card-bg);
            padding: 25px;
            border-radius: 12px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -2px rgba(0, 0, 0, 0.1);
        }
        h3 {
            margin-top: 0;
            color: var(--text-main);
            border-bottom: 2px solid #f1f5f9;
            padding-bottom: 10px;
            font-size: 18px;
        }
        .form-group {
            margin-bottom: 16px;
        }
        label {
            display: block;
            margin-bottom: 6px;
            font-weight: 600;
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: var(--text-muted);
        }
        input {
            width: 100%;
            padding: 10px 12px;
            box-sizing: border-box;
            border: 1px solid #cbd5e1;
            border-radius: 6px;
            font-size: 14px;
        }
        input:focus {
            outline: none;
            border-color: var(--primary);
            box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.15);
        }
        button {
            background: var(--primary);
            color: white;
            border: none;
            padding: 12px;
            width: 100%;
            cursor: pointer;
            border-radius: 6px;
            font-size: 15px;
            font-weight: 600;
            transition: background 0.2s;
            margin-top: 5px;
        }
        button:hover {
            background: var(--primary-hover);
        }
        .badge {
            display: inline-block;
            padding: 6px 12px;
            border-radius: 20px;
            font-weight: 700;
            font-size: 14px;
            margin-bottom: 15px;
        }
        .Safe { background: #dcfce7; color: #166534; }
        .Monitor { background: #e0f2fe; color: #0369a1; }
        .Critical { background: #fee2e2; color: #991b1b; }
        
        .advice-box {
            background: #f8fafc;
            border-left: 4px solid var(--primary);
            padding: 12px 16px;
            border-radius: 0 6px 6px 0;
            font-size: 14px;
            line-height: 1.5;
            margin-top: 10px;
        }
        .history-list {
            list-style: none;
            padding: 0;
            margin: 0;
            max-height: 250px;
            overflow-y: auto;
        }
        .history-item {
            padding: 8px 12px;
            border-bottom: 1px solid #f1f5f9;
            font-size: 13px;
            display: flex;
            justify-content: space-between;
        }
    </style>
</head>
<body>
    <div class="wrapper">
        <header>
            <h1>🎓 AI Academic Advisor System</h1>
            <p class="subtitle">CSC 303 Advanced Software Engineering & AI Integration Assignment</p>
        </header>

        <div class="grid">
            <!-- Left Card: Input Form -->
            <div class="card">
                <h3>Student Metrics Input</h3>
                <form method="POST">
                    <div class="form-group">
                        <label>Current GPA (0.0 - 5.0)</label>
                        <input type="number" step="0.1" min="0.0" max="5.0" name="gpa" value="{{ request.form.get('gpa', '3.5') }}" required>
                    </div>
                    
                    <div class="form-group">
                        <label>Weekly Study Hours</label>
                        <input type="number" min="0" max="168" name="hours" value="{{ request.form.get('hours', '18') }}" required>
                    </div>
                    
                    <div class="form-group">
                        <label>Previous Course Failures</label>
                        <input type="number" min="0" max="20" name="failures" value="{{ request.form.get('failures', '0') }}" required>
                    </div>
                    
                    <button type="submit">Run AI Evaluation</button>
                </form>
            </div>

            <!-- Right Card: Evaluation & Recommendations -->
            <div class="card">
                <h3>Advisor Evaluation & Recommendations</h3>
                {% if prediction %}
                    <div>
                        <span class="badge {{ prediction }}">Status: {{ prediction }}</span>
                        {% if confidence %}
                            <p style="font-size: 13px; color: var(--text-muted); margin: 0 0 10px 0;">Model Confidence: <strong>{{ confidence }}%</strong></p>
                        {% endif %}
                        
                        <div class="advice-box">
                            <strong>Personalized Academic Action Plan:</strong><br><br>
                            {{ advice|safe }}
                        </div>
                    </div>
                {% else %}
                    <p style="color: var(--text-muted); font-size: 14px; text-align: center; margin-top: 40px;">
                        Enter your metrics on the left and click <strong>Run AI Evaluation</strong> to generate your tailored advisory report.
                    </p>
                {% endif %}
            </div>
        </div>

        {% if history %}
        <div class="card" style="margin-top: 24px;">
            <h3>Recent Session Evaluations</h3>
            <ul class="history-list">
                {% for item in history %}
                <li class="history-item">
                    <span>GPA: <strong>{{ item.gpa }}</strong> | Study Hours: <strong>{{ item.hours }}h</strong> | Failures: <strong>{{ item.failures }}</strong></span>
                    <span class="</span><strong>{{ item.prediction }}</strong></span>
                </li>
                {% endfor %}
            </ul>
        </div>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    confidence = None
    advice = ""
    
    if request.method == 'POST':
        try:
            gpa = float(request.form['gpa'])
            hours = float(request.form['hours'])
            failures = int(request.form['failures'])
            
            if model:
                input_data = np.array([[gpa, hours, failures]])
                prediction = model.predict(input_data)[0]
                
                # Get prediction probabilities if available
                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba(input_data)[0]
                    confidence = round(float(max(probs)) * 100, 1)
                
                # Generate detailed recommendations based on prediction and stats
                if prediction == 'Safe':
                    advice = "✨ Excellent standing! You are maintaining a strong academic balance. Consider applying for undergraduate research assistantships, honors electives, or peer tutoring leadership roles."
                elif prediction == 'Monitor':
                    advice = "⚠️ Moderate standing. Your metrics indicate you are stable, but susceptible to burnout or grade slips. Try increasing your weekly study hours by 4-6 hours and review difficult problem sets proactively during office hours."
                else:
                    advice = "🚨 Critical intervention recommended. Your current trajectory poses academic risk. Please schedule a mandatory meeting with your faculty advisor this week, drop non-essential commitments, and utilize university peer-assisted study sessions."
                
                # Save to history log
                evaluation_history.insert(0, {'gpa': gpa, 'hours': hours, 'failures': failures, 'prediction': prediction})
                if len(evaluation_history) > 5:
                    evaluation_history.pop()
            else:
                prediction = "Error"
                advice = "Model file not found."
        except Exception as e:
            prediction = "Error"
            advice = "Invalid input values provided."
            
    return render_template_string(HTML_TEMPLATE, prediction=prediction, confidence=confidence, advice=advice, history=evaluation_history)

if __name__ == '__main__':
    app.run(debug=True)