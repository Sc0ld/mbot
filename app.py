from flask import Flask, request, redirect, url_for

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/grade', methods=['POST'])
def grade():
    question = request.form['question']
    essay = request.form['essay']
    
    # Process the question and essay
    # e.g., grading logic here
    
    return 'Your grade has been calculated.'

if __name__ == '__main__':
    app.run(debug=false)
