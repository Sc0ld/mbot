from flask import Flask, request, render_template
from ielts_grader import IELTSGrader

app = Flask(__name__)
grader = IELTSGrader()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/grade', methods=['POST'])
def grade_essay():
    question = request.form['question']
    essay = request.form['essay']
    result = grader.grade_essay(essay, question)
    return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)