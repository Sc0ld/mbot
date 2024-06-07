from flask import Flask, request, render_template, jsonify
from flask_cors import CORS
from ielts_grader import IELTSGrader

app = Flask(__name__)
CORS(app)
grader = IELTSGrader()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            question = request.form['question']
            essay = request.form['essay']
            result = grader.grade_essay(essay, question)
            return render_template('result.html', result=result)
        except KeyError as e:
            return jsonify({"error": f"Missing form parameter: {str(e)}"}), 400
        except Exception as e:
            return jsonify({"error": f"An error occurred: {str(e)}"}), 500
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
