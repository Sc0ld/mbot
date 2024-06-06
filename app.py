from flask import Flask, request, jsonify
from flask_cors import CORS
from ielts_grader import IELTSGrader

app = Flask(__name__)
CORS(app)
grader = IELTSGrader()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/grade', methods=['POST'])
def grade_essay():
    try:
        question = request.form['question']
        essay = request.form['essay']
        result = grader.grade_essay(essay, question)
        return jsonify(result)
    except KeyError as e:
        return jsonify({"error": f"Missing form parameter: {str(e)}"}), 400
    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)
