from flask import Flask, request, render_template, jsonify
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

app = Flask(__name__)

model_name = "facebook/bart-base-cnn"  # Use a smaller model

# Load the model and tokenizer during application initialization
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def summarize():
    text = request.form['text']
    max_length = int(request.form['max_length'])
    min_length = int(request.form['min_length'])

    # Create a summarization pipeline with the loaded model and tokenizer
    summarizer = pipeline("summarization", model=model, tokenizer=tokenizer)
    summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)

    return jsonify({"summary": summary[0]['summary_text']})

if __name__ == '__main__':
    app.run(debug=True)
