from flask import Flask, request, render_template, jsonify
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

app = Flask(__name__)


model_name = "facebook/bart-large-cnn"  

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def summarize():
    text = request.form['text']
    max_length = int(request.form['max_length'])  # Retrieve max_length from the form
    min_length = int(request.form['min_length'])  # Retrieve min_length from the form

    # Load the selected model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Create a summarization pipeline with the selected model and tokenizer
    summarizer = pipeline("summarization", model=model, tokenizer=tokenizer)
    summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)

    return jsonify({"summary": summary[0]['summary_text']})

if __name__ == '__main__':
    app.run(debug=True)
