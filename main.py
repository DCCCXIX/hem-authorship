#!/usr/bin/env python
import os
from flask import Flask, request, render_template, url_for, redirect
import predict
import train

app = Flask(__name__)

@app.route("/")
def main_form():
    return render_template("main_form.html")

@app.route("/", methods=["POST"])
def main_form_post():
        if 'Train the predictor' in request.form:
            print('train')
            return redirect(url_for("training_form"))
        elif 'Predict text authorship' in request.form:
            print('predict')
            return redirect(url_for("predict_form"))
        else:
            print('else')
            return render_template("main_form.html")

@app.route("/predict")
def predict_form():
    return render_template("prediction_form.html")

@app.route("/predict", methods=["POST"])
def predict_form_post():
    text = request.form["text"]
    #the same featurizing process which is used for training/testing data is applied to predictor's input text
    #it splits the text into batches of sentences, which is set in featurize()
    #predictor returns an average of predicted values for all the batches in the text
    #may add an option to pass unsplitted corpus to predictor later
    predictor = predict.Predictor(text, sentence_amount = 0, step = 0)
    try:
        result = predictor.predict_author()
        if result < 0.5:
            explanation = 'This text\'s authorship most likely belongs to Ernest Hemingway'
        else:
            explanation = 'This text\'s authorship most likely doesn\'t belong to Ernest Hemingway'
    except AttributeError:
        result = 'Train the predictor first'
        explanation = 'Predictor modules are missing'
    return render_template("prediction_result_form.html", result=result, explanation=explanation)

is_training = False
@app.route("/train")
def training_form():
    return render_template("training_form.html", is_training=is_training)

@app.route("/train", methods=["POST"])
def training_form_post():
    global is_training
    if 'train' in request.form:
        if is_training:
            return render_template("training_form.html", is_training=is_training)
        else:
            is_training = True
            trainer = train.Trainer(sentence_amount=5, step=1)
            clf_report = trainer.train_model()
            is_training = False
    return render_template("training_result_form.html", clf_report=clf_report)

if __name__ == '__main__':
    app.run(debug=True, port = int(os.environ.get('PORT', 33507)))
