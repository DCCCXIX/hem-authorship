#!/usr/bin/env python
from flask import Flask, request, render_template
import predict
import train

app = Flask(__name__)

@app.route("/")
def display_form():
    return render_template("main_form.html")

@app.route("/", methods=["POST"])
def my_form_post():
    text = request.form["text"]
    #the same featurizing process which is used for training/testing data is applied to predictor's input text
    #it splits the text into batches of sentenses, which is set in featurize()
    #predictor returns an average of predicted values for all the batches in the text
    #may add an option to pass unsplitted corpus to predictor later
    predictor = predict.Predictor(text)
    result = predictor.predict_author()    
    return f"Result: {result}"

@app.route("/train")
def training():    
    trainer = train.Trainer()
    trainer.train_model()

if __name__ == '__main__':
    app.run()
