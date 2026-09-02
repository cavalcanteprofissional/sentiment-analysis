import gradio as gr

from model import load_classifier

classifier = load_classifier()


def analyze_sentiment(text):
    if not text.strip():
        return "Please enter some text to analyze."
    result = classifier(text)
    return {result[0]["label"]: result[0]["score"]}


gradio_app = gr.Interface(
    analyze_sentiment,
    inputs=gr.Textbox(
        label="Enter text to analyze", placeholder="I love this product! It's amazing...", lines=3
    ),
    outputs=gr.Label(label="Sentiment Analysis Result"),
    title="Sentiment Analyzer",
    description="Analyzes the sentiment of text as POSITIVE or NEGATIVE",
    examples=[
        ["I love this course! The teacher is amazing."],
        ["This was a terrible experience, I'm very disappointed."],
        ["The movie was okay, not the best but not the worst."],
    ],
)

if __name__ == "__main__":
    gradio_app.launch()
