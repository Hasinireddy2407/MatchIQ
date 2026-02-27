from flask import Flask
from routes.matching import matching_bp

app = Flask(__name__)

app.register_blueprint(matching_bp)

if __name__ == "__main__":
    app.run(debug=True)