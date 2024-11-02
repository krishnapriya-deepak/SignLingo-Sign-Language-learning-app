
from flask import Flask, render_template, request, jsonify
from recommender import SignLingoKMeansRecommender
import pandas as pd

app = Flask(__name__)

# Initialize sample data and recommender
sample_data = pd.DataFrame({
    'video_link': [
        'https://www.youtube.com/watch?v=nuYcIMq8e5U', 
        'https://www.youtube.com/watch?v=JC80hJOObmg', 
        'https://www.youtube.com/watch?v=GYVZ3VpzJRI', 
        'https://www.youtube.com/watch?v=fVnCA91Bvwo', 
        'https://www.youtube.com/watch?v=wubfL2VbBLY',
        'https://www.youtube.com/watch?v=mKxgmzevEio',
        'https://www.youtube.com/watch?v=SUGgjeP54CQ',
        'https://www.youtube.com/watch?v=HufgJPpb1kQ',
        'https://www.youtube.com/watch?v=wDy5hBdrwoc'
    ],
    'word': ['apple', 'mango', 'banana', 'dog', 'cat', 'bird', 'car', 'bus', 'train'],
    'difficulty': [1, 2, 2, 3, 2, 2, 3, 3, 4],
    'views': [1000, 800, 900, 1200, 1100, 950, 1500, 1300, 1400],
    'rating': [4.5, 4.2, 4.3, 4.7, 4.6, 4.4, 4.8, 4.5, 4.6]
})

recommender = SignLingoKMeansRecommender(sample_data, n_clusters=3)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    word = request.form.get('word', '').lower().strip()
    
    if not word:
        return jsonify({'error': 'Please enter a word'})
    
    try:
        recommendations = recommender.recommend(word, top_n=5)
        # Convert to dictionary for JSON response
        results = recommendations.to_dict('records')
        return jsonify({'recommendations': results})
    except Exception as e:
        return jsonify({'error': f'Error getting recommendations: {str(e)}'})

if __name__ == '__main__':
    app.run(debug=True)



