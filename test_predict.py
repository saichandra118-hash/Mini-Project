import joblib
from preprocessor import clean_text

vec = joblib.load('model/tfidf_vectorizer.pkl')
model = joblib.load('model/fake_news_model.pkl')

test_samples = [
    'SHOCKING: Secret government lab exposed developing weather control weapons to manipulate world leaders!',
    'BOMBSHELL: Hollywood celebrities caught using ancient elixir from meteorite dust to reverse aging overnight!',
    'NASA admits the Moon is an artificial alien satellite broadcasting radio frequencies!',
    'Drinking tap water allows government satellites to read your thoughts directly via 5G towers!',
    'Miracle dandelion root cures all known diseases in forty-eight hours, big pharma furious!',
    'Aliens land on White House lawn demanding pizza and threatening to vaporize earth!',
    'Government prepares to ban the color purple nationwide starting this Friday!',
]

for s in test_samples:
    feat = vec.transform([clean_text(s)])
    pred = model.predict(feat)[0]
    prob = model.predict_proba(feat)[0]
    label = "REAL" if pred == 1 else "FAKE"
    print(f"Pred: {label:<4} (Fake: {prob[0]*100:5.1f}%, Real: {prob[1]*100:5.1f}%) | {s[:50]}...")
