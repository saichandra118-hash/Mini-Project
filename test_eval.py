import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import classification_report
from preprocessor import clean_text

df_fake = pd.read_csv('dataset/Fake.csv')
df_true = pd.read_csv('dataset/True.csv')
df_fake['label'] = 0
df_true['label'] = 1
df = pd.concat([df_fake, df_true], ignore_index=True)
df['cleaned'] = (df['title'] + ' ' + df['text']).apply(clean_text)

vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words='english', sublinear_tf=True)
X = vec.fit_transform(df['cleaned'])
y = df['label']

lr = LogisticRegression(class_weight='balanced', C=2.0, random_state=42).fit(X, y)

test_fakes = [
    'SHOCKING: Secret government lab exposed developing weather control weapons to manipulate world leaders!',
    'BOMBSHELL: Hollywood celebrities caught using ancient elixir from meteorite dust to reverse aging overnight!',
    'NASA admits the Moon is an artificial alien satellite broadcasting radio frequencies!',
    'Drinking tap water allows government satellites to read your thoughts directly via 5G towers!',
    'Miracle dandelion root cures all known diseases in forty-eight hours, big pharma furious!',
    'Aliens land on White House lawn demanding pizza and threatening to vaporize earth!',
    'Government prepares to ban the color purple nationwide starting this Friday!',
    'Supermarket barcode scanners contain microchips tracking your thoughts for shadow agencies!',
    'Top senator caught replacing official Constitution with fake replica made of cheese!',
    'Whistleblower claims smart refrigerators are secretly reporting midnight snacking to raise insurance rates!'
]

fake_preds = [lr.predict(vec.transform([clean_text(f)]))[0] for f in test_fakes]
print('Fake samples correctly predicted as FAKE (0):', fake_preds.count(0), 'out of', len(test_fakes))
for f, p in zip(test_fakes, fake_preds):
    tag = 'OK-FAKE' if p == 0 else 'FAIL-REAL'
    prob = lr.predict_proba(vec.transform([clean_text(f)]))[0]
    print(f"  [{tag}] Fake Prob: {prob[0]*100:.1f}% | {f[:60]}...")
