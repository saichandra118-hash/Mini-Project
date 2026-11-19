"""
Comprehensive System Verification Test Suite for Fake News Detection System.
Tests all endpoints, input validations, ML predictions, per-class metrics,
and per-user session history isolation.
"""

from app import app

# Test 1: Health Check
client1 = app.test_client()
h = client1.get('/health').get_json()
assert h['status'] == 'healthy' and h['model_loaded'] is True, f"Health check failed: {h}"
print(f"[TEST 1/10 PASSED] Health check: {h['status']} (Model Loaded: {h['model_loaded']})")

# Test 2: Model Info and Per-Class Metrics
m = client1.get('/model-info').get_json()
assert m['status'] == 'success', f"Model info failed: {m}"
metrics = m['metrics']
assert 'class_metrics' in metrics, "Missing class_metrics in /model-info"
assert 'fake_class' in metrics['class_metrics'] and 'real_class' in metrics['class_metrics'], "Missing fake or real class metrics"
print(f"[TEST 2/10 PASSED] Model Info: {metrics['selected_model']} | Overall Accuracy: {metrics['performance']['accuracy']}%")
print(f"               FAKE Class -> Precision: {metrics['class_metrics']['fake_class']['precision']}% | Recall: {metrics['class_metrics']['fake_class']['recall']}% | F1: {metrics['class_metrics']['fake_class']['f1_score']}%")
print(f"               REAL Class -> Precision: {metrics['class_metrics']['real_class']['precision']}% | Recall: {metrics['class_metrics']['real_class']['recall']}% | F1: {metrics['class_metrics']['real_class']['f1_score']}%")

# Test 3: Root HTML Rendering
r = client1.get('/')
assert r.status_code == 200 and b'Fake News Detection System' in r.data, f"Root route failed with status {r.status_code}"
assert b'Separate Class Performance Metrics' in r.data, "Root route missing class metrics elements"
print("[TEST 3/10 PASSED] Root HTML served successfully (200 OK)")

# Test 4: Empty Input Validation
e1 = client1.post('/predict', json={'news': '   '}).get_json()
assert 'message' in e1 and 'empty' in e1['message'].lower(), f"Empty validation failed: {e1}"
print("[TEST 4/10 PASSED] Empty input correctly rejected (400 Bad Request)")

# Test 5: Short Input Validation
e2 = client1.post('/predict', json={'news': 'Too short'}).get_json()
assert 'message' in e2 and 'too short' in e2['message'].lower(), f"Short validation failed: {e2}"
print("[TEST 5/10 PASSED] Short input (< 15 chars) correctly rejected (400 Bad Request)")

# Test 6: Noise/Symbol-only Input Validation
e3 = client1.post('/predict', json={'news': '!@#$%^&*()_+{}[]:;"<>,.?/~`'}).get_json()
assert 'message' in e3 and ('symbols' in e3['message'].lower() or 'too few words' in e3['message'].lower()), f"Symbol validation failed: {e3}"
print("[TEST 6/10 PASSED] Noise/symbol-only input correctly rejected (400 Bad Request)")

# Test 7: Genuine News Prediction
real_input = "WASHINGTON (Reuters) - The United States Senate approved a bipartisan infrastructure package providing funding for municipal water systems, roads, and bridges across multiple states."
p_real = client1.post('/predict', json={'news': real_input}).get_json()
assert p_real['status'] == 'success' and p_real['prediction'] == 'REAL', f"Expected REAL, got {p_real}"
assert p_real['confidence'] is not None and p_real['confidence'] > 50, f"Expected confidence > 50%, got {p_real['confidence']}"
print(f"[TEST 7/10 PASSED] Genuine news classified: REAL | Probability Estimate: {p_real['confidence']}% ({p_real['score_label']})")

# Test 8: Fake News Prediction
fake_input = "SHOCKING BOMBSHELL: Whistleblowers expose underground base where secret weather control machines are causing worldwide storms to manipulate financial markets!"
p_fake = client1.post('/predict', json={'news': fake_input}).get_json()
assert p_fake['status'] == 'success' and p_fake['prediction'] == 'FAKE', f"Expected FAKE, got {p_fake}"
assert p_fake['confidence'] is not None and p_fake['confidence'] > 50, f"Expected confidence > 50%, got {p_fake['confidence']}"
print(f"[TEST 8/10 PASSED] Fake news classified: FAKE | Probability Estimate: {p_fake['confidence']}% ({p_fake['score_label']})")

# Test 9: Per-User Session History Isolation
client2 = app.test_client()
# Client 1 should have 2 predictions in history
hist1 = client1.get('/history').get_json()
assert len(hist1['history']) == 2, f"Client 1 should have 2 history items, got {len(hist1['history'])}"

# Client 2 is a separate user/browser session and should have 0 items!
hist2 = client2.get('/history').get_json()
assert len(hist2['history']) == 0, f"Client 2 history should be completely empty and isolated, got {len(hist2['history'])}"
print(f"[TEST 9/10 PASSED] Per-user session isolation verified: Client 1 has {len(hist1['history'])} items, Client 2 has {len(hist2['history'])} items")

# Test 10: History Clear
client1.post('/history/clear')
hist1_after = client1.get('/history').get_json()
assert len(hist1_after['history']) == 0, "Client 1 history should be empty after clear"
print("[TEST 10/10 PASSED] Session history cleared successfully for current user")

print("\n" + "=" * 80)
print("  ALL 10 VERIFICATION TESTS COMPLETED AND PASSED WITH 100% SUCCESS!")
print("=" * 80)
