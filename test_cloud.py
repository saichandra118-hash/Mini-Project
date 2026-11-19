from cloud_service import analyze_news_with_cloud

fakes = [
    'Aliens land on White House lawn demanding pizza and threatening to vaporize earth!',
    'Government prepares to ban the color purple nationwide starting this Friday!',
    'Miracle dandelion root cures all known diseases in forty-eight hours, big pharma furious!'
]

reals = [
    'Senate passes bipartisan infrastructure package after months of negotiations providing federal funding for roads.',
    'European Union leaders agree on unified cybersecurity standards for digital platforms.'
]

print("=== TESTING FAKE NEWS CLOUD DETECTION ===")
for f in fakes:
    res = analyze_news_with_cloud(f)
    print(f"Verdict: {res['prediction']} ({res['confidence']}%) | Engine: {res['engine_name']} | {f[:45]}...")
    print(f"  Explanation: {res['explanation'][:80]}...")

print("\n=== TESTING REAL NEWS CLOUD DETECTION ===")
for r in reals:
    res = analyze_news_with_cloud(r)
    print(f"Verdict: {res['prediction']} ({res['confidence']}%) | Engine: {res['engine_name']} | {r[:45]}...")
    print(f"  Explanation: {res['explanation'][:80]}...")
