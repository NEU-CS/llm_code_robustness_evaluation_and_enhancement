from python_bleu import PythonBLEU
import json

bleu = PythonBLEU()
refs = []
preds = []
with open('test_results_test_same.json', 'r') as f:
    for line in f:
        data = json.loads(line)
        refs.append(data['gold'])
        preds.append(data['clean_code'])   
z = bleu._compute(preds, refs, smooth=False, max_order=4)
print(z)