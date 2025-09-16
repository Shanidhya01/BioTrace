import os, sys
try:
    import requests
except Exception:
    print('requests not installed, installing...')
    os.system(f'{sys.executable} -m pip install requests')
    import requests
# create a small CSV
csv_path = r'd:\EDNA\edna-backend\data\sample_input_for_test.csv'
with open(csv_path,'w',encoding='utf-8') as f:
    f.write('sequence\nATCGATCGATCG\n')
url = 'http://127.0.0.1:8000/api/upload-csv/'
print('Posting to', url)
with open(csv_path,'rb') as fh:
    files = {'file': ('sample.csv', fh, 'text/csv')}
    try:
        r = requests.post(url, files=files, timeout=10)
        print('status_code=', r.status_code)
        try:
            print('json=', r.json())
        except Exception:
            print('text=', r.text[:1000])
    except Exception as e:
        print('Request failed:', e)
