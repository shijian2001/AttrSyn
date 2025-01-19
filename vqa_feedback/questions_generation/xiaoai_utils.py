import http.client
import json
import time

def openai_setup(key_path='./_OAI_KEY.txt'):
    with open(key_path) as f:
        key = f.read().strip()
    print("Read key from", key_path)
    return key

def openai_completion(
    prompt,
    model='gpt-4',
    temperature=0,
    return_response=False,
    max_tokens=500,
    api_key=None,
    retries=3,
    delay=2
):
    conn = http.client.HTTPSConnection("xiaoai.plus")
    
    payload = json.dumps({
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "model": model,
        "temperature": temperature,
        "presence_penalty": 0,
        "frequency_penalty": 0,
        "top_p": 1
    })
    
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }
    
    for attempt in range(retries):
        try:
            conn.request("POST", "/v1/chat/completions", payload, headers)
            res = conn.getresponse()
            
            # Check for success status code
            if res.status != 200:
                raise ValueError(f"Request failed with status {res.status}. Response: {res.read().decode()}")
            
            data = res.read()
            response = json.loads(data.decode("utf-8"))
            
            # Check if 'choices' is in the response
            if 'choices' in response and len(response['choices']) > 0:
                return response['choices'][0].get('message', {}).get('content', 'No content found')
            else:
                raise ValueError("API response does not contain 'choices' or it is empty.")
        
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < retries - 1:
                time.sleep(delay)
            else:
                raise ValueError(f"All attempts failed. Last error: {e}")
    
    raise ValueError("Failed to get valid response after retries.")
