# test_phase1.py
import asyncio
import aiohttp
import base64

async def send_request(session, url, image_b64, req_num):
    try:
        async with session.post(url, json={"base64_data": image_b64}) as resp:
            result = await resp.json()
            if 'error' in result:
                print(f"Request {req_num}: ERROR - {result['error']}")
            else:
                print(f"Request {req_num}: SUCCESS - {result.get('number', 'N/A')}")
    except Exception as e:
        print(f"Request {req_num}: EXCEPTION - {e}")

async def main():
    # Load a test image (replace with your actual test image)
    with open("C:/Users/User/Documents/projects/ANPR/decoded_images/vehicles_new_250925/1_AT-AD010_20250912_103815150.png", "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()
    
    url = "http://localhost:8080/analyze"
    
    async with aiohttp.ClientSession() as session:
        # Send 10 concurrent requests
        tasks = [send_request(session, url, img_b64, i) for i in range(10)]
        await asyncio.gather(*tasks)
    
    # Check stats
    async with aiohttp.ClientSession() as session:
        async with session.get("http://localhost:8080/queue/stats") as resp:
            stats = await resp.json()
            print(f"\nFinal Stats: {stats}")

asyncio.run(main())