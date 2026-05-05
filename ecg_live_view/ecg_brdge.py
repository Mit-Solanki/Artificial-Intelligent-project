import serial, asyncio, websockets

PORT = "COM7"
BAUD = 115200

async def stream(websocket):
    ser = serial.Serial(PORT, BAUD, timeout=1)
    print(f"Connected to {PORT}")
    try:
        while True:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            if "Raw_ECG:" in line:
                try:
                    val = int(line.split(',')[0].split(':')[1])
                    await websocket.send(str(val))
                except:
                    pass
    finally:
        ser.close()

async def main():
    async with websockets.serve(stream, "localhost", 8765):
        print("Bridge running on ws://localhost:8765")
        await asyncio.Future()

asyncio.run(main())