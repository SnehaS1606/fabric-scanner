import sounddevice as sd

print(sd.query_devices())
print("\nDefault input device index:", sd.default.device[0])
