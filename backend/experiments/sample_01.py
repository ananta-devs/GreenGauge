from codecarbon import EmissionsTracker

tracker = EmissionsTracker(
    project_name="green-ai-sample-01",
    tracking_mode="process"
)

tracker.start()

total = 0
for i in range(10_000_000):
    total += i

emissions = tracker.stop()

print("Result:", total)
print("CO2e:", emissions, "kg")