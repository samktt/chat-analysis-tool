from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt

column_names = ['sender', 'message', 'date']
df = pd.read_csv('../conversation_export.txt', names=column_names)

df['date'] = pd.to_datetime(df['date'])


plt.figure(figsize=(10, 6))
day_of_week = daily_message_counts.index.to_series().apply(lambda x: x.weekday())  # 0=Monday, 6=Sunday

# Displaying the weekend bars (Saturday and Sunday) as red and the weekday bars as blue
colors = ['red' if day == 4 else 'blue' for day in day_of_week]
daily_message_counts.plot(kind='bar', color=colors)

num_ticks = 10
tick_positions = [int(i) for i in range(0, len(daily_message_counts), len(daily_message_counts) // num_ticks)]
tick_labels = [str(date) for date in daily_message_counts.index[tick_positions]]

# Visualizing the trend over time
plt.title('Histogram of Messages Sent per Day')
plt.xlabel('Date')
plt.ylabel('Number of Messages')
plt.xticks(tick_positions, tick_labels, rotation=45)
plt.tight_layout()
plt.show()

# Calculating average texts sent for each day of the week
day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
average_texts_per_day = []

for day_index in range(7):
    average_texts = df.groupby(df['date'].dt.date[df['date'].dt.dayofweek == day_index])['message'].count().mean()
    average_texts_per_day.append((day_names[day_index], average_texts))

# Printing the results
for day, average_texts in average_texts_per_day:
    print(f"Average texts sent on {day}: {average_texts:.2f}")