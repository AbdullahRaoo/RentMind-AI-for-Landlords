import pandas as pd
import numpy as np
from faker import Faker
from datetime import datetime, timedelta
import random
import os

fake = Faker('en_UK')

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)

# --- 1. Property Metadata ---
properties = []
cities = ['London', 'Manchester', 'Birmingham', 'Leeds', 'Bristol']
construction_types = ['Victorian Brick', 'Post-War Concrete', 'Modern Timber Frame', 'Edwardian Stone']

for i in range(1, 101):  # 100 properties
    age = random.randint(5, 120)  # UK homes can be very old
    properties.append({
        'property_id': i,
        'address': fake.street_address() + ', ' + random.choice(cities),
        'age_years': age,
        'construction_type': random.choice(construction_types),
        'hvac_age': min(age, random.randint(5, 25)),  # HVAC usually newer than building
        'plumbing_age': min(age, random.randint(10, 40)),
        'roof_age': min(age, random.randint(5, 30)),
        'last_inspection_date': fake.date_between(start_date='-2y', end_date='today').strftime('%Y-%m-%d')
    })

df_properties = pd.DataFrame(properties)

# --- 2. Maintenance Logs ---
maintenance_issues = [
    'Boiler Failure', 'Damp/Mould', 'Roof Leak', 'Electrical Fault',
    'Pipe Burst', 'Pest Infestation', 'Window Repair'
]
severity = ['Low', 'Medium', 'High', 'Critical']

logs = []
for pid in range(1, 101):
    for _ in range(random.randint(1, 5)):  # 1-5 logs per property
        issue_date = fake.date_between(
            start_date=datetime.strptime(df_properties.loc[pid-1, 'last_inspection_date'], '%Y-%m-%d'),
            end_date='today'
        ).strftime('%Y-%m-%d')
        logs.append({
            'log_id': len(logs) + 1,
            'property_id': pid,
            'issue_type': random.choice(maintenance_issues),
            'severity': random.choices(severity, weights=[0.3, 0.4, 0.2, 0.1])[0],
            'date': issue_date,
            'cost': round(random.uniform(50, 2000), 2),
            'resolved': random.choice([True, False]),
            'notes': fake.sentence()
        })

df_logs = pd.DataFrame(logs)

# --- 3. Tenant Reports ---
tenant_issues = [
    'No Heating', 'Damp Patch', 'Leaking Tap', 'Broken Lock',
    'Mould Growth', 'Toilet Blocked', 'Fuse Tripped'
]

reports = []
for pid in range(1, 101):
    for _ in range(random.randint(0, 3)):  # 0-3 reports per property
        reports.append({
            'report_id': len(reports) + 1,
            'property_id': pid,
            'issue_reported': random.choice(tenant_issues),
            'date': fake.date_between(start_date='-6m', end_date='today').strftime('%Y-%m-%d'),
            'urgency': random.randint(1, 5),
            'status': random.choice(['Pending', 'Resolved', 'Investigating'])
        })

df_reports = pd.DataFrame(reports)

# --- 4. Seasonal/Risk Factors ---
months = ['January', 'February', 'March', 'April', 'May', 'June',
          'July', 'August', 'September', 'October', 'November', 'December']
seasonal_data = []

for pid in range(1, 101):
    for month in months:
        avg_temp = random.randint(2, 22)  # UK temps in °C
        rainfall = round(random.uniform(0.5, 5.0), 1)  # Rainfall in mm
        # Lower probability of high/medium risk for more realistic data
        rand_val = random.random()
        if rand_val < 0.05:  # 5% chance of high risk
            risk_score = random.randint(8, 10)
        elif rand_val < 0.20:  # 15% chance of medium risk
            risk_score = random.randint(5, 7)
        else:  # 80% chance of low risk
            risk_score = random.randint(1, 4)
        seasonal_data.append({
            'property_id': pid,
            'month': month,
            'avg_temp_c': avg_temp,
            'rainfall_mm': rainfall,
            'pest_alert': random.choices(['None', 'Rodents', 'Damp Woodlice', 'Wasps'], weights=[0.7, 0.1, 0.1, 0.1])[0],
            'maintenance_risk_score': risk_score
        })

df_seasonal = pd.DataFrame(seasonal_data)

# --- 5. IoT Sensor Data (Optional) ---
sensor_types = ['Water Flow', 'Boiler Pressure', 'CO2', 'Humidity']
sensor_data = []

for pid in range(1, 101):
    for _ in range(random.randint(1, 3)):  # 1-3 sensors per property
        sensor_type = random.choice(sensor_types)
        reading = str(round(random.uniform(0, 100), 2)) + (
            ' gal/hr' if 'Water' in sensor_type else 
            ' psi' if 'Pressure' in sensor_type else 
            ' ppm' if 'CO2' in sensor_type else '%'
        )
        sensor_data.append({
            'property_id': pid,
            'sensor_type': sensor_type,
            'reading': reading,
            'timestamp': fake.date_time_between(start_date='-1m', end_date='now').strftime('%Y-%m-%d %H:%M:%S'),
            'alert_triggered': random.choice([True, False])
        })

df_sensors = pd.DataFrame(sensor_data)

# --- Export to CSV ---
data_dir = 'data/'
os.makedirs(data_dir, exist_ok=True)
df_properties.to_csv(data_dir + 'uk_properties.csv', index=False)
df_logs.to_csv(data_dir + 'uk_maintenance_logs.csv', index=False)
df_reports.to_csv(data_dir + 'uk_tenant_reports.csv', index=False)
df_seasonal.to_csv(data_dir + 'uk_seasonal_risks.csv', index=False)
df_sensors.to_csv(data_dir + 'uk_iot_sensors.csv', index=False)