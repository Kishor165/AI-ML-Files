#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install logging


# In[2]:


import logging


# In[4]:


# Set up basic logging 
logging.basicConfig( 
filename='etl_log.txt', 
level=logging.INFO, 
format='%(asctime)s - %(levelname)s - %(message)s' 
)


# In[5]:


import pandas as pd 
import numpy as np 
import mysql.connector 
import datetime


# In[6]:


csv_file_path = 'employees1.csv' 
df = pd.read_csv(csv_file_path) 
print("Raw data loaded:") 
print(df.head()) 
print(df.columns.tolist()) 
logging.info("CSV loaded successfully.")


# In[7]:


# Handle missing values 
df.fillna({ 
'EMAIL': 'not_provided@example.com', 
'PHONE_NUMBER': '0000000000', 
'HIRE_DATE': '01-Jan-00', 
'SALARY': 0 
}, inplace=True)


# In[8]:


# Standardize column names (optional) 
df.columns = [col.strip().lower().replace(' ', '_') for col in df.columns] 


# In[9]:


print(df.columns.tolist()) 
# Convert hire_date from 'dd-MMM-yy' to 'YYYY-MM-DD' 
df['hire_date'] = pd.to_datetime(df['hire_date'], format='%d-%b-%y', 
errors='coerce')


# In[10]:


# Replace invalid dates with a default 
df['hire_date'] = df['hire_date'].fillna(pd.to_datetime('2000-01-01')) 
# Replace non-numeric salaries with 0 
df['salary'] = pd.to_numeric(df['salary'], errors='coerce').fillna(0).astype(int) 
logging.info("Data cleaning completed.")


# In[11]:


# ----------------------------- 
# Step 3: Load - Insert into MySQL 
# ----------------------------- 
# MySQL connection


# In[41]:


mydb = mysql.connector.connect( 
    host="localhost", 
    user="root", 
    password="28082004", 
   database="employee"  
)


# In[42]:


cursor = mydb.cursor()


# In[43]:


# Create table if not exists 
cursor.execute(""" CREATE TABLE IF NOT EXISTS salary_2 ( empid INT, firstname VARCHAR(50), lastname VARCHAR(50), email VARCHAR(100), phone VARCHAR(20), hire_date DATE, job_id VARCHAR(20), salary INT ) """)


# In[44]:


# Create table if not exists 
cursor.execute(""" 
    CREATE TABLE IF NOT EXISTS salary_2 ( 
        empid INT, 
        firstname VARCHAR(50), 
        lastname VARCHAR(50), 
        email VARCHAR(100), 
        phone VARCHAR(20), 
        hire_date DATE, 
        job_id VARCHAR(20), 
        salary INT 
    ) 
""")


# In[50]:


# Insert rows
for _, row in df.iterrows():
    sql = """
        INSERT INTO salary_2 (empid, firstname, lastname, email, phone, hire_date, job_id, salary)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            firstname=%s,
            lastname=%s,
            email=%s,
            phone=%s,
            hire_date=%s,
            job_id=%s,
            salary=%s
    """


# In[51]:


values = (
    int(row['employee_id']),
    row['first_name'],
    row['last_name'],
    row['email'],
    row['phone_number'],
    row['hire_date'].date(),
    row['job_id'],
    int(row['salary']),
    row['first_name'],
    row['last_name'],
    row['email'],
    row['phone_number'],
    row['hire_date'].date(),
    row['job_id'],
    int(row['salary'])
)


# In[52]:


CREATE TABLE IF NOT EXISTS salary_2 (
    empid INT PRIMARY KEY,
    firstname VARCHAR(50),
    lastname VARCHAR(50),
    email VARCHAR(100),
    phone VARCHAR(20),
    hire_date DATE,
    job_id VARCHAR(20),
    salary INT
)


# In[ ]:




