#!/usr/bin/env python
# coding: utf-8

# In[13]:


# !pip install mysql-connector-python
import mysql.connector
import csv
import datetime


# In[14]:


# Connect to MySQL
mydb = mysql.connector.connect(
    host='localhost',
    user='root',
    password='28082004'
    # database='employee'  # Uncomment if already created
)


# In[15]:


cursor = mydb.cursor()


# In[16]:


# Step 1: Create Database
cursor.execute("CREATE DATABASE IF NOT EXISTS employee")


# In[17]:


# Step 2: Use Database
cursor.execute("USE employee")


# In[18]:


# Step 3: Create Table
create_table_sql = """
CREATE TABLE IF NOT EXISTS salary (
    empid INT,
    firstname VARCHAR(100),
    lastname VARCHAR(100),
    email VARCHAR(50),
    phone VARCHAR(15),
    hire_date DATE,
    job_id VARCHAR(15),
    salary INT
)
"""
cursor.execute(create_table_sql)


# In[21]:


# Step 4: Read CSV and Insert Data
filename = "employees1.csv"

with open(filename, 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    next(csvreader)  # Skip header row
    for row in csvreader:
        print(row)  # Optional: print row to verify
        empid = int(row[0])
        firstname = row[1]
        lastname = row[2]
        email = row[3]
        phone = row[4]
        hire_date = datetime.datetime.strptime(row[5], '%d-%b-%y').date()
        job_id = row[6]
        salary = int(row[7])

        insert_sql = """
        INSERT INTO salary (empid, firstname, lastname, email, phone, hire_date, job_id, salary)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """
        val = (empid, firstname, lastname, email, phone, hire_date, job_id, salary)
        cursor.execute(insert_sql, val)

mydb.commit()
print("Data inserted successfully!")


# In[22]:


import os
print(os.getcwd())


# In[23]:


#get_ipython().system('jupyter nbconvert --to script MYSQL-DATABASE.ipynb')


# In[ ]:




