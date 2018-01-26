#coding=utf-8
import sqlite3

conn = sqlite3.connect('db.sqlite3')
print ("Opened database successfully")
c = conn.cursor()
cursor = c.execute("SELECT id, question_text, pub_date from polls_question")
for row in cursor:
    print(row)
print ("Operation done successfully")
conn.close()