import sqlite3
import json

db_path = r"c:\Users\FQ1089AU\DocuShield_AI-1\backend\docushield.db"
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

cursor.execute("SELECT id, file_name, file_type, file_path, fraud_score, risk_level, extracted_text, metadata_json, explainable_ai_reasons FROM documents WHERE id IN (6, 7)")
rows = cursor.fetchall()

for row in rows:
    print(f"=== ID: {row[0]}, Name: {row[1]} ===")
    print(f"Path: {row[3]}")
    print(f"Risk score / level: {row[4]}% / {row[5]}")
    print("Extracted text:")
    print(row[6].replace('\u20b9', 'Rs.'))
    print("Metadata JSON:")
    print(row[7])
    print("Explainable AI Reasons:")
    print(row[8])
    print("-" * 50)

conn.close()
