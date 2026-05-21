# Explainable AI Data Analyst System

## Overview

ระบบ AI Analyst ที่สามารถ:

- รับคำถามด้วยภาษาธรรมชาติ
- ค้นหาไฟล์ข้อมูลที่เกี่ยวข้อง
- วิเคราะห์ schema ของข้อมูล
- สร้าง Python/Pandas code อัตโนมัติ
- รันโค้ดจริง
- วิเคราะห์ผลลัพธ์
- อธิบายวิธีคิด (Reasoning)
- สรุป Insight ให้ผู้ใช้เข้าใจ

---

# Example User Question

```text
ขอข้อมูลการพยายามฆ่าตัวตาย จังหวัดอุบล ปี 2565-2567
```

---

# Core Concept

ระบบนี้ไม่ใช่แค่ AI ตอบคำถาม

แต่เป็น:

```text
Explainable AI Analyst
```

ที่สามารถแสดง:

- วิธีคิด
- ขั้นตอนวิเคราะห์
- dataset ที่ใช้
- code ที่ generate
- ผลลัพธ์จริง
- insight สุดท้าย

---

# High-Level Workflow

```text
User Question
    ↓
Reasoning Narrator Agent
    ↓
File Finder Agent
    ↓
Schema Analyst Agent
    ↓
Python Code Generator Agent
    ↓
Python Execution Tool
    ↓
Insight Analyst Agent
    ↓
Final Answer
```

---

# Recommended Multi-Agent Architecture

```text
┌──────────────────────────┐
│ Reasoning Narrator       │
│ อธิบายขั้นตอนวิเคราะห์   │
└─────────────┬────────────┘
              ↓
┌──────────────────────────┐
│ File Finder Agent        │
│ ค้นหา dataset            │
└─────────────┬────────────┘
              ↓
┌──────────────────────────┐
│ Schema Analyst Agent     │
│ วิเคราะห์ schema         │
└─────────────┬────────────┘
              ↓
┌──────────────────────────┐
│ Python Generator Agent   │
│ สร้าง Pandas Code        │
└─────────────┬────────────┘
              ↓
┌──────────────────────────┐
│ Python Executor Tool     │
│ รันโค้ดจริง              │
└─────────────┬────────────┘
              ↓
┌──────────────────────────┐
│ Insight Analyst Agent    │
│ วิเคราะห์ผลลัพธ์         │
└──────────────────────────┘
```

---

# Detailed Workflow

---

# STEP 1 — User Question

ผู้ใช้ถาม:

```text
ขอข้อมูลการพยายามฆ่าตัวตาย จังหวัดอุบล ปี 2565-2567
```

---

# STEP 2 — AI Reasoning

AI อธิบายสิ่งที่กำลังทำ

ตัวอย่าง:

```text
กำลังค้นหา dataset ที่เกี่ยวข้องกับ:
- การพยายามฆ่าตัวตาย
- จังหวัดอุบลราชธานี
- ปี 2565-2567
```

---

# STEP 3 — File Finder Agent

AI ค้นหาไฟล์ที่เกี่ยวข้อง

ตัวอย่าง:

```text
พบไฟล์:
D2_Mental Health/ฆ่าตัวตาย/merged_suicide_attempts_all_5_provinces_2022_2025.csv
```

---

# STEP 4 — Schema Analyst Agent

AI อ่าน schema ของ dataset

ตัวอย่าง:

```python
df.columns
```

ผลลัพธ์:

```python
[
    "province",
    "year",
    "male",
    "female",
    "total",
    "sub-province"
]
```

---

# STEP 5 — Analysis Plan

AI อธิบายแผนการวิเคราะห์

```text
แผนการวิเคราะห์:
1. Filter จังหวัดอุบลราชธานี
2. Filter ปี 2565-2567
3. Aggregate รายปี
4. วิเคราะห์อำเภอที่พบมากที่สุด
```

---

# STEP 6 — Python Code Generation

AI สร้าง Python/Pandas code

ตัวอย่าง:

```python
import pandas as pd

df = pd.read_csv(
    'merged_suicide_attempts_all_5_provinces_2022_2025.csv'
)

ubon_df = df[
    (df['province'] == 'อุบลราชธานี') &
    (df['year'].isin([2022, 2023, 2024]))
]

summary_by_year = (
    ubon_df.groupby('year')[['male', 'female', 'total']]
    .sum()
    .reset_index()
)

summary_by_sub = (
    ubon_df.groupby('sub-province')[['total']]
    .sum()
    .sort_values(by='total', ascending=False)
    .reset_index()
)

print(summary_by_year)
print(summary_by_sub.head())
```

---

# STEP 7 — Python Execution

ระบบรันโค้ดจริง

ผลลัพธ์:

```text
Summary by year:
   year   male  female  total
0  2022  207.0   227.0  434.0
1  2023  252.0   231.0  483.0
2  2024  275.0   288.0  563.0
```

---

# STEP 8 — Insight Analysis

AI วิเคราะห์ผลลัพธ์

ตัวอย่าง:

```text
จำนวนผู้พยายามฆ่าตัวตายในจังหวัดอุบลราชธานี
เพิ่มขึ้นต่อเนื่องจาก 434 รายในปี 2565
เป็น 563 รายในปี 2567

พื้นที่ที่พบมากที่สุดคืออำเภอเมืองอุบลราชธานี
```

---

# Recommended UI/UX

ควรแสดงผลลัพธ์แบบนี้:

```text
[🧠 AI Reasoning]
กำลังค้นหา dataset...

[📂 Selected Dataset]
merged_suicide_attempts_all_5_provinces_2022_2025.csv

[📊 Analysis Plan]
1. Filter จังหวัด
2. Filter ปี
3. Aggregate

[🐍 Generated Python Code]
import pandas as pd
...

[⚡ Execution Result]
...

[📈 Final Insight]
...
```

---

# Recommended CrewAI Process

แนะนำ:

```python
Process.sequential
```

เหตุผล:
- แต่ละขั้นต้องใช้ output ของขั้นก่อนหน้า
- workflow เป็นลำดับชัดเจน

---

# Recommended Agents

---

## 1. Reasoning Narrator Agent

### Goal

```text
Explain analysis process to user
```

### Responsibilities

- อธิบาย workflow
- อธิบาย reasoning
- แสดงขั้นตอนการวิเคราะห์

---

## 2. File Finder Agent

### Goal

```text
Find the most relevant dataset
```

### Responsibilities

- ค้นหาไฟล์
- วิเคราะห์ชื่อไฟล์
- เลือก dataset

### Tools

- DirectorySearchTool
- FileReadTool

---

## 3. Schema Analyst Agent

### Goal

```text
Understand dataset schema
```

### Responsibilities

- อ่าน columns
- อ่าน sample rows
- วิเคราะห์ data types

---

## 4. Python Code Generator Agent

### Goal

```text
Generate executable pandas code
```

### Responsibilities

- filter data
- aggregate data
- summarize data
- generate analysis code

---

## 5. Python Executor Tool

### Responsibilities

- รัน Python code
- return stdout
- return dataframe
- handle errors

---

## 6. Insight Analyst Agent

### Goal

```text
Explain findings in Thai language
```

### Responsibilities

- วิเคราะห์ trend
- สรุป insight
- เขียนรายงาน

---

# Recommended Output Format

แนะนำใช้ Markdown

ตัวอย่าง:

````md
# 🧠 AI Reasoning

กำลังค้นหา dataset ที่เกี่ยวข้อง...

# 📂 Selected Dataset

merged_suicide_attempts_all_5_provinces_2022_2025.csv

# 📊 Analysis Plan

1. Filter จังหวัดอุบลราชธานี
2. Filter ปี 2565-2567
3. Aggregate รายปี

# 🐍 Generated Python Code

```python
...
```

# ⚡ Execution Result

```text
...
```

# 📈 Final Insight

...