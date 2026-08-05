"""Deterministic sample Employee Central dataset for the mock server.

Internal date fields use the `_ms` suffix (epoch milliseconds); the server
serializes them as SuccessFactors legacy JSON dates: "/Date(1691200000000)/".
"""

import random
from datetime import datetime, timedelta, timezone

random.seed(42)

EMPLOYEE_COUNT = 120

FIRST_NAMES = [
    "Aiden", "Bela", "Carlos", "Divya", "Elena", "Farid", "Grace", "Hiro",
    "Ines", "Jonas", "Kavya", "Liam", "Mona", "Nikhil", "Olga", "Pedro",
    "Quinn", "Rosa", "Sanjay", "Tara",
]
LAST_NAMES = [
    "Almeida", "Brandt", "Chen", "Dutta", "Eriksen", "Fischer", "Gupta",
    "Haas", "Iyer", "Jansen", "Kim", "Lopez", "Mehta", "Novak", "Okafor",
]
DEPARTMENTS = [
    ("DEP_ENG", "Engineering", "Technology"),
    ("DEP_SLS", "Sales", "Go-To-Market"),
    ("DEP_HR", "Human Resources", "Corporate"),
    ("DEP_FIN", "Finance", "Corporate"),
    ("DEP_MKT", "Marketing", "Go-To-Market"),
    ("DEP_OPS", "Operations", "Technology"),
]
JOB_TITLES = [
    "Software Engineer", "Senior Software Engineer", "Account Executive",
    "HR Business Partner", "Financial Analyst", "Marketing Specialist",
    "Operations Manager", "Data Engineer", "Product Manager",
]
LOCATIONS = ["Berlin", "Amsterdam", "Bangalore", "New York", "London"]

_BASE = datetime(2026, 6, 1, tzinfo=timezone.utc)


def _millis(dt):
    return int(dt.timestamp() * 1000)


def _modified(i):
    """Spread lastModifiedDateTime over ~60 days so delta filters bite."""
    return _millis(_BASE + timedelta(days=i % 60, hours=i % 24))


def _build():
    per_person, per_personal, emp_employment, emp_job = [], [], [], []
    for i in range(1, EMPLOYEE_COUNT + 1):
        person_id = f"P{i:05d}"
        user_id = f"U{i:05d}"
        first = FIRST_NAMES[i % len(FIRST_NAMES)]
        last = LAST_NAMES[i % len(LAST_NAMES)]
        dept_code, dept_name, division = DEPARTMENTS[i % len(DEPARTMENTS)]
        hired = _BASE - timedelta(days=random.randint(90, 2000))
        modified = _modified(i)

        per_person.append({
            "personIdExternal": person_id,
            "userId": user_id,
            "lastModifiedDateTime_ms": modified,
        })
        per_personal.append({
            "personIdExternal": person_id,
            "firstName": first,
            "lastName": last,
            "gender": random.choice(["M", "F", "U"]),
            "lastModifiedDateTime_ms": modified,
        })
        emp_employment.append({
            "userId": user_id,
            "personIdExternal": person_id,
            "startDate_ms": _millis(hired),
            "lastModifiedDateTime_ms": modified,
        })
        emp_job.append({
            "userId": user_id,
            "jobTitle": JOB_TITLES[i % len(JOB_TITLES)],
            "department": dept_name,
            "division": division,
            "location": random.choice(LOCATIONS),
            "managerId": f"U{max(1, i // 10):05d}",
            "startDate_ms": _millis(hired),
            "lastModifiedDateTime_ms": modified,
        })

    fo_department = [
        {
            "externalCode": code,
            "name": name,
            "status": "A",
            "lastModifiedDateTime_ms": _modified(idx),
        }
        for idx, (code, name, _division) in enumerate(DEPARTMENTS)
    ]
    return {
        "PerPerson": per_person,
        "PerPersonal": per_personal,
        "EmpEmployment": emp_employment,
        "EmpJob": emp_job,
        "FODepartment": fo_department,
    }


DATASET = _build()
