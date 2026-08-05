"""Employee Central entities extracted by the pipeline.

Each entry maps the OData entity set name to the $select projection and
the business key used when merging records into the curated layer.
"""

ENTITIES = {
    "PerPerson": {
        "select": "personIdExternal,userId,lastModifiedDateTime",
        "key": "personIdExternal",
    },
    "PerPersonal": {
        "select": "personIdExternal,firstName,lastName,gender,lastModifiedDateTime",
        "key": "personIdExternal",
    },
    "EmpEmployment": {
        "select": "userId,personIdExternal,startDate,lastModifiedDateTime",
        "key": "userId",
    },
    "EmpJob": {
        "select": (
            "userId,jobTitle,department,division,location,managerId,"
            "startDate,lastModifiedDateTime"
        ),
        "key": "userId",
    },
    "FODepartment": {
        "select": "externalCode,name,status,lastModifiedDateTime",
        "key": "externalCode",
    },
}
